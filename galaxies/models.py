import os
import json
import re
import shutil
from urllib.parse import quote_plus

import astropy.units as u
import constants
import numpy as np
import pandas as pd
import requests
import utils
from astropy.coordinates import SkyCoord
from astropy.io import ascii
from astropy.table import Table
from astroquery.exceptions import RemoteServiceError, TableParseError
from astroquery.ned import Ned
from astroquery.simbad import Simbad
from bs4 import BeautifulSoup
from django.contrib.auth.models import User
from django.core.exceptions import MultipleObjectsReturned, ObjectDoesNotExist
from django.db import models
from django.db.models import Q
from tqdm import tqdm

# Create your models here.


class CatalogEntry(models.Model):
    """
    Abstract for making generic methods
    """

    galaxy = models.ForeignKey("galaxies.Galaxy", null=True, on_delete=models.SET_NULL)
    ra = models.FloatField()
    dec = models.FloatField()

    class Meta:
        abstract = True

    def save(self, *args, **kwargs):
        if hasattr(self, "target") and self.target:
            self.galaxy = None
        elif not hasattr(self, "galaxy") or not self.galaxy:
            g = Galaxy.objects.create(ra=self.ra, dec=self.dec)
            self.galaxy = g
            g.z, g.z_err, g.z_flag = g.get_redshift(query=False)
        super(CatalogEntry, self).save(*args, **kwargs)


class HsfEntry(CatalogEntry):
    HSF_Z_CHOICES = [
        ("s", "Strong features, reliable redshift"),
        ("w", "Weak features, unreliable redshift"),
        ("n", "No features"),
        ("?", "Uninspected"),
    ]
    date = models.DateField()
    templates_used = models.ManyToManyField("data.TemplateSpectrum")
    z = models.FloatField(null=True)
    z_err = models.FloatField(null=True)
    z_flag = models.CharField(max_length=1, choices=HSF_Z_CHOICES, default="?")
    plus = models.FloatField(null=True)
    minus = models.FloatField(null=True)
    r = models.FloatField(null=True)

    class Meta:
        abstract = True

    def prep_spectrum(self):
        pass

    def weightedCC(
        self,
        template_spectra_qset="sdss",
        zmin=0.01,
        zmax=0.1,
        dz=None,
        pull_cut=200,
        binAngstroms=0,
        plot=False,
        show=False,
        verbose=False,
        convert_to_helio=True,
        outdir=constants.DATA_DIR + "/reduced_spectra",
    ):
        import shutil

        from data.models import TemplateSpectrum
        from data.weightedCC import weightedCC, weightedCC_mesh

        if template_spectra_qset in ("bc95", "brown", "kc96", "sdss", "swire"):
            template_spectra_qset = TemplateSpectrum.objects.filter(
                path__contains=template_spectra_qset
            )

        inspec = self.prep_spectra()
        if dz is None:
            z, plus, minus, r, d, _ = weightedCC_mesh(
                inspec=inspec,
                intemp=[ts.path for ts in template_spectra_qset],
                zmin=zmin,
                zmax=zmax,
                pull_cut=pull_cut,
                binAngstroms=binAngstroms,
                plot=plot,
                show=show,
                verbose=verbose,
            )
        else:
            z, plus, minus, r, d = weightedCC(
                inspec=inspec,
                intemp=[ts.path for ts in template_spectra_qset],
                zmin=zmin,
                zmax=zmax,
                dz=dz,
                pull_cut=pull_cut,
                binAngstroms=binAngstroms,
                plot=plot,
                show=show,
                verbose=verbose,
            )
        if convert_to_helio:
            z = utils.convert_z(
                z, self.ra, self.dec, z0="obs", z1="hel", mjd=self.spectrum.mjd
            )
        if plot and outdir is None:
            os.remove("default_wcc.png")
            os.remove("default_comparison.png")
        elif plot and outdir is not None:
            shutil.move(
                "default_wcc.png",
                f"{outdir}/{self.galaxy.targets.first().TNS_name}_wcc.png",
            )
            shutil.move(
                "default_comparison.png",
                f"{outdir}/{self.galaxy.targets.first().TNS_name}_comparison.png",
            )
        return z, plus, minus, r, d

    def plot(self, *args, **kwargs):
        if hasattr(self.spectrum, "plot"):
            return self.spectrum.plot(*args, **kwargs)


class SnifsEntry(HsfEntry):
    spectrum = models.ForeignKey("data.SnifsSpectrum", on_delete=models.CASCADE)
    reduction_date = models.DateField()

    class Meta:
        default_related_name = "snifs_entries"
        constraints = [
            models.UniqueConstraint(name="snifs_spectrum", fields=["spectrum"])
        ]

    def __str__(self):
        return self.exp_str

    @property
    def exp_str(self):
        return self.spectrum.exp_str

    def prep_spectra(self):
        data_path = (
            f"{constants.DATA_DIR}/reduced_spectra/"
            f"{self.galaxy.targets.first().TNS_name}.dat"
        )
        shutil.copy(self.spectrum.path, data_path)
        inspec = []
        with open(data_path, "r") as f:
            lines = f.readlines()
        for i in ["blue", "red"]:
            with open(data_path.replace(".dat", f"_{i}.dat"), "w") as f:
                if i[0].upper() not in lines[3]:  # CHANNEL = B+R
                    continue
                f.write("wave flux err\n")
                for line in lines[15:]:  # data
                    if "nan" in line:
                        continue
                    wl = float(line.split()[0])
                    if i == "blue":
                        outside_dichroic = wl < constants.SNIFS_DICHROIC[0]
                    elif i == "red":
                        outside_dichroic = wl > constants.SNIFS_DICHROIC[1]
                    outside_tellurics = True
                    for t_mask in constants.TELLURICS.values():
                        outside_tellurics *= wl < t_mask[0] or wl > t_mask[1]
                    if outside_dichroic and outside_tellurics:
                        f.write(line)
            inspec.append(data_path.replace(".dat", f"_{i}.dat"))
        return inspec


class FocasEntry(HsfEntry):
    spectrum = models.ForeignKey("data.FocasSpectrum", on_delete=models.CASCADE)

    class Meta:
        default_related_name = "focas_entries"
        constraints = [
            models.UniqueConstraint(name="focas_spectrum", fields=["spectrum"])
        ]

    def prep_spectra(self):
        data_path = (
            f"{constants.DATA_DIR}/reduced_spectra/"
            f"{self.galaxy.targets.first().TNS_name}_focas.dat"
        )
        inspec = []
        with open(data_path, "r") as f:
            lines = f.readlines()
        with open(data_path.replace(".dat", ".masked.dat"), "w") as f:
            f.write("wave flux\n")
            for line in lines:  # data
                if "nan" in line:
                    continue
                wl = float(line.split()[0])
                outside_tellurics = True
                for t_mask in constants.TELLURICS.values():
                    outside_tellurics *= wl < t_mask[0] or wl > t_mask[1]
                if outside_tellurics:
                    f.write(line)
        inspec.append(data_path.replace(".dat", ".masked.dat"))
        return inspec


class GladeEntry(CatalogEntry):
    TYPE_CHOICES = [
        ("Q", "the source is from the SDSS-DR16Q catalog"),
        ("G", "Not identified as a quasar."),
    ]
    DIST_CHOICES = [
        ("0", "the galaxy has no measured redshift or distance value"),
        (
            "1",
            "it has a measured photometric redshift from which we have calculated"
            "its luminosity distance",
        ),
        (
            "2",
            "it has a measured luminosity distance value from which we have calculated"
            "its redshift",
        ),
        (
            "3",
            "it has a measured spectroscopic redshift from which we have calculated"
            "its luminosity distance",
        ),
    ]
    glade_no = models.IntegerField(primary_key=True)
    pgc_no = models.IntegerField(null=True)
    gwgc_name = models.CharField(max_length=28, null=True)
    hyperleda_name = models.CharField(max_length=29, null=True)
    twomass_name = models.CharField(max_length=16, null=True)
    wisexscos_name = models.CharField(max_length=19, null=True)
    sdss_dr16q_name = models.CharField(max_length=18, null=True)
    object_type_flag = models.CharField(max_length=1, choices=TYPE_CHOICES, null=True)
    b = models.FloatField(null=True)
    b_err = models.FloatField(null=True)
    b_flag = models.BooleanField(default=False)
    b_abs = models.FloatField(null=True)
    j = models.FloatField(null=True)
    j_err = models.FloatField(null=True)
    h = models.FloatField(null=True)
    h_err = models.FloatField(null=True)
    k = models.FloatField(null=True)
    k_err = models.FloatField(null=True)
    w1 = models.FloatField(null=True)
    w1_err = models.FloatField(null=True)
    w2 = models.FloatField(null=True)
    w2_err = models.FloatField(null=True)
    w1_flag = models.BooleanField(default=False)
    b_j = models.FloatField(null=True)
    b_j_err = models.FloatField(null=True)
    z_helio = models.FloatField(null=True)
    z_cmb = models.FloatField(null=True)
    z_flag = models.BooleanField(default=0, null=True)
    v_err = models.FloatField(null=True)
    z_err = models.FloatField(null=True)
    d_l = models.FloatField(null=True)
    d_l_err = models.FloatField(null=True)
    dist_flag = models.CharField(max_length=1, choices=DIST_CHOICES)
    m_sun = models.FloatField(null=True)
    m_sun_err = models.FloatField(null=True)
    m_sun_flag = models.BooleanField(default=0, null=True)
    merger_rate = models.FloatField(null=True)
    merger_rate_error = models.FloatField(null=True)

    class Meta:
        default_related_name = "glade_entries"

    @property
    def names(self):
        return (
            self.glade_no,
            self.pgc_no,
            self.gwgc_name,
            self.hyperleda_name,
            self.twomass_name,
            self.wisexscos_name,
            self.sdss_dr16q_name,
        )

    def get_redshift(self):
        if self.dist_flag in ("0", "2"):
            return (None, None, "n", None)
        if self.dist_flag == "1":
            z_flag = "p"
        elif self.dist_flag == "3":
            z_flag = "sp"
        if self.z_err:
            z_flag += "u"
        return (self.z_helio, self.z_err, z_flag, f"glade_{self.glade_no}")

    def get_mag(self, bandpass="b"):
        if hasattr(self, bandpass):
            return getattr(self, bandpass)
        else:
            return None


class PanStarrsEntry(CatalogEntry):
    """PanStarrsEntry.

    for field names see https://outerspace.stsci.edu/display/PANSTARRS/PS1+ObjectThin+table+fields
    and https://outerspace.stsci.edu/display/PANSTARRS/PS1+StackObjectThin+table+fields
    for flags see https://outerspace.stsci.edu/display/PANSTARRS/PS1+Object+Flags
    """

    objname = models.CharField(max_length=32)
    objid = models.PositiveBigIntegerField(primary_key=True)
    objaltname1 = models.CharField(max_length=32, null=True)
    objaltname2 = models.CharField(max_length=32, null=True)
    objaltname3 = models.CharField(max_length=32, null=True)
    ramean = models.FloatField()
    decmean = models.FloatField()
    ndetections = models.SmallIntegerField(null=True)
    primarydetection = models.PositiveSmallIntegerField(default=255)
    objinfoflag = models.PositiveIntegerField(default=0)
    qualityflag = models.PositiveSmallIntegerField(default=0)
    ng = models.SmallIntegerField(null=True)
    nr = models.SmallIntegerField(null=True)
    ni = models.SmallIntegerField(null=True)
    nz = models.SmallIntegerField(null=True)
    ny = models.SmallIntegerField(null=True)
    gpsfmag = models.FloatField(null=True)
    gpsfmagerr = models.FloatField(null=True)
    gkronmag = models.FloatField(null=True)
    gkronmagerr = models.FloatField(null=True)
    gfpsfflux = models.FloatField(null=True)
    gfpsffluxerr = models.FloatField(null=True)
    gfkronflux = models.FloatField(null=True)
    gfkronfluxerr = models.FloatField(null=True)
    rpsfmag = models.FloatField(null=True)
    rpsfmagerr = models.FloatField(null=True)
    rkronmag = models.FloatField(null=True)
    rkronmagerr = models.FloatField(null=True)
    rfpsfflux = models.FloatField(null=True)
    rfpsffluxerr = models.FloatField(null=True)
    rfkronflux = models.FloatField(null=True)
    rfkronfluxerr = models.FloatField(null=True)
    ipsfmag = models.FloatField(null=True)
    ipsfmagerr = models.FloatField(null=True)
    ikronmag = models.FloatField(null=True)
    ikronmagerr = models.FloatField(null=True)
    ifpsfflux = models.FloatField(null=True)
    ifpsffluxerr = models.FloatField(null=True)
    ifkronflux = models.FloatField(null=True)
    ifkronfluxerr = models.FloatField(null=True)
    zpsfmag = models.FloatField(null=True)
    zpsfmagerr = models.FloatField(null=True)
    zkronmag = models.FloatField(null=True)
    zkronmagerr = models.FloatField(null=True)
    zfpsfflux = models.FloatField(null=True)
    zfpsffluxerr = models.FloatField(null=True)
    zfkronflux = models.FloatField(null=True)
    zfkronfluxerr = models.FloatField(null=True)
    ypsfmag = models.FloatField(null=True)
    ypsfmagerr = models.FloatField(null=True)
    ykronmag = models.FloatField(null=True)
    ykronmagerr = models.FloatField(null=True)
    yfpsfflux = models.FloatField(null=True)
    yfpsffluxerr = models.FloatField(null=True)
    yfkronflux = models.FloatField(null=True)
    yfkronfluxerr = models.FloatField(null=True)

    class Meta:
        default_related_name = "ps1_entries"

    def get_mag(self, bandpass="g"):
        if hasattr(self, f"{bandpass}psfmag"):
            return getattr(self, f"{bandpass}psfmag")
        else:
            return None


class SimbadEntry(CatalogEntry):
    """SimbadEntry.

    see https://simbad.u-strasbg.fr/Pages/guide/sim-fscript.htx
    wavelengths: R=Radio, m=mm, s=sub-mm, F=FIR, I=IR, N=NIR, O=Optical, U=UV, g=gamma
    angular major/minor axes 25 mag isophots
    """

    target = models.ForeignKey("targets.Target", null=True, on_delete=models.SET_NULL)

    objid = models.BigIntegerField(null=True)
    main_id = models.CharField(max_length=39, primary_key=True)
    otype_3 = models.CharField(max_length=3, choices=constants.SIMBAD_TYPE_CHOICES)
    all_types = models.JSONField(default=list)
    diameter = models.IntegerField(default=0)
    distance = models.IntegerField(default=0)
    velocities = models.IntegerField(default=0)

    coo_err_maja = models.FloatField(null=True)
    coo_err_mina = models.FloatField(null=True)
    coo_err_angle = models.IntegerField(null=True)
    coo_qual = models.CharField(max_length=1, null=True)
    coo_wavelength = models.CharField(max_length=1, null=True)
    coo_bibcode = models.CharField(max_length=19, null=True)
    rvz_type = models.CharField(max_length=1, null=True)
    rvz_radvel = models.FloatField(null=True)
    rvz_error = models.FloatField(null=True)
    rvz_qual = models.CharField(max_length=1, null=True)
    rvz_wavelength = models.CharField(max_length=1, null=True)
    rvz_bibcode = models.CharField(max_length=19, null=True)
    galdim_majaxis = models.FloatField(null=True)
    galdim_minaxis = models.FloatField(null=True)
    galdim_angle = models.IntegerField(null=True)
    galdim_qual = models.CharField(max_length=1, null=True)
    galdim_wavelength = models.CharField(max_length=15, null=True)
    galdim_bibcode = models.CharField(max_length=19, null=True)
    distance_distance = models.FloatField(null=True)
    distance_q = models.CharField(max_length=1, null=True)
    distance_unit = models.CharField(max_length=3, null=True)
    distance_merr = models.FloatField(null=True)
    distance_perr = models.FloatField(null=True)
    distance_method = models.CharField(max_length=8, null=True)
    distance_bibcode = models.CharField(max_length=19, null=True)
    ids = models.TextField()
    morph_type = models.CharField(max_length=6, null=True)
    morph_qual = models.CharField(max_length=1, null=True)
    morph_bibcode = models.CharField(max_length=19, null=True)
    flux_b = models.FloatField(null=True)
    flux_error_b = models.FloatField(null=True)
    flux_system_b = models.CharField(max_length=4, null=True)
    flux_bibcode_b = models.CharField(max_length=19, null=True)
    flux_var_b = models.CharField(max_length=2, null=True)
    flux_mult_b = models.CharField(max_length=1, null=True)
    flux_qual_b = models.CharField(max_length=3, null=True)
    flux_unit_b = models.CharField(max_length=3, null=True)
    flux_v = models.FloatField(null=True)
    flux_error_v = models.FloatField(null=True)
    flux_system_v = models.CharField(max_length=4, null=True)
    flux_bibcode_v = models.CharField(max_length=19, null=True)
    flux_var_v = models.CharField(max_length=2, null=True)
    flux_mult_v = models.CharField(max_length=1, null=True)
    flux_qual_v = models.CharField(max_length=3, null=True)
    flux_unit_v = models.CharField(max_length=3, null=True)
    flux_j = models.FloatField(null=True)
    flux_error_j = models.FloatField(null=True)
    flux_system_j = models.CharField(max_length=4, null=True)
    flux_bibcode_j = models.CharField(max_length=19, null=True)
    flux_var_j = models.CharField(max_length=2, null=True)
    flux_mult_j = models.CharField(max_length=1, null=True)
    flux_qual_j = models.CharField(max_length=3, null=True)
    flux_unit_j = models.CharField(max_length=3, null=True)
    flux_g = models.FloatField(null=True)
    flux_error_g = models.FloatField(null=True)
    flux_system_g = models.CharField(max_length=4, null=True)
    flux_bibcode_g = models.CharField(max_length=19, null=True)
    flux_var_g = models.CharField(max_length=2, null=True)
    flux_mult_g = models.CharField(max_length=1, null=True)
    flux_qual_g = models.CharField(max_length=3, null=True)
    flux_unit_g = models.CharField(max_length=3, null=True)
    flux_r = models.FloatField(null=True)
    flux_error_r = models.FloatField(null=True)
    flux_system_r = models.CharField(max_length=4, null=True)
    flux_bibcode_r = models.CharField(max_length=19, null=True)
    flux_var_r = models.CharField(max_length=2, null=True)
    flux_mult_r = models.CharField(max_length=1, null=True)
    flux_qual_r = models.CharField(max_length=3, null=True)
    flux_unit_r = models.CharField(max_length=3, null=True)
    flux_i = models.FloatField(null=True)
    flux_error_i = models.FloatField(null=True)
    flux_system_i = models.CharField(max_length=4, null=True)
    flux_bibcode_i = models.CharField(max_length=19, null=True)
    flux_var_i = models.CharField(max_length=2, null=True)
    flux_mult_i = models.CharField(max_length=1, null=True)
    flux_qual_i = models.CharField(max_length=3, null=True)
    flux_unit_i = models.CharField(max_length=3, null=True)
    script_number_id = models.IntegerField(null=True)

    class Meta:
        default_related_name = "simbad_entries"

    def get_redshift(self, clip=True, average=True, query=True):
        # parse processed redshift data
        z, z_err, z_flag, bibcode = (
            self.rvz_radvel,
            self.rvz_error,
            "n",
            self.rvz_bibcode,
        )
        if z and self.rvz_type in ("v", "cz", "c"):
            z /= constants.C
            z_flag = "l"
        if z_err and self.rvz_type in ("v", "cz", "c"):
            z_err /= constants.C
            z_flag = "lu"
        if not self.redshifts.exists():
            if self.velocities == 0 or not query:
                return z, z_err, z_flag, bibcode
            elif query:
                self.query_measurements()

        zsets = {}
        zsets["l"] = self.redshifts.all()
        if zsets["l"].exists():
            z_flag = "l"
        else:
            return z, z_err, z_flag, bibcode
        zsets["p"] = zsets["l"].filter(nat__startswith="p")
        if zsets["p"].exists():
            z_flag = "p"
        zsets["sp"] = zsets["l"].filter(nat__startswith="s")
        if zsets["sp"].exists():
            z_flag = "sp"

        zset = zsets[z_flag]
        if zset.filter(me__gt=0).exists():
            zset = zset.filter(me__gt=0)
            z_flag += "u"

        z = np.array([], dtype=float)
        z_err = np.array([], dtype=float)
        bibcodes = np.array([], dtype="<U19")
        for entry in zset:
            z = np.append(z, entry.z)
            if entry.z_err:
                z_err = np.append(z_err, entry.z_err)
            else:
                z_err = np.append(z_err, 99)
            bibcodes = np.append(bibcodes, entry.bibcode)
        if clip:
            z, z_err, idx, _ = utils.sigmaclip(z, z_err)
            bibcodes = bibcodes[idx]
        if average:
            z = np.average(z, weights=1 / z_err)
            z_err = np.average(z_err, weights=1 / z_err)
        return z, z_err, z_flag, bibcodes

    def get_mag(self, bandpass="g"):
        if hasattr(self, f"flux_{bandpass}"):
            return getattr(self, f"flux_{bandpass}")
        else:
            return None

    def query_measurements(self):
        soup = utils.simbad_object_search(self.objid, self.main_id)
        basic_data = soup.find("td", id="basic_data").find("table")
        all_types = {}
        for row in basic_data.find_all("tr"):
            if "Other object types" in row.text:
                tts = row.find_all("tt")
                for i in range(len(tts) // 2):
                    all_types[tts[2 * i + 1].text.strip("\n ")] = (
                        tts[2 * i].get("title").split(",")
                    )

        def parse_row(row, table_type):
            if table_type not in ("velocities", "distance", "diameter"):
                return
            cells = row.split("|")
            floats = []
            ints = []
            if table_type == "velocities":
                object_manager = self.redshifts
                floats = ["value", "me", "res"]
                ints = [
                    "nmes",
                ]
                # handling cells separated by |
                defaults = {
                    "objtype": cells[1],
                    "date": cells[5],
                    "rem": cells[6],
                    "origin": cells[7],
                    "bibcode": cells[8],
                }
                defaults["acc"], defaults["nmes"] = cells[3].split("(")
                defaults["nmes"] = defaults["nmes"].strip(")")
                defaults["nat"], defaults["q"], defaults["dom"], res_d = cells[4].split(
                    ","
                )

                # handling intracell entries with no delims
                value_r_me = cells[2].split()
                defaults["value"] = value_r_me.pop(0)
                rds = res_d.split()
                for one, multi, sp in zip(("r", "d"), ("me", "res"), (value_r_me, rds)):
                    defaults[one] = ""
                    defaults[multi] = ""
                    if len(sp) == 1:
                        tmp = sp[0].strip()
                        if len(tmp) == 1:
                            defaults[one] = tmp
                        else:
                            defaults[multi] = tmp
                    elif len(sp) == 2:
                        if one == "r":
                            defaults[one], defaults[multi] = sp
                        elif one == "d":
                            defaults[multi], defaults[one] = sp
                if defaults["r"] not in ("?", ":"):
                    defaults["r"] = ""
                    defaults["me"] = defaults["r"]
            elif table_type == "distance":
                object_manager = self.distances
                floats = ["value", "merr", "perr"]
                merr_perr = cells[2].split()
                if len(merr_perr) == 2:
                    merr = merr_perr[0]
                    perr = merr_perr[1]
                elif len(merr_perr) == 1:
                    merr = merr_perr[0]
                    perr = merr_perr[0]
                else:
                    merr = ""
                    perr = ""
                defaults = {
                    "merr": merr,
                    "perr": perr,
                    "method": cells[3],
                    "bibcode": cells[4],
                }
            elif table_type == "diameter":
                object_manager = self.diameters
                floats = ["value", "error"]
                defaults = {
                    "error": cells[2],
                    "filt": cells[3],
                    "method": cells[4],
                    "bibcode": cells[5],
                }
            if table_type in ("distance", "diameter"):
                dist_q_unit = cells[1].split()
                defaults["value"] = dist_q_unit.pop(0)
                defaults["unit"] = dist_q_unit.pop(-1)
                if dist_q_unit:
                    defaults["q"] = dist_q_unit[0]
                else:
                    defaults["q"] = ""

            for key, val in defaults.items():
                val = val.strip()
                if val == "":
                    defaults[key] = None
                elif key in floats and val:
                    defaults[key] = float(val)
                elif key in ints and val:
                    defaults[key] = int(val)
                elif val:
                    defaults[key] = val

            object_manager.create(**defaults)

        pre_idx = []
        p_pre = soup.find_all(["p", "pre"])
        for i, element in enumerate(p_pre):
            if element.name == "pre":
                pre_idx.append(i)
        for i in pre_idx:
            table = p_pre[i].text.strip("\n ").split("\n")
            table_type = p_pre[i - 2].find("b").text.strip("\n ")
            if table_type == "velocities":
                self.redshifts.all().delete()
            elif table_type == "distance":
                self.distances.all().delete()
            elif table_type == "diameters":
                self.diameters.all().delete()
            for row in table[2:]:
                parse_row(row, table_type)


class SimbadSubTable(models.Model):
    """SimbadSubTable.

    for field names see https://simbad.cds.unistra.fr/simbad/sim-display?data=meas
    """

    entry = models.ForeignKey(SimbadEntry, on_delete=models.CASCADE)
    q = models.CharField(max_length=1, null=True)
    bibcode = models.CharField(max_length=19, null=True)
    value = models.FloatField()

    class Meta:
        abstract = True


class SimbadRedshift(SimbadSubTable):
    objtype = models.CharField(max_length=3)
    r = models.CharField(max_length=1, null=True)
    me = models.FloatField(null=True)
    acc = models.CharField(max_length=1, null=True)
    nmes = models.IntegerField(null=True)
    nat = models.CharField(max_length=2, null=True)
    dom = models.CharField(max_length=4, null=True)
    res = models.FloatField(null=True)
    d = models.CharField(max_length=1, null=True)
    date = models.FloatField(null=True)
    rem = models.CharField(max_length=7, null=True)
    origin = models.CharField(max_length=2, null=True)

    class Meta:
        default_related_name = "redshifts"

    @property
    def z(self):
        if self.objtype in ("v", "cz"):
            return self.value / constants.C
        else:
            return self.value

    @property
    def z_err(self):
        if not self.me:
            return None
        if self.objtype in ("v", "cz"):
            return self.me / constants.C
        else:
            return self.me


class SimbadDiameter(SimbadSubTable):
    unit = models.CharField(max_length=4)
    error = models.FloatField(null=True)
    filt = models.CharField(max_length=8, null=True)
    method = models.CharField(max_length=8, null=True)

    class Meta:
        default_related_name = "diameters"


class SimbadDistance(SimbadSubTable):
    unit = models.CharField(max_length=4)
    merr = models.FloatField(null=True)
    perr = models.FloatField(null=True)
    method = models.CharField(max_length=8, null=True)

    class Meta:
        default_related_name = "distances"


class NedEntry(CatalogEntry):
    """NedEntry.

    Contains data from the tables produced by  utils.ned_cone_search()
    aliases field must be fetched individually with get_aliases()
    see https://ned.ipac.caltech.edu/tap/sync?QUERY=SELECT+*+FROM+TAP_SCHEMA.columns+WHERE+table_name=%27NEDTAP.objdir%27&REQUEST=doQuery&LANG=ADQL&FORMAT=text
    """

    target = models.ForeignKey("targets.Target", null=True, on_delete=models.SET_NULL)

    objid = models.BigIntegerField(primary_key=True)
    prefname = models.CharField(max_length=32)
    pretype = models.CharField(max_length=6, choices=constants.NED_TYPE_CHOICES)
    z = models.FloatField(null=True)
    zunc = models.FloatField(null=True)
    zflag = models.CharField(max_length=5, choices=constants.NED_Z_CHOICES, default="-")
    n_crosref = models.IntegerField(default=0)
    n_notes = models.IntegerField(default=0)
    n_gphot = models.IntegerField(default=0)
    n_posd = models.IntegerField(default=0)
    n_zdf = models.IntegerField(default=0)
    n_ddf = models.IntegerField(default=0)
    n_assoc = models.IntegerField(default=0)
    n_images = models.IntegerField(default=0)
    n_spectra = models.IntegerField(default=0)
    n_dist = models.IntegerField(default=0)
    n_class = models.IntegerField(default=0)
    aliases = models.JSONField(default=list)

    class Meta:
        default_related_name = "ned_entries"

    def __str__(self):
        return self.prefname

    def get_aliases(self):
        """get_aliases.

        Use NED's query_object with aliases flag.
        """
        info = utils.ned_aliases(self.prefname)
        if not info:
            return
        if "Aliases" in info:
            self.aliases = info["Aliases"]
            self.save()

    def query_table(self, table_type):
        """query_table.

        Parameters
        ----------
        table_type :
            table_type
        """
        if table_type not in (
            "all",
            "diameters",
            "redshifts",
            "photometry",
            "kinematics",
            "luminosity_class",
            "morphology",
            "classification",
            "distance",
        ):
            raise SyntaxError(
                "table_type must be one of the following: all, diameters, redshifts, photometry, kinematics, luminosity_class, morphology, classification, distance"
            )
        if table_type == "all":
            for param in (
                "diameters",
                "redshifts",
                "photometry",
                "classification",
                "distance",
            ):
                self.query_table(param)
            return
        if (
            (table_type == "diameters" and not self.n_ddf)
            or (table_type == "redshifts" and not self.n_zdf)
            or (table_type == "photometry" and not self.n_gphot)
            or (table_type == "dist" and not self.n_dist)
            or (
                table_type
                in ("classification", "kinematics", "luminosity_class", "morphology")
                and not self.n_class
            )
            or (table_type == "distance" and not self.n_dist)
        ):
            print(f"No NED {table_type} table available for {self.prefname}")
            return
        getattr(self, table_type).all().delete()
        if table_type in ("diameters", "redshifts", "photometry"):
            self.query_astroquery_table(table_type)
        if table_type in (
            "kinematics",
            "luminosity_class",
            "morphology",
            "classification",
        ):
            self.query_classification_table()
        if table_type == "distance":
            self.query_distance_table()

    def query_astroquery_table(self, table_type):
        """query_astroquery_table.

        Parameters
        ----------
        table_type : str
            table_type
        """
        try:
            table = Ned.get_table(self.prefname.lower(), table=table_type)
        except (IndexError, RemoteServiceError, TableParseError) as error:
            print(f"Error on {table_type} table for {self.prefname}")
            print(error)
            return

        def parse_row(row):
            defaults = {}
            for key in table.keys():
                if key in ("No."):
                    continue
                val = utils.generic_format(row[key])
                # diameters constraints
                if key == "NED cos-1_axis_ratio":
                    small = "ned_inv_cos_axis_ratio"
                elif key == "Unc. Significance":
                    small = "unc_significance"
                else:
                    small = key.lower().replace(" ", "_")
                # Fixing RA/Dec issues
                if key in ("Targeted RA", "Targeted Dec"):
                    if val == "":
                        val = None
                    elif val:
                        val = float(val)
                if key in ("Published RA", "Published Dec") and val is not None:
                    val = val.replace(" ", "")
                    # Ned has messy Char data for published dec.
                    # See NGC 2363 1992CORV..C...0000F
                    if not re.findall("[0-9]+", val):
                        val = None
                    elif len(val.split(".")[0]) < 6:
                        val = float(val)
                    else:
                        if key.endswith("RA"):
                            val = utils.HMS2deg(ra=f"{val[:2]}:{val[2:4]}:{val[4:]}")
                        elif key.endswith("Dec"):
                            sign = ""
                            if val[0] in ("+", "-"):
                                sign = val[0]
                                val = val[1:]
                                val = utils.HMS2deg(
                                    dec=f"{sign}{val[:2]}:{val[2:4]}:{val[4:]}"
                                )
                defaults[small] = val

            try:
                getattr(self, table_type).create(**defaults)
            except:
                print(defaults)
                raise

        for row in table:
            parse_row(row)

    def query_classification_table(self):
        encoded_name = quote_plus(self.prefname)
        url = f"https://ned.ipac.caltech.edu/cgi-bin/NEDatt?objname={encoded_name}"
        page = requests.get(url)
        soup = BeautifulSoup(page.text, "html.parser")
        if soup.find("h1").text.startswith("No Classifications currently avaiable"):
            return

        def parse_row(row):
            if row.text == "Kinematics":
                subtable = NedKinematics
            elif row.text == "Luminosity Class":
                subtable = NedLuminosityClass
            elif row.text == "Galaxy Morphology":
                subtable = NedMorphology
            elif row.text == "Distance Indicator":
                return
            cells = row.find_all("td")
            if len(cells) < 7:
                return
            subtable.objects.create(
                entry=self,
                classification=cells[2].text.strip("\xa0i\xa0\xa0"),
                refcode=cells[1].text,
                flag=cells[3].text,
                bandpass=cells[4].text,
                region=cells[5].text,
                notes=cells[6].text,
            )

        for row in soup.find_all("table")[-1].find_all("tr"):
            parse_row(row)

    def query_distance_table(self):
        """query_distance_table.

        Scrape NED distance tables and create subtables
        """
        encoded_name = quote_plus(self.prefname)
        url = f"https://ned.ipac.caltech.edu/cgi-bin/nDistance?name={encoded_name}"
        page = requests.get(url)
        soup = BeautifulSoup(page.text, "html.parser")
        if soup.find("h2").text.startswith("0 Distances found"):
            return

        def parse_row(row):
            cells = row.find_all("td")
            if len(cells) < 9:
                return
            NedDistance.objects.create(
                entry=self,
                distance_modulus=float(cells[0].text),
                distance_modulus_error=float(cells[1].text),
                method=cells[3].text,
                refcode=cells[4].text,
                notes=cells[5].text,
                sn_name=cells[6].text,
            )

        for row in soup.find_all("table")[-1].find_all("tr"):
            parse_row(row)

    def get_redshift(self, average=True, clip=True, query=True):
        """get_redshift.

        Parse available redshift measurements and return the best
        """
        z, z_err, z_flag, bibcode = self.z, self.zunc, "n", f"ned_{self.objid}"
        if not self.redshifts.exists():
            if self.n_zdf == 0 or not query:
                return z, z_err, z_flag, bibcode
            else:
                self.query_table("redshifts")

        # weird filtering because there are bad data in the ned entries
        # WISEA J021927.84+345751.5 is one of many objects with negative uncertainty
        # VIII Zw 066 has a redshift entry over 2 where others are under 0.01
        # this entry has the comments "Low-confidence redshift"
        zsets = {}
        zsets["l"] = self.redshifts.filter(
            published_redshift_uncertainty__gte=0
        ).exclude(comments__startswith="Low-confidence")
        if zsets["l"].exists():
            z_flag = "l"
        else:
            return z, z_err, z_flag, bibcode
        zsets["p"] = zsets["l"].filter(refcode__in=constants.PHOTO_Z_REFCODES)
        if zsets["p"].exists():
            z_flag = "p"
        zsets["sp"] = zsets["l"].exclude(
            spectrograph__isnull=True,
            measurement_mode_features__isnull=True,
            spectral_range__isnull=True,
        )
        if zsets["sp"].exists():
            z_flag = "sp"
            # filter for cross-correlation as opposed to single line fitting
            # see SDSS J110626.48+142149.9 which has 2 measurements from
            # 2006SDSS5.C...0000: off by a magnitude
            zsets["cc"] = zsets["sp"].filter(
                measurement_mode_technique__startswith="Cross-correlation"
            )
            if zsets["cc"].exists():
                z_flag = "cc"

        zset = zsets[z_flag]
        if z_flag == "cc":
            z_flag = "sp"
        if zset.filter(published_redshift_uncertainty__gt=0).exists():
            zset = zset.filter(published_redshift_uncertainty__gt=0)
            z_flag += "u"

        z, z_err, bibcodes = np.array(
            zset.values_list(
                "published_redshift", "published_redshift_uncertainty", "refcode"
            )
        ).T
        z = np.array(z, dtype=float)
        z_err = np.array(z_err, dtype=float)
        bibcodes = np.array(bibcodes, dtype="<U19")
        for i, ze in enumerate(z_err):
            if ze == 0:
                z_err[i] = 99
        if clip:
            z, z_err, idx, _ = utils.sigmaclip(z, z_err)
            bibcodes = bibcodes[idx]
        if average:
            z = np.average(z, weights=1 / z_err)
            z_err = np.average(z_err, weights=1 / z_err)
        return z, z_err, z_flag, bibcodes

    def get_mag(self, bandpass="g"):
        phot_qset = self.photometry.filter(observed_passband__contains=bandpass)
        if phot_qset.exists():
            return np.median(phot_qset.values_list("photometry_measurement", flat=True))
        else:
            return None


class NedSubTable(models.Model):
    """NedSubTable.

    Abstract holder for tables not featured in astroquery
    """

    entry = models.ForeignKey(NedEntry, on_delete=models.CASCADE)
    refcode = models.CharField(max_length=19)

    class Meta:
        abstract = True


class NedDiameter(NedSubTable):
    """NedDiameter.

    From astroquery
    """

    frequency_targeted = models.CharField(max_length=30)
    major_axis = models.FloatField(null=True)
    major_axis_flag = models.CharField(max_length=3, null=True)
    major_axis_unit = models.CharField(max_length=11, null=True)
    major_axis_uncertainty = models.FloatField(null=True)
    ned_major_axis = models.FloatField(null=True)
    ned_major_axis_uncertainty = models.FloatField(null=True)
    minor_axis = models.FloatField(null=True)
    minor_axis_flag = models.CharField(max_length=6, null=True)
    minor_axis_unit = models.CharField(max_length=11, null=True)
    minor_axis_uncertainty = models.FloatField(null=True)
    ned_minor_axis = models.FloatField(null=True)
    ned_minor_axis_uncertainty = models.FloatField(null=True)
    axis_ratio = models.FloatField(null=True)
    axis_ratio_flag = models.CharField(max_length=10, null=True)
    axis_ratio_uncertainty = models.FloatField(null=True)
    ned_axis_ratio = models.FloatField(null=True)
    ned_axis_ratio_uncertainty = models.FloatField(null=True)
    ellipticity = models.FloatField(null=True)
    ellipticity_uncertainty = models.FloatField(null=True)
    ned_ellipticity = models.FloatField(null=True)
    ned_ellipticity_uncertainty = models.FloatField(null=True)
    eccentricity = models.FloatField(null=True)
    eccentricity_uncertainty = models.FloatField(null=True)
    ned_eccentricity = models.FloatField(null=True)
    ned_eccentricity_uncertainty = models.FloatField(null=True)
    position_angle = models.FloatField(null=True)
    position_angle_uncertainty = models.FloatField(null=True)
    ned_position_angle = models.FloatField(null=True)
    ned_position_angle_uncertainty = models.FloatField(null=True)
    equinox = models.CharField(max_length=5, null=True)
    reference_level = models.CharField(max_length=30, null=True)
    ned_inv_cos_axis_ratio = models.FloatField(null=True)
    significance = models.CharField(max_length=30)
    frequency = models.FloatField()
    frequency_unit = models.CharField(max_length=7)
    frequency_mode = models.CharField(max_length=45)
    ned_frequency = models.FloatField()
    detector_type = models.CharField(max_length=34)
    fitting_technique = models.CharField(max_length=50)
    features = models.CharField(max_length=40, null=True)
    measured_quantity = models.CharField(max_length=33)
    measurement_qualifiers = models.CharField(max_length=44, null=True)
    targeted_ra = models.FloatField(null=True)
    targeted_dec = models.FloatField(null=True)
    targeted_equinox = models.CharField(max_length=5, null=True)
    ned_qualifiers = models.TextField(null=True)
    ned_comment = models.TextField(null=True)

    class Meta:
        default_related_name = "diameters"

    def __str__(self):
        return f"{self.frequency_targeted} {self.refcode}"


class NedPhotometry(NedSubTable):
    """NedPhotometry.

    from astroquery
    """

    observed_passband = models.CharField(max_length=20)
    photometry_measurement = models.FloatField(null=True)
    uncertainty = models.CharField(max_length=13, null=True)
    units = models.CharField(max_length=20)
    frequency = models.FloatField()
    flux_density = models.FloatField(null=True)
    upper_limit_of_uncertainty = models.FloatField(null=True)
    lower_limit_of_uncertainty = models.FloatField(null=True)
    upper_limit_of_flux_density = models.FloatField(null=True)
    lower_limit_of_flux_density = models.FloatField(null=True)
    ned_uncertainty = models.CharField(max_length=13, null=True)
    ned_units = models.CharField(max_length=5)
    significance = models.CharField(max_length=30)
    published_frequency = models.CharField(max_length=21, null=True)
    frequency_mode = models.CharField(max_length=92)
    coordinates_targeted = models.CharField(max_length=39, null=True)
    spatial_mode = models.CharField(max_length=60)
    qualifiers = models.TextField(null=True)
    comments = models.TextField(null=True)

    class Meta:
        default_related_name = "photometry"

    def __str__(self):
        return f"{self.observed_passband} {self.photometry_measurement}"


class NedRedshifts(NedSubTable):
    """NedRedshifts.

    from astroquery
    """

    frequency_targeted = models.CharField(max_length=20, null=True)
    published_velocity = models.IntegerField()
    published_velocity_uncertainty = models.IntegerField()
    published_redshift = models.FloatField()
    published_redshift_uncertainty = models.FloatField()
    name_in_publication = models.CharField(max_length=32, null=True)
    published_ra = models.FloatField(null=True)
    published_dec = models.FloatField(null=True)
    published_equinox = models.CharField(max_length=5, null=True)
    unc_significance = models.CharField(max_length=17, null=True)
    spectral_range = models.CharField(max_length=7, null=True)
    spectrograph = models.CharField(max_length=40, null=True)
    measurement_mode_features = models.CharField(max_length=45, null=True)
    measurement_mode_technique = models.CharField(max_length=45, null=True)
    spatial_mode = models.CharField(max_length=81, null=True)
    epoch = models.CharField(max_length=7, null=True)
    reference_frame = models.CharField(max_length=53, null=True)
    apex = models.CharField(max_length=3, null=True)
    longitude_of_the_apex = models.CharField(max_length=5, null=True)
    latitude_of_the_apex = models.CharField(max_length=5, null=True)
    apex_coordinate_system = models.CharField(max_length=8, null=True)
    qualifiers = models.TextField(null=True)
    comments = models.TextField(null=True)

    class Meta:
        default_related_name = "redshifts"

    def __str__(self):
        return f"{self.published_redshift}"


class NedKinematics(NedSubTable):
    """NedKinematics.

    from scraping
    """

    class Meta:
        default_related_name = "kinematics"


class NedLuminosityClass(NedSubTable):
    """NedLuminosityClass.

    from scraping
    """

    class Meta:
        default_related_name = "luminosity_classes"


class NedMorphology(NedSubTable):
    """NedMorphology.

    from scraping
    """

    class Meta:
        default_related_name = "morphologies"


class NedDistance(NedSubTable):
    """NedDistance.

    from scraping
    """

    distance_modulus = models.FloatField()
    distance_modulus_error = models.FloatField()
    method = models.CharField(max_length=18)
    notes = models.CharField(max_length=120, blank=True)
    sn_name = models.CharField(max_length=8, blank=True)

    class Meta:
        default_related_name = "distances"


class GalaxyQueries(models.Manager):
    """GalaxyQueries.

    Bundles query calls to various catalogs
    """

    def _ps1search(
        self,
        table="mean",
        release="dr2",
        fmt="csv",
        columns=None,
        baseurl="https://catalogs.mast.stsci.edu/api/v0.1/panstarrs",
        verbose=False,
        **kw,
    ):
        """Do a general search of the PS1 catalog

        Parameters
        ----------
        table (string): mean, stack, or detection
        release (string): dr1 or dr2
        fmt (string): format, either csv, votable, or json
        columns: list of column names to include (None means use defaults)
        baseurl: base URL for the request
        verbose: print info about request
        **kw: other parameters (e.g., 'nDetections.min':2).  Note this is required!
        """

        data = kw.copy()
        if not data:
            raise ValueError("You must specify some parameters for search")
        self._checklegal(table, release)
        if fmt not in ("csv", "votable", "json"):
            raise ValueError("Bad value for fmt (format)")
        url = "{baseurl}/{release}/{table}.{fmt}".format(**locals())

        if columns:
            # check that column values are legal
            # create a dictionary to speed this up
            dcols = {}
            for col in self._ps1metadata(table, release)["name"]:
                dcols[col.lower()] = 1
            badcols = []
            for col in columns:
                if col.lower().strip() not in dcols:
                    badcols.append(col)
            if badcols:
                raise ValueError(
                    "Some columns not found in table: {}".format(", ".join(badcols))
                )
            # two different ways to specify a list of column values in the API
            # data['columns'] = columns
            data["columns"] = "[{}]".format(",".join(columns))
        with requests.Session() as session:
            r = session.get(url, params=data)
        if verbose:
            print(r.url)
        r.raise_for_status()
        if fmt == "json":
            return r.json()
        else:
            return r.text

    def _checklegal(self, table, release):
        """Checks if this combination of table and release is acceptable

        Raises a ValueError exception if there is problem
        """
        releaselist = ("dr1", "dr2")
        if release not in ("dr1", "dr2"):
            raise ValueError(
                "Bad value for release (must be one of {})".format(
                    ", ".join(releaselist)
                )
            )
        if release == "dr1":
            tablelist = ("mean", "stack")
        else:
            tablelist = ("mean", "stack", "detection", "forced_mean")
        if table not in tablelist:
            raise ValueError(
                "Bad value for table (for {} must be one of {})".format(
                    release, ", ".join(tablelist)
                )
            )

    def _ps1metadata(
        self,
        table="mean",
        release="dr1",
        baseurl="https://catalogs.mast.stsci.edu/api/v0.1/panstarrs",
    ):
        """Return metadata for the specified catalog and table

        Parameters
        ----------
        table (string): mean, stack, or detection
        release (string): dr1 or dr2
        baseurl: base URL for the request

        Returns an astropy table with columns name, type, description
        """
        self._checklegal(table, release)
        url = "{baseurl}/{release}/{table}/metadata".format(**locals())
        r = requests.get(url)
        r.raise_for_status()
        v = r.json()
        # convert to astropy table
        tab = Table(
            rows=[(x["column_name"], x["type"], x["description"]) for x in v],
            names=("name", "type", "description"),
        )
        return tab

    def _cone_search_ps1(
        self,
        rad=1.5,
        table="mean",
        release="dr2",
        fmt="csv",
        columns=None,
        baseurl="https://catalogs.mast.stsci.edu/api/v0.1/panstarrs",
        verbose=False,
        **kw,
    ):
        """Do a cone search of the PS1 catalog

        Parameters
        ----------
        radius (float): (degrees) Search RAdius (<= 0.5 degrees)
        table (string): mean, stack, or detection
        release (string): dr1 or dr2
        fmt (string): format, either csv, votable, or json
        columns: list of column names to include (None means use defaults)
        baseurl: base URL for the request
        verbose: print info about request
        **kw: other parameters (e.g., 'nDetections.min':2)
        """

        data = kw.copy()
        # data['ra'] = coord.ra.value
        # data['dec'] = coord.dec.value
        # data['radius'] = rad/60.
        cone_search = self._ps1search(
            table=table,
            release=release,
            fmt=fmt,
            columns=columns,
            baseurl=baseurl,
            verbose=verbose,
            **data,
        )
        if cone_search == "":
            return
        tab_stack = ascii.read(cone_search)
        return tab_stack.to_pandas()

    def ps1(
        self,
        coord=None,
        ra=None,
        dec=None,
        radius=1.5,
        release="dr2",
        baseurl="https://catalogs.mast.stsci.edu/api/v0.1/panstarrs",
        verbose=False,
        **kw,
    ):
        """ps1.

        Query PanStarrs

        Parameters
        ----------
        coord : SkyCoord or None
            coord
        ra : Float-like or None
            ra
        dec : Float-like or None
            dec
        radius : Float-like
            radius in arcmin
        release :
            release
        baseurl :
            baseurl
        verbose :
            verbose
        kw :
            kw
        """
        # converting host_list to actual galaxy
        # Get host lists for Stacked and Forced phot

        data = kw.copy()
        if coord and (ra or dec):
            return SyntaxError(
                "Pass either a SkyCoord object, or ra and dec values, not both"
            )
        if not coord and not ra and not dec:
            return SyntaxError("Need Coordinates")
        if coord:
            data["ra"] = coord.ra.value
            data["dec"] = coord.dec.value
        elif ra and dec:
            data["ra"] = ra
            data["dec"] = dec
        data["radius"] = radius / 60.0
        # require at least two single-epoch image detection in any filter. Weeds out artifacts.
        data["nDetections.gt"] = 0

        # Stacked photometry
        # strip blanks and weed out blank and commented-out values
        columns_base = "objID,objName,objInfoFlag,qualityFlag,objAltName1,objAltName2,objAltName3,raMean,decMean,nDetections,ng,nr,ni,nz,ny"
        columns_stack = columns_base.split(",")
        for filt in "grizy":
            for mag_type in ("PSF", "Kron"):
                columns_stack.append(f"{filt}{mag_type}Mag")
                columns_stack.append(f"{filt}{mag_type}MagErr")
        columns_stack.append("primaryDetection")
        columns_stack = [x.strip() for x in columns_stack]
        columns_stack = [x for x in columns_stack if x and not x.startswith("#")]
        df_stack = self._cone_search_ps1(
            table="stack",
            release=release,
            columns=columns_stack,
            baseurl=baseurl,
            verbose=verbose,
            **data,
        )

        # Forced photometry
        # strip blanks and weed out blank and commented-out values
        columns_forced = columns_base.split(",")
        for filt in "grizy":
            for mag_type in ("PSF", "Kron"):
                columns_forced.append(f"{filt}F{mag_type}Flux")
                columns_stack.append(f"{filt}F{mag_type}FluxErr")
        columns_forced = [x.strip() for x in columns_forced]
        columns_forced = [x for x in columns_forced if x and not x.startswith("#")]
        df_forced = self._cone_search_ps1(
            table="forced_mean",
            release=release,
            columns=columns_forced,
            baseurl=baseurl,
            verbose=verbose,
            **data,
        )

        # Merge by objName
        if type(df_stack) == type(None) or type(df_forced) == type(None):
            return []

        # raMean and decMean vary from stacked to forced
        # will produce raMean_x/y and decMean_x/y, but better than dropping rows
        ps1 = pd.merge(
            left=df_stack,
            right=df_forced,
            on=columns_base.replace("raMean,", "").replace("decMean,", "").split(","),
        )
        ps1 = ps1.dropna()
        self._make_entry_from_pd(ps1, "ps1")
        return ps1

        """ned.

        Parameters
        ----------
        coord :
            coord
        ra :
            ra
        dec :
            dec
        radius :
            radius
        """

    def ned(self, coord=None, ra=None, dec=None, radius=1.5):
        """ned.

        Query Ned

        Parameters
        ----------
        coord :
            coord
        ra :
            ra
        dec :
            dec
        radius :
            radius
        """
        if coord and (ra or dec):
            return SyntaxError(
                "Pass either a SkyCoord object, or ra and dec values, not both"
            )
        if not coord and not ra and not dec:
            return SyntaxError("Need Coordinates")
        if coord and not (ra or dec):
            ra = coord.ra.value
            dec = coord.dec.value
        ned = utils.ned_cone_search(ra, dec, sr_deg=radius / 60.0)
        self._make_entry_from_pd(ned, "ned")
        return ned

    def simbad(self, coord=None, ra=None, dec=None, radius=1.5):
        """simbad.

        Query Simbad

        Parameters
        ----------
        coord :
            coord
        ra :
            ra
        dec :
            dec
        radius :
            radius
        """
        if coord and (ra or dec):
            return SyntaxError(
                "Pass either a SkyCoord object, or ra and dec values, not both"
            )
        if not coord and not ra and not dec:
            return SyntaxError("Need Coordinates")
        if not coord:
            coord = SkyCoord(ra, dec, unit="deg")
        Simbad.reset_votable_fields()
        Simbad.remove_votable_fields("coordinates")
        Simbad.add_votable_fields(
            "velocity",
            "dimensions",
            "otype(3)",
            "distance",
            "ids",
            "morphtype",
            "coo(d;ICRS;J2000.0)",
        )
        for filt in "BVGJgri":
            Simbad.add_votable_fields(f"fluxdata({filt})")
        sb = Simbad.query_region(coord, radius / 60 * u.deg)
        res = utils.simbad_cone_search(coord.ra.deg, coord.dec.deg, sr_arcmin=radius)
        if sb is None or res is None:
            return
        df = sb.to_pandas()
        # df["MAIN_ID"] = df["MAIN_ID"].str.decode("utf-8")
        df = pd.merge(left=df, right=res, left_on="MAIN_ID", right_on="main_id")
        self._make_entry_from_pd(df, "simbad")
        return df

    def _make_entry_from_pd(self, gal_obj, catalog):
        """_make_entry_from_pd.

        Parse results from a query and create the appropriate objects.

        Parameters
        ----------
        gal_obj :
            gal_obj
        catalog :
            catalog
        """
        if isinstance(gal_obj, pd.DataFrame):
            for i, row in gal_obj.iterrows():
                self._make_entry_from_pd(row, catalog=catalog)
            return
        defaults = {}
        for key, val in gal_obj.items():
            val = utils.generic_format(val)
            # getting null objid for simbad objects
            # if key in ("No.", "MAIN_ID", "objid") or (
            #     catalog == "simbad" and key.startswith("FILTER_NAME_")
            # ):
            #     continue
            if catalog == "simbad" and (
                key.startswith("FILTER_NAME_") or key == "MAIN_ID"
            ):
                continue
            elif catalog in ("ned", "ps1") and key in ("No.", "objid"):
                continue
            # sorting keys
            if catalog == "simbad" and key.endswith("_d_ICRS_J2000_0"):
                small = key[:-15].lower()
            else:
                small = key.lower().replace(" ", "_")
            # NED redshift flag should be - when unavailable
            if key == "zflag" and not val:
                val = "-"
            # ps1 null/blank replaced with -999
            if val == -999:
                val = None
            defaults[small] = val
        if catalog == "simbad":
            obj, new = SimbadEntry.objects.update_or_create(
                main_id=utils.generic_format(gal_obj["MAIN_ID"]), defaults=defaults
            )
        elif catalog == "ned":
            obj, new = NedEntry.objects.update_or_create(
                objid=utils.generic_format(gal_obj["objid"]), defaults=defaults
            )
            obj.get_aliases()
            obj.query_table("diameters")
        elif catalog == "ps1":
            defaults["ra"] = defaults["ramean"] = defaults.pop("ramean_x")
            defaults["dec"] = defaults["decmean"] = defaults.pop("decmean_x")
            del defaults["ramean_y"]
            del defaults["decmean_y"]
            obj, new = PanStarrsEntry.objects.update_or_create(
                objid=utils.generic_format(gal_obj["objID"]), defaults=defaults
            )
        g = obj.galaxy
        if g:
            g.z, g.z_err, g.z_flag = obj.galaxy.get_redshift(query=False)
            g.save()

    def all_catalogs(self, coord=None, ra=None, dec=None, radius=1.5):
        """all_catalogs.

        Query all catalogs

        Parameters
        ----------
        coord :
            coord
        ra :
            ra
        dec :
            dec
        radius :
            radius
        """
        if coord and (ra or dec):
            return SyntaxError(
                "Pass either a SkyCoord object, or ra and dec values, not both"
            )
        if not coord and not ra and not dec:
            return SyntaxError("Need Coordinates")
        if not coord:
            coord = SkyCoord(ra, dec, unit="deg")
        self.ps1(coord=coord, radius=radius)
        self.ned(coord=coord, radius=radius)
        self.simbad(coord=coord, radius=radius)


class GalaxyQuerySet(models.query.QuerySet):
    """GalaxyQuerySet.

    Scripts producing or acting on entire querysets
    """

    def box_search(self, ra, dec, box_size):
        """box_search.

        Square search around (ra, dec)

        Parameters
        ----------
        ra (deg):
            ra
        dec (deg):
            dec
        box_size (deg):
            box_size
        """
        qset = self.filter(dec__gt=dec - box_size, dec__lt=dec + box_size)
        ra_box_size = box_size * np.cos(dec * np.pi / 180)
        if ra + ra_box_size < 360 and ra - ra_box_size > 0:
            return qset.filter(ra__gt=ra - ra_box_size, ra__lt=ra + ra_box_size)
        else:
            return qset.filter(
                Q(ra__lt=(ra + ra_box_size) % 360) | Q(ra__gt=(ra - ra_box_size) % 360)
            )
        return

    def with_targets(self):
        return self.filter(target_set__isnull=False)

    def get_normalized_axes(self):
        """get_normalized_axes.

        Since the observations are heterogeneous, can't just average.
        Need to find either a frequency_targeted all sources have in common
        or get a rough table of ratios between the different measurements
        to put everything on one relative scale.
        """
        pks = np.zeros((self.count(), 2), dtype=object)
        for i, g in enumerate(self):
            for j, catalog in enumerate(("ned", "simbad")):
                entries = g.choose_galaxy_entry(catalog, return_qset=True)
                if entries is None or not entries.exists():
                    continue
                pks[i][j] = np.array(entries.values_list("pk", flat=True))

        neds = NedEntry.objects.filter(
            pk__in=self.values_list("ned_entries", flat=True)
        )
        simbads = SimbadEntry.objects.filter(
            pk__in=self.values_list("simbad_entries", flat=True)
        )

        ned_fset = np.array(
            neds.values_list("diameters__frequency_targeted", flat=True).distinct()
        )
        ned_fset = ned_fset[ned_fset != np.array([None])]
        simbad_fset = np.array(simbads.values_list("galdim_wavelength", flat=True))
        simbad_fset = simbad_fset[simbad_fset != np.array([None])]
        diameter_fset = np.append(ned_fset, simbad_fset)
        filt_num = len(diameter_fset)
        gal_num = self.count()
        # using axis ratios to avoid normalization issues between major and minor
        return_dict = {
            "major": [None for i in range(gal_num)],
            "axis_ratio": [1 for i in range(gal_num)],
            "phi": [0 for i in range(gal_num)],
        }
        if not filt_num:
            # dphi set to 90 if phi=0, this may work against galaxies with phi=0
            return return_dict
        # dict for major, minor, phi, value is ndarray
        # ndarrays are frequencies targeted by number of galaxies
        # units to be normalized out since different filters, methodologies.
        param_dict = {}
        for param in ("major", "axis_ratio", "phi"):
            param_dict[param] = np.zeros((filt_num, gal_num))
        for i, g in enumerate(self):
            for j, f in enumerate(ned_fset):
                if isinstance(pks[i][0], int):  # int means 0
                    continue
                diam_pks = []
                obj_neds = NedEntry.objects.filter(pk__in=pks[i][0])
                for o_n in obj_neds:
                    diam_pks += list(
                        o_n.diameters.filter(frequency_targeted=f).values_list(
                            "pk", flat=True
                        )
                    )
                diam_set = NedDiameter.objects.filter(pk__in=diam_pks)
                # grab most recent refcode if multiple observations in same filter
                # even across crossmatches
                if diam_set.exists():
                    diam = diam_set.order_by("refcode").last()
                else:
                    continue
                for param, ned_param in zip(
                    ("major", "axis_ratio", "phi"),
                    ("ned_major_axis", "ned_axis_ratio", "ned_position_angle"),
                ):
                    val = getattr(diam, ned_param)
                    if val is not None and not np.isnan(val):
                        param_dict[param][j, i] = val
            for k, f in enumerate(simbad_fset):
                if isinstance(pks[i][1], int):
                    continue
                obj_simbads = SimbadEntry.objects.filter(
                    pk__in=pks[i][1], galdim_wavelength=f
                )
                for param, simbad_param in zip(
                    ("major", "axis_ratio", "phi"),
                    ("galdim_majaxis", "galdim_minaxis", "galdim_angle"),
                ):
                    vals = obj_simbads.values_list(simbad_param, flat=True)
                    if [i for i in vals if i] == []:
                        continue
                    val = np.average([i for i in vals if i])
                    if param == "axis_ratio":
                        val = val / param_dict["major"][len(ned_fset) + k, i]
                    param_dict[param][len(ned_fset) + k, i] = val
        # arrays probably sparse, but if there's a filter where all galaxies have data,
        # don't need to normalize and can just return the full slice
        for param in ("major", "axis_ratio", "phi"):
            full_idx = []
            for i, arr in enumerate(param_dict[param]):
                if 0 not in arr:
                    full_idx.append(i)
            # harmonic means if multiple full arrays
            if len(full_idx) > 1:
                return_dict[param] = np.prod(param_dict[param][full_idx], axis=0) ** (
                    1 / len(full_idx)
                )
                break
            # or just the one full array
            elif len(full_idx) == 1:
                return_dict[param] = param_dict[param][full_idx[0]]
                break
            # terrible situation
            # if no common filters, need to make a matrix of ratios between different
            # filters of common targets to eventually push things into a common system
            # start with pxp grids for each object with all valid ratios between
            # non-zero bands and nans elsewhere
            else:
                narr = param_dict[param]
                # for each galaxy, look at each filter/filter combination
                # populate grid with ratios, x and y interchangeable but
                # will eventually turn a * into a /
                grid = np.zeros((filt_num, filt_num, gal_num))
                for f1 in range(filt_num):
                    for f2 in range(filt_num):
                        for gal in range(gal_num):
                            # avoid div 0
                            if narr[f2, gal] != 0:
                                grid[f1, f2, gal] = narr[f1, gal] / narr[f2, gal]
                # marginalize over the galaxy dimension with an average.
                # This ignores different galaxy profiles, but should generally
                # put different measurements on a somewhat normalized system
                aves = np.zeros((filt_num, filt_num))
                for f1 in range(filt_num):
                    for f2 in range(filt_num):
                        # avoid empy averages
                        if len(grid[f1, f2, :][grid[f1, f2, :] != 0]):
                            aves[f1, f2] = np.average(
                                grid[f1, f2, :][grid[f1, f2, :] != 0]
                            )
                # determine bandpass with most targets
                common_idx = 0
                non_zeros = 0
                for f in range(filt_num):
                    new_non_zeros = len(narr[f][narr[f] != 0])
                    if new_non_zeros > non_zeros:
                        common_idx = f
                        non_zeros = new_non_zeros
                # If available, use common bandpass measurement, otherwise, infer using
                # all other available measurements and the ratio matrix
                return_dict[param] = np.zeros(gal_num)
                for i, obj_idx in enumerate(range(gal_num)):
                    if narr[common_idx][obj_idx] != 0:
                        return_dict[param][i] = narr[common_idx][obj_idx]
                    else:
                        all_bands = narr[:, obj_idx]
                        normed = all_bands * aves[common_idx]
                        if len(normed[np.nan_to_num(normed) != 0]):
                            return_dict[param][i] = np.average(normed[normed != 0])
        return return_dict


class Galaxy(models.Model):
    """Galaxy.

    Basic demographic information and functions for compiling catalog entries.
    """

    Z_CHOICES = [
        ("n", "No redshift available"),
        ("l", "Literature redshift available, unknown methodology."),
        ("lu", "Literature redshift and uncertainty available, unknown methodology."),
        ("p", "Photometric redshift from literature"),
        ("pu", "Photometric redshift and uncertainty from literature"),
        ("sp", "Spectroscopic redshift from literature"),
        ("spu", "Spectroscopic redshift and uncertainty from literature"),
        ("sn1", "SNIFS spectrum available, not yet reduced"),  # deprecateing sn1/2/3
        ("sn2", "SNIFS spectrum available, reduced"),
        ("sn3", "SNIFS spectrum available, reduced, SNR too low"),
        ("su1", "Subaru spectrum available, not yet reduced"),  # deprecating su1/2/3
        ("su2", "Subaru spectrum available, reduced"),
        ("su3", "Subaru spectrum available, reduced, SNR too low"),
        ("spx1", "Literature spectrum available not yet reduced"),  # deprecating spx1/2
        ("spx2", "Literature spectrum available, reduced"),
        ("d", "Marked for deletion"),
    ]
    pgc_no = models.IntegerField(null=True)
    leda_v = models.FloatField(null=True)
    leda_v_err = models.FloatField(null=True)
    ra = models.FloatField()
    dec = models.FloatField()
    manually_inspected = models.ManyToManyField(User)
    z = models.FloatField(null=True)
    v = models.FloatField(null=True)
    z_err = models.FloatField(null=True)
    v_err = models.FloatField(null=True)
    z_flag = models.CharField(max_length=4, choices=Z_CHOICES, default="n")

    query = GalaxyQueries()
    objects = GalaxyQuerySet.as_manager()

    class Meta:
        indexes = [
            models.Index(fields=["ra"]),
            models.Index(fields=["dec"]),
            models.Index(fields=["z_flag"]),
            models.Index(fields=["pgc_no"]),
        ]

    def __str__(self):
        return f"{self.ra} {self.dec}"

    @property
    def target(self):
        if self.targets.count() > 1:
            raise MultipleObjectsReturned
        elif not self.targets.count():
            return
        return self.targets.first()

    @property
    def entries(self):
        """entries.

        returns all catalog entries as a tuple.
        """
        return (
            self.snifs_entries.all(),
            self.focas_entries.all(),
            self.ned_entries.all(),
            self.simbad_entries.all(),
            self.ps1_entries.all(),
            self.glade_entries.all(),
        )

    @property
    def snifs(self):
        if self.snifs_entries.count() > 1:
            raise MultipleObjectsReturned
        return self.snifs_entries.first()

    @property
    def focas(self):
        if self.focas_entries.count() > 1:
            raise MultipleObjectsReturned
        return self.focas_entries.first()

    @property
    def ned(self):
        if self.ned_entries.count() > 1:
            raise MultipleObjectsReturned
        return self.ned_entries.first()

    @property
    def simbad(self):
        if self.simbad_entries.count() > 1:
            raise MultipleObjectsReturned
        return self.simbad_entries.first()

    @property
    def ps1(self):
        if self.ps1_entries.count() > 1:
            raise MultipleObjectsReturned
        return self.ps1_entries.first()

    @property
    def glade(self):
        if self.glade_entries.count() > 1:
            raise MultipleObjectsReturned
        return self.glade_entries.first()

    @property
    def names(self):
        """names

        Search through catalog entries to find all aliases and set names field.
        """
        names = []
        for n in self.ned_entries.all():
            names += n.aliases
        for s in self.simbad_entries.all():
            names += s.ids.split("|")
        for p in self.ps1_entries.all():
            names.append(p.objname)
            for i in range(1, 4):
                alt_name = getattr(p, f"objaltname{i}")
                if alt_name:
                    names.append(alt_name)
        for g in self.glade_entries.all():
            for prefix, attr in (
                ("GLADE ", "glade_no"),
                ("PGC ", "pgc_no"),
                ("", "gwgc_name"),
                ("", "hyperleda_name"),
                ("2MASX ", "twomass_name"),
                ("WISEA ", "wisexscos_name"),
                ("SDSS ", "sdss_dr16q_name"),
            ):
                if hasattr(g, attr):
                    names.append(f"{prefix}{getattr(g, attr)}")

        def lower(s):
            # checking for uniqueness through caps and space variation
            return s.replace(" ", "").lower()

        clean = set()
        small = np.array(list(map(lower, names)))
        for small_name in small:
            idx = np.where(np.array(small) == small_name)[0]
            if len(idx) == 1:
                clean.add(names[idx[0]])
            else:
                # prefer catalog names with spaces using min for alpha sorting
                # 'SDSS J123456' < 'SDSSJ123456'
                # also prefers lower-case if those exist
                clean.add(min([names[i] for i in idx]))
        return sorted(list(clean))

    def non_defaults(self):
        if self.names != []:
            print(f"names = {self.names}")
        for entry in ("snifs", "focas", "ned", "simbad", "ps1", "glade"):
            if getattr(self, f"{entry}_entries").exists():
                print(getattr(self, f"{entry}_entries").all())
        for param in ("z", "z_err", "z_flag"):
            if getattr(self, param):
                print(f"{param} = {getattr(self, param)}")
        if self.manually_inspected.all().exists():
            print(f"Inspected by {self.manually_inspected.all()}")

    @property
    def redshifts(self):
        for cat_name, catalog in zip(
            ("SNIFS", "FOCAS", "NED", "Simbad", "PanSTARRS", "GLADE"), self.entries
        ):
            if cat_name == "PanSTARRS" or not catalog.exists():
                continue
            print(cat_name)
            for entry in catalog:
                if cat_name in ("SNIFS", "FOCAS"):
                    print(entry, entry.z, entry.z_err, entry.z_flag)
                else:
                    print(entry, entry.get_redshift())

    def HMS_coords(self):
        """HMS_coords.

        Return coordinates in sexagecimal form.
        """
        HMS = utils.deg2HMS(ra=self.ra, dec=self.dec, rounding=2)
        if HMS == "":
            ra = "00:00:00.0"
            dec = "00:00:00.00"
        else:
            ra = "%02d:%02d:%05.2f" % tuple([float(x) for x in HMS[0].split()])
            dec = "%02d:%02d:%05.2f" % tuple([float(x) for x in HMS[1].split()])
        return (ra, dec)

    def get_redshift(self, leda_priority=0, average=True, clip=True, query=True):
        """get_redshift.

        Parameters
        ----------
        average :
            average
        clip :
            clip
        """
        """get_redshift.

        Parse through catalog entries and find the best redshift
        leda_priority is
            0: prioritize leda values over all else,
            1: average leda and snifs/focas, 
            2: average leda with other catalog values,
        """
        # Assimilate catalog redshifts
        d = {
            "z": np.array([], dtype=float),
            "z_err": np.array([], dtype=float),
            "z_flag": np.array([], dtype="<U3"),
            "bibcodes": np.array([], dtype="<U19"),
        }
        if self.leda_v is not None:
            d["z"] = np.append(d["z"], self.leda_v / constants.C)
            if self.leda_v_err is not None:
                d["z_err"] = np.append(d["z_err"], self.leda_v_err / constants.C)
            else:
                d["z_err"] = np.append(d["z_err"], 60 / constants.C)
            d["z_flag"] = np.append(d["z_flag"], "spu")
            d["bibcodes"] = np.append(d["bibcodes"], "HYPERLEDA")
        if not ("HYPERLEDA" in d["bibcodes"] and leda_priority == 0):
            if self.snifs_entries.filter(z_flag="s").exists():
                snifs = self.snifs_entries.filter(z_flag="s").first()
                d["z"] = np.append(d["z"], snifs.z)
                d["z_err"] = np.append(d["z_err"], (snifs.plus + snifs.minus) / 2)
                d["z_flag"] = np.append(d["z_flag"], "spu")
                d["bibcodes"] = np.append(d["bibcodes"], f"SNIFS_{snifs.date}")
            if self.focas_entries.filter(z_flag="s").exists():
                focas = self.focas_entries.filter(z_flag="s").first()
                d["z"] = np.append(d["z"], focas.z)
                d["z_err"] = np.append(d["z_err"], (focas.plus + focas.minus) / 2)
                d["z_flag"] = np.append(d["z_flag"], "spu")
                d["bibcodes"] = np.append(d["bibcodes"], f"FOCAS_{focas.date}")
        for catalog in ("ned", "simbad", "glade"):
            if "HYPERLEDA" in d["bibcodes"] and leda_priority in (0, 1):
                continue
            if catalog == "glade":
                c_set = self.glade_entries.filter(object_type_flag="G")
                get_redshift_args = []
            else:
                c_set = self.choose_galaxy_entry(catalog, return_qset=True)
                get_redshift_args = [False, False, query]
            for c in c_set:
                c_z, c_z_err, c_z_flag, c_bibcodes = c.get_redshift(*get_redshift_args)
                d["z"] = np.append(d["z"], c_z)
                d["z_err"] = np.append(d["z_err"], c_z_err)
                # z_flag is a str, need to make array-like
                # using c_bibcodes as proxy
                if isinstance(c_bibcodes, str) or c_bibcodes is None:
                    d["z_flag"] = np.append(d["z_flag"], c_z_flag)
                else:
                    d["z_flag"] = np.append(d["z_flag"], [c_z_flag for i in c_bibcodes])
                d["bibcodes"] = np.append(d["bibcodes"], c_bibcodes)
        # get measurements for best z_flag set
        idx_arr = None
        for z_flag in ("spu", "sp", "pu", "p", "lu", "l", "n"):
            if z_flag in d["z_flag"]:
                idx_arr = np.where(d["z_flag"] == z_flag)[0]
                break
        if idx_arr is None:
            return None, None, "n"
        d["z"] = np.array(d["z"])[idx_arr]
        d["z_err"] = np.array(d["z_err"])[idx_arr]
        d["bibcodes"] = np.array(d["bibcodes"][idx_arr])

        # remove duplicate bibcode entries
        z, z_err, bibcodes = [], [], []
        for tmp_z, tmp_z_err, tmp_bibcode in zip(d["z"], d["z_err"], d["bibcodes"]):
            if tmp_bibcode in bibcodes:
                continue
            z.append(tmp_z)
            z_err.append(tmp_z_err)
            bibcodes.append(tmp_bibcode)

        if not z_flag.endswith("u"):
            z_err = np.ones(len(z))
        if clip:
            if None in np.append(z, z_err) or np.nan in np.append(z, z_err):
                return None, None, z_flag
            z, z_err, _, _ = utils.sigmaclip(z, z_err)
        if average:
            if None in np.append(z, z_err) or np.nan in np.append(z, z_err):
                return None, None, z_flag
            z = np.average(z, weights=1 / z_err)
            z_err = np.average(z_err, weights=1 / z_err)
        return z, z_err, z_flag

    def get_leda_redshift(self):
        url = "https://leda.univ-lyon1.fr/ledacat.cgi?o=%23" + str(self.pgc_no)
        resp = requests.get(url)
        soup = BeautifulSoup(resp.text, "html.parser")
        dt = soup.find_all("table", class_="datatable")[1]
        for row in dt.find_all("tr"):
            if row.find("td").find("a") and row.find("td").find("a").text == "v":
                cell_text = row.find_all("td")[1].text.split()
                self.leda_v = float(cell_text[0])
                self.leda_v_err = float(cell_text[2])
                self.save()
                return

    def choose_galaxy_entry(self, catalog, return_qset=True):
        """choose_galaxy_entry.

        When multiple sources lie within some threshold,
        there must be a way to figure out which one to use
        Parameters
        ----------
        catalog : str
            catalog
        return_qset : bool
            return_qset
        """
        qset = getattr(self, f"{catalog}_entries").all()
        if not qset.exists():
            if return_qset:
                return qset
            return
        elif qset.count() == 1:
            if return_qset:
                return qset
            return qset.first()
        if catalog == "ps1":
            # For ps1, using psf - kron > 0.05 as a flag for galaxies
            # https://outerspace.stsci.edu/display/PANSTARRS/How+to+separate+stars+and+galaxies
            phot_dict = {}
            for filt in "grizy":
                phot_dict[f"{filt}_psf"] = np.array(
                    qset.values_list(f"{filt}psfmag", flat=True), dtype=float
                )
                phot_dict[f"{filt}_psferr"] = np.array(
                    qset.values_list(f"{filt}psfmagerr", flat=True), dtype=float
                )
                phot_dict[f"{filt}_kron"] = np.array(
                    qset.values_list(f"{filt}kronmag", flat=True), dtype=float
                )
                phot_dict[f"{filt}_kronerr"] = np.array(
                    qset.values_list(f"{filt}kronmagerr", flat=True), dtype=float
                )
                psf_mask = np.where(~np.isnan(phot_dict[f"{filt}_psf"]))[0]
                psferr_mask = np.where(~np.isnan(phot_dict[f"{filt}_psferr"]))[0]
                kron_mask = np.where(~np.isnan(phot_dict[f"{filt}_kron"]))[0]
                kronerr_mask = np.where(~np.isnan(phot_dict[f"{filt}_kronerr"]))[0]
                m = list(
                    set(psf_mask)
                    & set(psferr_mask)
                    & set(kron_mask)
                    & set(kronerr_mask)
                )
                idx_arr = np.array(m)
                phot_dict[f"{filt}"] = idx_arr[
                    np.where(
                        phot_dict[f"{filt}_psf"][m]
                        - phot_dict[f"{filt}_kron"][m]
                        - np.sqrt(  # 1 std below
                            phot_dict[f"{filt}_psferr"][m] ** 2
                            + phot_dict[f"{filt}_kronerr"][m] ** 2
                        )
                        > 0.05
                    )[0]
                ]
            # Coverage varies by filter.
            # find galaxies in all filters, if not, iterate removing least covered
            gals_in_filter = sorted([(len(phot_dict[i]), i) for i in "grizy"])[::-1]
            if gals_in_filter[0][0] == 0:
                if return_qset:
                    return qset.exclude(pk__in=qset)
                return
            for i in range(1, len(gals_in_filter)):
                intersection = set(phot_dict[gals_in_filter[0][1]])
                for j in range(i, len(gals_in_filter)):
                    intersection = intersection & set(phot_dict[gals_in_filter[-j][1]])
                if len(intersection) > 0:
                    break
            pks = np.array(qset.values_list("pk", flat=True))
            good_pks = pks[list(intersection)]
            if return_qset:
                return qset.filter(pk__in=good_pks)
            else:
                return qset.get(pk__in=good_pks)
        elif catalog == "ned":
            # For NED and Simbad, give first available from the hierarchy:
            # Ensembles, galaxy, multisystem, part of galaxy, candidates,
            # bandpass classifications, unclassified
            # Ignoring QSOs, AGN, Seyferts, Lensed images
            type_list = (
                ("GGroup", "GClstr"),
                ("G",),
                ("GPair", "GTrpl"),
                ("PofG",),
                ("AbLS", "EmLS", "IrS", "UvS"),
                ("-",),
            )
        elif catalog == "simbad":
            type_list = (
                ("GrG", "ClG", "CGG", "SCG"),
                ("G", "GiG", "GiC", "BiC", "rG", "H2G", "LSB", "EmG", "SBG", "bCG"),
                ("PaG", "IG", "mul"),
                ("PoG",),
                ("G?", "SC?", "C?G", "Gr?"),
                (
                    "Rad",
                    "cm",
                    "mm",
                    "smm",
                    "IR",
                    "FIR",
                    "MIR",
                    "NIR",
                    "blu",
                    "UV",
                    "X",
                    "ULX",
                    "gam",
                ),
                ("?"),
            )
        for types in type_list:
            if catalog == "ned":
                retset = qset.filter(pretype__in=types)
            elif catalog == "simbad":
                ret_pks = []
                for entry in qset:
                    if (
                        not set(entry.all_types).isdisjoint(types)
                        or entry.otype_3 in types
                    ):
                        ret_pks.append(entry.pk)
                retset = qset.filter(pk__in=ret_pks)
            if not retset.exists():
                continue
            if return_qset:
                return retset
            else:
                return qset.get(pk__in=list(retset.values_list("pk", flat=True)))

    def get_nearby(self, arcsec=2):
        """get_nearby.

        Parameters
        ----------
        arcsec :
            arcsec
        """
        qset = Galaxy.objects.box_search(self.ra, self.dec, arcsec / 3600).exclude(
            pk=self.pk
        )
        return qset

    def check_for_common_names(self, galaxy):
        """check_for_common_names.

        Figure out whether two galaxies have names in common with each other.
        Can't just make sets and see if not disjoint
        Need a way to compare strings such that
        'WISEA J073807.39+174219.0' == 'WISEA J073807+174219.0'
        Inconsistent coord representation means need regex and it's slow
        Also not sure if can look for str matches or rounding

        Parameters
        ----------
        galaxy :
            galaxy
        """
        for name in self.names:
            o_name = [n for n in galaxy.names if name.split()[0].upper() in n.upper()]
            if o_name == []:
                continue
            else:
                o_name = o_name[0]
            own_coords = re.search(
                "(J|B)(\d*\.\d*|\d*)([+\-]\d*\.\d*|[+\-]\d*)", name.split()[-1]
            )
            other_coords = re.search(
                "(J|B)(\d*\.\d*|\d*)([+\-]\d*\.\d*|[+\-]\d*)", o_name.split()[-1]
            )
            if own_coords:
                e1, ra1, dec1 = own_coords.groups()
            if other_coords:
                e2, ra2, dec2 = other_coords.groups()
            if not own_coords or not other_coords:
                if (
                    name.split()[-1] in o_name.split()[-1]
                    or o_name.split()[-1] in name.split()[-1]
                ):
                    return True
            elif (
                e1 == e2
                and (
                    (ra1 in ra2 or ra2 in ra1)
                    or (np.round(float(ra1)) == np.round(float(ra2)))
                )
                and (
                    (dec1 in dec2 or dec2 in dec1)
                    or (np.round(float(dec1)) == np.round(float(dec2)))
                )
            ):
                return True
        return False

    def merge(self, galaxy: object):
        """merge.

        Compare galaxy with self.
        Move all catalog entries to galaxy with more entries.

        Parameters
        ----------
        galaxy :
            Galaxy object
        """
        # strip catalog entries and attach to self
        entries = 0
        o_entries = 0
        for c in ("ned", "simbad", "ps1", "glade"):
            entries += getattr(self, f"{c}_entries").count()
            o_entries += getattr(galaxy, f"{c}_entries").count()
        if entries >= o_entries:
            primary = self
            secondary = galaxy
        elif entries < o_entries:
            primary = galaxy
            secondary = self
        for c in ("ned", "simbad", "ps1", "glade"):
            getattr(secondary, f"{c}_entries").all().update(galaxy=primary)
        secondary.targets.all().update(galaxy=primary)
        secondary.z_flag = "d"
        primary.z, primary.z_err, primary.z_flag = primary.get_redshift(query=False)
        secondary.save()

    def merge_near(self, arcsec=1.5):
        """merge_near.

        Parameters
        ----------
        arcsec :
            Radius to search
        """
        near = self.get_nearby(arcsec=arcsec)
        for g in near:
            if self.check_for_common_names(g):
                self.merge(g)

    def get_mass(self, mu="z", H0=72, q0=-0.53, C=1.04):
        m_k, e_m_k = self.get_2mass_m_k()
        if mu == "z":
            mu = utils.mu_lcdm(
                self.z, utils.convert_z(self.z, self.ra, self.dec), H0, q0
            )
        if m_k is not None:
            return -0.4 * (m_k - mu) + C, 0.4 * e_m_k
        neill09_mass = self.get_neill09_mass()
        if neill09_mass[0] is not None:
            return neill09_mass
        chang15_mass = self.get_chang15_mass()
        if chang15_mass[0] is not None:
            return chang15_mass
        return None, None

    def get_2mass_m_k(self):
        if not self.ned_entries.exists():
            return None, None
        ned_set = self.ned_entries.exclude(prefname__startswith="SN ")
        n = ned_set.first()
        phot_set = n.photometry.filter(
            observed_passband="K_s",
            refcode="20032MASX.C.......:",
            qualifiers__endswith="arcsec integration area.",
        )
        if phot_set.exists():
            m_k = phot_set[0].photometry_measurement
            e_m_k = float(str(phot_set[0].uncertainty).replace("+/-", ""))
            return m_k, e_m_k
        return None, None

    def get_neill09_mass(self, TNS_name="first", neill09_sne=None, neill09_gal=None):
        if neill09_sne is None:
            neill09_sne = pd.read_csv(f"{constants.HSF_DIR}/ado/neil09_sne.dat")
        if neill09_gal is None:
            neill09_gal = pd.read_csv(f"{constants.HSF_DIR}/ado/neil09_masses.dat")
        if TNS_name == "first":
            TNS_name = self.targets.first().TNS_name
        name_idx = np.where(
            neill09_sne["TNS_name"] == self.targets.get(TNS_name=TNS_name)
        )
        if len(name_idx[0]):
            host_name = neill09_sne["Host"][name_idx[0][0]]
            host_idx = np.where(neill09_gal["name"] == host_name)
            if len(host_idx[0]):
                m = neill09_gal["m"][host_idx[0][0]]
                e_m = (
                    neill09_gal["m+"][host_idx[0][0]]
                    - neill09_gal["m-"][host_idx[0][0]]
                ) / 2
                return m, e_m
        return None, None

    def get_chang15_mass(self, threshold=1, sw_input=None, sw_output=None):
        from astropy.io import fits

        if sw_input is None:
            sw_input = fits.open(f"{constants.HSF_DIR}/ado/sw_input.fits")
        if sw_output is None:
            sw_output = fits.open(f"{constants.HSF_DIR}/ado/sw_output.fits")
        ra, dec = self.ra, self.dec
        dist = np.sqrt(
            (ra - sw_input[1].data["ra"]) ** 2 * np.cos(dec * np.pi / 180) ** 2
            + (dec - sw_input[1].data["dec"]) ** 2
        )
        if min(dist) * 3600 < threshold:
            m = sw_output[1].data["lmass50_all"][np.argmin(dist)]

            e_m = (
                sw_output[1].data["lmass84_all"][np.argmin(dist)]
                - sw_output[1].data["lmass16_all"][np.argmin(dist)]
            ) / 2
            return m, e_m
        return None, None

    @staticmethod
    def import_data(path, delimiter="\t"):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        class_dict = {
            "galaxy": Galaxy,
            "snifsentry": SnifsEntry,
            "focasentry": FocasEntry,
            "gladeentry": GladeEntry,
            "panstarrsentry": PanStarrsEntry,
            "simbadentry": SimbadEntry,
            "simbadredshift": SimbadRedshift,
            "simbaddiameter": SimbadDiameter,
            "simbaddistance": SimbadDistance,
            "nedentry": NedEntry,
            "neddiameter": NedDiameter,
            "nedphotometry": NedPhotometry,
            "nedredshifts": NedRedshifts,
            "nedkinematics": NedKinematics,
            "nedluminosityclass": NedLuminosityClass,
            "nedmorphology": NedMorphology,
            "neddistance": NedDistance,
        }
        with open(path, "r") as f:
            for line in tqdm(f.readlines()):
                line = line.strip("\n")
                line_class_name = line.split(delimiter)[0]
                if line_class_name not in class_dict:
                    continue
                line_class = class_dict[line_class_name]
                defaults = {}
                for field, val in zip(
                    line_class._meta.fields, line.split(delimiter)[1:]
                ):
                    if val == "None":
                        defaults[field.name] = None
                    elif field.name == "galaxy":
                        defaults[field.name] = Galaxy.objects.get_or_create(
                            pk=int(val)
                        )[0]
                    elif field.name == "entry":
                        if "ned" in line_class_name:
                            defaults[field.name] = NedEntry.objects.get_or_create(
                                pk=int(val)
                            )[0]
                        elif "simbad" in line_class_name:
                            defaults[field.name] = SimbadEntry.objects.get_or_create(
                                pk=int(val)
                            )[0]
                    elif isinstance(field, models.fields.json.JSONField):
                        defaults[field.name] = json.loads(val)
                    elif isinstance(field, models.fields.DateTimeField):
                        defaults[field.name] = datetime.fromisoformat(val)
                    elif isinstance(field, models.fields.FloatField):
                        defaults[field.name] = float(val)
                    elif isinstance(
                        field,
                        (
                            models.fields.IntegerField,
                            models.fields.SmallIntegerField,
                            models.fields.BigIntegerField,
                            models.fields.PositiveIntegerField,
                            models.fields.PositiveSmallIntegerField,
                            models.fields.PositiveBigIntegerField,
                        ),
                    ):
                        defaults[field.name] = int(val)
                    else:
                        defaults[field.name] = str(val)
                if line_class_name == "gladeentry":
                    pk = "glade_no"
                elif line_class_name in ("panstarrsentry", "nedentry"):
                    pk = "objid"
                elif line_class_name == "simbadentry":
                    pk = "main_id"
                else:
                    pk = "id"
                pk = defaults.pop(pk)
                line_class.objects.update_or_create(pk=pk, defaults=defaults)
