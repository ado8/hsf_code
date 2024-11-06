import glob
import os
import warnings
from datetime import date

import astropy.units as u
import constants
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
import utils
from astropy.io import fits
from astropy.nddata import StdDevUncertainty
from astropy.time import Time
from astropy.utils.exceptions import AstropyWarning
from astropy.wcs import WCS
from django.core.exceptions import ObjectDoesNotExist
from django.db import models
from lightcurves.models import UkirtDetection
from rascal.calibrator import Calibrator
from rascal.util import refine_peaks
from scipy.signal import find_peaks
from specutils import Spectrum1D
from specutils.analysis import template_redshift
from utils import print_verb

warnings.simplefilter("ignore", AstropyWarning)

# Create your models here.
class FitsFileGoodManager(models.Manager):
    def get_queryset(self):
        return super(FitsFileGoodManager, self).get_queryset().filter(data_flag=True)


class FitsFileBadManager(models.Manager):
    def get_queryset(self):
        return super(FitsFileBadManager, self).get_queryset().filter(data_flag=False)


class FitsFile(models.Model):
    """
    Abstract class for data in FITS format
    """

    path = models.CharField(max_length=200, blank=True)
    name = models.CharField(max_length=40, default="NEED NAME", primary_key=True)
    date = models.DateField(null=True)
    mjd = models.FloatField(null=True)
    program = models.CharField(max_length=20, null=True)
    data_flag = models.BooleanField(default=True)

    objects = FitsFileGoodManager()
    objects_bad = FitsFileBadManager()
    objects_all = models.Manager()

    class Meta:
        abstract = True
        constraints = [models.UniqueConstraint(name="fits_file_path", fields=["path"])]

    def ds9(self):
        os.system(f"ds9 {self.path}")

    @property
    def header(self, extension="PRIMARY"):
        return fits.open(self.path)[extension].header

    @property
    def data(self, extension="PRIMARY"):
        return fits.open(self.path)[extension].data


class Observation(FitsFile):
    """
    Class representing one pointing of UKIRT.
    """

    STATUS_CHOICES = [
        ("science", "Science exposure"),
        ("pending", "Observed on UKIRT but not reduced"),
        ("reference", "Reference image for galaxy subtraction"),
        ("calibration", "Reference image for galaxy subtraction"),
        ("defocus", "Defocused image"),
        ("coords_err", "Issue with coordinate input"),
        ("name_err", "Issue with name"),
        ("bad_data", "Issue with data"),
        ("other", "Special, see target comment"),
    ]
    target = models.ForeignKey(
        "targets.Target", related_name="observations", on_delete=models.CASCADE
    )
    status = models.CharField(max_length=12, choices=STATUS_CHOICES, default="science")
    sn_mef = models.IntegerField(null=True)
    bandpass = models.CharField(max_length=10)
    conf_path = models.CharField(max_length=100, blank=True)
    cat_path = models.CharField(max_length=100, blank=True)
    exp_time = models.IntegerField(null=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(name="observation_path", fields=["path"])
        ]

    def save(self, *args, **kwargs):
        if self.name == "NEED NAME":
            self.name = self.path.split("/")[-1].split("_s")[0]
        if self.status != "pending":
            if self.mjd is None:
                self.get_mjd()
            if self.date is None:
                self.get_date()
            if self.program is None:
                self.get_program()
            if self.sn_mef is None:
                self.get_sn_mef()
            if self.bandpass == "":
                self.get_bandpass()
            if self.exp_time is None:
                self.get_exp_time()
            if os.path.exists(self.path[:-4] + "_conf.fit"):
                self.conf_path = self.path[:-4] + "_conf.fit"
            if os.path.exists(self.path[:-4] + "_cat.fits"):
                self.cat_path = self.path[:-4] + "_cat.fits"
        super(Observation, self).save(*args, **kwargs)

    def __str__(self):
        return f"{self.target}: {self.name}"

    def fits(self, mef=None):
        if mef is None:
            return fits.open(self.path)
        elif isinstance(mef, int) or isinstance(mef, str):
            return fits.open(self.path)[mef]
        elif isinstance(mef, tuple) or isinstance(mef, list):
            return [fits.open(obs.path)[m] for m in mef]

    def get_date(self):
        obs_date = self.fits("PRIMARY").header["DATE-OBS"]
        y, m, d = obs_date.split("T")[0].split("-")
        self.date = date(int(y), int(m), int(d))

    def get_mjd(self):
        self.mjd = self.fits("PRIMARY").header["MJD-OBS"]

    def get_program(self):
        self.program = self.fits("PRIMARY").header["PROJECT"]

    def get_sn_mef(self):
        contains, mef = self.contains(self.target.ra, self.target.dec)
        if contains:
            self.sn_mef = mef
        else:
            print(
                f"WARNING: {self.name} is linked to {self.target}, but does not contain its coordinates. Consider setting status to coords_err."
            )

    def get_bandpass(self):
        self.bandpass = self.fits("PRIMARY").header["FILTER"]

    def get_exp_time(self):
        self.exp_time = int(self.fits("PRIMARY").header["EXP_TIME"])

    def contains(self, ra, dec):
        for mef in range(1, 5):
            hdr = self.fits(mef).header
            [x, y] = utils.sky2pix(hdr, ra, dec)
            if x > 0 and x < hdr["NAXIS1"] and y > 0 and y < hdr["NAXIS2"]:
                return True, mef
        return False, None

    def make_image(self, mef, force=True, galaxy=True):
        if isinstance(mef, tuple):
            for m in mef:
                self.make_image(mef=m, force=force, galaxy=galaxy)
            return
        if mef == "sn":
            mef = self.sn_mef
        if not force and self.images.filter(mef=mef).exists():
            return
        hdul = fits.open(self.path)
        obs = hdul[mef]
        obs_im = obs.data
        obs_header = obs.header
        for key in ["MJD-OBS", "AMSTART", "AMEND", "EXP_TIME"]:
            obs_header[key] = hdul["PRIMARY"].header[key]

        if self.target.sn_type.name == "CALSPEC":
            subdir = "non_transient"
        else:
            subdir = f"20{self.target.TNS_name[:2]}"
        out_dir = f"{constants.DATA_DIR}/{subdir}/{self.target.TNS_name}/ukirt/{self.name}/{mef}"
        out_path = f"{out_dir}/im.fits"
        os.makedirs(out_dir, exist_ok=True)

        fits.writeto(out_path, obs_im, header=obs_header, overwrite=True)
        image, _ = Image.objects.get_or_create(observation=self, mef=mef)
        image.path = out_path
        image.status = "?"
        try:
            conf = fits.open(self.path.replace(".fit", "_conf.fit"))[mef]
            fits.writeto(
                out_path[:-4] + "conf.fits",
                conf.data,
                header=conf.header,
                overwrite=True,
            )
            image.conf = out_path[:-4] + "conf.fits"
        except FileNotFoundError:
            print(f"{out_path[:-4]}conf.fits not found")
        except IndexError:
            print(f"{out_path[:-4]}conf.fits doesn't have enough extensions")
        image.save()
        image.plot(force=True, galaxy=galaxy)

    @property
    def sn_image(self):
        qset = self.images.filter(mef=self.sn_mef)
        if qset.exists():
            return qset.first()


class Image(models.Model):
    """
    Single image from the WFCAM 4 chip array.
    """

    STATUS_CHOICES = [
        ("g", "good"),
        ("f", "fixable"),
        ("b", "bad"),
        ("?", "uninspected"),
    ]
    observation = models.ForeignKey(
        Observation, related_name="images", on_delete=models.CASCADE
    )
    mef = models.IntegerField(default=0)
    path = models.CharField(max_length=200)
    status = models.CharField(max_length=2, choices=STATUS_CHOICES)
    conf_path = models.CharField(max_length=100, blank=True)
    cat_path = models.CharField(max_length=100, blank=True)
    sn_ra = models.FloatField(default=0)
    sn_dec = models.FloatField(default=0)
    rot_ra = models.FloatField(default=0)
    rot_dec = models.FloatField(default=0)
    major = models.FloatField(null=True)
    minor = models.FloatField(null=True)
    phi = models.FloatField(null=True)
    zp_guess = models.FloatField(null=True)
    zp_err_guess = models.FloatField(null=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                name="observation_mef", fields=["observation", "mef"]
            ),
            models.UniqueConstraint(name="image_path", fields=["path"]),
        ]
        indexes = [
            models.Index(fields=["observation"]),
            models.Index(fields=["mef"]),
        ]

    def __save__(self, *args, **kwargs):
        if os.path.exists(self.path[:-4] + "_conf.fit"):
            self.conf_path = self.path[:-4] + "_conf.fit"
        if os.path.exists(self.path[:-4] + "_cat.fits"):
            self.conf_path = self.path[:-4] + "_cat.fits"
        super(Image, self).save(*args, **kwargs)

    def __str__(self):
        return f"{self.observation} chip {self.mef}"

    def get_path(self, sub_type="nosub"):
        if sub_type == "nosub":
            return self.path
        elif sub_type == "rotsub":
            return self.path[:-4] + "rotsub.fits"
        elif sub_type == "refsub":
            return self.path[:-4] + "refsub.fits"

    def fits(self, sub_type="nosub"):
        return fits.open(self.get_path(sub_type))["PRIMARY"]

    def contains_sky(
        self, ra=None, dec=None, coords=None, coord_axis=0, sub_type="nosub"
    ):
        header = self.fits(sub_type).header
        pix_coords = utils.sky2pix(
            self.fits(sub_type).header,
            ra=ra,
            dec=dec,
            coords=coords,
            coord_axis=coord_axis,
        )
        m1 = pix_coords[0] > 0
        m2 = pix_coords[1] > 0
        m3 = pix_coords[0] < header["NAXIS1"]
        m4 = pix_coords[1] < header["NAXIS2"]
        return m1 & m2 & m3 & m4

    def contains_pix(self, x=None, y=None, coords=None, coord_axis=0, sub_type="nosub"):
        header = self.fits(sub_type).header
        if x is not None and y is not None:
            try:
                x = float(x)
                y = float(y)
                pix_coords = np.array([[x, y]]).T
            except TypeError:
                if len(x) != len(y):
                    raise SyntaxError(
                        "Number of x coords not equal to the number of y coords"
                    )
                pix_coords = np.array([x, y])
        elif coords is not None:
            if coord_axis == 0:
                pix_coords = np.array(coords).T
            elif coord_axis == 1:
                pix_coords = np.array(coords)
        m1 = pix_coords[0] > 0
        m2 = pix_coords[1] > 0
        m3 = pix_coords[0] < header["NAXIS1"]
        m4 = pix_coords[1] < header["NAXIS2"]
        return m1 & m2 & m3 & m4

    def get_rotsub(
        self, center_of_rotation=None, force=False, padding=constants.IM_PAD
    ):
        # just sn for now, but can imagine generalizing for galaxy subtraction tests
        data = self.fits().data
        header = self.fits().header
        out_path = self.get_path("rotsub")
        if os.path.exists(out_path) and not force:
            print(
                f"Already see a rotsub file for this image. Consider running again with force=True"
            )
            return
        if center_of_rotation is None:
            gra = self.observation.target.galaxy.ra
            gdec = self.observation.target.galaxy.dec
            if not self.contains_sky(gra, gdec):
                raise ValueError("Center of rotation is not in the image")
            [x, y] = utils.sky2pix(
                header,
                self.observation.target.galaxy.ra,
                self.observation.target.galaxy.dec,
            ).astype(int)
        elif type(center_of_rotation) in [tuple, list]:
            if len(center_of_rotation) == 2:
                x, y = center_of_rotation
            else:
                raise SyntaxError(
                    "Center of rotation coordinates improperly defined. Provide a tuple or list of ra/dec in degrees"
                )
        [sn_x, sn_y] = utils.sky2pix(
            header, self.observation.target.ra, self.observation.target.galaxy.dec
        ).astype(int)
        dx = np.abs(sn_x - x)
        dy = np.abs(sn_y - y)
        padding = max(dx, dy) + padding
        if padding:
            utils.crop_fits(
                x - padding,
                x + padding,
                y - padding,
                y + padding,
                data,
                header,
                out_path=out_path,
                overwrite=True,
            )
        else:
            fits.writeto(out_path, data, header)
        utils.rotational_subtract(
            out_path,
            0,
            self.observation.target.galaxy.ra,
            self.observation.target.galaxy.dec,
            self.observation.target.ra,
            self.observation.target.dec,
            self.rot_ra,
            self.rot_dec,
            out_path,
        )
        self.plot(sub_type="rotsub", crop_around="galaxy", force=True)
        self.save()

    @property
    def bandpass(self):
        return self.observation.bandpass

    def get_isis_refsub(self, reference, force=True):  # move to image
        if not self.target.reference_set.exists():
            return
        out_path = f"{constants.DATA_DIR}/20{self.observation.target.TNS_name[:2]}/{self.observation.target.TNS_name}/ukirt/{self.observation.name}/{self.mef}/refsub.fits"
        hdr = fits.open(self.path).header
        refhdr = fits.open(reference.path).header
        convolve_target = ""
        if hdr["SEEING"] < refhdr["SEEING"]:
            convolve_target = "-c i"
        os.system(
            f"{constants.HSF_DIR}/env/hotpants/hotpants -inim {self.path} -tmplim {reference.path} -outim {out_path} -tg {refhdr['GAIN']} -tr {refhdr['READNOIS']} -ig {hdr['GAIN']} -ir {hdr['READNOIS']} {convolve_target} -n i"
        )

    def get_isis_refsub(self, reference, force=True):  # move to image
        if not self.target.reference_set.exists():
            return
        try:
            ref_header = utils.fit_images(self.path, reference.path, rebin=(2, 2))
        except ValueError:
            return
        utils.crop_bright(
            a_path=f"{constants.ISIS_DIR}/new.fits",
            b_path=f"{constants.ISIS_DIR}/ref.fits",
            sn_ra=self.target.ra,
            sn_dec=self.target.dec,
        )
        utils.ISIS(
            a_path="{constants.ISIS_DIR}/new.fits",
            b_path=f"{constants.ISIS_DIR}/ref.fits",
        )
        out_path = f"{constants.DATA_DIR}/20{self.observation.target.TNS_name[:2]}/{self.observation.target.TNS_name}/ukirt/{self.observation.name}/{self.mef}/refsub.fits"
        fits.writeto(
            out_path,
            -fits.open(f"{constants.ISIS_DIR}/conv_new.fits")["PRIMARY"].data,
            header=ref_header,
            overwrite=os.path.exists(out_path),
        )
        image = Image.objects.get_or_create(
            observation=self.observation, sub_type="refsub", mef=mef
        )[0]
        image.path = out_path
        image.status = "?"
        image.save()
        image.plot(force=True)

    def ds9(self, sub_type="nosub"):
        os.system(f"ds9 {self.get_path(sub_type)}")

    def get_reference_stars(self, force=False, verbose=False):
        """
        download list of reference stars using refcat
        """
        rs_set = self.reference_stars.all()
        initial_rs_count = rs_set.count()
        if rs_set.exists() and not force:
            if verbose:
                print(
                    f"Already see {initial_rs_count} reference stars. Consider running again with force=True"
                )
            return

        header = self.fits().header
        n1 = header["NAXIS1"]
        n2 = header["NAXIS2"]
        corners = utils.pix2sky(header, [0, n1, 0, n1], [0, 0, n2, n2]).T
        ra = corners.ra.value
        dec = corners.dec.value
        dr = (max(ra) - min(ra)) / 2 * constants.REFCAT_PADDING
        dd = (max(dec) - min(dec)) / 2 * constants.REFCAT_PADDING
        rc = utils.refcat(np.average(ra), np.average(dec), dr=dr, dd=dd, all_cols=True)

        # Trim rc to just objects in image
        pix_coords = utils.sky2pix(header, ra=rc["ra"], dec=rc["dec"])
        m = self.contains_sky(rc["ra"], rc["dec"])
        pix_coords = np.array([pix_coords[0][m], pix_coords[1][m]])
        rc = rc[m]
        rc["projected_x"] = pix_coords[0]
        rc["projected_y"] = pix_coords[1]
        rc = rc.reset_index(drop=True)
        for i, row in rc.iterrows():
            defaults = {}
            for idx in row.index:
                if idx not in ["ra", "dec", "projected_x", "projected_y"]:
                    defaults[idx] = row[idx]
            rs, _ = ReferenceStar.objects.update_or_create(
                ra=row["ra"], dec=row["dec"], defaults=defaults
            )
            m1 = np.abs(rc["ra"] - row["ra"]) < np.cos(row["dec"] * np.pi / 180) / 3600
            m2 = np.abs(rc["dec"] - row["dec"]) < 1 / 3600
            if len(rc[m1 & m2]) > 1:
                continue
            rsd, _ = ReferenceStarDetails.objects.update_or_create(
                image=self,
                reference_star=rs,
                defaults={
                    "projected_x": row["projected_x"],
                    "projected_y": row["projected_y"],
                },
            )
        if verbose:
            print(
                f"Found {self.reference_stars.count() - initial_rs_count} new stars on top of {initial_rs_count} already known stars"
            )

    def get_psf(self, force=False, verbose=False):
        """
        Create a default tphot catalog of image
        Apply cuts to major axis, minor axis, eccentricity, flux
        """
        if (
            not force
            and self.major is not None
            and self.minor is not None
            and self.phi is not None
        ):
            if verbose:
                print(
                    f"Already see PSF parameters major={self.major} minor={self.minor} phi={self.phi}. Consider running again with force=True."
                )
            return
        if self.observation.target.sn_type.name == "CALSPEC":
            subdir = "non_transient"
        else:
            subdir = f"20{self.observation.target.TNS_name[:2]}"
        tph = utils.tphot(
            path=self.path,
            force=force,
            tph_force=False,
            arg_dict={
                "out": f"{constants.DATA_DIR}/{subdir}/{self.observation.target.TNS_name}/ukirt/{self.observation.name}/{self.mef}/default.tph"
            },
        )
        e_lim = np.quantile(
            tph["major"] / tph["minor"], constants.PSF_ECCENTRICITY_QUANTILE
        )
        size_lim = np.quantile(tph["major"], constants.PSF_SIZE_QUANTILE)
        flux_lim = np.quantile(tph["flux"], constants.PSF_FLUX_QUANTILE)
        mask1 = tph["major"] > size_lim[0]
        mask2 = tph["major"] < size_lim[1]
        mask3 = tph["major"] / tph["minor"] < e_lim
        mask4 = tph["flux"] < flux_lim
        mask = mask1 & mask2 & mask3 & mask4
        self.major = np.median(tph["major"][mask])
        self.minor = np.median(tph["minor"][mask])
        self.phi = np.median(tph["phi"][mask])
        self.save()
        if verbose:
            print(f"Found major={self.major} minor={self.minor} phi={self.phi}")

    def rubin_get_psf(self, psfsize=27, verb=False):
        import sep
        import tqdm
        from astropy.nddata import NDData
        from astropy.table import Table
        from fitting.RubinsNM import miniNM_new, save_img
        from photutils.psf import EPSFBuilder, extract_stars
        from scipy.interpolate import RectBivariateSpline

        hdul = fits.open(self.path)
        f = hdul["PRIMARY"]

        dat = f.data * 1.0

        w0 = WCS(f.header)

        filt = f.header["SKYSUB"].split(".fit")[0][-1]

        # ADUSCALE = f.header["ADUSCALE"]

        arcsecperpix = (
            max(
                np.abs(f.header["CD1_1"]),
                np.abs(f.header["CD1_2"]),
                np.abs(f.header["CD2_1"]),
                np.abs(f.header["CD2_2"]),
            )
            * 3600.0
        )  # not right, but close enough here

        print_verb(verb, "arcsecperpix", arcsecperpix)

        dat -= sep.Background(dat, bw=128)
        the_NMAD = utils.NMAD(dat)
        print_verb(verb, "the_NMAD", the_NMAD)

        objs = sep.extract(dat, thresh=the_NMAD * 2.0)

        print_verb(verb, "objs", objs)
        print_verb(verb, "objs x", objs["x"])
        print_verb(verb, len(objs), "objects found")

        im_dir = "/".join(self.path.split("/")[:-1])
        f_reg = open(f"{im_dir}/2M_stars.reg", "w")
        f_reg.write(
            """# Region file format: DS9 version 4.1
    global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1
    physical
    """
        )

        good_obj_mask = (
            (objs["peak"] < 10000)
            & (objs["peak"] > the_NMAD * 100)
            & (objs["x"] > 55)
            & (objs["x"] < len(dat[0]) - 55)
            & (objs["y"] > 55)
            & (objs["y"] < len(dat) - 55)
        )

        for i in tqdm.trange(len(good_obj_mask)):

            if good_obj_mask[i]:
                ij = [
                    int(np.around(objs["y"][i])) - 1,
                    int(np.around(objs["x"][i])) - 1,
                ]
                pixels_around_peak = dat[ij[0] - 5 : ij[0] + 6, ij[1] - 5 : ij[1] + 6]
                # print_verb(verb, "pixels_around_peak", pixels_around_peak, ij, objs["x"][i], objs["y"][i])

                if pixels_around_peak.max() < objs["peak"][i] * 0.5:
                    # No maximum nearby!
                    good_obj_mask[i] = 0

            for j in range(len(good_obj_mask)):
                if i != j:
                    delta_pix = np.sqrt(
                        (objs["x"][i] - objs["x"][j]) ** 2.0
                        + (objs["y"][i] - objs["y"][j]) ** 2.0
                    )
                    delta_arcsec = delta_pix * arcsecperpix

                    if delta_arcsec < 15:
                        # Does j contaminate i?
                        i_at_j = (
                            objs["peak"][i] / (1.0 + (delta_arcsec / 2.0) ** 2.0) ** 2.0
                        )

                        if (
                            i_at_j < objs["peak"][j] * 50
                            and i_at_j / objs["peak"][i] > 0.005
                        ):
                            good_obj_mask[i] = 0

        print_verb(verb, "good objects", sum(good_obj_mask))

        for i in range(len(objs["x"])):
            f_reg.write(
                "circle(%f,%f,%i)\n"
                % (objs["x"][i], objs["y"][i], 2 + 8 * good_obj_mask[i])
            )
        f_reg.close()

        hdul.close()

        stars_tbl = Table()
        stars_tbl["x"] = objs["x"][good_obj_mask]
        stars_tbl["y"] = objs["y"][good_obj_mask]

        stamps = extract_stars(NDData(dat), stars_tbl, size=35)

        save_img(
            [stamps[i].data for i in range(len(stamps))], f"{im_dir}/stamps.fits",
        )

        epsf_builder = EPSFBuilder(oversampling=1.0, maxiters=15, progress_bar=True)
        epsf, fitted_stars = epsf_builder(stamps)

        print_verb(verb, "epsf.data", np.sum(epsf.data))
        the_psf = epsf.data  # Junk around the edges, so cutout

        psf_center = the_psf * 1.0
        psf_center[:5] *= 0.0
        psf_center[-5:] *= 0.0
        psf_center[:, :5] *= 0.0
        psf_center[:, -5:] *= 0.0

        brightest_ij = np.where(psf_center == psf_center.max())

        the_psf = the_psf[
            brightest_ij[0][0] - 13 : brightest_ij[0][0] + 14,
            brightest_ij[1][0] - 13 : brightest_ij[1][0] + 14,
        ]

        print_verb(verb, "the_psf", the_psf.sum())

        the_psf /= the_psf.sum()

        print_verb(verb, "the_psf", the_psf.sum())

        save_img(the_psf, f"{im_dir}/epsf.fits")

        subxs = np.arange(len(the_psf), dtype=np.float64) / 1.0
        subxs -= np.median(subxs)

        xs = np.arange(psfsize, dtype=np.float64)
        xs -= np.median(xs)

        ifn = RectBivariateSpline(subxs, subxs, the_psf, kx=2, ky=2)

        evalxs = np.arange(psfsize * 3, dtype=np.float64) / 3.0
        evalxs -= np.mean(evalxs)

        save_img(ifn(evalxs, evalxs), f"{im_dir}/epsfx3.fits")

        evalxs = np.arange(psfsize * 3 * 9, dtype=np.float64) / 27.0
        evalxs -= np.mean(evalxs)

        tmp_PSF = ifn(evalxs, evalxs) / (9.0 ** 2)
        ePSFsubpix = 0.0

        for i in range(9):
            for j in range(9):
                ePSFsubpix += tmp_PSF[i::9, j::9]

        save_img(ePSFsubpix, f"{im_dir}/epsfx3subpix.fits")

        f_phot = open(f"{im_dir}/field_stars.txt", "w",)
        all_stars = []

        ZP_dict = {"2MASSJ": [], "2MASSdJ": [], "UKInstJ": []}

        def chi2fn(P, alldat, ifn=ifn, xs=xs):
            mod = P[0] * ifn(xs - P[1], xs - P[2])
            resid = alldat[0] - mod
            resid -= np.mean(resid)
            return np.sum(resid ** 2.0)

        rs = self.reference_stars.all()
        for ind in np.where(good_obj_mask > 0)[0]:
            print_verb(verb, "ind", ind)
            stardat = dat[
                int(np.around(objs["y"][ind]))
                - 13 : int(np.around(objs["y"][ind]))
                + 14,
                int(np.around(objs["x"][ind]))
                - 13 : int(np.around(objs["x"][ind]))
                + 14,
            ]

            P, NA, NA = miniNM_new(
                ministart=[np.sum(stardat), 0.0, 0.0],
                miniscale=[np.sum(stardat) / 10.0, 2.0, 2.0],
                passdata=stardat,
                chi2fn=chi2fn,
                verbose=False,
                compute_Cmat=False,
            )

            mod = P[0] * ifn(xs - P[1], xs - P[2])
            all_stars.append(
                np.concatenate((stardat, mod, (stardat - mod) / mod.max()))
            )
            # save_img([stardat, mod, stardat - mod], "stardat.fits")
            RA_Dec = w0.all_pix2world([[objs["x"][ind], objs["y"][ind]]], 1)[0]
            print_verb(verb, "RA_Dec", RA_Dec)

            # TMJ, TMdJ, TMH, TMdH = get_2MASS(RA_Dec[0], RA_Dec[1])
            temp_rs = rs.filter(
                ra__gt=RA_Dec[0] - 0.002,
                ra__lt=RA_Dec[0] + 0.002,
                dec__gt=RA_Dec[1] - 0.002,
                dec__lt=RA_Dec[1] + 0.002,
            )
            if temp_rs.count() == 1:
                trs = temp_rs.first()
                TMJ, TMdJ, TMH, TMdH = trs.j, trs.dj, trs.h, trs.dh
            else:
                TMJ, TMdJ, TMH, TMdH = [0 for i in range(4)]

            if TMJ > 1 and TMH > 1 and P[0] > 0:
                if TMJ == TMH:
                    print_verb(verb, "Warning! J-H = 0")

                if filt == "J":
                    ZP_dict["2MASSJ"].append(TMJ - 0.065 * (TMJ - TMH))
                    ZP_dict["2MASSdJ"].append(
                        np.sqrt(TMdJ ** 2.0 + (0.065 * TMdH) ** 2.0)
                    )
                elif filt == "H":
                    ZP_dict["2MASSJ"].append(TMH + 0.070 * (TMJ - TMH) - 0.030)
                    ZP_dict["2MASSdJ"].append(
                        np.sqrt(TMdH ** 2.0 + (0.070 * TMdJ) ** 2.0)
                    )
                elif filt == "Y":
                    ZP_dict["2MASSJ"].append(TMJ + 0.500 * (TMJ - TMH) + 0.080)
                    ZP_dict["2MASSdJ"].append(
                        np.sqrt((1.5 * TMdJ) ** 2.0 + (0.50 * TMdH) ** 2.0)
                    )

                ZP_dict["UKInstJ"].append(-2.5 * np.log10(P[0]))

            towrite = [
                RA_Dec[0],
                RA_Dec[1],
                objs["x"][ind],
                objs["y"][ind],
                P[0],
                TMJ,
                TMdJ,
                TMH,
                TMdH,
            ]
            towrite = [str(item) for item in towrite]

            f_phot.write("  ".join(towrite) + "\n")
        f_phot.close()
        save_img(all_stars, f"{im_dir}/all_stars.fits")

    def rubin_get_zp(self):
        import multiprocessing
        import sys

        import pystan
        from fitting.FileRead import readcol
        from scipy.stats import scoreatpercentile

        multiprocessing.set_start_method("fork")
        stan_code = """
        data {
            int n_obs;
            int n_star;
            int n_im;

            int <lower = 0, upper = n_im - 1> im_inds [n_obs];
            int <lower = 0, upper = n_star - 1> star_inds [n_obs];

            vector [n_obs] instr_mags;
            vector [n_obs] x_norm;
            vector [n_obs] y_norm;

            vector [n_star] twomass_mags; // 0 for missing
            vector [n_star] twomass_dmags;
        }

        parameters {
            vector [n_star] star_mags;
            vector [n_im] zeropoints;
            vector [n_im] dz_dx;
            vector [n_im] dz_dy;

            vector [n_im] dz_dxx;
            vector [n_im] dz_dxy;
            vector [n_im] dz_dyy;

            real <lower = 0.01, upper = 0.1> ext_floor;
            real <lower = 0.2, upper = 2> ext_outl;
            real <lower = 0.001, upper = 0.1> ext_frac;

            vector <lower = 0.002, upper = 0.05> [n_im] int_floor;
            real <lower = 0.1, upper = 2> int_outl;
            vector <lower = 0.001, upper = 0.999> [n_star] int_frac;
        }

        transformed parameters {
            vector [n_obs] instr_model_mags;

            for (i in 1:n_obs) {
                instr_model_mags[i] = star_mags[star_inds[i] + 1] - (
                                                                    zeropoints[im_inds[i] + 1] + dz_dx[im_inds[i] + 1]*x_norm[i] + dz_dy[im_inds[i] + 1]*y_norm[i]
                                                                    + dz_dxx[im_inds[i] + 1]*x_norm[i]^2 + dz_dxy[im_inds[i] + 1]*x_norm[i]*y_norm[i] + dz_dyy[im_inds[i] + 1]*y_norm[i]^2 
                                                                    );
            }
        }

        model {
            for (i in 1:n_star) {
                if (twomass_mags[i] > 2) {
                    target += log_sum_exp(
                                         log(1. - ext_frac) + normal_lpdf(star_mags[i] | twomass_mags[i], sqrt(twomass_dmags[i]^2 + ext_floor^2)),
                                         log(ext_frac) + normal_lpdf(star_mags[i] | twomass_mags[i], sqrt(twomass_dmags[i]^2 + ext_outl^2))
                                         );
                }
            }

            for (i in 1:n_obs) {
                if (instr_mags[i] < 5) {
                    target += log_sum_exp(
                                         log(1. - int_frac[star_inds[i] + 1]) + normal_lpdf(instr_mags[i] | instr_model_mags[i], int_floor[im_inds[i] + 1]),
                                         log(int_frac[star_inds[i] + 1]) + normal_lpdf(instr_mags[i] | instr_model_mags[i], int_outl)
                                         );
                }
            }

            star_mags ~ normal(15, 5);
            dz_dx ~ normal(0, 0.05);
            dz_dy ~ normal(0, 0.05);
            int_frac ~ cauchy(0, 0.03);

            dz_dxx ~ normal(0, 0.007); // Approximately +0.021 mag at the edges
            dz_dxy ~ normal(0, 0.007);
            dz_dyy ~ normal(0, 0.007);
        }

        """

        def get_ang(ra1, dec1, ra2, dec2):
            if ra1 == ra2 and dec1 == dec2:
                return 0.0

            r1 = np.array(
                [
                    np.cos(ra1 * np.pi / 180.0) * np.cos(dec1 * np.pi / 180.0),
                    np.sin(ra1 * np.pi / 180.0) * np.cos(dec1 * np.pi / 180.0),
                    np.sin(dec1 * np.pi / 180.0),
                ],
                dtype=np.float64,
            )
            r2 = np.array(
                [
                    np.cos(ra2 * np.pi / 180.0) * np.cos(dec2 * np.pi / 180.0),
                    np.sin(ra2 * np.pi / 180.0) * np.cos(dec2 * np.pi / 180.0),
                    np.sin(dec2 * np.pi / 180.0),
                ],
                dtype=np.float64,
            )

            ang = np.arccos(np.dot(r1, r2)) * 180.0 / np.pi
            return ang

        def show_residuals_by_star(all_obs, fit_params):
            median_residuals = all_obs["instr_mags"] - np.median(
                fit_params["instr_model_mags"], axis=0
            )

            for star_ind in np.sort(np.unique(all_obs["star_inds"])):
                inds = np.where(all_obs["star_inds"] == star_ind)
                print("star_ind", star_ind, "resids", list(median_residuals[inds]))

        [TNS_name, reference, path, filts, sn_RA, sn_Dec, gal_RA, gal_Dec] = readcol(
            "observations.dat", "aaaa,ff,ff"
        )

        path = [item.split("/")[-1] for item in path]

        for i in range(len(TNS_name)):
            TNS_name[i] += "_" + filts[i]

        TNS_name = np.array(TNS_name)
        path = np.array(path)

        print("SN ", sys.argv[1])
        inds = np.where(sys.argv[1] == TNS_name)

        this_RA = sn_RA[inds][0]
        this_Dec = sn_Dec[inds][0]

        filt = sys.argv[1].split("_")[-1]
        fls = list(np.unique(path[inds]))
        fls.sort()

        print("fls", fls)

        for i in range(len(fls))[::-1]:
            field_star_fl = "2M_stars/field_stars_" + fls[i].replace(".fits", ".txt")
            if glob.glob(field_star_fl) == []:
                print("Couldn't find stars for ", fls[i])
                del fls[i]
            else:
                [RA, Dec, x, y] = readcol(field_star_fl, "ff,ff")
                if len(RA) < 3:
                    del fls[i]

        print("fls", fls)

        unique_stars = dict(RAs=[], Decs=[], Js=[], dJs=[], Hs=[], dHs=[], n_obs=[])
        all_obs = dict(
            xs=[],
            ys=[],
            delt_xs=[],
            delt_ys=[],
            star_inds=[],
            im_inds=[],
            instr_mags=[],
        )

        SN_x_ys = []

        for fl in fls:
            # 2M_stars/field_stars_w20190322_01321.2.txt
            # 175.84667519253503  20.045464883452794  2162.383076872126  473.65542348295793  17879.640090395114  13.695  0.021  13.379  0.032

            fitsfl = fits.open("ukirt_data/" + fl)
            w0 = WCS(fitsfl[1].header)
            fitsfl.close()

            SN_x_y = w0.all_world2pix([[this_RA, this_Dec]], 1)[0]
            print("SN_x_y", SN_x_y)
            SN_x_ys.append(SN_x_y)

            [RA, Dec, x, y, ampl, J, dJ, H, dH] = readcol(
                "2M_stars/field_stars_" + fl.replace(".fits", ".txt"), "ff,ff,f,ff,ff"
            )

            for i in range(len(RA)):
                if get_ang(RA[i], Dec[i], this_RA, this_Dec) > 0.001:
                    closest_point = 100
                    for j in range(len(unique_stars["RAs"])):
                        ang = get_ang(
                            unique_stars["RAs"][j],
                            unique_stars["Decs"][j],
                            RA[i],
                            Dec[i],
                        )

                        if ang < closest_point:
                            closest_point = ang
                            best_j = j

                    if closest_point > 0.0005:
                        best_j = len(unique_stars["n_obs"])
                        unique_stars["RAs"].append(RA[i])
                        unique_stars["Decs"].append(Dec[i])
                        unique_stars["Js"].append(J[i])
                        unique_stars["dJs"].append(dJ[i])
                        unique_stars["Hs"].append(H[i])
                        unique_stars["dHs"].append(dH[i])
                        unique_stars["n_obs"].append(1)

                    else:
                        unique_stars["n_obs"][best_j] += 1

                    all_obs["xs"].append(x[i])
                    all_obs["ys"].append(y[i])

                    all_obs["delt_xs"].append(x[i] - SN_x_y[0])
                    all_obs["delt_ys"].append(y[i] - SN_x_y[1])

                    all_obs["star_inds"].append(best_j)
                    all_obs["im_inds"].append(fls.index(fl))
                    if ampl[i] > 0:
                        all_obs["instr_mags"].append(-2.5 * np.log10(ampl[i]))
                    else:
                        all_obs["instr_mags"].append(10.0)

        for key in all_obs:
            all_obs[key] = np.array(all_obs[key])

        for key in unique_stars:
            unique_stars[key] = np.array(unique_stars[key])

        all_obs["n_star"] = len(unique_stars["RAs"])
        all_obs["n_obs"] = len(all_obs["xs"])
        all_obs["n_im"] = len(fls)

        if filt == "J":
            all_obs["twomass_mags"] = unique_stars["Js"] - 0.065 * (
                unique_stars["Js"] - unique_stars["Hs"]
            )
            all_obs["twomass_dmags"] = np.sqrt(
                (unique_stars["dJs"] * 0.935) ** 2.0
                + (unique_stars["dHs"] * 0.065) ** 2.0
            )
        elif filt == "Y":
            all_obs["twomass_mags"] = (
                unique_stars["Js"]
                + 0.500 * (unique_stars["Js"] - unique_stars["Hs"])
                + 0.080
            )
            all_obs["twomass_dmags"] = np.sqrt(
                (unique_stars["dJs"] * 1.5) ** 2.0 + (unique_stars["dHs"] * 0.5) ** 2.0
            )
        elif filt == "H":
            all_obs["twomass_mags"] = (
                unique_stars["Hs"]
                + 0.070 * (unique_stars["Js"] - unique_stars["Hs"])
                - 0.030
            )
            all_obs["twomass_dmags"] = np.sqrt(
                (unique_stars["dHs"] * 0.93) ** 2.0
                + (unique_stars["dJs"] * 0.07) ** 2.0
            )
        else:
            could_not_find_filt

        sm = pystan.StanModel(model_code=stan_code)

        all_obs["x_norm"] = all_obs["delt_xs"] / np.std(all_obs["delt_xs"])
        all_obs["y_norm"] = all_obs["delt_ys"] / np.std(all_obs["delt_ys"])

        fit_is_good = 0

        iteration = 5000

        while fit_is_good == 0:
            fit = sm.sampling(data=all_obs, iter=iteration, chains=4, refresh=100)
            print(fit)
            fit_params = fit.extract(permuted=True)

            highest_Rhat = 0
            str_fit = str(fit).split("\n")
            for line in str_fit:
                if line.count("zeropoints"):
                    parsed = line.split(None)
                    print(parsed)
                    highest_Rhat = max(highest_Rhat, float(parsed[-1]))

            if highest_Rhat < 1.05:
                fit_is_good = 1
            else:
                if (
                    highest_Rhat <= 1.07
                    and np.std(fit_params["zeropoints"], axis=0).max() < 0.005
                ):
                    fit_is_good = 1
                elif (
                    highest_Rhat <= 1.06
                    and np.std(fit_params["zeropoints"], axis=0).max() < 0.006
                ):
                    fit_is_good = 1
                elif (
                    highest_Rhat <= 1.05
                    and np.std(fit_params["zeropoints"], axis=0).max() < 0.007
                ):
                    fit_is_good = 1
                else:
                    fit_is_good = 0
                    show_residuals_by_star(all_obs, fit_params)

            iteration += 500

        zps_minus_mean = (
            fit_params["zeropoints"].T - np.mean(fit_params["zeropoints"], axis=1)
        ).T

        for i in range(len(fls)):
            print(
                "ZP",
                i,
                fls[i],
                np.std(fit_params["zeropoints"][:, i], ddof=1),
                np.std(zps_minus_mean[:, i], ddof=1),
            )

        median_residuals = all_obs["instr_mags"] - np.median(
            fit_params["instr_model_mags"], axis=0
        )

        f = open("2M_stars/zeropoints_" + sys.argv[1] + ".txt", "w")
        for i in range(len(fls)):
            towrite = [
                fls[i],
                filt,
                sys.argv[1],
                np.median(fit_params["zeropoints"][:, i]),
                np.std(fit_params["zeropoints"][:, i], ddof=1),
            ]
            towrite = [str(item) for item in towrite]
            f.write("  ".join(towrite) + "\n")
        f.close()

        number_of_fls = max(len(fls), 3)

        plt.figure(figsize=(3 * 2 + 0.5, 3 * number_of_fls + 0.5))

        print("SN_x_ys", SN_x_ys)

        for i in range(len(fls)):
            plt.subplot(number_of_fls, 2, 2 * i + 1)
            inds = np.where(
                (all_obs["im_inds"] == i) * (np.abs(median_residuals) < 0.15)
            )
            plt.plot(SN_x_ys[i][0], SN_x_ys[i][1], "*", color="green", zorder=1)
            plt.scatter(
                all_obs["xs"][inds],
                all_obs["ys"][inds],
                c=median_residuals[inds],
                zorder=0,
                cmap="magma",
                label="%.3f mag" % np.median(fit_params["int_floor"][:, i]),
            )
            plt.colorbar()

            inds = np.where(
                (all_obs["im_inds"] == i) * (np.abs(median_residuals) >= 0.15)
            )
            plt.plot(
                all_obs["xs"][inds],
                all_obs["ys"][inds],
                ".",
                color=[0.7] * 3,
                zorder=0.5,
            )
            plt.ylabel(fls[i])
            plt.legend(loc="best", prop={"size": 6})

        plt.subplot(number_of_fls, 2, 2)
        star_mags = np.median(fit_params["star_mags"], axis=0)

        inds = np.where(all_obs["twomass_mags"] > 2)

        plt.plot(
            all_obs["twomass_mags"][inds],
            star_mags[inds] - all_obs["twomass_mags"][inds],
            ".",
            color="b",
        )
        plt.axhline(0, color="k", linewidth=0.8)
        plt.ylim(-0.15, 0.15)
        plt.ylabel("UKIRT $-$ 2MASS")
        plt.xlabel("2MASS")
        plt.tick_params(
            axis="y", labelleft=False, labelright=True, left=False, right=True
        )

        plt.subplot(number_of_fls, 2, 4)
        plt.plot(
            np.median(fit_params["zeropoints"], axis=0),
            np.median(fit_params["int_floor"], axis=0),
            ".",
            color="b",
        )
        for i in range(len(fls)):
            plt.plot(
                [np.median(fit_params["zeropoints"][:, i])] * 2,
                scoreatpercentile(fit_params["int_floor"][:, i], [15.8655, 84.1345]),
                color="b",
            )

        plt.ylabel("Floor")
        plt.xlabel("Zeropoint")
        plt.tick_params(
            axis="y", labelleft=False, labelright=True, left=False, right=True
        )

        plt.subplot(number_of_fls, 2, 6)

        inds = np.where(all_obs["twomass_mags"] > 2)

        plt.scatter(
            unique_stars["RAs"][inds],
            unique_stars["Decs"][inds],
            c=star_mags[inds] - all_obs["twomass_mags"][inds],
            zorder=0,
            cmap="magma",
            vmin=-0.1,
            vmax=0.1,
        )
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])

        plt.savefig("2M_stars/resid_" + sys.argv[1] + ".png", bbox_inches="tight")
        plt.close()

    def detect_stars(
        self,
        target="reference_stars",
        psf="median",
        force=False,
        clean=False,
        verbose=False,
    ):
        """
        Use tphot on the image with customizable target and psf shapes.
        target can be:
            'reference_stars' to force tphot to look at the projected coordinates of the reference stars
            'all' to allow tphot to look for targets across the image
            a tuple of two floats to define the pixel coordinates x and y
            a tuple of two array-likes to define the pixel coordinates to search
        psf can be:
            a tuple of three floats to define the major axis, minor axis, and phi,
            'median' to use force tphot to use the median psf shape from a default tphot run,
            'free' to allow tphot to figure out a psf for each detection.
        """
        if self.detected_stars.all().exists() and not force:
            print_verb(
                verbose,
                "Already see detected stars for this image. Consider rerunning with force=True",
            )
            return
        if force:
            self.detected_stars.all().delete()
        initial_ds_count = self.detected_stars.count()

        # handle psf choice
        if isinstance(psf, tuple):
            major, minor, phi = psf
        elif psf == "median":
            if not self.major or not self.minor or not self.phi:
                raise ValueError(
                    "Image does not have a median psf yet. Consider running get_psf"
                )
            major, minor, phi = self.major, self.minor, self.phi
        elif psf == "free":
            major, minor, phi = None, None, None

        # handle target choice
        if self.observation.target.sn_type.name == "CALSPEC":
            subdir = "non_transient"
        else:
            subdir = f"20{self.observation.target.TNS_name[:2]}"
        write_to = f"{constants.DATA_DIR}/{subdir}/{self.observation.target.TNS_name}/ukirt/{self.observation.name}/{self.mef}/"
        tph_force = True
        if isinstance(target, tuple):
            x, y = target
            write_to += "custom.tph"
        elif target == "reference_stars":
            if not self.reference_stars.exists():
                raise ObjectDoesNotExist(
                    "There are no reference stars in the database yet. Consider running get_reference_stars on the observation."
                )
            rsd_set = ReferenceStarDetails.objects.filter(image=self).select_related(
                "reference_star"
            )
            x, y = np.array([(rsd.projected_x, rsd.projected_y) for rsd in rsd_set]).T
            write_to += "refcat.tph"
        elif target == "all":
            x, y = None, None
            tph_force = False
            write_to += "targets.tph"

        # call tphot and add detected stars to database
        tph = utils.tphot(
            path=self.path,
            x=x,
            y=y,
            force=force,
            tph_force=tph_force,
            arg_dict={"out": write_to, "major": major, "minor": minor, "phi": phi},
        )
        if clean:
            os.remove(write_to)
        header = self.fits().header
        for i, row in tph.iterrows():
            if (
                row["x"] < 0
                or row["y"] < 0
                or row["x"] > header["NAXIS1"]
                or row["y"] > header["NAXIS2"]
            ):
                continue
            defaults = {}
            for idx in row.index:
                if idx not in ["x", "y"]:
                    defaults[idx] = row[idx]
            DetectedStar.objects.update_or_create(
                image=self, x=row["x"], y=row["y"], defaults=defaults
            )
        print_verb(
            verbose,
            f"Detected {self.detected_stars.count() - initial_ds_count} new stars on top of {initial_ds_count} old ones.",
        )

    def crossmatch(self, thresh=constants.TPHOT_RAD, force=False, verbose=False):
        if force:
            self.detected_stars.all().update(reference_star=None)
        initial_ls_count = self.detected_stars.filter(
            reference_star__isnull=False
        ).count()
        if not force and initial_ls_count:
            print_verb(
                verbose,
                f"Already see {initial_ls_count} linked stars. Consider rerunning with force=True",
            )
            return

        # check to see if there are stars to link
        rsd_set = ReferenceStarDetails.objects.filter(image=self).select_related(
            "reference_star"
        )
        if not rsd_set.exists():
            raise ObjectDoesNotExist(
                f"There are no reference stars linked to this image in the database yet. Consider running get_reference_stars."
            )
        ds_set = self.detected_stars.all()
        if not ds_set.exists():
            raise ObjectDoesNotExist(
                "There are no detected stars in the database yet. Consider running detect_stars"
            )

        # match with wcs
        rs_world = rsd_set.values_list("reference_star__ra", "reference_star__dec")
        rs_pix = utils.sky2pix(self.fits().header, coords=rs_world)
        for ds in ds_set:
            dist = np.sqrt((ds.x - rs_pix[0]) ** 2 + (ds.y - rs_pix[1]) ** 2)
            if dist.min() < thresh:
                ds.reference_star = rsd_set[int(dist.argmin())].reference_star
            ds.save()
        print_verb(
            verbose,
            f"Linked {self.detected_stars.filter(reference_star__isnull=False).count() - initial_ls_count} new stars on top of {initial_ls_count} old links",
        )

    def get_zp_info(
        self,
        force=False,
        color_dict=constants.HODGKIN_COLOR_CORRECTION,
        edge_pad=constants.EDGE_PAD,
        verbose=False,
    ):
        """
        calculate zero point using linked reference/detected_stars and given color_dict
        """
        if not force and self.zp_guess is not None:
            print_verb(
                verbose, "Already see a zp estimate. Consider rerunning with force=True"
            )
            return
        if not self.reference_stars.exists():
            raise ObjectDoesNotExist(
                "There are no reference stars in the database yet. Consider running get_reference_stars."
            )
        if not self.major or not self.minor or not self.phi:
            raise ValueError(
                "Image does not have a median psf yet. Consider running get_psf."
            )
        d_stars = self.detected_stars.all()
        if not d_stars.exists():
            raise ObjectDoesNotExist(
                "There are no detected stars in the database yet. Consider running detect_stars."
            )
        # Tonry 2018, Virtually all galaxies can be rejected by selecting objects for which Gaia provides a nonzero proper-motion uncertainty, dpmra and dpmdec, at the cost of about 0.7% of all real stars.
        linked_stars = (
            d_stars.exclude(reference_star__isnull=True)
            .exclude(reference_star__dupvar=1)
            .exclude(reference_star__dpmra=0)
            .exclude(reference_star__dpmdec=0)
        )
        header = self.fits().header
        linked_stars = linked_stars.filter(
            x__gt=edge_pad,
            y__gt=edge_pad,
            x__lt=header["NAXIS1"] - edge_pad,
            y__lt=header["NAXIS2"] - edge_pad,
        )
        if not linked_stars.exists():
            raise ObjectDoesNotExist(
                "The detected stars do not match with any valid reference stars. Consider running crossmatch."
            )

        # calculate zero point
        zp, zp_err, mag, dmag = [[] for i in range(4)]
        k = header["EXTINCT"]
        airmass = (header["AMSTART"] + header["AMEND"]) / 2
        extinction = k * (airmass - 1)
        cc = color_dict[self.observation.bandpass]
        for i, ls in enumerate(linked_stars):
            rs = ls.reference_star
            mag_2mass = getattr(rs, cc[0].lower())
            mag_2mass_color1 = getattr(rs, cc[2].lower())
            mag_2mass_color2 = getattr(rs, cc[3].lower())
            dmag_2mass = getattr(rs, f"d{cc[0].lower()}")
            dmag_2mass_color1 = getattr(rs, f"d{cc[2].lower()}")
            dmag_2mass_color2 = getattr(rs, f"d{cc[3].lower()}")
            if (
                mag_2mass == 0
                or mag_2mass_color1 == 0
                or mag_2mass_color2 == 0
                or dmag_2mass == 0
                or dmag_2mass_color1 == 0
                or dmag_2mass_color2 == 0
            ):
                continue
            m = mag_2mass + cc[1] * (mag_2mass_color1 - mag_2mass_color2) + cc[4]
            mag.append(m)
            dm = np.sqrt(
                dmag_2mass ** 2
                + (cc[1] * dmag_2mass_color1) ** 2
                + (cc[1] * dmag_2mass_color2) ** 2
            )
            dmag.append(dm)
            # eqn 3 of Hodgkin et al. 2009
            zp.append(m - ls.instr_mag - ls.radial_distortion + extinction)
            zp_err.append(np.sqrt(dm ** 2 + ls.instr_mag_err ** 2))
        mag = np.array(mag)
        dmag = np.array(dmag)
        zp = np.array(zp)
        zp_err = np.array(zp_err)
        mask1 = zp < np.median(zp + constants.ZP_SIGMA_FROM_MEDIAN * np.median(zp_err))
        mask2 = zp > np.median(zp - constants.ZP_SIGMA_FROM_MEDIAN * np.median(zp_err))
        mask_both = mask1 & mask2
        self.zp_guess = np.median(zp[mask_both])
        # Hodgkin et al. 20019 section 2.3, para 4
        self.zp_err_guess = 1.48 * np.median(
            np.abs(zp[mask_both] - np.median(zp[mask_both]))
        )
        self.save()
        return (
            {"zp": zp, "zp_err": zp_err, "mag": mag, "dmag": dmag},
            {"zp_low": mask1, "zp_high": mask2, "both": mask_both},
        )

    def photometry(
        self,
        sub_type="nosub",
        color_dict=constants.HODGKIN_COLOR_CORRECTION,
        sn=True,
        x=None,
        y=None,
        ra=None,
        dec=None,
        check_area=False,
        thresh=constants.TPHOT_RAD,
        force=False,
        verbose=False,
    ):
        """
        Point tphot at the SN or at a list of ra/decs or x/y and get photometry.
        If SN, save to results to db.
        """
        # sort out use cases
        header = self.fits(sub_type).header
        if sn:
            # check to see if need to do work at all.
            det_set = UkirtDetection.objects.filter(
                lc__bandpass=self.observation.bandpass,
                mjd=self.observation.mjd,
                lc__variant="tphot",
            )
            if det_set.exists() and not force:
                print_verb(
                    verbose,
                    f"Already see {det_set.first()}. Consider rerunning with force=True",
                )
                return
            if x is not None or y is not None or ra is not None or dec is not None:
                raise ValueError(
                    "To run photometry on just the sn, do not enter other coordinates"
                )
            ra = self.observation.target.ra + self.sn_ra
            dec = self.observation.target.dec + self.sn_dec
            if self.observation.target.TNS_name in constants.CALSPEC_PM:
                pmra, pmdec = constants.CALSPEC_PM[self.observation.target.TNS_name]
                sc = self.observation.target.apply_space_motion(
                    pmra, pmdec, "2015-06-01", Time(self.observation.mjd, format="mjd")
                )
                ra = sc.ra.value
                dec = sc.dec.value
        if ra is not None and dec is not None:
            if x is not None or y is not None:
                raise ValueError("Provide either ra/dec or pixel coordinates, not both")
            if isinstance(ra, str) and isinstance(dec, str):
                ra, dec = utils.HMS2deg(ra, dec)

            [x, y] = utils.sky2pix(header, ra, dec)

        # Make sure there is a zero point, potentially recalculate
        if self.zp_guess is None or force:
            self.get_zp_info()

        # Run photometry
        if sn:
            target_string = sub_type
            if sub_type == "nosub":
                target_string = "sn"
        else:
            target_string = "custom"
        if self.observation.target.sn_type.name == "CALSPEC":
            subdir = "non_transient"
        else:
            subdir = f"20{self.observation.target.TNS_name[:2]}"
        path_head = f"{constants.DATA_DIR}/{subdir}/{self.observation.target.TNS_name}/ukirt/{self.observation.name}/{self.mef}/{target_string}"
        arg_dict = {
            "out": f"{path_head}.tph",
            "resid": f"{path_head}.resid.fits",
            "major": self.major,
            "minor": self.minor,
            "phi": self.phi,
        }
        phot = utils.tphot(
            path=self.get_path(sub_type),
            x=x,
            y=y,
            force=force,
            tph_force=True,
            arg_dict=arg_dict,
        )
        fits.writeto(
            arg_dict["resid"],
            fits.open(arg_dict["resid"])["PRIMARY"].data,
            header,
            overwrite=True,
        )
        grid_arg_dict = {
            "out": f"{path_head}.resid.tph",
            "major": self.major,
            "minor": self.minor,
            "phi": self.phi,
            "npar": 2,
        }
        x_box = np.linspace(x - self.major, x + self.major, 30)
        y_box = np.linspace(y - self.major, y + self.major, 30)
        x_rect = []
        y_rect = []
        for xx in x_box:
            for yy in y_box:
                x_rect.append(xx)
                y_rect.append(yy)
        grid = utils.tphot(
            arg_dict["resid"],
            x=x_rect,
            y=y_rect,
            force=force,
            tph_force=True,
            arg_dict=grid_arg_dict,
            instr_mag=False,
        )
        if check_area:
            triang = mtri.Triangulation(grid["x"], grid["y"])
            refiner = mtri.UniformTriRefiner(triang)
            tri_refi, z_test_refi = refiner.refine_field(grid["peakfit"], subdiv=4)
            plt.tricontourf(tri_refi, z_test_refi)
            plt.colorbar()
            plt.scatter(x, y, marker="x")
            plt.grid(color="black")
            plt.show()
            return
        extinction = header["EXTINCT"] * ((header["AMSTART"] + header["AMEND"]) / 2 - 1)
        if len(phot):
            mag = (
                self.zp_guess
                + phot["instr_mag"]
                + phot["radial_distortion"]
                - extinction
            )
            dmag = np.sqrt(self.zp_err_guess ** 2 + phot["instr_mag_err"] ** 2)
            ujy, dujy = utils.ab_to_ujy(
                mag + constants.MAG_TO_FLUX[self.observation.bandpass], dmag
            )
        else:
            mag, dmag, ujy, dujy = [None for i in range(4)]

        if not sn:
            return mag, dmag, ujy, dujy

        # if sn, add data to lightcurve and get rid of old data.
        # Not a simple replacement if sn goes from detection to nondeteciton
        lc, _ = self.observation.target.lightcurves.get_or_create(
            source="observed",
            bandpass=self.observation.bandpass,  # + utils.sub_type_suffix(sub_type),
            variant="tphot",
        )
        if det_set.exists():
            det_set.delete()
        ud = UkirtDetection.objects.create(
            lc=lc,
            observation=self.observation,
            mjd=self.observation.mjd,
            zp=self.zp_guess,
            dzp=self.zp_err_guess,
        )
        if not len(phot):
            print_verb(verbose, f"Non-detection. mjd={ud.mjd} zp={ud.zp} dzp={ud.dzp}")
            ud.status = "non"
        elif np.sqrt((phot["x"][0] - x) ** 2 + (phot["y"][0] - y) ** 2) > thresh:
            print_verb(
                verbose,
                f"Non-detection. tphot detects an object at {phot['x'][0]} {phot['y'][0]}. Projected coordinates were {x} {y}",
            )
            ud.status = "miss"
        elif (
            np.isnan(mag[0])
            or np.isnan(dmag[0])
            or np.isnan(ujy[0])
            or np.isnan(dujy[0])
        ):
            raise ValueError(
                f"Detected {phot} but there is a NaN. mag={mag[0]} dmag={dmag[0]} ujy={ujy[0]} dujy={dujy[0]}"
            )
        else:
            ud.mag = mag[0]
            ud.dmag = dmag[0]
            ud.ujy = ujy[0]
            ud.dujy = dujy[0]
            ud.x = phot["x"][0]
            ud.y = phot["y"][0]
            ud.status = "det"
            print_verb(
                verbose,
                f"Measured mag={mag[0]} dmag={dmag[0]} flux={ujy[0]} dflux={dujy[0]}",
            )
        self.save()
        ud.save()
        lc.save()

    def plot(
        self,
        sub_type="nosub",
        crop_around="sn",
        padding=constants.IM_PAD,
        sn=True,
        galaxy=True,
        reference_stars=True,
        detected_stars=True,
        force=False,
        out_path=None,
        show=False,
        verbose=False,
    ):
        """
        Plot of image with options for cropping, objects to be overplotted, and image alignment.

        Parameters:
            sub_type: 'nosub', 'rotsub', or 'refsub'
            crop_around: center thumbnail around 'sn', 'galaxy', or do not crop around if None
            padding: How big to make the thumbnail
            sn: Mark the SN on the image
            galaxy: Mark the galaxy on the image
            reference_stars: Mark the reference stars on the image
            detected_stars: Mark the detected stars on the image
            force: Continue if file already exists
            out_path: File location. None leads to default.
            show: Display thumbnail with matplotlib
            verbose: Print comments
        """
        # Option to skip if file exists
        if out_path is None:
            fname = sub_type
            if sub_type == "nosub":
                fname = "sn"
            if self.observation.target.sn_type.name == "CALSPEC":
                subdir = "non_transient"
            else:
                subdir = f"20{self.observation.target.TNS_name[:2]}"
            file_path = f"{constants.DATA_DIR}/{subdir}/{self.observation.target.TNS_name}/ukirt/{self.observation.name}/{self.mef}/{fname}.png"
        else:
            file_path = out_path
        if os.path.exists(file_path) and not force and not show:
            print_verb(
                verbose,
                "Already see a thumbnail for this image. Consider rerunning with force=True",
            )
            return

        # Making the image
        header = self.fits(sub_type).header
        data = self.fits(sub_type).data
        fig = plt.figure(figsize=(5, 5))
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        data_scaling = data  # for determining vmin and vmax. Uncropped by default.
        crop_x, crop_y = None, None

        # Centering around sn or galaxy
        if (sn or crop_around == "sn") and self.contains_sky(
            self.observation.target.ra, self.observation.target.dec
        ):
            raw_sn_coords = utils.sky2pix(
                header, self.observation.target.ra, self.observation.target.dec
            )
            sn_coords = utils.sky2pix(
                header,
                self.observation.target.ra
                + self.sn_ra * np.cos(self.observation.target.dec * np.pi / 180),
                self.observation.target.dec + self.sn_dec,
            )
            raw_x = raw_sn_coords[0]
            raw_y = raw_sn_coords[1]
            x = sn_coords[0]
            y = sn_coords[1]
            if sn:
                # projected coords and with manual offsets
                ax.scatter(raw_x, raw_y, marker="X", s=50, color="cyan")
                ax.scatter(x, y, marker="X", s=50, color="green")
                # ellipse for SN
                ud = UkirtDetection.objects.filter(
                    lc__bandpass=self.observation.bandpass
                    + utils.sub_type_suffix(sub_type),
                    observation=self.observation,
                    x__isnull=False,
                    y__isnull=False,
                )
                if ud.exists():
                    reg = pd.DataFrame(
                        data={
                            "x": (ud.first().x,),
                            "y": (ud.first().y,),
                            "major": (self.major,),
                            "minor": (self.minor,),
                            "phi": (self.phi,),
                        }
                    )
                    ax = utils.add_ellipses(ax, reg, "red")
            if crop_around == "sn":
                crop_x = int(x)
                crop_y = int(y)

        if (galaxy or crop_around == "galaxy") and self.contains_sky(
            self.observation.target.galaxy.ra, self.observation.target.galaxy.dec
        ):
            raw_gal_coords = utils.sky2pix(
                header,
                self.observation.target.galaxy.ra,
                self.observation.target.galaxy.dec,
            )
            gal_coords = utils.sky2pix(
                header,
                self.observation.target.galaxy.ra
                - self.rot_ra
                * np.cos(self.observation.target.dec * np.pi / 180)
                / 3600,
                self.observation.target.galaxy.dec - self.rot_dec / 3600,
            )
            raw_gal_x = raw_gal_coords[0]
            raw_gal_y = raw_gal_coords[1]
            gal_x = gal_coords[0]
            gal_y = gal_coords[1]
            if galaxy:
                ax.scatter(raw_gal_x, raw_gal_y, marker="+", s=50, color="cyan")
                ax.scatter(gal_x, gal_y, marker="+", s=50, color="green")
            if crop_around == "galaxy":
                crop_x = int(gal_x)
                crop_y = int(gal_y)

            # rotsub images are small, and may not require cropping
            if (
                crop_around is not None
                and 2 * padding < data.shape[0]
                and 2 * padding < data.shape[1]
            ):
                # print(data)
                # print(data.shape)
                # print(f'{crop_x} {crop_y} {padding}')
                data_scaling = data[
                    max(0, crop_y - padding) : min(data.shape[1], crop_y + padding),
                    max(0, crop_x - padding) : min(data.shape[0], crop_x + padding),
                ]
                # print(data_scaling)
                percentiles = np.percentile(data_scaling, (20, 80))
                data_scaling = data_scaling[data_scaling > percentiles[0]]
                data_scaling = data_scaling[data_scaling < percentiles[1]]
                # print(percentiles)
                # print(
                #     f"median = {np.median(data_scaling)} std = {np.std(data_scaling)}"
                # )
        vmax = np.median(data_scaling) + 2 * np.std(data_scaling)
        if sub_type == "rotsub":
            vmin = -vmax
        else:
            vmin = np.median(data_scaling) - np.std(data_scaling)
        ax.imshow(data, vmin=vmin, vmax=vmax, origin="lower", cmap=plt.get_cmap("gray"))

        # scatter plot of reference star coords projected onto image
        if reference_stars:
            ref_set = ReferenceStarDetails.objects.filter(image=self)
            if ref_set.exists():
                ax.scatter(
                    *utils.sky2pix(
                        header,
                        coords=ref_set.values_list(
                            "reference_star__ra", "reference_star__dec"
                        ),
                    ),
                    alpha=0.2,
                    color="cyan",
                )
        # ellipses for detected stars
        if detected_stars:
            det_set = self.detected_stars.all()
            if det_set.exists():
                det_dict = {}
                for param in ("major", "minor", "phi"):
                    det_dict[param] = np.array(det_set.values_list(param)).T[0]
                sc = utils.pix2sky(
                    self.fits("nosub").header,
                    coords=self.detected_stars.values_list("x", "y"),
                )
                det_dict["x"], det_dict["y"] = utils.sky2pix(
                    header, ra=sc.ra.value, dec=sc.dec.value
                )
                reg = pd.DataFrame(data=det_dict)
                ax = utils.add_ellipses(ax, reg, "blue")

        # cropping
        if crop_x is not None and crop_y is not None:
            ax.set_xlim(crop_x - padding, crop_x + padding)
            ax.set_ylim(crop_y - padding, crop_y + padding)

        # image prep
        ax.set_axis_off()
        fig.add_axes(ax)
        plt.savefig(file_path, format="png")
        if show:
            plt.show()
        plt.close(plt.gcf())
        utils.align_image(file_path, header)

    def inspect(self, sub_type="rotsub", verbose=True, uninspected_only=True):
        print_verb(verbose, f"{self}: {sub_type}")
        if uninspected_only:
            try:
                ud = UkirtDetection.objects.get(
                    observation=self.observation,
                    lc__bandpass=self.observation.bandpass
                    + utils.sub_type_suffix(sub_type),
                    mjd=self.observation.mjd,
                )
            except UkirtDetection.DoesNotExist:
                if sub_type == "rotsub":
                    self.get_rotsub()
                self.photometry(sub_type=sub_type)
                ud = UkirtDetection.objects.get(
                    observation=self.observation,
                    lc__bandpass=self.observation.bandpass
                    + utils.sub_type_suffix(sub_type),
                    mjd=self.observation.mjd,
                )
            if ud.status != "?":
                return

        status = "start"
        while status not in ["det", "non", "miss"]:
            self.plot(
                sub_type=sub_type,
                show=True,
                sn=True,
                galaxy=True,
                reference_stars=False,
                detected_stars=False,
            )
            plt.show()
            status = input("Status: ")
            if status == "phot" or status == "p":
                self.photometry(sub_type=sub_type, force=True)
            if (
                status.split()[0] == "rot"
                or status.split()[0] == "r"
                and sub_type == "rotsub"
            ):
                dx = float(status.split()[1])
                dy = float(status.split()[2])
                ra = self.observation.target.galaxy.ra
                dec = self.observation.target.galaxy.dec
                old_pix = utils.sky2pix(
                    self.fits(sub_type).header,
                    ra + self.rot_ra * np.cos(dec * np.pi / 180) / 3600,
                    dec + self.rot_dec / 3600,
                )
                new_coords = utils.pix2sky(
                    self.fits(sub_type).header, old_pix[0] + dx, old_pix[1] + dy
                )
                dra = (
                    new_coords.ra.value
                    - (ra + self.rot_ra * np.cos(dec * np.pi / 180) / 3600)
                ) * 3600
                ddec = (new_coords.dec.value - (dec + self.rot_dec / 3600)) * 3600
                self.rot_ra += dra
                self.rot_dec += ddec
                self.save()
                self.get_rotsub(force=True)
                self.photometry(sub_type=sub_type, force=True)
            if status == "help" or status == "h":
                print(
                    "Enter 'help' to print this message, "
                    "'phot' to redo photometry, "
                    "'rot x y' to redo rot subtraction with pixel offsets x and y, "
                    "'quit' to stop, "
                    "or make a judgment on the image: "
                    "'det' for a good detection, "
                    "'non' for a nondetection, and"
                    "'miss' for a detection on the wrong object"
                )
            if status == "quit" or status == "q":
                return "done"
        ud = UkirtDetection.objects.get(
            observation=self.observation,
            lc__bandpass=self.observation.bandpass + utils.sub_type_suffix(sub_type),
            mjd=self.observation.mjd,
        )
        ud.status = status
        ud.save()

    def print_phot(self, sub_type="nosub"):
        print(
            self.observation.target.lightcurve.ir_bands.get(
                bandpass=self.observation.bandpass + utils.sub_type_suffix(sub_type)
            ).detections[str(self.observation.mjd)]
        )

    def process(
        self, force=False, wipe_first=False, clean=False, verbose=True, galaxy=True
    ):
        self.get_reference_stars(force=force, verbose=verbose)
        self.get_psf(force=force, verbose=verbose)
        self.detect_stars(force=force, clean=clean, verbose=verbose)
        self.crossmatch(force=force, verbose=verbose)
        if self.mef == self.observation.sn_mef:
            self.photometry(force=force, verbose=verbose)
        self.get_zp_info(force=force, verbose=verbose)
        self.plot(force=force, verbose=verbose, galaxy=galaxy)

    def validate(self):
        rs_set = self.reference_stars.all()
        if rs_set.exists():
            rs_set_in = self.contains_sky(coords=rs_set.values_list("ra", "dec"))
            for i, rs in enumerate(rs_set):
                # Reference stars must be in the image
                if not rs_set_in[i]:
                    print(f"removing {rs} because it's not in the image.")
                    self.reference_stars.remove(rs)
                # Reference stars should not be too close to each other. 1"
                c = np.cos(rs.dec * np.pi / 180)
                near = rs_set.filter(
                    ra__gt=rs.ra - c / 3600,
                    ra__lt=rs.ra + c / 3600,
                    dec__gt=rs.dec - 1 / 3600,
                    dec__lt=rs.dec + 1 / 3600,
                )
                if near.count() > 1:
                    print(
                        f"removing {rs} from {self} because it's within 1\" of another reference star."
                    )
                    self.reference_stars.remove(rs)

        # Detected stars must be in the image
        ds_set = self.detected_stars.all()
        if ds_set.exists():
            ds_set_in = self.contains_pix(coords=ds_set.values_list("x", "y"))
            for i, ds in enumerate(ds_set):
                if not ds_set_in[i]:
                    print(f"removing {ds} because it's not in the image.")
                    ds.delete()

            # Over 70% (arbitrary) of the detected stars should be matched to reference stars.
        if rs_set.exists() and ds_set.exists():
            if (
                ds_set.filter(reference_star__isnull=False).count()
                < 0.7 * ds_set.count()
            ):
                print(
                    f"Of the {ds_set.count()} stars detected, only {ds_set.filter(reference_star__isnull=False).count()} are linked to reference stars. Reprocessing."
                )
                obs = self.observation
                mef = self.mef
                obs.make_image(mef=mef, force=True)
                self = Image.objects.get(observation=obs, mef=mef)
                self.process(force=True)


class ReferenceStar(models.Model):
    """
    Catalog stars contained in the image
    """

    images = models.ManyToManyField(
        "Image", through="ReferenceStarDetails", related_name="reference_stars"
    )
    ra = models.FloatField()
    dec = models.FloatField()
    plx = models.FloatField()
    dplx = models.FloatField()
    pmra = models.FloatField()
    dpmra = models.FloatField()
    pmdec = models.FloatField()
    dpmdec = models.FloatField()
    gaia = models.FloatField()
    dgaia = models.FloatField()
    bp = models.FloatField()
    dbp = models.FloatField()
    rp = models.FloatField()
    drp = models.FloatField()
    teff = models.FloatField()
    agaia = models.FloatField()
    dupvar = models.FloatField()
    ag = models.FloatField()
    rp1 = models.FloatField()
    r1 = models.FloatField()
    r10 = models.FloatField()
    g = models.FloatField()
    dg = models.FloatField()
    gchi = models.FloatField()
    gcontrib = models.IntegerField()
    r = models.FloatField()
    dr = models.FloatField()
    rchi = models.FloatField()
    rcontrib = models.IntegerField()
    i = models.FloatField()
    di = models.FloatField()
    ichi = models.FloatField()
    icontrib = models.IntegerField()
    z = models.FloatField()
    dz = models.FloatField()
    zchi = models.FloatField()
    zcontrib = models.IntegerField()
    nstat = models.FloatField()
    j = models.FloatField()
    dj = models.FloatField()
    h = models.FloatField()
    dh = models.FloatField()
    k = models.FloatField()
    dk = models.FloatField()

    class Meta:
        constraints = [
            models.UniqueConstraint(name="reference_coords", fields=["ra", "dec"])
        ]

    def __str__(self):
        return f"{self.ra} {self.dec}"

    def plot(self, padding=10):
        num = self.images.count()
        n2 = int(np.sqrt(num) + 1)
        fig, ax = plt.subplots(n2, n2, sharex=True, sharey=True, figsize=(5, 5))
        for i, im in enumerate(self.images.all()):
            rsd = ReferenceStarDetails.objects.get(image=im, reference_star=self)
            hdu = fits.open(im.path)
            tmp_ax = ax[i // n2][i % n2]
            tmp_ax.imshow(
                hdu["PRIMARY"].data[
                    int(rsd.projected_y - padding) : int(rsd.projected_y + padding),
                    int(rsd.projected_x - padding) : int(rsd.projected_x + padding),
                ]
            )
            tmp_ax.set_xlim(0, padding * 2)
            tmp_ax.set_ylim(0, padding * 2)
            tmp_ax.set_title(
                f"{im.observation.name}: {np.round(rsd.projected_x)} {np.round(rsd.projected_y)}"
            )
            tmp_ax.set_axis_off()
        for j in range(i, n2 ** 2):
            tmp_ax = ax[j // n2][j % n2]
            tmp_ax.set_axis_off()
        plt.show()


class ReferenceStarDetails(models.Model):
    """
    Many-to-many table linking reference stars and images with coordinates
    """

    image = models.ForeignKey("Image", on_delete=models.CASCADE)
    reference_star = models.ForeignKey("ReferenceStar", on_delete=models.CASCADE)
    projected_x = models.FloatField(null=True)
    projected_y = models.FloatField(null=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                name="reference_details_image", fields=["reference_star", "image"]
            )
        ]


class DetectedStar(models.Model):
    """
    Sources detected in an image with tphot
    """

    image = models.ForeignKey(
        Image, on_delete=models.CASCADE, related_name="detected_stars"
    )
    reference_star = models.ForeignKey(
        ReferenceStar, related_name="detections", null=True, on_delete=models.SET_NULL
    )
    x = models.FloatField()
    y = models.FloatField()
    peakval = models.FloatField()
    skyval = models.FloatField()
    peakfit = models.FloatField()
    dpeak = models.FloatField()
    skyfit = models.FloatField()
    flux = models.FloatField()
    dflux = models.FloatField()
    major = models.FloatField()
    minor = models.FloatField()
    phi = models.FloatField()
    err = models.FloatField()
    chin = models.FloatField()
    instr_mag = models.FloatField(null=True)
    instr_mag_err = models.FloatField(null=True)
    radial_distortion = models.FloatField(default=0)

    class Meta:
        constraints = [
            models.UniqueConstraint(name="detected_coords", fields=["image", "x", "y"])
        ]

    def __str__(self):
        return f"{self.image} {self.x} {self.y}"

    def plot(self, padding=20):
        hdu = fits.open(self.image.path)
        plt.imshow(
            hdu[0].data[
                int(self.y - padding) : int(self.y + padding),
                int(self.x - padding) : int(self.x + padding),
            ]
        )
        plt.title(
            f"{self.image.observation.target.TNS_name} - {self.image.observation.name}: {self.x} {self.y}"
        )
        plt.show()


class Spectrum(FitsFile):
    target = models.ForeignKey("targets.Target", on_delete=models.CASCADE)
    TNS_name = models.CharField(max_length=20, null=True)
    exp_time = models.FloatField()
    reduced = models.BooleanField(default=False)

    def __str__(self):
        return f"{self.TNS_name}: {self.name}"

    class Meta:
        abstract = True
        constraints = [
            models.UniqueConstraint(name="spectrum_file_path", fields=["path"])
        ]


class CalspecSpectrum(Spectrum):
    """CalspecSpectrum.

    Files from https://archive.stsci.edu/hlsps/reference-atlases/cdbs/current_calspec/
    """

    instr = models.CharField(max_length=20)
    wavelength = models.JSONField(default=list)
    flux = models.JSONField(default=list)
    staterror = models.JSONField(default=list)
    syserror = models.JSONField(default=list)
    fwhm = models.JSONField(default=list)
    dataqual = models.JSONField(default=list)
    totexp = models.JSONField(default=list)

    class Meta:
        constraints = [models.UniqueConstraint(name="calspec_path", fields=["path"])]

    def save(self, *args, **kwargs):
        if not len(self.wavelength):
            path_fits = fits.open(self.path)
            for attr in (
                "WAVELENGTH",
                "FLUX",
                "STATERROR",
                "SYSERROR",
                "FWHM",
                "DATAQUAL",
                "TOTEXP",
            ):
                setattr(
                    self,
                    attr.lower(),
                    list(np.array(path_fits[1].data[attr], dtype=float)),
                )
        super(CalspecSpectrum, self).save(*args, **kwargs)

    def synthetic_flux(
        self,
        filt,
        atmos_wl=None,
        atmos_transmission=None,
        ra=None,
        dec=None,
        mjd=None,
        mag_sys="vega",
    ):
        from astropy.coordinates import EarthLocation, SkyCoord
        from astropy.time import Time

        if ra is not None and dec is not None and mjd is not None:
            sc = SkyCoord(ra=ra, dec=dec, unit="deg", frame="icrs")
            obstime = Time(mjd, format="mjd")
            loc = EarthLocation.from_geodetic(
                lat=constants.MK_LAT * u.deg,
                lon=constants.MK_LON * u.deg,
                height=constants.MK_HEIGHT * u.m,
            )
            heliocorr = sc.radial_velocity_correction(
                "heliocentric", obstime=obstime, location=loc
            )
            spec_wl = np.array(self.wavelength) / (
                1 + heliocorr.to(u.km / u.s).value / constants.C
            )
        else:
            spec_wl = np.array(self.wavelength)
        if isinstance(filt, str):
            f = pd.read_csv(
                f"{constants.STATIC_DIR}/filters/{filt}.dat",
                delim_whitespace=True,
                header=None,
                names=["wl", "trans"],
            )
        wl = f["wl"]
        transmission = f["trans"]
        if atmos_wl is not None and atmos_transmission is not None:
            a_trans = np.interp(wl, atmos_wl, atmos_transmission)
            transmission = transmission * a_trans
        flux = np.interp(wl, spec_wl, np.array(self.flux))
        stat_err = np.interp(wl, spec_wl, np.array(self.staterror))
        sys_err = np.interp(wl, spec_wl, np.array(self.syserror))
        err = np.sqrt(sum(sys_err ** 2 + stat_err ** 2))
        flux = np.sum(flux * transmission)
        if mag_sys == "flux_only":
            return flux, err
        elif mag_sys == "vega":
            ref_flux, _ = CalspecSpectrum.objects.get(
                name="alpha_lyr_stis_011.fits"
            ).synthetic_flux(filt=filt, mag_sys="flux_only")
        elif mag_sys == "ab":
            ref = pd.read_csv(
                f"{constants.STATIC_DIR}/filters/ab-spec.dat",
                header=None,
                names=["wl", "flux"],
                delimiter="\s+",
            )
            ref_flux = np.sum(np.interp(wl, ref["wl"], ref["flux"]) * transmission)
        mag = -2.5 * np.log10(flux / ref_flux)
        mag_err = 2.5 * np.log10(1 + err / flux)
        return flux, err, mag, mag_err


class SnifsSpectrum(Spectrum):
    """
    Unaltered raw data.
    """

    target = models.ForeignKey(
        "targets.Target", related_name="snifs_spectra", on_delete=models.CASCADE
    )
    airmass = models.FloatField(null=True)
    channels = models.CharField(max_length=3, null=True)
    fluxcal_rms_b = models.FloatField(null=True)
    fluxcal_rms_r = models.FloatField(null=True)
    exp_num = models.CharField(max_length=8)
    # exp_str = models.CharField(max_length=10)

    class Meta:
        constraints = [models.UniqueConstraint(name="snifs_path", fields=["path"])]

    @property
    def wl(self):
        return np.array(self.get_pd()["wl"])

    @property
    def flux(self):
        return np.array(self.get_pd()["flux"])

    @property
    def err(self):
        return np.array(self.get_pd()["flux_err"])

    @property
    def exp_str(self):
        return "_".join(
            (
                str(self.date.year)[2:],
                str(self.date.timetuple().tm_yday).zfill(3),
                self.exp_num,
            )
        )

    @property
    def mask(self):
        nan_mask = np.where(np.isnan(self.flux + self.err))[0]
        dichroic_mask = np.where(
            (self.wl > constants.SNIFS_DICHROIC[0])
            & (self.wl < constants.SNIFS_DICHROIC[1])
        )[0]
        telluric_masks = [
            np.where((self.wl > telluric[0]) & (self.wl < telluric[1]))[0]
            for telluric in constants.TELLURICS.values()
        ]
        bad_idx = set(np.concatenate([nan_mask, dichroic_mask, *telluric_masks]))
        # extra () to match output of np.where
        return (np.array([i for i in range(len(self.wl)) if i not in bad_idx]),)

    def save(self, *args, **kwargs):
        if self.name == "NEED NAME":
            self.name = self.path.split("/")[-1].split(".fits")[0]
        if self.date is None:
            self.date = self.get_date()
        if self.exp_time is None:
            self.get_exp_time()
        super(SnifsSpectrum, self).save(*args, **kwargs)

    def get_date(self):
        with open(self.path, "r") as f:
            y, m, d = f.readlines()[6].split()[3].split("T")[0].split("-")
        self.date = date(int(y), int(m), int(d))

    def get_exp_time(self):
        with open(self.path, "r") as f:
            self.exp_time = int(float(f.readlines()[7].split()[3].strip("s")))

    def get_mjd(self):
        with open(self.path, "r") as f:
            self.mjd = float(f.readlines()[5].split()[3])

    def get_pd(self):
        return pd.read_csv(
            self.path,
            skiprows=15,
            delim_whitespace=True,
            header=None,
            names=["wl", "flux", "flux_err"],
        )

    def get_spec1d(self):
        df = self.get_pd()
        return Spectrum1D(
            spectral_axis=np.array(df["wl"])[self.mask] * u.AA,
            flux=np.array(df["flux"])[self.mask] * u.erg / u.s / u.cm ** 2 / u.AA,
            uncertainty=StdDevUncertainty(
                np.array(df["flux_err"])[self.mask] * u.erg / u.s / u.cm ** 2 / u.AA
            ),
            velocity_convention="optical",
            rest_value=6000 * u.AA,
        )

    # Deprecated for galaxies.SnifsEntry.weightedCC
    # def get_z(self, tspec, max_iters=7):
    #     ospec = self.get_spec1d()
    #     z = None
    #     z_arr = None
    #     for i in range(2, max_iters):
    #         if not z:
    #             z = 0.05
    #         z_range = np.linspace(z - 2 * 10 ** (-i), z + 2 * 10 ** (-i), 41)
    #         z, _, chi2 = template_redshift(ospec, tspec, z_range)
    #         if z_arr is None:
    #             z_arr = z_range
    #             chi2_arr = chi2
    #         else:
    #             z_arr = np.append(z_arr, z_range)
    #             chi2_arr = chi2_arr + chi2
    #     z_arr, chi2_arr = np.array([[x, y] for x, y in sorted(zip(z_arr, chi2_arr))]).T
    #     return z, z_arr, chi2_arr

    def get_continuum_subtracted_spec1d(self):
        from specutils.fitting import fit_generic_continuum

        sp = self.get_spec1d()
        generic_fit = fit_generic_continuum(sp)
        continuum_flux = generic_fit(sp.spectral_axis)
        return Spectrum1D(
            spectral_axis=sp.spectral_axis,
            flux=sp.flux - continuum_flux,
            uncertainty=sp.uncertainty,
            velocity_convention=sp.velocity_convention,
            rest_value=sp.rest_value,
        )

    def get_continuum_normed_spec1d(self):
        from specutils.fitting import fit_generic_continuum

        sp = self.get_spec1d()
        generic_fit = fit_generic_continuum(sp)
        continuum_flux = generic_fit(sp.spectral_axis)
        return Spectrum1D(
            spectral_axis=sp.spectral_axis,
            flux=sp.flux / continuum_flux.value,
            uncertainty=sp.uncertainty,
            velocity_convention=sp.velocity_convention,
            rest_value=sp.rest_value,
        )

    def plot(self, z=None, tspec=None):
        fig, ax = plt.subplots(1)
        ax.plot(self.wl, self.flux, label="observed", alpha=0.5)
        if tspec:
            ax.plot(
                tspec.wavelength,
                tspec.flux * np.median(self.flux) / np.median(tspec.flux),
                label="template",
                alpha=0.5,
            )
        if z:
            ax.plot(
                self.wl / (1 + z), self.flux, label="rest-frame", alpha=0.5,
            )
        ax.legend()
        return fig


class FocasImage(FitsFile):
    target = models.ForeignKey(
        "targets.Target",
        related_name="focas_images",
        on_delete=models.CASCADE,
        null=True,
    )
    TNS_name = models.CharField(max_length=20, null=True)
    exp_time = models.FloatField()

    class Meta:
        constraints = [
            models.UniqueConstraint(name="focas_image_path", fields=["path"])
        ]

    def save(self, *args, **kwargs):
        if self.name == "NEED NAME":
            self.name = self.path.split("/")[-1].split(".fits")[0]
        if self.date is None:
            self.date = self.get_date()
        if self.exp_time is None:
            self.get_exp_time()
        super(FocasImage, self).save(*args, **kwargs)

    def get_date(self):
        mjd = fits.open(self.path)["PRIMARY"].header["MJD"]
        year, month, day = utils.MJD_to_ut(mjd)
        self.date = date(year, month, int(day))

    def get_exp_time(self):
        self.exp_time = int(fits.open(self.path)["PRIMARY"].header["EXPTIME"])


class FocasSpectrum(Spectrum):
    """
    Unaltered raw data.
    """

    target = models.ForeignKey(
        "targets.Target",
        related_name="focas_spectra",
        on_delete=models.CASCADE,
        null=True,
    )
    obs_type = models.CharField(max_length=10, default="OBJECT")

    class Meta:
        constraints = [models.UniqueConstraint(name="focas_path", fields=["path"])]

    def save(self, *args, **kwargs):
        if self.name == "NEED NAME":
            self.name = self.path.split("/")[-1].split(".fits")[0]
        if self.date is None:
            self.date = self.get_date()
        if self.exp_time is None:
            self.get_exp_time()
        super(FocasSpectrum, self).save(*args, **kwargs)

    def get_date(self):
        mjd = fits.open(self.path)["PRIMARY"].header["MJD"]
        year, month, day = utils.MJD_to_ut(mjd)
        self.date = date(year, month, int(day))

    def get_exp_time(self):
        self.exp_time = int(fits.open(self.path)["PRIMARY"].header["EXPTIME"])

    def create_bias(self, outpath="default", force=False):
        import glob

        if outpath == "default":
            dir_path = "/".join(self.path.split("/")[:-2])
            outpath = dir_path + "/stdbias.fits"
        if os.path.exists(outpath) and not force:
            return
        bias_data = []
        for path in glob.glob(dir_path + "/chip2/*"):
            f = fits.open(path)["PRIMARY"]
            if f.header["OBJECT"] == "BIAS" and f.data.shape == self.data.shape:
                bias_data.append(f.data)
        if len(bias_data):
            bias_data = np.median(bias_data, axis=0)
            fits.PrimaryHDU(bias_data).writeto(outpath, overwrite=force)

    def get_bias(self, bias="default"):
        dir_path = "/".join(self.path.split("/")[:-2])
        if bias == "default":
            self.create_bias()
            if not os.path.exists(dir_path + "/stdbias.fits"):
                return fits.open(constants.SUBARU_HSF_DIR + "/general_bias.fits")[
                    "PRIMARY"
                ].data
            return fits.open(dir_path + "/stdbias.fits")["PRIMARY"].data
        elif isinstance(bias, str) and os.path.exists(bias):
            return fits.open(bias)["PRIMARY"].data
        elif isinstance(bias, np.ndarray) and bias.shape == self.data.shape:
            return bias
        elif bias is not None:
            try:
                return self.data - bias
            except TypeError:
                raise TypeError(
                    "bias not recognized. Provide the string 'default', a string that is a path to a fits file, a np.ndarray the same shape as the data, or a number"
                )

    def create_flat(self, outpath="default", force=False):
        import glob

        if outpath == "default":
            dir_path = "/".join(self.path.split("/")[:-2])
            outpath = dir_path + "/flat.fits"
        if os.path.exists(outpath) and not force:
            return
        flat_data = []
        for path in glob.glob(dir_path + "/chip2/*"):
            f = fits.open(path)["PRIMARY"]
            if f.header["OBJECT"] == "DOMEFLAT" and f.data.shape == self.data.shape:
                flat_data.append(f.data)
        if len(flat_data):
            med_flat = np.median(flat_data, axis=0)
            fits.PrimaryHDU(med_flat).writeto(outpath, overwrite=force)

    def get_flat(self, flat="default"):
        dir_path = "/".join(self.path.split("/")[:-2])
        if flat == "default":
            self.create_flat()
            if not os.path.exists(dir_path + "/flat.fits"):
                return fits.open(constants.SUBARU_HSF_DIR + "/general_flat.fits")[
                    "PRIMARY"
                ].data
            return fits.open(dir_path + "/flat.fits")[0].data
        elif isinstance(flat, str) and os.path.exists(flat):
            return fits.open(flat)["PRIMARY"].data
        elif isinstance(flat, np.ndarray) and flat.shape == self.data.shape:
            return flat
        else:
            raise TypeError(
                "flat not recognized. Provide the string 'default', a string that is a path to a fits file, or a np.ndarray the same shape as the data"
            )

    def clean_cosmic_rays(
        self,
        cr_threshold=5,
        neighbor_threshold=5,
        readnoise=40,
        force=False,
        outpath="default",
    ):
        from data.lacosmic import lacosmic

        if outpath == "default":
            dir_path = "/".join(self.path.split("/")[:-2]) + "/cr"
            outpath = dir_path + "/" + self.name + ".fits"
            os.makedirs(dir_path, exist_ok=True)
        if os.path.exists(outpath) and not force:
            return
        data, crmask = lacosmic(
            np.array(self.data, dtype=float),
            contrast=constants.FOCAS_LACOSMIC_CONTRAST,
            cr_threshold=cr_threshold,
            neighbor_threshold=neighbor_threshold,
            effective_gain=self.header["GAIN"],
            readnoise=readnoise,
        )
        fits.PrimaryHDU(data, header=self.header).writeto(outpath, overwrite=force)

    def prep_for_iraf(
        self,
        remove_cosmic_rays=True,
        bias="default",
        flat="default",
        outpath="default",
        force=False,
    ):
        data = self.data
        dir_path = "/".join(self.path.split("/")[:-2])
        if remove_cosmic_rays:
            self.clean_cosmic_rays()
            data = fits.open(dir_path + "/cr/" + self.name + ".fits")["PRIMARY"].data
        data = (
            (data - self.get_bias(bias))
            * np.average(self.get_flat(flat) - self.get_bias(bias))
            / (self.get_flat(flat) - self.get_bias(bias))
        )
        if outpath == "default":
            outpath = dir_path + "/ff/" + self.name + ".fits"
            os.makedirs(dir_path + "/ff", exist_ok=True)
        fits.PrimaryHDU(data=data, header=self.header).writeto(outpath, overwrite=force)

    def wl_calibration(
        self, image=None, xmin=300, xmax=500, ymin=40, ymax=2640, order=5, plot=True
    ):
        if image is None:
            data = self.data
        elif image in ("cr", "ff", "dc"):
            data = fits.open(self.path.replace("chip2", image))[0].data
        spectrum = np.median(data[ymin:ymax, xmin:xmax], axis=1)[::-1]
        peaks, _ = find_peaks(spectrum, prominence=50)
        peaks_refined = refine_peaks(spectrum, peaks)
        c = Calibrator(peaks_refined, spectrum)
        c.set_calibrator_properties(
            num_pix=len(spectrum), plotting_library="matplotlib", log_level="info"
        )
        c.set_hough_properties(
            num_slopes=10000,
            xbins=1000,
            ybins=1000,
            min_wavelength=5000,
            max_wavelength=8600,
            range_tolerance=500,
            linearity_tolerance=50,
        )
        wl, names, intensity = np.array(constants.SKYLINE_CAL).T
        wl = np.array(wl, dtype=float)
        intensity = np.array(intensity, dtype=int)
        c.add_user_atlas(
            names,
            wl,
            intensity,
            pressure=self.header["OUT-PRS"] * 100,
            temperature=self.header["DOM-TMP"],
            relative_humidity=self.header["OUT-HUM"],
        )
        c.set_known_pairs(pix=peaks_refined[0], wave=5577.34)
        c.do_hough_transform()
        (
            fit_coeff,
            matched_peaks,
            matched_atlas,
            rms,
            residual,
            peak_util,
            atlas_util,
        ) = c.fit(max_tries=1000, fit_deg=order)
        if plot:
            c.plot_fit(fit_coeff)
        return fit_coeff


class TemplateSpectrum(FitsFile):
    wl = models.JSONField(default=list)
    flux = models.JSONField(default=list)
    err = models.JSONField(default=list)

    class Meta:
        constraints = [models.UniqueConstraint(name="template_path", fields=["path"])]

    def get_spec1d(self, max_wl=10000):
        if self.name.startswith("spDR2"):
            return Spectrum1D(
                spectral_axis=self.wl * u.AA,
                flux=self.flux * 1e-17 * u.erg / u.s / u.cm ** 2 / u.AA,
            )

        data = fits.open(self.path)[1].data
        mt = np.where(data["WAVELENGTH"] < max_wl)
        return Spectrum1D(
            spectral_axis=np.array(data["WAVELENGTH"])[mt] * u.AA,
            flux=np.array(data["FLUX"])[mt] * u.erg / u.s / u.cm ** 2 / u.AA,
        )
