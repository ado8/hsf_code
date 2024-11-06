import os

import astropy.units as u
import constants
import dotenv
import numpy as np
import pandas as pd
import utils
from alerce.core import Alerce
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from django.db import models
from django.db.models import F
from hsf.settings import BASE_DIR
from scipy import interpolate

dotenv_file = os.path.join(BASE_DIR, ".env")
if os.path.isfile(dotenv_file):
    dotenv.load_dotenv(dotenv_file)

STATUS_CHOICES = [
    ("g", "gold"),
    ("s", "silver"),
    ("b", "bronze"),
    ("det", "Detection"),
    ("non", "Nondetection"),
    ("miss", "Detecting the wrong thing"),
    ("?", "uninspected"),
]
# Create your models here.


class Lightcurve(models.Model):
    BANDPASS_CHOICES = [
        ("c", "cyan"),
        ("o", "orange"),
        ("ztfg", "ZTF g"),
        ("ztfr", "ZTF r"),
        ("ztfi", "ZTF i"),
        ("asg", "ASAS-SN g"),
        ("Z", "WFCAM Z"),
        ("Y", "WFCAM Y"),
        ("J", "WFCAM J"),
        ("H", "WFCAM H"),
        ("H", "WFCAM H"),
        ("K", "WFCAM K"),
    ]
    VAR_CHOICES = [
        ("none", "None"),
        ("magpsf", "ZTF magpsf"),
        ("magap", "ZTF magap"),
        ("magap_big", "ZTF magap_big"),
        ("rot", "rotational subtraction"),
        ("ref", "reference subtraction"),
        ("0D", "Rubin forward modeled, no galaxy"),
        ("1D", "Rubin forward modeled, no reference"),
        ("1D2", "Rubin forward modeled, no reference, scale=2"),
        ("1D3", "Rubin forward modeled, no reference, scale=3"),
        ("1D4", "Rubin forward modeled, no reference, scale=4"),
        ("2D", "Rubin forward modeled, with reference"),
        ("dehvils", "Photometry from DEHVILS DR1"),
        ("imcore_pk", "CASU provided photometry"),
        ("imcore_ap1", "CASU provided photometry"),
        ("imcore_ap2", "CASU provided photometry"),
        ("imcore_ap3", "CASU provided photometry"),
        ("imcore_ap4", "CASU provided photometry"),
        ("imcore_ap5", "CASU provided photometry"),
        ("imcore_ap6", "CASU provided photometry"),
        ("imcore_ap7", "CASU provided photometry"),
    ]
    SOURCE_CHOICES = [
        ("ATLAS", "ATLAS"),
        ("ZTF", "ZTF"),
        ("ASAS-SN", "ASAS-SN"),
        ("UKIRT", "UKIRT"),
        ("model", "Model"),
    ]
    MODEL_CHOICES = [
        ("snpy_max_model", "SNooPy max_model"),
        ("snpy_ebv_model2", "SNooPy ebv_model2"),
        ("salt3", "salt3"),
        ("salt2-extended", "salt2-extended"),
        ("salt3-nir", "salt3-nir"),
    ]
    CALIBRATION_CHOICES = [
        (0, "u-band, st > 0.5"),
        (1, "u-band, (B-V) < 0.3"),
        (2, "u-band, all objects"),
        (3, "no u-band, st > 0.5"),
        (4, "no u-band, (B-V) < 0.3"),
        (5, "no u-band, all objects"),
        (6, "Full sample (2018)"),
        (7, "(B-V) < 0.5 (2018)"),
        (8, "st > 0.5 (2018)"),
        (9, "st > 0.5 and (B-V) < 0.5 (2018)"),
        (10, "dm15 0"),
        (11, "dm15 1"),
        (12, "dm15 2"),
        (13, "dm15 3"),
        (14, "dm15 4"),
        (15, "dm15 5"),
    ]
    REDLAW_CHOICES = [
        # Following dust_extinction naming conventions
        ("CCM89", "Cardelli, Clayton, & Mathis (1989) Milky Way R(V) dependent model"),
        ("O94", "O'Donnell (1994) Milky Way R(V) dependent model"),
        ("F99", "Fitzpatrick (1999) Milky Way R(V) dependent model"),
        ("F04", "Fitzpatrick (2004) Milky Way R(V) dependent model"),
        ("VCG04", "Valencic, Clayton, & Gordon (2004) Milky Way R(V) dependent model"),
        ("GCC09", "Grodon, Cartledge, & Clayton (2009) Milky Way R(V) dependent model"),
        ("M14", "Maiz Apellaniz et al (2014) Milky Way & LMC R(V) dependent model"),
        ("G16", "Gordon et al (2016) Milky Way, L/SMC R(V) and f_A dependent model"),
        ("F19", "Fitzpatrick et al (2019) extinction model calculation"),
        ("D22", "Decleir et al (2022) extinction model calculation"),
    ]
    target = models.ForeignKey(
        "targets.Target", on_delete=models.CASCADE, related_name="lightcurves"
    )
    bandpass = models.CharField(max_length=10, choices=BANDPASS_CHOICES)
    eff_wl = models.FloatField(null=True)
    variant = models.CharField(max_length=10, choices=VAR_CHOICES, default="none")
    early = models.FloatField(null=True)
    late = models.FloatField(null=True)
    tmax = models.FloatField(null=True)
    source = models.CharField(max_length=8, choices=SOURCE_CHOICES)
    fit_result = models.ForeignKey(
        "fitting.FitResults",
        on_delete=models.CASCADE,
        related_name="model_lightcurves",
        null=True,
    )

    class Meta:
        constraints = [
            models.UniqueConstraint(
                name="lc_source_bandpass_variant",
                fields=[
                    "target",
                    "source",
                    "bandpass",
                    "variant",
                    "fit_result",
                ],
            ),
        ]
        indexes = [
            models.Index(fields=["target"]),
            models.Index(fields=["source"]),
            models.Index(fields=["bandpass"]),
            models.Index(fields=["variant"]),
            models.Index(fields=["fit_result"]),
        ]

    def __save__(self, *args, **kwargs):
        if self.eff_wl is None:
            self.eff_wl = constants.EFF_WL_DICT[self.bandpass]
        super(Lightcurve, self).save(*args, **kwargs)

    def __str__(self):
        if self.source == "model":
            return f"{self.fit_result}: {self.bandpass}"
        return f"{self.target.TNS_name}: {self.source} {self.bandpass} {self.variant}"

    def calculate_coords(
        self,
        astrometry_radius=0.3,
        pixel_threshold=1,
        statistic="average",
        weight="rms_ivar",
    ):
        if self.survey != "ATLAS":
            return
        ra_arr, dec_arr, rms_arr = [np.array([]) for i in range(3)]
        for det in self.detections():
            if not det.coordinates.filter(astrometry_radius=astrometry_radius).count():
                det.get_coords(astrometry_radius=astrometry_radius)
            try:
                coord = det.coordinates.get(
                    astrometry_radius=astrometry_radius, nearest_ddc__lt=pixel_threshold
                )
                ra_arr = np.append(ra_arr, coord.ra)
                dec_arr = np.append(dec_arr, coord.dec)
                rms_arr = np.append(rms_arr, coord.rms)
            except:
                continue
        if weight == "rms_ivar":
            weights = 1 / rms_arr**2
        elif weight in [None, "none"]:
            weights = np.ones(len(ra_arr))
        else:
            raise TypeError("weights should be 'rms_ivar', None, or 'none'")
        if statistic == "average":
            ra = np.average(ra_arr, weights=weights)
            dec = np.average(dec_arr, weights=weights)
        elif statistic == "median":
            ra = np.median(ra_arr)
            dec = np.median(dec_arr)
        else:
            raise TypeError("statistic should be 'average' or 'median'")
        dist_arcsec = (
            np.sqrt(
                ((ra_arr - ra) * np.cos(self.target.dec * np.pi / 180)) ** 2
                + (dec_arr - dec) ** 2
            )
            * 3600
        )
        return (
            ra,
            dec,
            np.average(dist_arcsec / rms_arr),
            [ra_arr, dec_arr, rms_arr, dist_arcsec],
        )

    def bin(
        self,
        bin_period="mjd",
        statistic="median",
        weight="inverse variance",
        clip=3,
        use_pull=True,
    ):
        # replace intranight observations with a single photometric point
        bin_dict = {}
        for i in zip(self.mjd, self.ujy, self.dujy, self.mag, self.dmag):
            if bin_period in ["mjd", "day", "daily"]:
                bin_str = str(int(i[0]))
            elif bin_period in ["none", "None"]:
                bin_str = str(i[0])
            else:
                print(f"bin_period={bin_period} not valid, see help if it exists")
                print("proceeding with mjd binning")
                return self.bin(
                    bin_period="mjd", statistic=statistic, weight=weight, clip=clip
                )
            if bin_str not in bin_dict:
                bin_dict[bin_str] = []
            bin_dict[bin_str].append(i)
        holder = {}
        for i, param in enumerate(["mjd", "ujy", "dujy", "mag", "dmag"]):
            holder[param] = []
            for b in bin_dict.keys():
                nparr = np.array(bin_dict[b])
                if len(nparr) == 1:
                    holder[param].append(nparr[0, i])
                    continue
                for j in range(len(nparr[:, 2])):
                    if float(nparr[:, 2][j]) == 0.0:
                        nparr[:, 2][j] = 1e5
                if clip:
                    clip_results = utils.sigmaclip(
                        data=nparr[:, 1],
                        errors=nparr[:, 2],
                        sigmalow=clip,
                        sigmahigh=clip,
                        use_pull=use_pull,
                    )
                    if not sum(clip_results[2]):
                        continue
                    nparr = nparr[clip_results[2] == 1]
                if str(weight).lower() in [
                    "inverse variance",
                    "ivar",
                    "iv",
                    "least squares",
                    "least square",
                    "ls",
                ]:
                    w = 1 / nparr[:, 2] ** 2
                elif str(weight).lower() in ["none", "flat", "even"]:
                    w = np.ones(len(nparr[:, 2]))
                else:
                    print(f"weight={weight} not valid, see help if it exists")
                    print("proceeding with no weights")
                    w = np.ones(len(nparr[:, 2]))
                if statistic in ["ave", "average", "mean"]:
                    value = np.average(nparr[:, i], weights=w)
                elif statistic in ["median", "med"]:
                    value = utils.weighted_quantile(
                        nparr[:, i], [0.5], sample_weight=w
                    )[0]
                holder[param].append(value)
        if self.bandpass[0] not in ["Y", "J", "H", "K"]:
            for i in range(len(holder["mjd"])):
                mag, dmag = utils.ujy_to_ab(holder["ujy"][i], holder["dujy"][i])
                holder["mag"][i] = mag
                holder["dmag"][i] = dmag
        for key, val in holder.items():
            holder[key] = np.array(val)
        return holder

    def values_at_time(self, today=True, mjd=None, ymdhms=None, d=None, dt=None):
        if mjd:
            t = mjd
        else:
            t = utils.MJD(today=today, ymdhms=ymdhms, d=d, dt=dt)
        values = {}
        if not len(self.mjd):
            print("No detections for this lightcurve")
            return None
        if t < min(self.mjd) - 7 or t > max(self.mjd) + 7:
            print(f"{t} is more than 7 days away from the nearest photometric point.")
            return None
        if self.source == "model":
            mjd = self.mjd
            mag = self.mag
            ujy = self.ujy
            dmag = self.dmag
            dujy = self.dujy
        else:
            blc = self.bin()
            mjd = blc["mjd"]
            mag = blc["mag"]
            ujy = blc["ujy"]
            dmag = blc["dmag"]
            dujy = blc["dujy"]
        m1 = mjd > self.target.detection_date - 15
        m2 = mag != 99
        m3 = mag > 0
        m = m1 & m2 & m3
        if sum(m) < 2:
            return None
        elif sum(m) < 10:
            k = 1
        else:
            k = 3
        values["m"] = float(
            interpolate.splev(
                t,
                interpolate.splrep(
                    mjd[m], mag[m], 1 / (0.0001 + np.array(dujy[m])), k=k
                ),
                ext=0,
            )
        )
        values["dm_dt"] = float(
            interpolate.splev(
                t,
                interpolate.splrep(
                    mjd[m], mag[m], 1 / (0.0001 + np.array(dujy[m])), k=k
                ),
                ext=0,
                der=1,
            )
        )
        values["ujy"] = float(
            interpolate.splev(
                t,
                interpolate.splrep(
                    mjd[m], ujy[m], 1 / (0.0001 + np.array(dujy[m])), k=k
                ),
                ext=0,
            )
        )
        values["dujy_dt"] = float(
            interpolate.splev(
                t,
                interpolate.splrep(
                    mjd[m], dujy[m], 1 / (0.0001 + np.array(dujy[m])), k=k
                ),
                ext=0,
                der=1,
            )
        )
        # rough and dirty errors takes average and if time to nearest epoch is greater than 1 day, multiplies dmag by that difference cubed for the cubic spline
        values["dm"] = np.average(dmag[m]) * max(min(np.abs(t - mjd[m])), 1) ** 3
        values["dujy"] = np.average(dujy[m]) * max(min(np.abs(t - mjd[m])), 1) ** 3
        return values

    def get_peak_values(self):
        peak_vals = {"mjd": 0, "mag": 0, "dmag": 99, "ujy": 0, "dujy": 9999}
        try:
            maxima_idx = utils.get_extrema_idx(self.ujy, which="max")
            idxmax = maxima_idx[self.ujy[maxima_idx].argmax()]
        except (ValueError, IndexError):
            return peak_vals
        peak_vals["mjd"] = self.mjd[idxmax]
        peak_vals["mag"] = self.mag[idxmax]
        peak_vals["ujy"] = self.ujy[idxmax]
        if self.dmag[idxmax] != 0:
            peak_vals["dmag"] = self.dmag[idxmax]
        else:
            peak_vals["dmag"] = 0.3
        if self.dujy[idxmax] != 0:
            peak_vals["dujy"] = self.dujy[idxmax]
        else:
            peak_vals["dujy"] = self.ujy[idxmax] * 0.3
        return peak_vals

    def detections(self, use_uninspected=False, ATLAS_edge_clip=40):
        if use_uninspected:
            status_list = ("det", "?")
        else:
            status_list = ("det",)
        if self.source == "ATLAS":
            d = self.atlas_detections.filter(
                x__gt=ATLAS_edge_clip,
                y__gt=ATLAS_edge_clip,
                x__lt=constants.ATLAS_CHIP_SIZE - ATLAS_edge_clip,
                y__lt=constants.ATLAS_CHIP_SIZE - ATLAS_edge_clip,
                major__lt=constants.ATLAS_AXIS_RATIO_LIM * F("minor"),
                sky__gt=constants.ATLAS_SKY_LIM,
            )
        elif self.source == "ZTF":
            d = self.ztf_detections
        elif self.source == "ASAS-SN":
            d = self.asassn_detections
        elif self.source == "UKIRT":
            d = self.ukirt_detections
        elif self.source == "model":
            return self.model_detections.all().order_by("mjd")
        return d.filter(status__in=status_list).exclude(dujy=0).order_by("mjd")

    def nondetections(self):
        if self.source == "ATLAS":
            return self.atlas_detections.filter(status="non")
        elif self.source == "ZTF":
            return self.ztf_detections.filter(status="non")
        elif self.source == "ASAS-SN":
            return self.asassn_detections.filter(status="non")
        elif self.source == "UKIRT":
            return self.ukirt_detections.filter(status="non")
        elif self.source == "model":
            return

    def detection_attr(self, attr):
        return np.array([getattr(det, attr) for det in self.detections()])

    def refresh_from_file(self, path=None):
        if self.source != "UKIRT":
            return
        if path is None:
            if self.target.sn_type.name == "CALSPEC":
                year_dir = "non_transient"
            else:
                year_dir = f"20{self.target.TNS_name[:2]}"
            path = f"{constants.DATA_DIR}/{year_dir}/{self.target.TNS_name}/ukirt/photometry/{self.bandpass}_{self.variant}/results.txt"
        self.detections().delete()
        self.nondetections().delete()
        if not os.path.exists(path):
            return
        with open(path, "r") as f:
            flux, err, mjd = [[] for _ in range(3)]
            for line in f.readlines():
                if line.startswith("SN"):
                    flux.append(float(line.split()[1]))
                    err.append(float(line.split()[2]))
                elif line.startswith("MJD"):
                    mjd.append(float(line.split()[1]))
        obs = self.target.observations.all()
        obs_mjds = np.array(obs.values_list("mjd", flat=True))
        for f, e, m in zip(flux, err, mjd):
            mag = 24 - 2.5 * np.log10(f)
            dmag = 2.5 * np.log10(1 + e / f)
            i = int(np.abs(obs_mjds - m).argmin())
            status = "det"
            if np.isnan(mag) or dmag == 0 or e == 0:
                mag, dmag, status = 99, 99, "non"
            UkirtDetection.objects.create(
                lc=self,
                mjd=m,
                observation=obs[i],
                status=status,
                mag=mag,
                dmag=dmag,
                ujy=f,
                dujy=e,
                zp=24,
            )

    def dump_as_str(self, delimiter="\t"):
        outlist = [
            utils.dump_as_str(Lightcurve, delimiter),
        ]
        for manager in (
            self.atlas_detections,
            self.ztf_detections,
            self.asassn_detections,
            self.ukirt_detections,
            self.model_detections,
        ):
            for detection in manager.all():
                outlist.append(utils.dump_as_str(type(detection), delimiter=delimiter))
        return "\n".join(outlist)

    @property
    def mjd(self):
        return np.array([det.mjd for det in self.detections()])

    @property
    def mag(self):
        return np.array([det.mag for det in self.detections()])

    @property
    def dmag(self):
        return np.array([det.dmag for det in self.detections()])

    @property
    def ujy(self):
        return np.array([det.ujy for det in self.detections()])

    @property
    def dujy(self):
        return np.array([det.dujy for det in self.detections()])


class AtlasDetection(models.Model):
    CAMERA_CHOICES = [
        ("01a", "Mauna Loa"),
        ("02a", "Haleakala"),
        ("03a", "Southern Telescope 1"),
        ("04a", "Southern Telescope 2"),
    ]
    lc = models.ForeignKey(
        "Lightcurve", related_name="atlas_detections", on_delete=models.CASCADE
    )
    status = models.CharField(max_length=4, default="g", choices=STATUS_CHOICES)
    camera = models.CharField(max_length=3, choices=CAMERA_CHOICES)
    exp_num = models.PositiveIntegerField()
    mjd = models.FloatField()
    mag = models.FloatField(null=True)
    dmag = models.FloatField(null=True)
    ujy = models.IntegerField(null=True)
    dujy = models.PositiveIntegerField(null=True)
    err = models.BooleanField(default=False)
    chin = models.FloatField(null=True)
    ra = models.FloatField(null=True)
    dec = models.FloatField(null=True)
    x = models.FloatField(null=True)
    y = models.FloatField(null=True)
    major = models.FloatField(null=True)
    minor = models.FloatField(null=True)
    phi = models.FloatField(null=True)
    apfit = models.FloatField(null=True)
    mag5sig = models.FloatField(null=True)
    sky = models.FloatField(null=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                name="atlas_detection_lc_epoch_camera_expnum",
                fields=["lc", "mjd", "camera", "exp_num"],
            )
        ]

    def __str__(self):
        return f"{self.lc.target.TNS_name} - {self.mjd}"

    def get_coords(self, astrometry_radius=0.3):
        # read ddc to find closest source
        # change TSK(x,y) to Numpy/C(x,y), then pixel_to_world
        w = self.get_astrometry(threshold=astrometry_radius)
        ddc = self.get_ddc(sort_by="dist")
        sc = w.pixel_to_world(ddc["x"][0] - 0.5, ddc["y"][0] - 0.5)
        # get rms
        phot = self.get_phot(threshold=astrometry_radius)
        sc_list = SkyCoord(phot["raref"], phot["decref"], unit="deg")
        sc_calc = w.pixel_to_world(phot["x"] - 0.5, phot["y"] - 0.5)
        separations = sc_list.separation(sc_calc)
        rms = np.sqrt(np.average(separations**2)).to("arcsec").value
        defaults = ddc.T[0].to_dict()
        for key, val in defaults.items():
            if int(val) == val:
                defaults[key] = int(val)
        defaults.pop("dist")
        defaults["rms"] = rms
        defaults["nearest_ddc"] = ddc["dist"][0]
        defaults["ra"] = sc.ra.deg
        defaults["dec"] = sc.dec.deg
        AtlasCoordinate.objects.update_or_create(
            detection=self, astrometry_radius=astrometry_radius, defaults=defaults
        )

    @property
    def atlas_obs_name(self):
        return f"{self.camera}{int(self.mjd)}o{str(self.exp_num).zfill(4)}{self.lc.bandpass}"

    def get_phot(self, threshold=None, no_err=True):
        # x,y in TSK convention
        # center of lower left pixel at 0.5,0.5
        # sky2pix -TSKpix raref decref gives x/y+dx/dy
        file_path = f"{constants.MEDIA_DIR}/{self.lc.target.TNS_name}/atlas_files/{self.atlas_obs_name}.phot"
        if not os.path.exists(file_path):
            utils.dl_atlas_file(
                f"{os.environ['ATLAS_PHOT_PATH']}/{self.camera}/{int(self.mjd)}/{self.atlas_obs_name}.phot",
                file_path,
            )
        phot = pd.read_csv(
            file_path, delim_whitespace=True, skiprows=1, names=constants.PHOT_COLUMNS
        )
        if threshold is not None:
            phot = phot[
                np.sqrt(
                    (
                        (phot["raref"] - self.lc.target.ra)
                        * np.cos(self.lc.target.dec * np.pi / 180)
                    )
                    ** 2
                    + (phot["decref"] - self.lc.target.dec) ** 2
                )
                < threshold
            ].reset_index(drop=True)
        if no_err:
            phot = phot[phot["err"] == 0]
        return phot

    def get_ddc(self, sort_by="dist"):
        # TSK convention
        # center of lower left pixel at 0.5,0.5
        # sky2pix -TSKpix raref decref gives x/y+dx/dy
        file_path = f"{constants.MEDIA_DIR}/{self.lc.target.TNS_name}/atlas_files/{self.atlas_obs_name}.ddc"
        if not os.path.exists(file_path):
            utils.dl_atlas_file(
                f"{os.environ['ATLAS_DDC_PATH']}/{self.camera}/{int(self.mjd)}/{self.atlas_obs_name}.ddc",
                file_path,
            )
        ddc = pd.read_csv(
            file_path,
            delim_whitespace=True,
            skiprows=38,
            names=constants.DDC_COLUMNS,
            dtype=float,
        )
        ddc["dist"] = np.sqrt((ddc["x"] - self.x) ** 2 + (ddc["y"] - self.y) ** 2)
        ddc = ddc.sort_values(sort_by).reset_index(drop=True)
        return ddc

    def get_astrometry(self, threshold=0.3, match_threshold=1):
        # returns astrometric solution between TSK(x,y)-0.5 and ra/dec
        # transforming ra/dec with world_to_pixel gives TSK(x,y)-0.5 or WCS(x,y)-1 because it uses 0 based indexing (C, Numpy)
        phot = self.get_phot(threshold=threshold)
        rc = self.lc.target.get_refcat(rad=threshold)
        sc = SkyCoord(
            rc["RA"] * u.deg,
            rc["Dec"] * u.deg,
            pm_ra_cosdec=np.array(rc["pmra"])
            * np.cos(self.lc.target.dec * np.pi / 180)
            * u.mas
            / u.yr,
            pm_dec=np.array(rc["pmdec"]) * u.mas / u.yr,
            equinox="J2000",
            obstime="J2015.5",
        )
        y, m, d = utils.MJD_to_ut(self.mjd)
        new_sc = sc.apply_space_motion(new_obstime=Time(f"{y}-{m}-{d}"))
        with open(f"/tmp/hsf/{self.atlas_obs_name}.xyrd", "w") as f:
            for i, row in phot.iterrows():
                nearest_arr = np.sqrt(
                    (
                        (row["raref"] - sc.ra.value)
                        * np.cos(self.lc.target.dec * np.pi / 180)
                    )
                    ** 2
                    + (row["decref"] - sc.dec.value) ** 2
                )
                idx = nearest_arr.argmin()
                if nearest_arr.min() * 3600 > match_threshold:
                    continue
                # f.write(f"{row['x']} {row['y']} {row['raref']} {row['decref']}\n")
                f.write(
                    f"{row['x']} {row['y']} {new_sc[idx].ra.value} {new_sc[idx].dec.value}\n"
                )
        # halfpix changes TSK x,y to WCS x,y by adding 0.5
        os.system(
            f"mapsipv /tmp/hsf/{self.atlas_obs_name}.xyrd wcs=/tmp/hsf/{self.atlas_obs_name}.wcs halfpix"
        )
        hdu = fits.PrimaryHDU()
        with open(f"/tmp/hsf/{self.atlas_obs_name}.wcs", "r") as f:
            for line in f.readlines():
                key = line.split()[0].strip("=")
                val = line.split("=")[1].split()[0].strip("'")
                if key not in ["RADECSYS", "CTYPE1", "CTYPE2", "CUNIT1", "CUNIT2"]:
                    val = float(val)
                comment = line.split("/")[1].strip("\n")
                hdu.header.append((key, val, comment))
        hdu.header["CTYPE1"] = "RA---TAN-SIP"
        hdu.header["CTYPE2"] = "DEC--TAN-SIP"
        return WCS(hdu.header)

    def fit_spline_to_phot(self):
        phot = self.get_phot()
        tck = interpolate.bisplrep(phot["x"], phot["y"], phot["dx"])
        znew = interpolate.bisplev(
            np.linspace(min(phot["x"]), max(phot["x"]), 100),
            np.linspace(min(phot["y"]), max(phot["y"]), 100),
        )
        plt.imshow(znew)
        plt.colorbar()
        plt.show()

    def vector_field(self):
        phot = self.get_phot()
        plt.quiver(phot["x"], phot["y"], phot["dx"], phot["dy"])
        plt.show()


class AtlasCoordinate(models.Model):
    detection = models.ForeignKey(
        "AtlasDetection", related_name="coordinates", on_delete=models.CASCADE
    )
    astrometry_radius = models.FloatField()
    nearest_ddc = (
        models.FloatField()
    )  # pixel distance between sn coordinates projected to ddc file and nearest ddc source
    ra = models.FloatField()
    dec = models.FloatField()
    rms = models.FloatField()
    mag = models.FloatField()
    dmag = models.FloatField()
    x = models.FloatField()
    y = models.FloatField()
    major = models.FloatField()
    minor = models.FloatField()
    phi = models.FloatField()
    det = models.IntegerField()
    chin = models.FloatField()
    pvr = models.IntegerField()
    ptr = models.IntegerField()
    pmv = models.IntegerField()
    pkn = models.IntegerField()
    pno = models.IntegerField()
    pbn = models.IntegerField()
    pcr = models.IntegerField()
    pxt = models.IntegerField()
    psc = models.IntegerField()
    dup = models.IntegerField()
    wpflx = models.FloatField()
    dflx = models.FloatField()

    class Meta:
        constraints = [
            models.UniqueConstraint(
                name="coordinate_detection_radius",
                fields=["detection", "astrometry_radius"],
            )
        ]


class ZtfDetection(models.Model):
    # https://alerce.readthedocs.io/_/downloads/en/develop/pdf/
    g = 1
    r = 2
    i = 3
    FID_CHOICES = [
        (g, "ztfg"),
        (r, "ztfr"),
        (i, "ztfi"),
    ]
    SCI_MINUS_REF = 1
    REF_MINUS_SCI = -1
    ISDIFFPOS_CHOICES = [
        (SCI_MINUS_REF, "Candidate is from positive (sci minus ref) subtraction"),
        (REF_MINUS_SCI, "Candidate is from negative (ref minus sci) subtraction"),
    ]
    tid = models.CharField(max_length=5, blank=True)
    mjd = models.FloatField()
    fid = models.PositiveIntegerField(choices=FID_CHOICES)
    diffmaglim = models.FloatField()
    lc = models.ForeignKey(
        "Lightcurve", related_name="ztf_detections", on_delete=models.CASCADE
    )
    status = models.CharField(max_length=4, default="?", choices=STATUS_CHOICES)
    ujy = models.FloatField(null=True)
    dujy = models.FloatField(null=True)
    candid = models.CharField(max_length=20, null=True)
    pid = models.PositiveBigIntegerField(null=True)
    isdiffpos = models.FloatField(choices=ISDIFFPOS_CHOICES, null=True)
    nid = models.SmallIntegerField(null=True)
    distnr = models.FloatField(null=True)
    ra = models.FloatField(null=True)
    dec = models.FloatField(null=True)
    rb = models.FloatField(null=True)
    rbversion = models.CharField(max_length=9, null=True)
    drb = models.FloatField(null=True)
    rfid = models.PositiveBigIntegerField(null=True)
    has_stamp = models.BooleanField(null=True)
    corrected = models.BooleanField(null=True)
    dubious = models.BooleanField(null=True)
    candid_alert = models.CharField(max_length=20, null=True)
    step_id_corr = models.CharField(max_length=16)
    phase = models.FloatField(null=True)
    parent_candid = models.BigIntegerField(null=True)
    magpsf = models.FloatField(null=True)
    sigmapsf = models.FloatField(null=True)
    magpsf_corr = models.FloatField(null=True)
    sigmapsf_corr = models.FloatField(null=True)
    magpsf_corr_ext = models.FloatField(null=True)
    sigmapsf_corr_ext = models.FloatField(null=True)
    magap = models.FloatField(null=True)
    sigmagap = models.FloatField(null=True)
    magap_corr = models.FloatField(null=True)
    sigmagap_corr = models.FloatField(null=True)
    magapbig = models.FloatField(null=True)
    sigmagapbig = models.FloatField(null=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                name="ztf_detection_lc_epoch", fields=["lc", "fid", "mjd"]
            )
        ]

    def __str__(self):
        return f"{self.lc.target.TNS_name} - {self.mjd}"

    @property
    def mag(self):
        if self.status == "det":
            return self.magpsf
        elif self.status == "non":
            return self.diffmaglim

    @property
    def dmag(self):
        if self.status == "det":
            return self.sigmapsf
        elif self.status == "non":
            return utils.ab_to_ujy(self.diffmaglim, 0)[0]

    def save(self, *args, **kwargs):
        if (
            (self.ujy is None or self.dujy is None)
            and self.magpsf is not None
            and self.sigmapsf is not None
        ):
            self.ujy, self.dujy = utils.ab_to_ujy(self.magpsf, self.sigmapsf)
        super(ZtfDetection, self).save(*args, **kwargs)


class AsassnDetection(models.Model):
    lc = models.ForeignKey(
        "Lightcurve", related_name="asassn_detections", on_delete=models.CASCADE
    )
    status = models.CharField(max_length=4, default="?", choices=STATUS_CHOICES)
    mjd = models.FloatField()
    hjd = models.FloatField()
    ut_date = models.DateTimeField()
    image = models.CharField(max_length=15)
    camera = models.CharField(max_length=2)
    exp_num = models.PositiveIntegerField()
    fwhm = models.FloatField()
    diff = models.FloatField()
    limit = models.FloatField()
    mag = models.FloatField()
    dmag = models.FloatField()
    counts = models.FloatField()
    count_err = models.FloatField()
    ujy = models.IntegerField()
    dujy = models.PositiveIntegerField()

    class Meta:
        constraints = [
            models.UniqueConstraint(
                name="asassn_detection_lc_epoch_image", fields=["lc", "mjd", "image"]
            )
        ]

    def __str__(self):
        return f"{self.lc.target.TNS_name} - {self.mjd}"


class UkirtDetection(models.Model):
    lc = models.ForeignKey(
        "Lightcurve",
        null=True,
        related_name="ukirt_detections",
        on_delete=models.CASCADE,
    )
    observation = models.ForeignKey("data.Observation", on_delete=models.CASCADE)
    mjd = models.FloatField()
    status = models.CharField(max_length=4, default="?", choices=STATUS_CHOICES)
    mag = models.FloatField(null=True)
    dmag = models.FloatField(null=True)
    ujy = models.FloatField(null=True)
    dujy = models.FloatField(null=True)
    x = models.FloatField(null=True)
    y = models.FloatField(null=True)
    zp = models.FloatField(null=True)
    dzp = models.FloatField(null=True)

    class Meta:
        """Meta."""

        constraints = [
            models.UniqueConstraint(
                name="ukirt_detection_lc_epoch",  # mjd is degenerate with observation
                fields=["lc", "mjd"],
            )
        ]

    def __str__(self):
        return f"{self.lc.target.TNS_name} - {self.mjd}"


class ModelDetection(models.Model):
    lc = models.ForeignKey(
        "Lightcurve", related_name="model_detections", on_delete=models.CASCADE
    )
    mjd = models.FloatField()
    mag = models.FloatField()
    dmag = models.FloatField(default=0)
    ujy = models.FloatField()
    dujy = models.FloatField(default=0)
