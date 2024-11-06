import copy
import os
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta

import astropy.table as at
import astropy.units as u
import constants
import dotenv
import gspread
import numpy as np
import pytz
import requests
import utils
from apiclient.http import MediaFileUpload, MediaIoBaseDownload
from astropy.coordinates import SkyCoord
from bs4 import BeautifulSoup
from django.db import models
from googleapiclient.discovery import build
from hsf.settings import BASE_DIR
from oauth2client.service_account import ServiceAccountCredentials

dotenv_file = os.path.join(BASE_DIR, ".env")
if os.path.isfile(dotenv_file):
    dotenv.load_dotenv(dotenv_file)

xmlns = "{https://ukirt.ifa.hawaii.edu/omp/schema/TOML}"


# Create your models here.
class MSBList(models.Model):
    targets = models.ManyToManyField(
        "targets.Target", through="TargetOTDetails", related_name="msb_list"
    )
    previous = models.ForeignKey(
        "self", blank=True, null=True, related_name="next", on_delete=models.SET_NULL
    )
    date = models.DateField(primary_key=True)
    last_updated = models.DateField(auto_now=True)
    program = models.CharField(max_length=10, default="H02")
    semester = models.CharField(max_length=3, default="23A")

    def import_from_previous(self):
        """
        copy over the m2m table from the previous list
        """
        for tod in self.previous.target_details.filter(
            rejection_date__isnull=True, remaining__gt=0, drop=False
        ):
            tod.pk = None  # creates new pk/object on save
            tod.msblist = self
            tod.save()

    def import_from_xml(self, file_path):
        # to avoid circuluar imports, putting in method
        from targets.models import Target

        root = ET.parse(file_path).getroot()
        xmlns = root.tag.split("Sp")[0]
        for msb in root.findall(f"{xmlns}SpMSB"):
            if msb.find(f"{xmlns}title").text in [
                "NGC4242 - YJH band - WFCAM_FLIP_SLOW",
                "AT20uxz - YJHK and CALSPEC standards",
            ]:
                continue
            site_quality = msb.find(f"{xmlns}SpSiteQualityObsComp")
            cloud_choice = "a"
            cloud_max = site_quality.find(f"{xmlns}cloud").find(f"{xmlns}max").text
            if cloud_max == "20":
                cloud_choice = "tc"
            elif cloud_max == "0":
                cloud_choice = "p"
            moon = site_quality.find(f"{xmlns}moon")
            moon_choice = "dc"
            if moon:
                moon_max = moon.find(f"{xmlns}max").text
                if moon_max == "25":
                    moon_choice = "g"
                elif moon_max == "0":
                    moon_choice = "d"
            seeing = site_quality.find(f"{xmlns}seeing")
            seeing_min = None
            seeing_max = None
            if seeing:
                seeing_min = float(seeing.find(f"{xmlns}min").text)
                seeing_max = float(seeing.find(f"{xmlns}max").text)
            cso_tau = site_quality.find(f"{xmlns}csoTau")
            cso_tau_min = None
            cso_tau_max = None
            if cso_tau:
                cso_tau_min = float(cso_tau.find(f"{xmlns}min").text)
                cso_tau_max = float(cso_tau.find(f"{xmlns}max").text)
            j_sky = site_quality.find(f"{xmlns}skyBrightness")
            j_sky_min = None
            j_sky_max = None
            if j_sky:
                j_sky_min = float(j_sky.find(f"{xmlns}min").text)
                j_sky_max = float(j_sky.find(f"{xmlns}max").text)
            scheduling = msb.find(f"{xmlns}SpSchedConstObsComp")
            source_hour_angle = site_quality.find(f"{xmlns}meridianApproach")
            if source_hour_angle is None:
                source_hour_angle = "dc"
            else:
                source_hour_angle = source_hour_angle[0]
            coord_dict = {}
            for component in (
                msb.findall(f"{xmlns}SpObs")[1]
                .find(f"{xmlns}SpTelescopeObsComp")
                .findall(f"{xmlns}BASE")
            ):
                coord_comp = component.find(f"{xmlns}target").find(
                    f"{xmlns}spherSystem"
                )
                ra = coord_comp.find(f"{xmlns}c1").text
                dec = coord_comp.find(f"{xmlns}c2").text
                coords = f"{ra} {dec}"
                t = component.attrib["TYPE"]
                coord_dict[t] = coords
            for t in [
                "BASE",
                "GUIDE",
                "GUIDE2",
                "SKY0",
                "SKYGUIDE0",
                "SKYGUIDE1",
                "SKYGUIDE2",
                "SKYGUIDE3",
            ]:
                if t not in coord_dict:
                    coord_dict[t] = ""
            filters = ""
            exp_times = ""
            coadds = ""
            for component in msb.findall(f"{xmlns}SpObs")[1::2]:
                filters += (
                    component.find(f"{xmlns}SpInstWFCAM").find(f"{xmlns}filter").text
                )
                exp_times += (
                    component.find(f"{xmlns}SpInstWFCAM")
                    .find(f"{xmlns}exposureTime")
                    .text
                    + " "
                )
                coadds += (
                    component.find(f"{xmlns}SpInstWFCAM").find(f"{xmlns}coadds").text
                    + " "
                )
            TargetOTDetails.objects.get_or_create(
                msblist=self,
                target=Target.quick_get(
                    utils.get_TNS_name(msb.find(f"{xmlns}title").text.split()[0])
                ),
                defaults=dict(
                    remaining=int(msb.attrib["remaining"]),
                    priority=int(msb.find(f"{xmlns}priority").text),
                    cloud_status=cloud_choice,
                    moon_status=moon_choice,
                    seeing_min=seeing_min,
                    seeing_max=seeing_max,
                    cso_tau_min=cso_tau_min,
                    cso_tau_max=cso_tau_max,
                    j_sky_min=j_sky_min,
                    j_sky_max=j_sky_max,
                    earliest_date=utils.ukirt_str_to_datetime(
                        scheduling.find(f"{xmlns}earliest").text
                    ),
                    latest_date=utils.ukirt_str_to_datetime(
                        scheduling.find(f"{xmlns}latest").text
                    ),
                    elevation_max=float(scheduling.find(f"{xmlns}maxEl").text),
                    elevation_min=float(scheduling.find(f"{xmlns}minEl").text),
                    reschedule_every=float(scheduling.find(f"{xmlns}period").text),
                    source_hour_angle=source_hour_angle,
                    filters=filters,
                    exp_times=exp_times,
                    coadds=coadds,
                    base_coords=coord_dict["Base"],
                    guide_coords=coord_dict["GUIDE"],
                    guide_2_coords=coord_dict["GUIDE2"],
                    sky_coords=coord_dict["SKY0"],
                    sky_guide_0_coords=coord_dict["SKYGUIDE0"],
                    sky_guide_1_coords=coord_dict["SKYGUIDE1"],
                    sky_guide_2_coords=coord_dict["SKYGUIDE2"],
                    sky_guide_3_coords=coord_dict["SKYGUIDE3"],
                ),
            )

    def process_ukirt_log(self, copy_current_targets=True):
        # to avoid circular imports, importing in method
        from targets.models import Target

        self.target_details.filter(observed=True).update(observed=False)

        # get list of MSBs in queue and number of remaining observations from web page
        soup = utils.read_ukirt_page(self.semester, self.program)
        if copy_current_targets:
            targets = [
                utils.get_TNS_name(i.text.split()[0].strip("-"))
                for i in soup.find_all("table")[-1].find_all("td")[8::7]
            ]
            filters = [
                i.text.replace("/", "")
                for i in soup.find_all("table")[-1].find_all("td")[9::7]
            ]
            remaining = [
                int(i.text) for i in soup.find_all("table")[-1].find_all("td")[13::7]
            ]
            # remove old targets not on project site.
            self.target_details.exclude(target__TNS_name__in=targets).delete()
            for t, f, r in zip(targets, filters, remaining):
                coadds = "".join(["1 " for i in f])
                tod, _ = self.target_details.get_or_create(
                    target=Target.quick_get(t),
                    defaults={"filters": f, "coadds": coadds},
                )
                previous_tod = self.previous.target_details.filter(target=tod.target)
                if previous_tod.exists():
                    tod.reference = previous_tod.first().reference
                tod.remaining = r
                tod.save()

        # read specific log to get list of observations
        dates = [
            i.text
            for i in soup.find_all("table")[4].find_all("a")
            if i.attrs["href"].startswith("utprojlog")
        ]
        if f"{self.date.year}-{self.date.month:02d}-{self.date.day:02d}" not in dates:
            print(f"{self.date} not in {dates}")
            return
        observed_list = utils.read_ukirt_log(
            year=self.date.year,
            month=self.date.month,
            day=self.date.day,
            semester=self.semester,
            program=self.program,
        )
        for name, bps in observed_list:
            obj = Target.quick_get(name)
            for bp in bps:
                observations = obj.observations.filter(date=self.date, bandpass=bp)
                if observations.exists():
                    continue
                name = f"{obj.TNS_name}.{bp}.{self.date.year}{str(self.date.month).zfill(2)}{str(self.date.day).zfill(2)}"
                obj.observations.create(
                    status="pending",
                    bandpass=bp,
                    name=name,
                    date=self.date,
                    mjd=utils.MJD(
                        year=self.date.year, month=self.date.month, day=self.date.day
                    ),
                    program=f"U/{self.semester}/{self.program}",
                    path=f"placeholder for {name}",
                )
            tod, new = self.target_details.get_or_create(target=obj, filters=bps)
            tod.observed = True
            tod.earliest_date = datetime.now().astimezone(
                pytz.timezone("UTC")
            ) + timedelta(days=tod.reschedule_every - 1)
            if self.previous.target_details.filter(target=tod.target).exists():
                prev = self.previous.target_details.get(target=obj)
                tod.remaining = prev.remaining - 1
                tod.exp_times = prev.exp_times
            tod.save()

    def update_tns_info(self):
        """
        Check TNS for objects classified within past two days
        Remove objects classified as non-Ias
        Ignore TNS update for references
        """
        for tod in self.target_details.filter(reference=False):
            obj = tod.target
            obj.update_TNS()
            if (
                not obj.sn_type.subtype_of("Ia")
                and not obj.sn_type.name == "?"
                and obj.TNS_name not in constants.OT_SKIP
            ):
                tod.drop = True
                tod.save()
                obj.status_reason += (
                    f"Classified as {obj.sn_type} on {utils.MJD(today=True)}\n"
                )
                obj.save()

    def update_phase(self):
        """
        Check targets still at 3 observation, drop old ones
        """
        for tod in self.target_details.filter(reference=False, remaining=3):
            obj = tod.target
            if not obj.tmax:
                continue
            if (
                obj.tmax < utils.MJD(today=True)
                and obj.TNS_name not in constants.OT_SKIP
            ):
                tod.drop = True
                tod.save()
                obj.status_reason = f"Lightcurve automatically rejected for phase on {utils.MJD(today=True)}"
                obj.save()

    def update_exp_times(self):
        """
        Parse tree to find exposure times and adjust that field in the target.
        Does not affect tree
        """
        for tod in self.target_details.filter(reference=False):
            obj = tod.target
            obj.update_current_values()
            obj.save()
            tod.get_exp_times()
            for f, et in zip(tod.filters, tod.exp_times.split()):
                if float(et) == 99 and obj.TNS_name not in constants.OT_SKIP:
                    tod.drop = True
                    tod.save()
                    obj.status_reason = f"Mag estimate exceeded upper limit in {f} on {utils.MJD(today=True)}"
                    obj.save()

    def write_out(self):
        """
        Write tree to media directory
        """
        root = ET.parse(
            f"{constants.STATIC_DIR}/program_head.xml"
        ).getroot()
        root.find(f"{xmlns}projectID").text = f"U/{self.semester}/{self.program}"
        for tod in self.target_details.filter(
            remaining__gt=0, drop=False, rejection_date__isnull=True
        ):
            tod.get_coords(force=False)
            root.append(tod.new_msb())

        # enforce id sequence
        for i, msb in enumerate(root.findall(f"{xmlns}SpMSB")):
            msb.find(f"{xmlns}SpSchedConstObsCompRef").attrib["idref"] = f"{2*i+1}"
            msb.find(f"{xmlns}SpSiteQualityObsCompRef").attrib["idref"] = f"{2*i}"
            msb.find(f"{xmlns}SpSiteQualityObsComp").attrib["id"] = f"{2*i}"
            msb.find(f"{xmlns}SpSchedConstObsComp").attrib["id"] = f"{2*i+1}"
        ET.register_namespace("", xmlns[1:-1])
        name = (
            f"{int(self.date.year-2000)}"
            f"{int(self.date.month):02d}"
            f"{int(self.date.day):02d}.xml"
        )
        xml_out = f"{constants.MEDIA_DIR}/new_{name}"
        ET.ElementTree(root).write(xml_out)
        utils.upload_file(xml_out)

    def add_target(
        self, target, filters="YJ", exp_times="", reference=False, priority=2, epochs=3
    ):
        tod, new = self.target_details.get_or_create(
            target=target,
            defaults={
                "filters": filters,
                "reference": reference,
                "priority": priority,
                "remaining": epochs,
            },
        )
        tod.get_coords()
        if exp_times == "":
            tod.get_exp_times()
        else:
            tod.exp_times = exp_times
        tod.save()

    def update_current_targets_sheet(self):
        """
        update current UKIRT targets for sharing purposes
        """
        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive",
        ]
        creds = ServiceAccountCredentials.from_json_keyfile_name(
            f"{constants.HSF_DIR}/hawaii-supernova-flows-1332a295a90b.json",
            scope,
        )
        client = gspread.authorize(creds)
        sheet = client.open_by_key(os.environ["CURRENT_TARGETS_ID"])
        sheet_instance = sheet.get_worksheet(0)

        sheet_instance.clear()

        def safe_insert_row(
            sheet_instance, row, index, value_input_option="USER_ENTERED"
        ):
            try:
                sheet_instance.insert_row(
                    row, index=index, value_input_option=value_input_option
                )
            except gspread.client.APIError:
                print("Waiting 60s and trying again")
                time.sleep(60)
                sheet_instance.insert_row(
                    row, index=index, value_input_option=value_input_option
                )

        i = 1
        safe_insert_row(sheet_instance, [f"Date of last update: {str(self.date)}"], i)
        for category, l in zip(
            [
                "Objects still in the queue",
                "Objects added to queue",
                "Objects removed from the queue",
            ],
            [
                self.get_in_set(output="targets").filter(TNS_name__gt="22"),
                self.get_add_set(output="targets").filter(TNS_name__gt="22"),
                self.targets.filter(
                    TNS_name__in=self.target_details.filter(drop=True).values_list(
                        "target__TNS_name"
                    )
                ),
            ],
        ):
            i += 1
            safe_insert_row(sheet_instance, [category], i)
            for obj in l:
                i += 1
                epochs = [
                    str(d)
                    for d in obj.observations.values_list("date", flat=True).distinct()
                ]
                rmax = None
                obj.update_peak_values()
                if "ztfr" in obj.peak_values:
                    if "mag" in obj.peak_values["ztfr"]:
                        rmax = np.average(obj.peak_values["ztfr"]["mag"])
                row = [
                    f"20{obj.TNS_name}",
                    utils.deg2HMS(ra=obj.ra, rounding=1).replace(" ", ":"),
                    utils.deg2HMS(dec=obj.dec, rounding=2).replace(" ", ":"),
                    rmax,
                ]
                row += epochs
                safe_insert_row(sheet_instance, row, index=i, value_input_option="RAW")
            if not l.exists():
                i += 1
                safe_insert_row(sheet_instance, ["None"], index=i)

    def get_xml_as_bytes(self):
        date_str = f"{self.date.year-2000}{str(self.date.month).zfill(2)}{str(self.date.day).zfill(2)}"
        xml_out = f"{constants.MEDIA_DIR}/new_{date_str}.xml"
        if not os.path.exists(xml_out):
            self.write_out()
        with open(xml_out, "rb") as f:
            return f.read()

    def get_add_set(self, output="details"):
        add_set = self.target_details.filter(rejection_date__isnull=True).exclude(
            target__in=self.previous.targets.all()
        )
        if output == "details":
            return add_set
        elif output == "targets":
            return self.targets.filter(
                TNS_name__in=add_set.values_list("target__TNS_name")
            )
        elif output == "names":
            return add_set.values_list("target__TNS_name", flat=True)

    def get_in_set(self, output="details"):
        in_set = self.target_details.filter(
            target__in=self.previous.targets.all(),
            remaining__gt=0,
            drop=False,
            rejection_date__isnull=True,
        )
        if output == "details":
            return in_set
        elif output == "targets":
            return self.targets.filter(
                TNS_name__in=in_set.values_list("target__TNS_name")
            )
        elif output == "names":
            return in_set.values_list("target__TNS_name", flat=True)

    @staticmethod
    def create_todays_list():
        """
        Get latest MSBList, overwrite date (pk), wipe lists
        """
        now = datetime.now(pytz.timezone("US/Hawaii"))
        today = date(now.year, now.month, now.day)
        prev = None
        prev_list = MSBList.objects.filter(date__lt=today)
        if prev_list.exists():
            prev = prev_list.latest("date")
        msb, created = MSBList.objects.get_or_create(date=today, previous=prev, last_updated=today)
        return msb

    @staticmethod
    def process_ukirt_program(semester="22A", program="H06"):
        from targets.models import Target

        soup = utils.read_ukirt_page(semester=semester, program=program)
        dates = [
            i.text
            for i in soup.find_all("table")[4].find_all("a")
            if i.attrs["href"].startswith("utprojlog")
        ]
        for dd in dates[1:]:
            y, m, d = dd.split("-")
            date_object = date(int(y), int(m), int(d))
            ol = utils.read_ukirt_log(y, m, d, semester, program)
            for name, bps in ol:
                obj = Target.quick_get(name)
                for bp in bps:
                    observations = obj.observations.filter(
                        date=date_object, bandpass=bp
                    )
                    if observations.exists():
                        continue
                    obj.observations.create(
                        status="pending",
                        bandpass=bp,
                        name=f"{obj.TNS_name}.{bp}.{y}{m}{d}",
                        date=date_object,
                        mjd=utils.MJD(year=int(y), month=int(m), day=int(d)),
                        program=f"U/{semester}/{program}",
                    )


class TargetOTDetails(models.Model):
    """
    Many-to-many table linking targets to MSBLists
    """

    MOON_CHOICES = [
        ("d", "Dark (moon not up)"),
        ("g", "Grey (<25% illuminated)"),
        ("dc", "Don't Care (Any)"),
    ]
    CLOUD_CHOICES = [
        ("p", "Photometric (no attenuation variability)"),
        ("tc", "Thin Cirrus (<20% attenuation variability)"),
        ("a", "Allocated"),
    ]
    HA_CHOICES = [
        ("r", "Rising"),
        ("s", "Setting"),
        ("dc", "Don't Care"),
    ]
    target = models.ForeignKey(
        "targets.Target",
        null=True,
        on_delete=models.SET_NULL,
        related_name="ot_details",
    )
    msblist = models.ForeignKey(
        "MSBList", on_delete=models.CASCADE, related_name="target_details"
    )
    remaining = models.IntegerField(default=3)
    priority = models.IntegerField(default=2)
    seeing_min = models.FloatField(null=True, default=0.0)
    seeing_max = models.FloatField(null=True, default=1.8)
    cso_tau_min = models.FloatField(null=True)
    cso_tau_max = models.FloatField(null=True)
    moon_status = models.CharField(default="dc", max_length=2, choices=MOON_CHOICES)
    cloud_status = models.CharField(default="tc", max_length=2, choices=CLOUD_CHOICES)
    j_sky_min = models.FloatField(null=True)
    j_sky_max = models.FloatField(null=True)
    earliest_date = models.DateTimeField(null=True)
    latest_date = models.DateTimeField(null=True)
    elevation_max = models.FloatField(default=90.0)
    elevation_min = models.FloatField(default=30.0)
    reschedule_every = models.FloatField(null=True, default=3)
    source_hour_angle = models.CharField(default="dc", max_length=2, choices=HA_CHOICES)

    base_coords = models.CharField(max_length=26, blank=True)
    guide_coords = models.CharField(max_length=26, blank=True)
    guide_2_coords = models.CharField(max_length=26, blank=True)
    sky_coords = models.CharField(max_length=26, blank=True)
    sky_guide_0_coords = models.CharField(max_length=26, blank=True)
    sky_guide_1_coords = models.CharField(max_length=26, blank=True)
    sky_guide_2_coords = models.CharField(max_length=26, blank=True)
    sky_guide_3_coords = models.CharField(max_length=26, blank=True)

    filters = models.CharField(max_length=5, default="YJ")
    exp_times = models.CharField(max_length=20, default="1 1 ")
    coadds = models.CharField(max_length=10, default="1 1 ")

    observed = models.BooleanField(default=False)
    drop = models.BooleanField(default=False)
    reference = models.BooleanField(default=False)
    rejection_date = models.DateField(null=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                name="tod_msblist_target", fields=["msblist", "target"]
            ),
        ]

    def __str__(self):
        if self.target:
            return f"{self.target.TNS_name} {self.msblist.date}"
        else:
            return f"Null target from {self.msblist.date}"

    def get_coords(self, force=False):
        """
         Modified from code from David Jones
         get a field center that allows enough guide stars
         """
        qset = (
            TargetOTDetails.objects.filter(target=self.target)
            .exclude(base_coords="")
            .exclude(guide_coords="")
            .exclude(sky_coords="")
            .exclude(sky_guide_0_coords="")
        )
        if qset.exists() and not force:
            tod = qset.first()
            if tod.base_coords != "":
                self.base_coords = tod.base_coords
                self.guide_coords = tod.guide_coords
                self.guide_2_coords = tod.guide_2_coords
                self.sky_coords = tod.sky_coords
                self.sky_guide_0_coords = tod.sky_guide_0_coords
                self.sky_guide_1_coords = tod.sky_guide_1_coords
                self.sky_guide_2_coords = tod.sky_guide_2_coords
                self.save()
            return
        if not force and self.base_coords != "":
            return

        sc = SkyCoord(self.target.ra, self.target.dec, unit=u.deg)
        # Each array covers 13.65 x 13.65 arcmin, gaps between are 12.83 arcmin
        guide_area = SkyCoord(
            (
                sc.ra.deg
                - (13.65 / 2.0 / 60.0 + 12.83 / 2.0 / 60.0)
                / np.cos(sc.dec.deg * np.pi / 180)
            )
            * u.deg,
            (sc.dec.deg - 13.65 / 2.0 / 60.0 - 12.83 / 2.0 / 60.0) * u.deg,
        )
        skyguide_area = SkyCoord(
            (
                sc.ra.deg
                - (13.65 / 2.0 / 60.0 + 12.83 / 2.0 / 60.0)
                / np.cos(sc.dec.deg * np.pi / 180)
            )
            * u.deg,
            (sc.dec.deg + 13.65 / 2.0 / 60.0 + 12.83 / 2.0 / 60.0) * u.deg,
        )
        # diamond is 4.25 arcmin on a side
        # approximating with circle of radius 2.1 arcmin but this is rough

        # define an ra/dec grid of +/-13.65/2 arcmin
        guide_rc = utils.refcat(
            guide_area.ra.deg, guide_area.dec.deg, rad=5 / 60, all_cols=True
        )
        skyguide_rc = utils.refcat(
            skyguide_area.ra.deg, skyguide_area.dec.deg, rad=5 / 60, all_cols=True
        )
        # Tonry 2018, Virtually all galaxies can be rejected by selecting objects for which Gaia provides a nonzero proper-motion uncertainty, dpmra and dpmdec, at the cost of about 0.7% of all real stars.
        m1 = guide_rc["dpmra"] > 0
        m2 = guide_rc["dpmdec"] > 0
        guide_rc = guide_rc[m1 & m2]
        m3 = skyguide_rc["dpmra"] > 0
        m4 = skyguide_rc["dpmdec"] > 0
        skyguide_rc = skyguide_rc[m3 & m4]

        scguidestar = SkyCoord(
            guide_rc["ra"][guide_rc["r"] < 15.5],
            guide_rc["dec"][guide_rc["r"] < 15.5],
            unit="deg",
        )
        scskyguidestar = SkyCoord(
            skyguide_rc["ra"][skyguide_rc["r"] < 15.5],
            skyguide_rc["dec"][skyguide_rc["r"] < 15.5],
            unit="deg",
        )

        # now loop over a grid, 0.5-arcmin steps?
        done = False
        step = 0.5
        for rstep in range(13):
            if done:
                break
            for dstep in range(13):
                if done:
                    break
                newguide1 = SkyCoord(
                    guide_area.ra.deg + step * rstep / 60,
                    guide_area.dec.deg + step * dstep / 60,
                    unit=u.deg,
                )
                newguide2 = SkyCoord(
                    guide_area.ra.deg - step * rstep / 60,
                    guide_area.dec.deg + step * dstep / 60,
                    unit=u.deg,
                )
                newguide3 = SkyCoord(
                    guide_area.ra.deg + step * rstep / 60,
                    guide_area.dec.deg - step * dstep / 60,
                    unit=u.deg,
                )
                newguide4 = SkyCoord(
                    guide_area.ra.deg - step * rstep / 60,
                    guide_area.dec.deg - step * dstep / 60,
                    unit=u.deg,
                )

                newskyguide1 = SkyCoord(
                    skyguide_area.ra.deg + step * rstep / 60,
                    skyguide_area.dec.deg + step * dstep / 60,
                    unit=u.deg,
                )
                newskyguide2 = SkyCoord(
                    skyguide_area.ra.deg - step * rstep / 60,
                    skyguide_area.dec.deg + step * dstep / 60,
                    unit=u.deg,
                )
                newskyguide3 = SkyCoord(
                    skyguide_area.ra.deg + step * rstep / 60,
                    skyguide_area.dec.deg - step * dstep / 60,
                    unit=u.deg,
                )
                newskyguide4 = SkyCoord(
                    skyguide_area.ra.deg - step * rstep / 60,
                    skyguide_area.dec.deg - step * dstep / 60,
                    unit=u.deg,
                )

                for guides in [
                    (newguide1, newskyguide1, step * rstep / 60, step * dstep / 60),
                    (newguide2, newskyguide2, -step * rstep / 60, step * dstep / 60),
                    (newguide3, newskyguide3, step * rstep / 60, -step * dstep / 60),
                    (newguide4, newskyguide4, -step * rstep / 60, -step * dstep / 60),
                ]:

                    delra = scguidestar.ra.deg - guides[0].ra.deg
                    deldec = scguidestar.dec.deg - guides[0].dec.deg

                    # 2'40" from center of guiding diamond to corner
                    # cutting back to stay away from edge
                    iGuide = np.where(
                        (scguidestar.ra.deg > guides[0].ra.deg - 2.4 / 60.0)
                        & (scguidestar.ra.deg < guides[0].ra.deg + 2.4 / 60.0)
                        & (scguidestar.dec.deg > guides[0].dec.deg - 2.4 / 60.0)
                        & (scguidestar.dec.deg < guides[0].dec.deg + 2.4 / 60.0)
                        & (deldec < delra + 2.4 / 60.0)
                        & (deldec < -delra + 2.4 / 60.0)
                        & (deldec > delra - 2.4 / 60)
                        & (deldec > -delra - 2.4 / 60)
                    )[0]

                    delskyra = scskyguidestar.ra.deg - guides[1].ra.deg
                    delskydec = scskyguidestar.dec.deg - guides[1].dec.deg

                    iSkyGuide = np.where(
                        (scskyguidestar.ra.deg > guides[1].ra.deg - 2.4 / 60.0)
                        & (scskyguidestar.ra.deg < guides[1].ra.deg + 2.4 / 60.0)
                        & (scskyguidestar.dec.deg > guides[1].dec.deg - 2.4 / 60.0)
                        & (scskyguidestar.dec.deg < guides[1].dec.deg + 2.4 / 60.0)
                        & (delskydec < delskyra + 2.4 / 60.0)
                        & (delskydec < -delskyra + 2.4 / 60.0)
                        & (delskydec > delskyra - 2.4 / 60)
                        & (delskydec > -delskyra - 2.4 / 60)
                    )[0]

                    if len(iGuide) >= 1 and len(iSkyGuide) >= 1:
                        final_coord = SkyCoord(
                            self.target.ra + guides[2],
                            self.target.dec + guides[3],
                            unit=u.deg,
                        )
                        # apply some kind of ranking thing
                        done = True
                        break

        if not done:
            raise RuntimeError("no acceptable field center coords")

        self.base_coords = utils.skycoord_to_hms_dgs(final_coord)
        self.guide_coords = utils.skycoord_to_hms_dgs(scguidestar[iGuide[0]])
        if len(iGuide) > 1:
            self.guide_2_coords = utils.skycoord_to_hms_dgs(scguidestar[iGuide[1]])
        sky_dec = final_coord.dec + 795 / 3600.0 * u.deg
        sky_ra = (
            final_coord.ra - 795 / 3600.0 / np.cos(sky_dec.deg * np.pi / 180) * u.deg
        )
        self.sky_coords = utils.skycoord_to_hms_dgs(
            sc=None, sc_ra=sky_ra, sc_dec=sky_dec
        )
        self.sky_guide_0_coords = utils.skycoord_to_hms_dgs(
            scskyguidestar[iSkyGuide[0]]
        )
        if len(iSkyGuide) > 1:
            self.sky_guide_1_coords = utils.skycoord_to_hms_dgs(
                scskyguidestar[iSkyGuide[1]]
            )
        if len(iSkyGuide) > 2:
            self.sky_guide_2_coords = utils.skycoord_to_hms_dgs(
                scskyguidestar[iSkyGuide[2]]
            )
        if len(iSkyGuide) > 3:
            self.sky_guide_3_coords = utils.skycoord_to_hms_dgs(
                scskyguidestar[iSkyGuide[3]]
            )
        self.save()

    def new_msb(self):
        """
        Clone sample msb, adjust with info from get_field_center
        """
        msb = ET.parse(f"{constants.STATIC_DIR}/msb_sample2.xml").getroot()

        # Set remaining, title, priority
        msb.attrib["remaining"] = str(self.remaining)
        msb.find(
            "title"
        ).text = f"AT{self.target.TNS_name} - {self.filters} band - WFCAM_FLIP_SLOW"
        msb.find("priority").text = str(self.priority)

        # Set site quality
        site_quality = msb.find("SpSiteQualityObsComp")
        c0 = 0
        if self.cloud_status == "a":
            c1 = 100
        elif self.cloud_status == "p":
            c1 = 0
        elif self.cloud_status == "tc":
            c1 = 20
        site_quality.find("cloud").find("min").text = str(c0)
        site_quality.find("cloud").find("max").text = str(c1)
        m0 = 0
        if self.moon_status == "dc":
            m1 = 100
        elif self.moon_status == "d":
            m1 = 0
        elif self.moon_status == "g":
            m1 = 25
        site_quality.find("moon").find("min").text = str(m0)
        site_quality.find("moon").find("max").text = str(m1)
        if self.seeing_min is not None and self.seeing_max is not None:
            site_quality.find("seeing").find("min").text = str(self.seeing_min)
            site_quality.find("seeing").find("max").text = str(self.seeing_max)
        else:
            site_quality.remove(site_quality.find("seeing"))
        if self.j_sky_min is not None and self.j_sky_max is not None:
            site_quality.find("skyBrightness").find("min").text = str(self.j_sky_min)
            site_quality.find("skyBrightness").find("max").text = str(self.j_sky_max)
        else:
            site_quality.remove(site_quality.find("skyBrightness"))
        if self.cso_tau_min is not None and self.cso_tau_max is not None:
            site_quality.find("csoTau").find("min").text = str(self.cso_tau_min)
            site_quality.find("csoTau").find("max").text = str(self.cso_tau_min)
        else:
            site_quality.remove(site_quality.find("csoTau"))

        # Set scheduling
        scheduling = msb.find("SpSchedConstObsComp")
        if self.earliest_date is not None:
            scheduling.find("earliest").text = (
                str(self.earliest_date).split("+")[0].replace(" ", "T")
            )
        else:
            scheduling.find(
                "earliest"
            ).text = f"20{self.msblist.semester[:2]}-01-01T00:00:00"
        if self.latest_date is not None:
            scheduling.find("latest").text = (
                str(self.latest_date).split("+")[0].replace(" ", "T")
            )
        else:
            scheduling.find(
                "latest"
            ).text = f"20{int(self.msblist.semester[:2])+1}-12-31T00:00:00"
        scheduling.find("maxEl").text = str(self.elevation_max)
        scheduling.find("minEl").text = str(self.elevation_min)
        scheduling.find("period").text = str(self.reschedule_every)
        if self.source_hour_angle == "r":
            scheduling.find("meridianApproach").text = "rising"
        elif self.source_hour_angle == "s":
            scheduling.find("meridianApproach").text = "setting"
        elif self.source_hour_angle == "dc":
            scheduling.remove(scheduling.find("meridianApproach"))

        # Adding change filter and flush, Observation
        for filt, exp_time, coadd in zip(
            self.filters, self.exp_times.split(), self.coadds.split()
        ):
            if exp_time == "99":
                continue
            change_and_flush = copy.deepcopy(msb.findall("SpObs")[0])
            change_and_flush.find("title").text = f"Change Filter and Flush - {filt}"
            change_and_flush.find("SpInstWFCAM").find("filter").text = filt
            change_and_flush.find("SpIterFolder").find("SpIterWFCAMCalObs").find(
                "filter"
            ).text = filt
            node = copy.deepcopy(msb.findall("SpObs")[1])
            node.find("title").text = filt
            node.find("SpInstWFCAM").find("filter").text = filt
            obs_comp = node.find("SpTelescopeObsComp")
            components = obs_comp.findall("BASE")
            coords_tup = (
                self.base_coords,
                self.guide_coords,
                self.guide_2_coords,
                self.sky_coords,
                self.sky_guide_0_coords,
                self.sky_guide_1_coords,
                self.sky_guide_2_coords,
                self.sky_guide_3_coords,
            )
            for component, coords, title, name in zip(
                components,
                coords_tup,
                (f"AT{self.target.TNS_name}", " ", " ", " ", " ", " ", " ", " "),
                (
                    "Base",
                    "GUIDE",
                    "GUIDE2",
                    "SKY0",
                    "SKYGUIDE0",
                    "SKYGUIDE1",
                    "SKYGUIDE2",
                    "SKYGUIDE3",
                ),
            ):
                if coords != "":
                    utils.edit_component_coords(
                        component, coords.split()[0], coords.split()[1], title, name
                    )
                else:
                    obs_comp.remove(component)
            if float(exp_time) >= 10:
                node.find("SpInstWFCAM").find("readMode").text = "NDR_1.0"
            node.find("SpInstWFCAM").find("coadds").text = f"{int(coadd)}"
            node.find("SpInstWFCAM").find("exposureTime").text = f"{float(exp_time)}"
            msb.append(change_and_flush)
            msb.append(node)
        # adding to the template, so need to take away originals
        for i in range(2):
            msb.remove(msb.findall("SpObs")[0])
        return msb

    def get_exp_times(self):
        self.exp_times = ""
        self.coadds = ""
        if self.reference:
            for filt in self.filters:
                coadds = 1
                max_time = max(
                    self.target.observations.filter(bandpass=filt)
                    .exclude(status="pending")
                    .values_list("exp_time", flat=True)
                )
                if max_time == 30:
                    max_time = 40
                self.exp_times += f"{max_time} "
                self.coadds += f"{coadds} "
            self.save()
            return
        self.target.update_current_values()
        for filt in self.filters:
            vals = self.target.current_values.get(filt)
            if not vals:
                for closest_filt in ("o", "ztfr", "c", "ztfg", "asg"):
                    vals = self.target.current_values.get(closest_filt)
                    if vals and vals.get("m"):
                        break
            if vals.get("m"):
                mag_ranks = sorted(constants.EXP_TIMES[filt] + [vals["m"]])
                exp_time = [1, 2, 5, 10, 15, 20, 99][
                    np.where(mag_ranks == vals["m"])[0][0]
                ]
                self.exp_times += f"{exp_time} "
                self.coadds += "1 "
            else:
                self.filters = self.filters.replace(filt, "")
        self.save()
