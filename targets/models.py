import glob
import os
import re
import warnings
from datetime import date, datetime, timedelta, timezone
from itertools import chain, combinations
import json

import astroplan
import astropy.units as u
import constants
import dotenv
import gspread
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
import requests
import utils
from alerce.core import Alerce
from astropy.coordinates import SkyCoord
from astropy.time import Time
from bokeh.io import curdoc
from bokeh.layouts import column, gridplot, layout, row
from bokeh.models import (
    Band,
    BooleanFilter,
    CDSView,
    ColumnDataSource,
    CustomJS,
    CustomJSFilter,
    GroupFilter,
    HoverTool,
    Legend,
    MultiLine,
    OpenURL,
    Span,
    TabPanel,
    Tabs,
    TapTool,
    Whisker,
)
from bokeh.models.widgets import CheckboxButtonGroup, Div
from bokeh.plotting import figure
from bokeh.themes import Theme
from data.models import SnifsSpectrum
from django.contrib.auth.models import User
from django.core.exceptions import ObjectDoesNotExist
from django.db import models
from django.db.models import F, Q
from django.urls import reverse
from dustmaps.sfd import SFDQuery
from fitting.models import (
    FitResults,
    salt_fit,
    snpy_fit,
    bayesn_fit,
    spline_fit,
)
from galaxies.models import Galaxy
from hsf.settings import BASE_DIR
from lightcurves.models import AsassnDetection, AtlasDetection, ZtfDetection
from oauth2client.service_account import ServiceAccountCredentials
from tqdm import tqdm
from utils import logger, print_verb

dotenv_file = os.path.join(BASE_DIR, ".env")
if os.path.isfile(dotenv_file):
    dotenv.load_dotenv(dotenv_file)

warnings.simplefilter(action="ignore", category=FutureWarning)


class TransientType(models.Model):
    """TransientType.

    TNS types
    """

    name = models.CharField(max_length=20, default="?", primary_key=True)
    slug = models.SlugField()
    parent = models.ForeignKey(
        "self", blank=True, null=True, related_name="children", on_delete=models.CASCADE
    )

    class Meta:
        # enforcing that there can not be two categories under a parent with same slug
        # __str__ method elaborated later in post.  use __unicode__ in place of
        # __str__ if you are using python 2
        unique_together = (
            "slug",
            "parent",
        )
        verbose_name_plural = "transient type"

    def __str__(self):
        full_path = [self.name]
        k = self.parent
        while k is not None:
            full_path.append(k.name)
            k = k.parent
        return " -> ".join(full_path[::-1])

    def subtype_of(self, name, parent=None):
        """
        Determine whether object has "name" in its parent tree
        """
        sn_type = TransientType.objects.get(name=name)
        if not parent:
            parent = self.parent
        if self == sn_type or parent == sn_type:
            return True
        elif parent is None or parent.parent is None:
            return False
        else:
            return self.subtype_of(name, parent.parent)

    def get_children(self):
        ids = [self.name]
        for t in TransientType.objects.all():
            if t.subtype_of(self.name):
                ids.append(t.name)
        return ids

    def initialize():
        """
        Generates models for each possible classification on TNS. Only need to run this once on a fresh db.
        """
        for parent in ["?", "O", "SN", "Computed", "Gap", "SLSN", "WR"]:
            if not TransientType.objects.filter(name=parent).exists():
                TransientType.objects.create(name=parent, slug=parent.replace(" ", "-"))
        for obj_type in ["Ia", "CC", "SN I", "SN I-faint", "SN I-rapid"]:
            if not TransientType.objects.filter(name=obj_type).exists():
                TransientType.objects.create(
                    name=obj_type,
                    parent=TransientType.objects.get(name="SN"),
                    slug=obj_type.replace(" ", "-"),
                )
        for obj_type in [
            "Afterglow",
            "AGN",
            "CV",
            "FBOT",
            "FRB",
            "Galaxy",
            "ILRT",
            "Impostor-SN",
            "Kilonova",
            "LBV",
            "Light-Echo",
            "LRN",
            "M dwarf",
            "Nova",
            "Other",
            "QSO",
            "Std-spec",
            "TDE",
            "Varstar",
        ]:
            if not TransientType.objects.filter(name=obj_type).exists():
                TransientType.objects.create(
                    name=obj_type,
                    parent=TransientType.objects.get(name="O"),
                    slug=obj_type.replace(" ", "-"),
                )
        for obj_type in [
            "SN Ia",
            "SN Ia-91bg-like",
            "SN Ia-91T-like",
            "SN Ia-CSM",
            "SN Ia-pec",
            "SN Ia-SC",
            "SN Iax[02cx-like]",
        ]:
            if not TransientType.objects.filter(name=obj_type).exists():
                TransientType.objects.create(
                    name=obj_type,
                    parent=TransientType.objects.get(name="Ia"),
                    slug=obj_type.replace(" ", "-"),
                )
        for obj_type in ["Ib", "Ic", "Ib/c", "II"]:
            if not TransientType.objects.filter(name=obj_type).exists():
                TransientType.objects.create(
                    name=obj_type,
                    parent=TransientType.objects.get(name="CC"),
                    slug=obj_type.replace(" ", "-"),
                )
        for obj_type in ["SN Ib", "SN Ib-Ca-rich", "SN Ib-pec", "SN Ibn"]:
            if not TransientType.objects.filter(name=obj_type).exists():
                TransientType.objects.create(
                    name=obj_type,
                    parent=TransientType.objects.get(name="Ib"),
                    slug=obj_type.replace(" ", "-"),
                )
        for obj_type in ["SN Ib/c", "SN Ibn/Icn"]:
            if not TransientType.objects.filter(name=obj_type).exists():
                TransientType.objects.create(
                    name=obj_type,
                    parent=TransientType.objects.get(name="Ib/c"),
                    slug=obj_type.replace(" ", "-"),
                )
        for obj_type in ["SN Ic", "SN Ic-BL", "SN Ic-pec", "SN Icn"]:
            if not TransientType.objects.filter(name=obj_type).exists():
                TransientType.objects.create(
                    name=obj_type,
                    parent=TransientType.objects.get(name="Ic"),
                    slug=obj_type.replace(" ", "-"),
                )
        for obj_type in [
            "SN II",
            "SN II-pec",
            "SN IIb",
            "SN IIL",
            "SN IIn",
            "SN IIn-pec",
            "SN IIP",
        ]:
            if not TransientType.objects.filter(name=obj_type).exists():
                TransientType.objects.create(
                    name=obj_type,
                    parent=TransientType.objects.get(name="II"),
                    slug=obj_type.replace(" ", "-"),
                )
        for obj_type in ["Gap I", "Gap II"]:
            if not TransientType.objects.filter(name=obj_type).exists():
                TransientType.objects.create(
                    name=obj_type,
                    parent=TransientType.objects.get(name="Gap"),
                    slug=obj_type.replace(" ", "-"),
                )
        for obj_type in [
            "Computed-Ia",
            "Computed-IIb",
            "Computed-IIn",
            "Computed-IIP",
            "Computed-PISN",
        ]:
            if not TransientType.objects.filter(name=obj_type).exists():
                TransientType.objects.create(
                    name=obj_type,
                    parent=TransientType.objects.get(name="Computed"),
                    slug=obj_type.replace(" ", "-"),
                )
        for obj_type in ["SLSN-I", "SLSN-II", "SLSN-R"]:
            if not TransientType.objects.filter(name=obj_type).exists():
                TransientType.objects.create(
                    name=obj_type,
                    parent=TransientType.objects.get(name="SLSN"),
                    slug=obj_type.replace(" ", "-"),
                )
        for obj_type in ["WR-WC", "WR-WN", "WR-WO"]:
            if not TransientType.objects.filter(name=obj_type).exists():
                TransientType.objects.create(
                    name=obj_type,
                    parent=TransientType.objects.get(name="WR"),
                    slug=obj_type.replace(" ", "-"),
                )


class TargetQuerySet(models.query.QuerySet):
    """
    Chainable filters for use in querysets.
    E.g. Target.objects.get_by_type('Ia').get_lean_queryset().filter(tmax__gt=59300)
    Returns basic info for all Ia and subtypes peaking after 59300.
    """

    def get_lean_queryset(self):
        """
        Returns diagnostic info for use in the List View.
        The lightcurves were taking too long to load with a significant queryset.
        """
        return
        # return (
        #     self.values(
        #         "TNS_name",
        #         "ra",
        #         "dec",
        #         "detection_date",
        #         "discovering_group",
        #         "galaxy",
        #         "sn_type",
        #         "current_mags",
        #         "airmass",
        #         "status",
        #         "tmax",
        #     )
        #     .prefetch_related("galaxy", "sn_type", "status",)
        #     .order_by("TNS_name")
        # )

    def get_by_type(self, sn_type, include_children=True, include_peculiar=False):
        """
        Has a weird error where this needs to be right after Target.objects or OperationalError: (1241, 'Operand should contain 1 column(s)')
        For selecting all SNe of a given type, optionally including subtypes
        """
        ret_qset = self.filter(sn_type__name=sn_type)
        if include_children:
            ret_qset = self.filter(sn_type__name=sn_type) | self.filter(
                sn_type__in=TransientType.objects.get(name=sn_type).children.all()
            )
        if not include_peculiar and "Ia" in sn_type:
            ret_qset = ret_qset.exclude(
                sn_type__name__in=(
                    "SN Ia-pec",
                    "SN Ia-SC",
                    "SN Iax[02cx-like]",
                    "SN Ia-CSM",
                )
            )
        return ret_qset

    def with_galaxies(self, queue_status="c"):
        """
        Filters by whether a target's galaxy is (c)lear, there are (m)ultiple
        possible hosts, or there is (n)o apparent galaxy.
        """
        if queue_status:
            return self.filter(galaxy_status=galaxy_status)
        return self.filter(galaxy__isnull=False)

    def with_good_host_z(self):
        return self.filter(
            Q(galaxy__z_flag__in=["sp", "spu", "spx1", "spx2", "su2"])
            | Q(galaxy__snifs_entries__z_flag__in=["?", "s"])
            | Q(galaxy__focas_entries__z_flag__in=["?", "s"])
        )

    def needs_host_z(self):
        return (
            self.filter(galaxy__z_flag__in=["n", "l", "lu", "p", "pu"])
            .exclude(galaxy__snifs_entries__z_flag__in=["?", "s"])
            .exclude(galaxy__focas_entries__z_flag__in=["?", "s"])
        )

    def number_of_observations(self, num=1, logic="gte"):

        observations = self.filter(observations__data_flag=True).annotate(
            models.Count("observations")
        )
        if logic == "eq":
            return observations.filter(observations__count=num)
        elif logic == "gt":
            return observations.filter(observations__count__gt=num)
        elif logic == "gte":
            return observations.filter(observations__count__gte=num)
        elif logic == "lt":
            return observations.filter(observations__count__lt=num)
        elif logic == "lte":
            return observations.filter(observations__count__lte=num)

        # bad_pks = []
        # for obj in self:
        #     if logic == "eq" and obj.observations.count() == num:
        #         continue
        #     if logic == "gt" and obj.observations.count() > num:
        #         continue
        #     if logic == "gte" and obj.observations.count() >= num:
        #         continue
        #     if logic == "lt" and obj.observations.count() < num:
        #         continue
        #     if logic == "lte" and obj.observations.count() <= num:
        #         continue
        #     bad_pks.append(obj.pk)
        # return self.exclude(pk__in=bad_pks).distinct()

    def box_search(self, ra, dec, box_size):
        qset = self.filter(dec__gt=dec - box_size, dec__lt=dec + box_size)
        ra_box_size = box_size * np.cos(dec * np.pi / 180)
        if ra + ra_box_size < 360 and ra - ra_box_size > 0:
            return qset.filter(ra__gt=ra - ra_box_size, ra__lt=ra + ra_box_size)
        else:
            return qset.filter(
                Q(ra__lt=(ra + ra_box_size) % 360) | Q(ra__gt=(ra - ra_box_size) % 360)
            )
        return

    def ukirt_candidates(self):
        qset = (
            self.filter(
                detection_date__gt=utils.MJD(today=True) - 7,
                dec__gt=-40,
                dec__lt=60,
                queue_status="c",
            )
            .filter(
                Q(lightcurves__atlas_detections__isnull=False)
                | Q(lightcurves__ztf_detections__isnull=False)
                | Q(lightcurves__asassn_detections__isnull=False)
            )
            .distinct()
        )
        return qset.get_by_type("Ia") | qset.get_by_type("?")

    def with_reference(self):
        return self.filter(
            Q(observations__data_flag=True),
            Q(observations__mjd__gt=F("detection_date") + 200)
            | Q(observations__mjd__lt=F("detection_date") - constants.PRE_SN),
        )

    def needs_reference(self):
        return (
            self.filter(observations__data_flag=True)
            .exclude(observations__mjd__gt=F("detection_date") + 200)
            .exclude(observations__mjd__lt=F("detection_date") - constants.PRE_SN)
        )

    def ra_range(self, date="today", r1=0, r2=0):
        r1, r2 = utils.get_observable_ra(date)
        if r1 > r2:
            return self.filter(ra__lt=r2) | self.filter(ra__gt=r1)
        return self.filter(ra__range=(r1, r2))

    def good(self):
        return (
            self.get_by_type("Ia")
            .with_good_host_z()
            .number_of_observations(logic="gte")
        )

    def faint(self, faint_m=19.0, include_unknown=False):
        faint_list = []
        for obj in self:
            if obj.galaxy.mags != {}:
                m_list = [val for val in obj.galaxy.mags.values() if val != None]
                if m_list == []:
                    if include_unknown:
                        faint_list.append(obj.TNS_name)
                    continue
                if min(m_list) > faint_m:
                    faint_list.append(obj.TNS_name)
            elif include_unknown:
                faint_list.append(obj.TNS_name)
        return Target.objects.filter(TNS_name__in=faint_list)

    def get_qset_as_bytes(self, cols=["ra", "dec", "z", "galaxy_ra", "galaxy_dec"]):
        qset_str = "#TNS_name"
        if "ra" in cols:
            qset_str += " ra"
        if "dec" in cols:
            qset_str += " dec"
        # if "DM" in cols:
        #     for model_name in ("snpy_max_model", "snpy_ebv_model2", "salt3"):
        #         qset_str += f" DM_{model_name}"
        # if "e_DM" in cols:
        #     for model_name in ("snpy_max_model", "snpy_ebv_model2", "salt3"):
        #         qset_str += f" e_DM_{model_name}"
        if "z" in cols:
            qset_str += " z"
        if "galaxy_ra" in cols:
            qset_str += " galaxy_ra"
        if "galaxy_dec" in cols:
            qset_str += " galaxy_dec"
        qset_str += "\n"
        for obj in self.all():
            try:
                obj_str = f"{obj.TNS_name}"
                if "ra" in cols:
                    obj_str += f" {obj.ra}"
                if "dec" in cols:
                    obj_str += f" {obj.dec}"
                # if "DM" in cols:
                #     for model_name in ("snpy_max_model", "snpy_ebv_model2", "salt3"):
                #         try:
                #             obj_str += f" {obj.fit_results.get(model_name=model_name).get('DM')}"
                #         except:
                #             obj_str += " N/A"
                # if "e_DM" in cols:
                #     for model_name in ("snpy_max_model", "snpy_ebv_model2", "salt3"):
                #         try:
                #             obj_str += f" {obj.fit_results.get(model_name=model_name).get('e_DM')}"
                #         except:
                #             obj_str += " N/A"
                if "z" in cols:
                    obj_str += f" {obj.galaxy.z}"
                if "z_err" in cols:
                    obj_str += f" {obj.galaxy.z_err}"
                if "z_flag" in cols:
                    obj_str += f" {obj.galaxy.z_flag}"
                if "galaxy_ra" in cols:
                    obj_str += f" {obj.galaxy.ra}"
                if "galaxy_dec" in cols:
                    obj_str += f" {obj.galaxy.dec}"
                if "pgc_id" in cols:
                    obj_str += f" {obj.galaxy.pgc_no}"
                if "disc_group" in cols:
                    obj_str += f" {obj.discovering_group}"
                if "disc_date" in cols:
                    obj_str += f" {obj.discovery_date}"
                if "class_group" in cols:
                    obj_str += f" {obj.classifying_group}"
                if "class_date" in cols:
                    obj_str += f" {obj.classification_date}"
                obj_str += "\n"
            except:
                continue
            qset_str += obj_str
        return str.encode(qset_str)

    def get_fitresults_qset(
        self,
        model_name,
        bandpasses=[],
        variants=[],
        calibration=None,
        redlaw="F19",
        acceptable_status=["g", "?", "n"],
        max_bandpass="J",
        max_color_1="V",
        max_color_2="r",
        cuts={},
        optional_args={},
        excluded_args={},
        verbose=False,
    ):
        fitresults_pks = np.zeros(self.count())
        junk = {}
        for junk_type in (
            "no_z",
            "missing_bps",
            "no_fit",
            "no_successful_fit",
            "clipped",
        ):
            junk[junk_type] = np.zeros(self.count(), dtype="<U6")
        junk["cut"] = {"any": np.zeros(self.count(), dtype="<U6")}
        for i, obj in tqdm(enumerate(self), total=self.count()):
            # Skip objects with no redshift
            if (
                not obj.galaxy
                or not obj.galaxy.z
                or obj.galaxy.z_flag not in ["sp", "spu"]
            ):
                print_verb(verbose, f"No galaxy redshift for {obj.TNS_name}")
                junk["no_z"][i] = obj.TNS_name
                continue

            # Skip objects without detections in the required bandpasses
            for bp in bandpasses:
                lc_set = obj.lightcurves.filter(bandpass=bp).exclude(source="model")
                all_vs = variants + ["none"]
                if "variants" in optional_args:
                    all_vs += optional_args["variants"]
                if len(all_vs):
                    lc_set = lc_set.filter(variant__in=all_vs)
                if bp in "ZYJHK":
                    lc_set = lc_set.filter(
                        ukirt_detections__isnull=False,
                        ukirt_detections__status__in=["det", "?"],
                    ).distinct()
                else:
                    lc_set = lc_set.filter(
                        **{
                            f"{lc_set.first().source.replace('-', '').lower()}_detections__isnull": False
                        }
                    ).distinct()
                if not lc_set.exists():
                    print_verb(verbose, f"Missing bps for {obj.TNS_name}")
                    junk["missing_bps"][i] = obj.TNS_name
                    break
            if obj.TNS_name in junk["missing_bps"]:
                continue

            # Filtering by required args
            results_qset = obj.fit_results.filter(
                model_name=model_name,
                calibration=calibration,
                redlaw=redlaw,
            )
            for arg_name, required_arg in zip(
                ("bandpasses", "variants"), (bandpasses, variants)
            ):
                if not len(required_arg):
                    continue
                for arg in required_arg:
                    results_qset = results_qset.filter(
                        **{f"{arg_name}_str__contains": arg}
                    )
            # Filtering by excluded arguments
            for arg_name, excluded_arg in excluded_args.items():
                for arg in excluded_arg:
                    results_qset = results_qset.exclude(
                        **{f"{arg_name}_str__contains": arg}
                    )
            if not results_qset.exists():
                print_verb(verbose, f"no fit for {obj.TNS_name} with required args")
                junk["no_fit"][i] = obj.TNS_name
                continue
            results_qset = results_qset.filter(
                success=True, status__in=acceptable_status
            )
            if not results_qset.exists():
                print_verb(
                    verbose, f"no successful fit for {obj.TNS_name} with required args"
                )
                junk["no_successful_fit"][i] = obj.TNS_name
                continue
            # Filtering by optional arguments, order matters
            for arg_name, optional_arg in optional_args.items():
                for arg in optional_arg:
                    tmp_results_qset = results_qset.filter(
                        **{f"{arg_name}_str__contains": arg}
                    )
                    if tmp_results_qset.exists():
                        results_qset = tmp_results_qset
            # Filtering down to just required args + optional args
            for bp in constants.FILT_SNPY_NAMES:
                if (
                    bp not in bandpasses
                    or (
                        "bandpasses" in optional_args
                        and bp not in optional_args.get("bandpasses")
                    )
                ) and results_qset.exclude(bandpasses_str__contains=bp).exists():
                    results_qset = results_qset.exclude(bandpasses_str__contains=bp)
            # Filtering by cuts
            # relatively slow, so putting at the end
            results_qset = results_qset.check_cuts(cuts)
            if isinstance(results_qset, list):
                print_verb(verbose, f"cutting {obj.TNS_name}")
                cut_reason = results_qset[0]
                junk["cut"]["any"][i] = obj.TNS_name
                if cut_reason not in junk["cut"]:
                    junk["cut"][cut_reason] = np.zeros(self.count(), dtype="<U20")
                junk["cut"][cut_reason][i] = obj.TNS_name
                continue

            best_res = results_qset.first()
            if model_name.startswith("snpy"):
                try:
                    best_res.get_snpy().model.rchisquare
                except AttributeError:
                    continue
                except FileNotFoundError:
                    print_verb(verbose, f"snpy file not found for {obj.TNS_name}")
                    junk["no_successful_fit"][i] = obj.TNS_name
                    continue
                if model_name == "snpy_max_model" and not (
                    f"{max_bandpass}max" in best_res.params
                    and f"{max_color_1}max" in best_res.params
                    and f"{max_color_2}max" in best_res.params
                ):
                    junk["missing_bps"][i] = obj.TNS_name
                    continue
            fitresults_pks[i] = best_res.pk

        junk_qsets = {"cut": {}}
        for key in junk:
            if key == "cut":
                continue
            junk_qsets[key] = self.filter(TNS_name__in=junk[key])
        for key in junk["cut"]:
            junk_qsets["cut"][key] = self.filter(TNS_name__in=junk["cut"][key])

        return (
            FitResults.objects.filter(pk__in=fitresults_pks).order_by(
                "target__TNS_name"
            ),
            junk_qsets,
        )

    def compare_common_targets(
        self,
        model_name="snpy_ebv_model2",
        bandpasses=[],
        variants=[],
        calibration=None,
        redlaw="F19",
        h="average",
        sigma=5,
        dm_bandpass=None,
        dm_ebvhost="snpy_ebv_model2",
        dm_e_ebvhost="snpy_ebv_model2",
        q0=-0.51,
        cuts={},
        optional_args={},
        excluded_args={},
        varied_args={},
        max_iters=3,
        verbose=False,
    ):
        args = {
            "model_name": model_name,
            "bandpasses": bandpasses,
            "variants": variants,
            "calibration": calibration,
            "redlaw": redlaw,
            "h": h,
            "sigma": sigma,
            "dm_bandpass": dm_bandpass,
            "dm_ebvhost": dm_ebvhost,
            "dm_e_ebvhost": dm_e_ebvhost,
            "q0": q0,
            "cuts": cuts,
            "optional_args": optional_args,
            "excluded_args": excluded_args,
        }
        stats_1, p_1, qset_1, _, _ = self.get_dispersion(**args)
        for key, val in args.items():
            if key not in varied_args:
                varied_args[key] = val
        stats_2, p_2, qset_2, _, _ = self.get_dispersion(**varied_args)
        print_verb(verbose, f"{qset_1.count()} targets for first qset, initial")
        print_verb(verbose, f"{qset_2.count()} targets for second qset, initial")
        iter_num = 0
        while list(qset_1) != list(qset_2):
            qset = qset_1.filter(pk__in=qset_2)
            print_verb(verbose, f"{qset.count()} targets in common, iter {iter_num}")
            iter_num += 1
            if iter_num > max_iters:
                print_verb(verbose, "maximum iterations reached")
                return qset, stats_1, stats_2, p_1, p_2, qset.count()
            stats_1, p_1, qset_1, _, _ = qset.get_dispersion(**args)
            stats_2, p_2, qset_2, _, _ = qset.get_dispersion(**varied_args)
            print_verb(
                verbose, f"{qset_1.count()} targets for first qset, iter {iter_num}"
            )
            print_verb(
                verbose, f"{qset_2.count()} targets for second qset, iter {iter_num}"
            )
        print_verb(verbose, stats_1)
        print_verb(verbose, stats_2)
        qset = qset_1.filter(pk__in=qset_2)
        return (stats_1, stats_2), (p_1, p_2), qset, self.exclude(TNS_name__in=qset)

    def quick_hubble(
        self,
        model_name="snpy_ebv_model2",
        bandpasses=[],
        variants=[],
        calibration=None,
        h="average",
        sigma=5,
        dm_bandpass=None,
        dm_ebvhost="snpy_ebv_model2",
        dm_e_ebvhost=0.06,
        verbose=False,
    ):
        """quick_hubble.

        Parameters
        ----------
        model_name :
            model_name
        bandpasses :
            bandpasses
        variants :
            variants
        h :
            h
        sigma :
            sigma
        dm_bandpass :
            dm_bandpass
        dm_ebvhost :
            dm_ebvhost
        dm_e_ebvhost :
            dm_e_ebvhost
        verbose :
            verbose
        """
        stats, p, qset, junk = self.get_dispersion(
            model_name=model_name,
            bandpasses=bandpasses,
            variants=variants,
            calibration=calibration,
            h=h,
            sigma=sigma,
            dm_bandpass=dm_bandpass,
            dm_ebvhost=dm_ebvhost,
            dm_e_ebvhost=dm_e_ebvhost,
        )
        fig, [ax0, ax1] = plt.subplots(nrows=2, sharex=True)
        fig.subplots_adjust(hspace=0)
        ax0.errorbar(x=p["z_cmb"], y=p["DM"], yerr=p["e_DM"], ls="none", color="orange")
        xs = np.linspace(min(p["z_cmb"]), max(p["z_cmb"]), 100)
        ys = 5 * np.log10(constants.C * xs / p["h"]) + 25
        ax0.plot(xs, ys, color="black")
        ax0.set_ylabel("DM")
        ax1.set_ylabel("Residual DM")
        ax1.errorbar(
            x=p["z_cmb"],
            y=p["resid_DM"],
            yerr=p["e_DM"],
            ls="none",
            marker=".",
            color="orange",
        )
        ax1.hlines(0, min(p["z_cmb"]), max(p["z_cmb"]), color="black")
        for i in [-3, -1, 1, 3]:
            ax1.hlines(
                i * np.std(p["resid_DM"]),
                min(p["z_cmb"]),
                max(p["z_cmb"]),
                alpha=0.2,
                ls="dashed",
                color="black",
            )
        print_verb(verbose, f"N={qset.count()}")
        print_verb(verbose, stats)
        plt.xlabel("z")
        plt.show()
        return stats, p, qset, junk, fig

    def get_hubble_diagram(
        self,
        model_name="snpy_ebv_model2",
        bandpasses=[],
        variants=[],
        calibration=None,
        h="average",
        sigma=5,
        dm_bandpass=None,
        dm_ebvhost=0,
        dm_e_ebvhost=0.06,
    ):
        stats, p, qset, junk = self.get_dispersion(
            model_name=model_name,
            bandpasses=bandpasses,
            variants=variants,
            calibration=calibration,
            h=h,
            sigma=sigma,
            dm_bandpass=dm_bandpass,
            dm_ebvhost=dm_ebvhost,
            dm_e_ebvhost=dm_e_ebvhost,
        )
        p["DM_lower"] = p["DM"] - p["e_DM"]
        p["DM_upper"] = p["DM"] + p["e_DM"]
        p["name"] = np.array(qset.values_list("TNS_name", flat=True))

        tooltips = [("name", "@name"), ("(x,y)", "($x, $y)")]
        plot_config = {
            "width": 4000,
            "height": 2500,
            "tools": [
                "pan",
                "wheel_zoom",
                "hover",
                "box_zoom,",
                "reset",
                "tap",
                "help",
            ],
            "active_scroll": "wheel_zoom",
            "tooltips": tooltips,
        }
        color = "white"
        li = []
        hubplot = figure(title=None, **plot_config)
        taptool = hubplot.select(type=TapTool)

        # plotting hubble
        args = []

        # Hubble parameter slider
        # hub_z = np.linspace(0.001, 0.12, 120)
        # hub_DM = 5*np.log10(constants.C*hub_z/75)+25
        # hub_source = ColumnDataSource(data=dict(x=hub_z, y=hub_DM))
        # h = hubplot.line('x', 'y', color='red', source=hub_source)
        # callback = CustomJS(args=dict(source=hub_source), code="""
        #    var data = source.data;
        #    var f = cb_obj.value
        #    var x = data['x']
        #    var y = data['y']
        #    for (var i = 0; i < x.length; i++) {
        #        y[i] = 5*Math.log10(299792.458*x[i]/f)+25
        #    }
        #    source.change.emit();
        #    """)
        # slider = Slider(start=60, end=90, value=75, step=0.1, title="Hubble Constant")
        # slider.js_on_change('value', callback)
        # li.append(('Hubble Constant', [h]))

        # plotting data
        # 1 std dev band
        df = pd.DataFrame(data=dict(x=p["z"], y=p["DM"])).sort_values(by="x")
        sem = lambda x: x.std() / np.sqrt(x.size)
        df2 = df.y.rolling(window=10).agg(
            {"y_mean": np.mean, "y_std": np.std, "y_sem": sem}
        )
        df2 = df2.fillna(method="bfill")
        df = pd.concat([df, df2], axis=1)
        df["lower"] = df.y_mean - df.y_std
        df["upper"] = df.y_mean + df.y_std
        band_source = ColumnDataSource(df.reset_index())

        # data
        data = ColumnDataSource(data=pd.DataFrame(p))
        h = hubplot.circle(
            "z", "DM", source=data, alpha=0.8, muted_alpha=0.2, color=color
        )
        hw = Whisker(
            base="z", upper="DM_upper", lower="DM_lower", line_color=color, source=data
        )
        hw.upper_head.line_color = color
        hw.lower_head.line_color = color
        band = Band(
            base="x",
            lower="lower",
            upper="upper",
            source=band_source,
            level="underlay",
            fill_alpha=0.3,
            line_width=1,
            line_color="blue",
        )
        hubplot.add_layout(hw)
        hubplot.add_layout(band)
        # Tie whisker visiblity to data with custom JS
        h.js_on_change(
            "visible", CustomJS(args=dict(ls=hw), code="ls.visible = cb_obj.visible;")
        )
        h.js_on_change(
            "visible", CustomJS(args=dict(ls=band), code="ls.visible = cb_obj.visible;")
        )
        li.append((f"PLACEHOLDER", [h]))
        taptool.callback = OpenURL(
            url="https://hawaiisupernovaflows.com/targets/@name/"
        )
        # args += [(f'h{i}',h)]
        # code += f"h{i}.visible = (active.includes({i%lenfit}) && active.includes({int(i/lenfit)+lenfit}) && active.includes({int(i%lenfit/lenfit)+lensub+lenfit}));"

        legend = Legend(items=li, location="center")
        legend.click_policy = "hide"
        legend.background_fill_color = "black"
        legend.label_text_color = color
        # hubplot.add_layout(legend, 'right')

        checkbox = CheckboxButtonGroup(labels=list(constants.MODEL_NAMES), active=[])
        # checkbox.js_on_click(CustomJS(args={key:value for key,value in args}, code=code))
        # arrange plots in a column
        bp = gridplot([[checkbox], [hubplot]], sizing_mode="scale_width")  # [slider],
        curdoc().theme = Theme(
            json={
                "attrs": {
                    "Figure": {
                        "background_fill_color": "black",
                        "border_fill_color": "black",
                        "outline_line_color": "black",
                    },
                    "Axis": {
                        "axis_line_color": "black",
                        "axis_label_text_color": "white",
                        "major_label_text_color": "white",
                    },
                    "Grid": {
                        "minor_grid_line_color": "#222222",
                        "grid_line_alpha": 0.4,
                    },
                    "Toolbar": {"autohide": True},
                }
            }
        )
        curdoc().add_root(bp)
        return bp, stats


class Target(models.Model):
    """Target.

    Transient events reported to TNS
    """

    QUEUE_CHOICES = [
        ("c", "Candidate"),
        ("q", "Queued"),
        ("j", "Junk"),
        ("d", "Done"),
    ]
    GALAXY_CHOICES = [
        ("?", "Uninspected"),
        ("c", "Clear host"),
        ("m", "Multiple possible hosts"),
        ("n", "No apparent galaxy"),
    ]
    QUALITY_CHOICES = [
        ("u", "Uninspected"),
        ("g", "Gold"),
        ("s", "Silver"),
        ("b", "Bronze"),
        ("t", "Trash"),
    ]
    TNS_name = models.CharField(max_length=60, primary_key=True)
    tns_sn_z = models.FloatField(null=True)
    tns_host_gal = models.CharField(max_length=40, null=True)
    tns_host_gal_z = models.FloatField(null=True)
    galaxy = models.ForeignKey(
        "galaxies.Galaxy", on_delete=models.SET_NULL, null=True, related_name="targets"
    )
    tns_creation_date = models.DateTimeField(null=True)
    tns_last_modified = models.DateTimeField(null=True)
    queue_status = models.CharField(max_length=1, choices=QUEUE_CHOICES, default="c")
    status_reason = models.CharField(max_length=200, default="")
    galaxy_status = models.CharField(max_length=1, choices=GALAXY_CHOICES, default="?")
    fit_status = models.CharField(max_length=1, choices=QUALITY_CHOICES, default="u")
    sub_status = models.CharField(max_length=1, choices=QUALITY_CHOICES, default="u")
    sn_type = models.ForeignKey(TransientType, on_delete=models.CASCADE)
    ra = models.FloatField(default=0)
    dec = models.FloatField(default=0)
    tns_ra = models.FloatField(default=0)
    tns_dec = models.FloatField(default=0)
    detection_date = models.FloatField(default=0)
    discovering_group = models.CharField(max_length=25, blank=True)
    other_detections = models.JSONField(default=dict)
    classification_date = models.FloatField(default=0)
    classifying_group = models.CharField(max_length=25, blank=True)
    other_classifications = models.JSONField(default=dict)
    ATLAS_name = models.CharField(max_length=20, null=True)
    ZTF_name = models.CharField(max_length=20, null=True)
    queue_date = models.FloatField(default=0)
    airmass = models.JSONField(default=dict)
    mwebv = models.FloatField(null=True)
    current_values = models.JSONField(default=dict)
    peak_values = models.JSONField(default=dict)
    tmax = models.JSONField(default=dict)
    manual_z = models.FloatField(null=True)
    manual_z_err = models.FloatField(null=True)
    comments = models.TextField(default="")

    objects = TargetQuerySet.as_manager()

    class Meta:
        ordering = ["TNS_name"]
        indexes = [
            models.Index(fields=["TNS_name"]),
            models.Index(fields=["ra"]),
            models.Index(fields=["dec"]),
            models.Index(fields=["sn_type"]),
            models.Index(fields=["queue_status"]),
            models.Index(fields=["sub_status"]),
            models.Index(fields=["galaxy_status"]),
        ]

    def __str__(self):
        return self.TNS_name

    def save(self, *args, **kwargs):
        for dir_name in [
            "atlas_stamps",
            "atlas_files",
            "fits",
            "ztf_stamps",
            "thumbnails",
        ]:
            os.makedirs(
                f"{constants.MEDIA_DIR}/{self.TNS_name}/{dir_name}", exist_ok=True
            )
        if self.ATLAS_name is None and "ATLAS" in self.other_detections.keys():
            self.ATLAS_name = self.other_detections["ATLAS"][1]
        if self.ZTF_name is None and "ZTF" in self.other_detections.keys():
            self.ZTF_name = self.other_detections["ZTF"][1]
        if self.ra == 0 and self.tns_ra != 0:
            self.ra = self.tns_ra
        if self.dec == 0 and self.tns_dec != 0:
            self.dec = self.tns_dec
        if self.mwebv is None:
            self.mwebv = SFDQuery()(SkyCoord(self.ra, self.dec, unit="deg"))
        super(Target, self).save(*args, **kwargs)

    def get_absolute_url(self):
        return reverse("targets:detail", kwargs={"TNS_name": self.TNS_name})

    # UTILS
    def quick_get(TNS_name):
        try:
            return Target.objects.get(TNS_name=TNS_name)
        except Target.DoesNotExist:
            raise Target.DoesNotExist(f"{TNS_name} does not exist")

    def quick_create(TNS_name, skip=["asassn"]):
        print(f"Adding {TNS_name} to the database")
        obj, new = Target.objects.get_or_create(
            TNS_name=TNS_name,
            defaults={
                "queue_status": "c",
                "sn_type": TransientType.objects.get(name="?"),
            },
        )
        if not new:
            print("Already exists")
            return obj
        print("Updating everything")
        obj.update_all(skip)
        if (
            obj.detection_date < utils.MJD(today=True) - 30
            or obj.dec > 60
            or obj.dec < -40
        ):
            obj.queue_status = "j"
        obj.save()
        return obj

    def quick_get_or_create(TNS_name):
        try:
            return Target.quick_get(TNS_name)
        except ObjectDoesNotExist:
            return Target.quick_create(TNS_name)

    def HMS_coords(self):
        HMS = utils.deg2HMS(ra=self.ra, dec=self.dec, rounding=2)
        if HMS == "":
            ra = "00:00:00.0"
            dec = "00:00:00.00"
        else:
            ra = "%02d:%02d:%05.2f" % tuple([float(x) for x in HMS[0].split()])
            dec = "%02d:%02d:%05.2f" % tuple([float(x) for x in HMS[1].split()])
        return (ra, dec)

    def add_comment(self, comment_str):
        self.comments = self.comments + f"{utils.MJD(today=True)}: {comment_str}\n\n"
        self.save()

    def reference_set(self):
        return self.observations.exclude(
            mjd__range=(
                self.detection_date - constants.PRE_SN,
                self.detection_date + 200,
            )
        )

    def apply_space_motion(self, pm_ra_cosdec, pm_dec, obstime, new_obstime):
        sc = SkyCoord(
            ra=self.ra,
            dec=self.dec,
            unit="deg",
            pm_ra_cosdec=pm_ra_cosdec * u.mas / u.yr,
            pm_dec=pm_dec * u.mas / u.yr,
            obstime=obstime,
        )
        return sc.apply_space_motion(new_obstime=new_obstime)

    @property
    def bandpasses(self):
        return (
            self.lightcurves.exclude(source="model")
            .values_list("bandpass", flat=True)
            .distinct()
        )

    @property
    def detected_bandpasses(self):
        return self.get_detected_bandpasses()

    def get_detected_bandpasses(
        self, binned=False, mag_cut=None, dmag_cut=None, excluded_variants=()
    ):
        good = []
        for lc in self.lightcurves.exclude(source="model").exclude(
            variant__in=excluded_variants
        ):
            if binned:
                blc = lc.bin()
                mags = blc["mag"]
                dmags = blc["dmag"]
            if not binned:
                mags = lc.mag
                dmags = lc.dmag
            if len(mags) < 0:
                continue
            mag_mask = np.ones(len(mags), dtype=bool)
            dmag_mask = np.ones(len(mags), dtype=bool)
            if mag_cut is not None:
                mag_mask = mags < mag_cut
            if dmag_cut is not None:
                dmag_mask = dmags < dmag_cut
            if sum(mag_mask & dmag_mask):
                good.append(lc.bandpass)
        return self.bandpasses.filter(bandpass__in=good)

    # UPDATES
    @logger.catch
    def update_all(self, skip=["asassn"]):
        print(f"running updates for {self.TNS_name}")
        if "TNS" not in skip:
            self.update_TNS()
        if not [
            value
            for value in ["atlas", "optical", "lightcurve", "lc"]
            if value.lower() in skip
        ]:
            self.update_atlas()
        if not [
            value
            for value in ["ztf", "optical", "lightcurve", "lc"]
            if value.lower() in skip
        ]:
            self.update_ztf()
        if not [
            value
            for value in ["asassn", "optical", "lightcurve", "lc"]
            if value.lower() in skip
        ]:
            # time bottleneck, and only useful for very bright SNe
            self.update_asassn()
        if not [
            value
            for value in ["ukirt", "ir", "lightcurve", "lc"]
            if value.lower() in skip
        ]:
            self.photometry()
        if "airmass" not in skip:
            self.get_airmass()
        if "galaxy" not in skip:
            if not self.galaxy:
                if not Galaxy.objects.box_search(self.ra, self.dec, 1.5 / 60).exists():
                    Galaxy.query.all_catalogs(SkyCoord(self.ra, self.dec, unit="deg"))
                g = self.get_closest_galaxy()
                if not self.galaxy:
                    g = Galaxy.objects.create(ra=self.ra, dec=self.dec)
                    self.galaxy = g
            for g in Galaxy.objects.box_search(self.ra, self.dec, 1.5 / 60):
                for n in g.ned_entries.all():
                    if not len(n.aliases):
                        n.get_aliases()
                    if n.n_ddf and not n.diameters.exists():
                        n.query_table("diameters")
                g.merge_near()
        if "fit" not in skip:
            self.fit(bandpasses=self.detected_bandpasses)
        if "current_values" not in skip:
            self.update_current_values()
        if "peak_values" not in skip:
            self.update_peak_values()
        self.save()

    def update_TNS(self):
        """
        queries TNS to get RA, Dec, host_z if precise to 5 digits, discovery_MJD, SN_type
        """
        day = datetime.utcnow()
        # assume date=2000+TNS_name[:2], if gt current year, assumption faulty
        # this dumb hack will start failing in 2076 for 76R
        old = int(self.TNS_name[:2]) + 2000 > day.year
        get_obj = [
            ("objname", f"20{self.TNS_name}"),
            ("objid", ""),
            ("photometry", "1"),
            ("spectra", "0"),
        ]
        if old:
            get_obj = [
                ("objname", f"19{self.TNS_name}"),
                ("objid", ""),
                ("photometry", "1"),
                ("spectra", "0"),
            ]
        data = utils.TNS_query("get", get_obj)
        self.tns_ra = data["radeg"]
        self.tns_dec = data["decdeg"]
        if data["decdeg"] < -40 or data["decdeg"] > 60:
            self.queue_status = "j"
        self.detection_date = utils.MJD(ymdhms=data["discoverydate"])
        self.discovering_group = data["reporting_group"]["group_name"]
        if self.discovering_group is None:
            self.discovering_group = ""
        if data["object_type"]["name"]:
            sn_type = data["object_type"]["name"]
        else:
            sn_type = "?"
        if self.queue_status == "c" and not (
            sn_type == "?" or sn_type.startswith("SN Ia")
        ):
            self.queue_status = "j"
            self.status_reason = f"Classified as {sn_type} on {str(day)}"
        self.sn_type = TransientType.objects.get(name=sn_type)
        self.tns_sn_z = data.get("redshift")
        self.tns_host_gal = data.get("hostname")
        self.tns_host_z = data.get("host_redshift")
        internal_names = data.get("internal_names")
        if internal_names is not None:
            internal_names = internal_names.split(", ")
        p = data["photometry"]
        # for figuring out which photometric report is
        # connected to which survey and which internal name
        phot_parse = [
            ["telescope", "ATLAS", "ATLAS"],
            ["instrument", "ZTF", "ZTF"],
            ["telescope", "PS", "Pan-STARRS1"],
            ["telescope", "Gaia", "Gaia"],
        ]
        for detection in p:
            if detection["flux"] == "":
                continue
            obs_date = float(str(detection["jd"])[2:])
            for pp in phot_parse:
                if detection[pp[0]]["name"].startswith(pp[1]):
                    int_name = [x for x in internal_names if x.startswith(pp[1])]
                    if not len(int_name):
                        continue
                    if pp[2] in self.other_detections:
                        if self.other_detections[pp[2]][0] < obs_date:
                            continue
                    self.other_detections[pp[2]] = [obs_date, int_name[0]]
        if "ATLAS" in self.other_detections:
            self.ATLAS_name = self.other_detections["ATLAS"][1]
        if "ZTF" in self.other_detections:
            self.ATLAS_name = self.other_detections["ZTF"][1]
        self.save()

    def update_TNS_classification(self):
        for d, t, g, c in zip(*utils.TNS_query("classification", self.TNS_name)):
            if t.text != self.sn_type.name:
                continue
            if self.classification_date == 0:
                self.classification_date = utils.MJD(ymdhms=d.text)
            if g.text != "None":
                if g.text not in self.other_classifications:
                    self.other_classifications[g.text] = [
                        utils.MJD(ymdhms=d.text),
                        t.text,
                    ]
                if self.classifying_group == "":
                    self.classifying_group = g.text
            elif c.text != "None":
                if c.text not in self.other_classifications:
                    self.other_classifications[c.text[:25]] = [
                        utils.MJD(ymdhms=d.text),
                        t.text,
                    ]
                if self.classifying_group == "":
                    self.classifying_group = c.text[:25]
        self.save()

    def add_to_google_sheet(self):
        """
        deprecated
        """
        print(f"{self.TNS_name}: Adding to Google Sheet")
        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive",
        ]
        creds = ServiceAccountCredentials.from_json_keyfile_name(
            f"{constants.HSF_DIR}/hawaii-supernova-flows-1332a295a90b.json",
            scope,
        )
        client = gspread.authorize(creds)
        sheet = client.open("Target list")
        sheet_instance = sheet.get_worksheet(0)
        if self.TNS_name in sheet.col_values(0):
            return
        sheet_instance.insert_row(
            [
                self.TNS_name,
                "",
                self.ra,
                self.dec,
                self.detection_date,
                self.galaxy.z,
                "",
                self.sn_type.name,
                utils.deg2HMS(ra=self.ra),
                utils.deg2HMS(dec=self.dec),
                utils.MJD(today=True),
                "",
                "",
                "",
                "",
                "",
                self.galaxy.ra,
                self.galaxy.dec,
            ],
            index=6,
            value_input_option="RAW",
        )

    def get_airmass(self):
        print(f"{self.TNS_name}: Calculating observability")
        coordinates = SkyCoord(self.ra * u.deg, self.dec * u.deg)
        target = astroplan.FixedTarget(name=self.TNS_name, coord=coordinates)
        observer = astroplan.Observer.at_site("ukirt", timezone="US/Hawaii")
        timegrid = astroplan.time_grid_from_range([Time.now(), Time.now() + 1 * u.d])
        airmass = np.array([observer.altaz(t, target).secz for t in timegrid])
        if [i for i in airmass if i > 0] == []:
            self.airmass["min_z"] = "Never up"
            self.airmass["start"] = "Never up"
            self.airmass["end"] = "Never up"
            return
        airmin = np.where(airmass == min([i for i in airmass if i > 0]))[0][0]
        airthresh = airmass[airmin] + 0.5
        for t in timegrid:
            if observer.altaz(t, target).secz < airthresh:
                start = np.where(airmass == observer.altaz(t, target).secz)[0][0]
                break
        for t in timegrid[timegrid > timegrid[start]]:
            if observer.altaz(t, target).secz > airthresh:
                end = np.where(airmass == observer.altaz(t, target).secz)[0][0]
                break
        if start == 0:
            for t in timegrid[timegrid > timegrid[end]]:
                if observer.altaz(t, target).secz < airthresh:
                    start = np.where(airmass == observer.altaz(t, target).secz)[0][0]
        start_time = pytz.utc.localize(timegrid[start].datetime).astimezone(
            pytz.timezone("US/Hawaii")
        )
        end_time = pytz.utc.localize(timegrid[end].datetime).astimezone(
            pytz.timezone("US/Hawaii")
        )
        self.airmass["min_z"] = np.round(airmass[airmin], 2)
        self.airmass["start"] = f"{start_time.hour}:{start_time.minute:02d}"
        self.airmass["end"] = f"{end_time.hour}:{end_time.minute:02d}"

    def update_peak_values(self, model_only=True):
        self.peak_values = {}
        if model_only:
            sources = ("model",)
        else:
            sources = ("model", "ATLAS", "ASAS-SN", "ZTF", "UKIRT")
        for bp in (
            self.lightcurves.filter(source__in=sources)
            .values_list("bandpass", flat=True)
            .distinct()
        ):
            self.peak_values[bp] = {
                "mjd": [],
                "mag": [],
                "dmag": [],
                "ujy": [],
                "dujy": [],
            }
            for lc in self.lightcurves.filter(bandpass=bp, source__in=sources):
                vals = lc.get_peak_values()
                if np.nan in vals.values():
                    continue
                for param in vals:
                    self.peak_values[bp][param].append(vals[param])
            tmp_dict = {}
            for param in self.peak_values[bp]:
                if not len(self.peak_values[bp][param]):
                    tmp_dict[param] = None
                else:
                    tmp_dict[param] = np.average(
                        self.peak_values[bp][param],
                        weights=1 / np.array(self.peak_values[bp]["dujy"]) + 0.001,
                    )
            for param, value in tmp_dict.items():
                if value is not None:
                    self.peak_values[bp][param] = value
                else:
                    self.peak_values[bp].pop(param)
        self.save()

    def update_current_values(self, model_only=True):
        self.current_values = {}
        if model_only:
            sources = ("model",)
        else:
            sources = ("model", "ATLAS", "ASAS-SN", "ZTF", "UKIRT")
        for bp in self.bandpasses:
            self.current_values[bp] = {}
            for lc in self.lightcurves.filter(bandpass=bp, source__in=sources):
                vals = lc.values_at_time(today=True)
                if vals is None:
                    continue
                for param in vals:
                    if param not in self.current_values[bp]:
                        self.current_values[bp][param] = []
                    if not np.isnan(vals[param]):
                        self.current_values[bp][param].append(vals[param])
        for bp in self.current_values:
            tmp_dict = {}
            for param in self.current_values[bp]:
                if not len(self.current_values[bp][param]):
                    tmp_dict[param] = np.nan
                else:
                    tmp_dict[param] = np.average(
                        self.current_values[bp][param],
                        weights=1 / np.array(self.current_values[bp][param]) + 0.0001,
                    )
            for param, value in tmp_dict.items():
                if not np.isnan(value):
                    self.current_values[bp][param] = value
                else:
                    self.current_values[bp][param]
        self.save()

    def update_optical(self, mjd0=None, mjd1=None, force=False):
        self.update_atlas(mjd0=mjd0, mjd1=mjd1, force=force)
        self.update_ztf(force=force)
        self.update_asassn(mjd0=mjd0, mjd1=mjd1, force=force)

    def update_atlas(
        self,
        mjd0=None,
        mjd1=None,
        force=False,
        get_stamps=False,
        force_stamps=False,
        stamp_size=64,
        force_kwargs={},
    ):
        """
        Update or create the o and c bandpasses between mjd0 and mjd1.
        If those already have data and force is False,
        Query dates between [mjd0, earliest_search_date) and (latest_search_data, mjd1]
        if those intervals exist.
        """
        new_data = False
        if not mjd0 and self.detection_date != 0:
            mjd0 = self.detection_date - constants.PRE_SN
        if not mjd1 and self.detection_date != 0:
            mjd1 = self.detection_date + constants.POST_SN
        tmp = {}
        for filt in ["c", "o"]:
            tmp[filt] = self.lightcurves.get_or_create(
                bandpass=filt, eff_wl=constants.EFF_WL_DICT[filt], source="ATLAS"
            )[0]
        epochs = utils.epoch_coverage(
            [mjd0, mjd1],
            [[tmp["c"].early, tmp["c"].late], [tmp["o"].early, tmp["o"].late]],
            force=force,
        )

        result = pd.DataFrame([], columns=constants.FORCE_COLUMNS)
        for e in epochs:
            new_res = utils.atlas_force(
                self.ra, self.dec, m0=e[0], m1=e[1], **force_kwargs
            )
            if new_res is not None:
                result = pd.concat([result, new_res])
            for filt in ["c", "o"]:
                tmp[filt].early = min(
                    np.array([e[0], tmp["c"].early, tmp["o"].early], dtype=float)
                )
                tmp[filt].late = max(
                    np.array([e[1], tmp["c"].late, tmp["o"].late], dtype=float)
                )
        if not len(result):
            return new_data
        commands = []
        dl_paths = []
        local_paths = []
        for i, row in result.iterrows():
            defaults = {}
            camera = row["obs"][:3]
            exp_num = int(row["obs"][9:-1])
            nan_flag = False
            for idx in row.index:
                if idx not in ["mjd", "obs", "bandpass"]:
                    if (
                        not np.isnan(row[idx])
                        and not (idx == "dujy" and row["dujy"] > 2147483647)
                        and not (
                            idx == "ujy" and np.abs(row["ujy"] + 0.5) > 2147483647.5
                        )
                    ):
                        defaults[idx] = row[idx]
                    else:
                        nan_flag = True
                        defaults[idx] = None
            if row["bandpass"] not in ["c", "o"]:
                tmp[row["bandpass"]] = self.lightcurves.get_or_create(
                    bandpass=row["bandpass"],
                    eff_wl=constants.EFF_WL_DICT[row["bandpass"]],
                    source="ATLAS",
                    early=mjd0,
                    late=mjd1,
                )[0]
            if not nan_flag and defaults["dujy"] != 0 and defaults["dmag"] != 0:
                defaults["status"] = "det"
                AtlasDetection.objects.update_or_create(
                    lc=tmp[row["bandpass"]],
                    mjd=np.round(row["mjd"], 6),
                    exp_num=exp_num,
                    camera=camera,
                    defaults=defaults,
                )
                new_data = True
            else:
                defaults["status"] = "non"
                AtlasDetection.objects.update_or_create(
                    lc=tmp[row["bandpass"]],
                    mjd=np.round(row["mjd"], 6),
                    camera=camera,
                    exp_num=exp_num,
                    defaults=defaults,
                )
            if get_stamps:
                stamp_mjd = f"{row['mjd']:6f}"
                path_head = (
                    f"{constants.DATA_DIR}/20{self.TNS_name[:2]}/{self.TNS_name}/atlas/"
                )
                if (
                    os.path.exists(f"{path_head}/{stamp_mjd}.diff.png")
                    and not force_stamps
                ):
                    continue
                commands.append(
                    f"python /atlas/catdev/UKIRT/code/stamp.py "
                    f"{row['obs']} {row['x']} {row['y']} {stamp_size}"
                )
                for dr in ("diff", "red"):
                    dl_paths.append(f"/tmp/hsf/{row['obs']}.{dr}.png")
                    local_paths.append(f"{path_head}/{stamp_mjd}.{dr}.png")
        utils.atlas_command(commands)
        utils.dl_atlas_file(dl_paths, local_paths)
        for lc in tmp.values():
            lc.save()
        return new_data

    def update_asassn(self, mjd0=None, mjd1=None, force=False):
        print("Getting ASAS-SN lightcurve")
        new_data = False
        if not mjd0:
            mjd0 = self.detection_date - constants.PRE_SN
        if not mjd1:
            mjd1 = self.detection_date + constants.POST_SN
        g = self.lightcurves.get_or_create(
            bandpass="asg", eff_wl=constants.EFF_WL_DICT["asg"], source="ASAS-SN"
        )[0]

        epochs = utils.epoch_coverage([mjd0, mjd1], [g.early, g.late], force=force)

        os.makedirs("/tmp/hsf", exist_ok=True)
        for e in epochs:
            os.system(
                f"sub_ap_phot.py --ra={self.ra} --dec={self.dec} --deg --filter g --mjdmin {e[0]} --mjdmax {e[1]} | tail -n +2 >> /tmp/hsf/{self.ra}{self.dec}.asassn.dat"
            )
            g.early = min(np.array([e[0], g.early], dtype=float))
            g.late = max(np.array([e[1], g.late], dtype=float))
        if not os.path.exists(f"/tmp/hsf/{self.ra}{self.dec}.asassn.dat"):
            return new_data
        with open(f"/tmp/hsf/{self.ra}{self.dec}.asassn.dat", "r") as f:
            for line in f.readlines():
                # Issue where some asassn lcs start with "could not open XWindow display" etc.
                if line == "\n":
                    continue
                if line[0] in ["c", "N", "G", "F", "#"]:
                    continue

                line_list = line.split()
                if line_list[12] == "nan":
                    continue
                (
                    jd,
                    hjd,
                    ut_date,
                    image,
                    fwhm,
                    diff,
                    limit,
                    mag,
                    mag_err,
                    counts,
                    count_err,
                    flux,
                    flux_err,
                ) = line_list
                ujy = int(float(flux) * 1000)
                dujy = int(float(flux_err) * 1000)
                mjd = float(jd) - 2400000.5
                # weird date sometimes has e-#,
                # 21lpf has 2021-05-23.5625e-0
                y, m, d = ut_date.split("e")[0].split("-")
                h = int(float(d) % 1 * 24)
                mi = int(float(d) % 1 * 24 * 60 % 60)
                s = float(d) % 1 * 24 * 60 * 60 % 60
                microsecond = round((s % 1) * 1e6)
                if microsecond == 1e6:
                    microsecond -= 1
                ut_date = datetime(
                    int(y),
                    int(m),
                    int(float(d)),
                    h,
                    mi,
                    int(s),
                    microsecond,
                    tzinfo=timezone.utc,
                )
                camera = image.split("_")[1][:2]
                exp_num = int(image.split("_")[1][2:])
                if mag[0] == ">":
                    AsassnDetection.objects.update_or_create(
                        lc=g,
                        mjd=mjd,
                        hjd=float(hjd),
                        ut_date=ut_date,
                        image=image,
                        camera=camera,
                        exp_num=exp_num,
                        fwhm=float(fwhm),
                        diff=float(diff),
                        limit=float(limit),
                        mag=float(mag[1:]),
                        dmag=mag_err,
                        counts=float(counts),
                        count_err=float(count_err),
                        ujy=ujy,
                        dujy=dujy,
                        status="non",
                    )
                else:
                    AsassnDetection.objects.update_or_create(
                        lc=g,
                        mjd=mjd,
                        hjd=float(hjd),
                        ut_date=ut_date,
                        image=image,
                        camera=camera,
                        exp_num=exp_num,
                        fwhm=float(fwhm),
                        diff=float(diff),
                        limit=float(limit),
                        mag=float(mag),
                        dmag=mag_err,
                        counts=float(counts),
                        count_err=float(count_err),
                        ujy=ujy,
                        dujy=dujy,
                        status="det",
                    )
                    new_data = True

        os.remove(f"/tmp/hsf/{self.ra}{self.dec}.asassn.dat")
        g.save()
        return new_data

    def update_ztf(self, force=False):
        """
        query ALERCE API for given object
        Inputs:
            TNS_name (str): AT20XXabc
            force (bool): If true, pull lightcurves from mjd0 to mjd1
                          if false, don't redo work.
        Outputs:
            Three LightCurve instances, one for each of g,r,i-band data
        """
        new_data = False
        tmp = {}
        tmp["1"] = self.lightcurves.get_or_create(
            source="ZTF", eff_wl=constants.EFF_WL_DICT["ztfg"], bandpass="ztfg"
        )[0]
        tmp["2"] = self.lightcurves.get_or_create(
            source="ZTF", eff_wl=constants.EFF_WL_DICT["ztfr"], bandpass="ztfr"
        )[0]
        tmp["3"] = self.lightcurves.get_or_create(
            source="ZTF", eff_wl=constants.EFF_WL_DICT["ztfi"], bandpass="ztfi"
        )[0]

        if (
            "ZTF" in self.other_detections.keys()
            or "ALeRCE" in self.other_detections.keys()
        ):
            if not self.ZTF_name.startswith("ZTF"):
                self.ZTF_name = self.other_detections["ZTF"][1]
            ZTF_name = self.ZTF_name
        else:
            return new_data

        try:
            alerce = Alerce()
            lightcurve = alerce.query_lightcurve(ZTF_name)
        except requests.exceptions.ConnectionError:
            return new_data

        for det in lightcurve["detections"]:
            defaults = {}
            for key, val in det.items():
                if key != "mjd":
                    defaults[key] = val
            defaults["status"] = "det"
            _, new = ZtfDetection.objects.update_or_create(
                lc=tmp[str(det["fid"])], mjd=det["mjd"], defaults=defaults
            )
            if new:
                new_data = True
        for nondet in lightcurve["non_detections"]:
            defaults = {"diffmaglim": nondet["diffmaglim"], "status": "non"}
            ZtfDetection.objects.update_or_create(
                lc=tmp[str(det["fid"])],
                mjd=nondet["mjd"],
                fid=nondet["fid"],
                defaults=defaults,
            )
        for lc in tmp.values():
            lc.save()
        return new_data

    # def update_ukirt(
    #     self, mjd0=None, mjd1=None, bandpass="J", force=False, sub_type="nosub"
    # ):
    #     print("Updating UKIRT lightcurve")
    #     if mjd0 is None:
    #         mjd0 = min(self.observations.values_list("mjd", flat=True))
    #     if mjd1 is None:
    #         mjd1 = max(self.observations.values_list("mjd", flat=True))
    #     self.lightcurves.get_or_create(
    #         source="UKIRT", bandpass=f"{bandpass}{utils.sub_type_suffix(sub_type)}"
    #     )
    #     for obs in self.observations.filter(mjd__range=(mjd0, mjd1)):
    #         obs.make_image(mef=obs.sn_mef, force=force)
    #         obs.sn_image.photometry(sub_type=sub_type, force=force)
    #     self.save()

    def get_usable_bandpasses(
        self, bandpasses=[], variants=[], exclude=[], use_uninspected=False
    ):
        no_detections = []
        all_lcs = self.lightcurves.exclude(source="model").filter(
            bandpass__in=bandpasses, variant__in=["none"] + variants
        )
        for bp in bandpasses:
            if all_lcs.filter(bandpass=bp).count() <= 1:
                continue
            for i, var in enumerate(constants.VARIANT_PREFERENCE):
                if all_lcs.filter(bandpass=bp, variant=var).exists():
                    lc = all_lcs.get(bandpass=bp, variant=var)
                    if not lc.detections(use_uninspected=use_uninspected).exists():
                        no_detections.append(lc.pk)
                        continue
                    all_lcs = all_lcs.exclude(
                        bandpass=bp, variant__in=constants.VARIANT_PREFERENCE[i + 1 :]
                    )
                    break
        for lc in all_lcs:
            if not lc.detections(use_uninspected=use_uninspected).exists():
                no_detections.append(lc.pk)
        all_lcs = all_lcs.exclude(pk__in=no_detections)
        no_detections = self.lightcurves.filter(pk__in=no_detections)
        return all_lcs, no_detections

    def write_for_snpy(
        self,
        bandpasses=[],
        variants=[],
        mag_lim=23,
        dmag_lim=1.5,
        binned=True,
        gal_z=None,
    ):
        all_lcs, _ = self.get_usable_bandpasses(
            bandpasses=bandpasses, variants=variants
        )
        if gal_z is None:
            gal_z = 0.04
            if self.galaxy and self.galaxy.z:
                gal_z = self.galaxy.z
        try:
            float(gal_z)
        except ValueError:
            raise
        lc_str = f"{self.TNS_name} {gal_z} {self.ra} {self.dec}" + "\n"
        for lc in all_lcs:
            suffix = ""
            if binned:
                blc = lc.bin(statistic="median")
                (
                    mjd,
                    mag,
                    dmag,
                ) = (
                    blc["mjd"],
                    blc["mag"],
                    blc["dmag"],
                )
            else:
                mjd, mag, dmag = lc.mjd, lc.mag, lc.dmag

            if len(mjd) > 0:
                bp_str = f"filter {constants.FILT_SNPY_NAMES[lc.bandpass]}" + "\n"
            for epoch in range(len(mjd)):
                if mag[epoch] < mag_lim and dmag[epoch] < dmag_lim:
                    bp_str += f"{mjd[epoch]} {mag[epoch]} {dmag[epoch]}\n"
            if bp_str.count("\n") > 1:
                lc_str += bp_str
            else:
                raise RuntimeError(
                    f"Lightcurve {lc} has no usable data given a magnitude limit of {mag_lim} and an error limit of {dmag_lim}."
                )
        if lc_str[-1:] == "\n":
            lc_str = lc_str[:-1]
        return lc_str

    def write_for_salt(self, bandpasses=[], variants=[], binned=True, gal_z=None):
        all_lcs, _ = self.get_usable_bandpasses(
            bandpasses=bandpasses, variants=variants
        )
        if gal_z is None and self.galaxy and self.galaxy.z:
            gal_z = self.galaxy.z
        elif gal_z is None:
            gal_z = constants.DEFAULT_Z
        else:
            try:
                gal_z = float(gal_z)
            except ValueError:
                raise
        lc_str = f"#{self.TNS_name} {gal_z} {self.ra} {self.dec}\nmjd bandpass flux fluxerr mag magerr  zp zpsys\n"
        for lc in all_lcs:
            if lc.source == "UKIRT":
                magsys = "vega"
                zp = 24.0
                if binned:
                    blc = lc.bin(statistic="average")
            else:
                magsys = "ab"
                zp = 23.9
                if binned:
                    blc = lc.bin()
            if binned:
                mjd, mag, dmag, ujy, dujy = (
                    blc["mjd"],
                    blc["mag"],
                    blc["dmag"],
                    blc["ujy"],
                    blc["dujy"],
                )
            else:
                mjd, mag, dmag, ujy, dujy = lc.mjd, lc.mag, lc.dmag, lc.ujy, lc.dujy
            if lc.bandpass == "asg":
                bp = "g"
            else:
                bp = lc.bandpass
            for i in range(len(ujy)):
                lc_str += f"{mjd[i]} {bp} {ujy[i]} {dujy[i]} {mag[i]} {dmag[i]} {zp} {magsys}\n"
        return lc_str

    def get_lc_as_bytes(self):
        z_flag_dict = {
            "n": "No z",
            "p": "Photometric redshift from literature",
            "sp": "Spectroscopic redshift from literature",
            "sn1": "SNIFS spectrum available, not yet reduced",
            "sn2": "SNIFS spectrum available, reduced",
            "su1": "Subaru spectrum available, not yet reduced",
            "su2": "Subaru spectrum available, reduced",
            "spx1": "Literature spectrum available not yet reduced",
            "spx2": "Literature spectrum available, reduced",
        }
        gal_identity = {
            "?": "Uninspected",
            "c": "Clear host",
            "m": "Multiple possible hosts",
            "no": "No apparent galaxy",
        }
        lc_str = "# Information\n"
        lc_str += "############################\n"
        lc_str += f"# TNS_name: {self.TNS_name}" + "\n"
        lc_str += f"# SN Type: {self.sn_type.name}" + "\n"
        lc_str += f"# SN RA (J2000d): {self.ra}" + "\n"
        lc_str += f"# SN Dec (J2000d): {self.dec}" + "\n"
        lc_str += f"# SN Detection Date (MJD): {self.detection_date}" + "\n"
        lc_str += f"# SN Discovering Group: {self.discovering_group}" + "\n"
        lc_str += (
            f"# Host Galaxy Inspection Status: {gal_identity[self.galaxy_status]}"
            + "\n"
        )
        lc_str += f"# Host Galaxy RA (J2000d): {self.galaxy.ra}" + "\n"
        lc_str += f"# Host Galaxy Dec (J2000d): {self.galaxy.dec}" + "\n"
        lc_str += f"# Host Galaxy Redshift (Helio): {self.galaxy.z}" + "\n"
        lc_str += f"# Host Galaxy Redshift error: {self.galaxy.z_err}" + "\n"
        lc_str += (
            f"# Host Galaxy Redshift Status: {z_flag_dict[self.galaxy.z_flag]}" + "\n"
        )
        lc_str += "############################\n"
        lc_str += "#\n"
        lc_str += "# Lightcurve\n"
        lc_str += "#mjd bandpass flux_ujy dflux_dujy mag_ab dmag_ab\n"
        for lc in self.lightcurves.filter(source__in=("ATLAS", "ASAS-SN", "ZTF")):
            for epoch in range(len(lc.mjd)):
                lc_str += (
                    f"{lc.mjd[epoch]} {lc.bandpass} {lc.ujy[epoch]} {lc.dujy[epoch]} {lc.mag[epoch]} {lc.dmag[epoch]}"
                    + "\n"
                )
        qs = self.lightcurves.filter(source="UKIRT")
        if qs.count() > 0:
            for lc in qs:
                for epoch in range(len(lc.mjd)):
                    lc_str += (
                        f"{lc.mjd[epoch]} {lc.bandpass} {lc.ujy[epoch]} {lc.dujy[epoch]} {lc.mag[epoch]} {lc.dmag[epoch]}"
                        + "\n"
                    )
        return str.encode(lc_str)

    def get_refcat(
        self, rad=None, dr=None, dd=None, all_cols=True, mlim=None, force=False
    ):
        file_path = f"{constants.DATA_DIR}/20{self.TNS_name[:2]}/{self.TNS_name}/atlas/refcat.dat"
        if os.path.exists(file_path):
            rc = pd.read_csv(file_path)
            if len(rc.columns) == 9 and all_cols:
                force = True
            if rad:
                rc_rad = np.sqrt(
                    ((rc["RA"] - self.ra) * np.cos(self.dec * np.pi / 180)) ** 2
                    + (rc["Dec"] - self.dec) ** 2
                ).max()
                if rc_rad * constants.REFCAT_PADDING < rad:
                    force = True
            if dr and dd:
                rc_dd = (rc["Dec"].max() - rc["Dec"].min()) / 2
                # handling RA wrap around at 360 deg
                if rc["RA"].min() < 1 and rc["RA"].max() > 359:
                    rc_dr = (
                        rc["RA"][rc["RA"] < 180].max()
                        - (rc["RA"][rc["RA"] > 180].min() - 360)
                    ) / 2
                else:
                    rc_dr = (rc["RA"].max() - rc["RA"].min()) / 2
                if (
                    rc_dr * constants.REFCAT_PADDING < dr
                    or rc_dd * constants.REFCAT_PADDING < dd
                ):
                    force = True
                    dr = max(dr, rc_dr * constants.REFCAT_PADDING)
                    dd = max(dd, rc_dd * constants.REFCAT_PADDING)
            if mlim:
                rc_mlim = max([rc[filt].max() for filt in "griz"])
                if rc_mlim + 2.5 * np.log10(constants.REFCAT_PADDING) < mlim:
                    force = True
        return utils.refcat(
            self.ra,
            self.dec,
            rad=rad,
            dr=dr,
            dd=dd,
            all_cols=all_cols,
            mlim=mlim,
            force=force,
            file_path=file_path,
        )

    # PLOTTING
    def plot(self, model_names="all"):
        if self.lightcurves.exclude(source="model").count() == 0:
            return None
        # plot params
        plot_tools = {
            "tools": ["pan", "wheel_zoom", "hover", "box_zoom,", "reset", "help"],
            "active_scroll": "wheel_zoom",
        }
        plot_tall = {"height": 300, "width": 900}
        plot_short = {"height": 125, "width": 900}

        # tooltip info
        mag_tooltips = [
            ("MJD", "$x{(0.00)}"),
            ("AB Mag", "$y"),
            ("Variant", "@variant"),
            # ("Model", "@model"),
        ]
        flux_tooltips = [
            ("MJD", "$x{(0.00)}"),
            ("Flux uJy", "$y"),
            ("Variant", "@variant"),
            # ("Model", "@model"),
        ]

        # generating plots
        magplot = figure(title=None, **plot_tall, **plot_tools, tooltips=mag_tooltips)
        magplot.y_range.flipped = True
        magresid = figure(
            title=None,
            x_range=magplot.x_range,
            y_range=[3, -3],
            **plot_short,
            **plot_tools,
        )
        fluxplot = figure(
            title=None,
            x_range=magplot.x_range,
            **plot_tall,
            **plot_tools,
            tooltips=flux_tooltips,
        )
        fluxresid = figure(
            title=None,
            x_range=magplot.x_range,
            y_range=[-3, 3],
            **plot_short,
            **plot_tools,
        )

        # checkboxes
        checkboxes = {}
        available_bps = list(sorted(self.detected_bandpasses))
        available_variants = list(
            self.lightcurves.values_list("variant", flat=True).distinct()
        )
        checkboxes["bp"] = CheckboxButtonGroup(
            labels=available_bps,
            active=[i for i in range(len(available_bps))],
            tags=["bandpasses"],
        )
        checkboxes["variants"] = CheckboxButtonGroup(
            labels=available_variants,
            active=[i for i in range(len(available_variants))],
            tags=["variants"],
        )
        # model checkboxes
        checkboxes["model"] = CheckboxButtonGroup(
            labels=list(("data",) + constants.MODEL_NAMES),
            active=[i for i in range(len(constants.MODEL_NAMES) + 1)],
            tags=["model"],
        )
        checkboxes["bp_str"] = CheckboxButtonGroup(
            labels=available_bps,
            active=[i for i in range(len(available_bps))],
            tags=["bp_str"],
        )
        checkboxes["v_str"] = CheckboxButtonGroup(
            labels=available_variants,
            active=[],
            tags=["v_str"],
        )

        d = {}
        detections, non_detections = self.data_cds()
        whisker_source, _ = self.data_cds()
        models, resids = self.model_cds()
        dummy_models, dummy_resids = self.model_cds()
        # Filters detections via indices, emits changes to dummy models and
        # resids to update those separately. There's surely a better way to do this.
        det_filt = CustomJSFilter(
            args=dict(
                cbox=checkboxes,
                det_source=detections,
                whisker_source=whisker_source,
                model_source=models,
                resid_source=resids,
                dm_source=dummy_models,
                dr_source=dummy_resids,
            ),
            code="""
            var actives = {};
            for (var i in cbox) {
                actives[i] = cbox[i].active.map(j=>cbox[i].labels[j]);
            }

            var data = {};
            data['det'] = det_source.data;
            data['model'] = model_source.data;
            data['resid'] = resid_source.data;

            // Reset the dummy sources to empty
            var dummy = {};
            dummy['det'] = whisker_source.data;
            dummy['model'] = dm_source.data;
            dummy['resid'] = dr_source.data;
            for (var i in dummy) {
                for (var j in dummy[i]) {
                    dummy[i][j] = [];
                }
            }

            let indices = [];
            for (var i in data) {
                for (let j=0; j<data[i]['bandpass'].length; j++){
                    let bp_str = actives['bp_str'].filter(n=>n).join('-');
                    let v_str = actives['v_str'].filter(n=>n).join('-');
                    console.log(bp_str);
                    if (
                        actives['bp'].includes(data[i]['bandpass'][j])
                        && actives['variants'].includes(data[i]['variant'][j])
                        && actives['model'].includes(data[i]['model'][j])
                        && (i == 'det' || bp_str == data[i]['bp_str'][j])
                        && (i == 'det' || v_str == data[i]['v_str'][j])
                    ) {
                        if (i=='det') {
                            indices.push(j);
                        }
                        for (var k in dummy[i]) {
                                dummy[i][k].push(data[i][k][j])
                        }
                    }
                }
            }

            // Call to update dummy source data
            whisker_source.change.emit();
            dm_source.change.emit();
            dr_source.change.emit();

            return indices;
            """,
        )
        for cb in checkboxes:
            checkboxes[cb].js_on_change(
                "active",
                CustomJS(
                    code="source.change.emit();",
                    args=dict(source=detections),
                ),
            )
        det_view = CDSView(source=detections)  # , filters=[det_filt])
        det_view.filter &= det_filt
        for scale, scale_name, units, plot, residplot in zip(
            ("m", "f"),
            ("mag", "flux"),
            ("mag", "ujy"),
            (magplot, fluxplot),
            (magresid, fluxresid),
        ):
            # detections
            d["detections"] = plot.circle(
                "mjd",
                units,
                color="color",
                source=detections,
                alpha=0.8,
                muted_alpha=0.2,
                view=det_view,
            )
            d["detections_w"] = Whisker(
                base="mjd",
                upper=f"{scale_name}upper",
                lower=f"{scale_name}lower",
                line_color="color",
                source=whisker_source,
            )
            d["detections_w"].upper_head.line_color = "color"
            d["detections_w"].lower_head.line_color = "color"
            plot.add_layout(d["detections_w"])

            # non-detections
            d["non_detections"] = plot.inverted_triangle(
                "tlim",
                f"{units}lim",
                source=non_detections,
                color="color",
                alpha=0.4,
                muted_alpha=0.1,
            )

            # model lcs
            d["models"] = plot.multi_line(
                xs="mjd",
                ys=units,
                line_dash="dotted",
                color="color",
                source=dummy_models,
            )

            # residuals
            d["resids"] = residplot.x(
                "mjd",
                f"{scale_name}resid",
                color="color",
                source=dummy_resids,
            )

        div_text = Div(text="""<h3>Model Lightcurve Parameters</h3>""")
        widgets = [
            row(checkboxes["bp"]),
            row(checkboxes["variants"]),
            row(div_text),
            row(checkboxes["model"]),
            row(checkboxes["bp_str"]),
            row(checkboxes["v_str"]),
        ]
        today_line = Span(
            location=utils.MJD(today=True),
            dimension="height",
            line_color="yellow",
            line_width=1,
        )
        detected_line = Span(
            location=self.detection_date,
            dimension="height",
            line_color="white",
            line_width=1,
        )
        h_line = Span(location=0, dimension="width", line_color="white", line_width=1)
        magplot.renderers.extend([today_line, detected_line])
        fluxplot.renderers.extend([today_line, detected_line])
        magresid.renderers.extend([h_line])
        fluxresid.renderers.extend([h_line])
        # legend
        # legend = Legend(items=li, location=(0, -30))
        # legend.click_policy = "hide"
        # legend.background_fill_color = "black"
        # legend.label_text_color = "white"
        # magplot.add_layout(legend, "right")
        # fluxplot.add_layout(legend, "right")
        # arrange plots in a column
        tab1 = TabPanel(
            child=column([fluxplot, fluxresid, *widgets], sizing_mode="scale_width"),
            title="Flux",
        )
        tab2 = TabPanel(
            child=column([magplot, magresid, *widgets], sizing_mode="scale_width"),
            title="Magnitudes",
        )
        p = Tabs(tabs=[tab1, tab2])
        curdoc().theme = Theme(
            json={
                "attrs": {
                    "Figure": {
                        "background_fill_color": "black",
                        "border_fill_color": "black",
                        "outline_line_color": "black",
                    },
                    "Axis": {
                        "axis_line_color": "black",
                        "axis_label_text_color": "white",
                        "major_label_text_color": "white",
                    },
                    "Grid": {
                        "minor_grid_line_color": "#222222",
                        "grid_line_alpha": 0.4,
                    },
                    "Toolbar": {"autohide": True},
                }
            }
        )
        curdoc().add_root(p)
        return p

    def data_cds(self):
        dfs = []
        # Detections
        det_data = {}
        for key in (
            "mjd",
            "mag",
            "ujy",
            "maglower",
            "magupper",
            "fluxlower",
            "fluxupper",
            "bandpass",
            "variant",
            "color",
            "model",
            "bp_str",
            "v_str",
        ):
            det_data[key] = []
        detections = ColumnDataSource(data=det_data)

        for lc in self.lightcurves.exclude(source="model"):
            lcb = lc.bin(bin_period="mjd", statistic="median", weight="ivar", clip=3)
            dfs.append(
                pd.DataFrame(
                    data={
                        "mjd": lcb["mjd"],
                        "mag": lcb["mag"],
                        "ujy": lcb["ujy"],
                        "maglower": [
                            mag - dmag for mag, dmag in zip(lcb["mag"], lcb["dmag"])
                        ],
                        "magupper": [
                            mag + dmag for mag, dmag in zip(lcb["mag"], lcb["dmag"])
                        ],
                        "fluxlower": [
                            ujy - dujy for ujy, dujy in zip(lcb["ujy"], lcb["dujy"])
                        ],
                        "fluxupper": [
                            ujy + dujy for ujy, dujy in zip(lcb["ujy"], lcb["dujy"])
                        ],
                        "bandpass": lc.bandpass,
                        "variant": lc.variant,
                        "color": constants.COLOR_DICT[lc.bandpass],
                        "model": "data",
                        "bp_str": lc.bandpass,
                        "v_str": lc.variant,
                    }
                )
            )
        if dfs:
            detections = ColumnDataSource(data=pd.concat(dfs))

        # Non detections
        dfs = []
        non_det_data = {}
        for key in ("tlim", "maglim", "ujylim", "bandpass", "variant", "color"):
            non_det_data[key] = []
        non_detections = ColumnDataSource(data=non_det_data)
        for lc in self.lightcurves.exclude(source="model"):
            if lc.nondetections():
                dfs.append(
                    pd.DataFrame(
                        data={
                            "tlim": [nondet.mjd for nondet in lc.nondetections()],
                            "maglim": [nondet.mag for nondet in lc.nondetections()],
                            "ujylim": [nondet.ujy for nondet in lc.nondetections()],
                            "bandpass": lc.bandpass,
                            "variant": lc.variant,
                            "color": constants.COLOR_DICT[lc.bandpass],
                        }
                    )
                )
        if dfs:
            non_detections = ColumnDataSource(data=pd.concat(dfs))
        return detections, non_detections

    def model_cds(self):
        lcs = self.lightcurves.filter(
            fit_result__bandpasses_str__in=(
                # "-".join([bp for bp in self.detected_bandpasses if bp != "asg"]),
                # "J-asg-c-o-ztfg-ztfr",
                # "J-asg-c-o",
                # "J-asg-ztfg-ztfr",
                # "asg-c-o",
                # "asg-ztfg-ztfr",
                "J-c-o-ztfg-ztfr",
                "J-c-o",
                "J-ztfg-ztfr",
                "c-o",
                "ztfg-ztfr",
            ),
        )
        model = ColumnDataSource(
            dict(
                mjd=[list(lc.mjd) for lc in lcs],
                mag=[list(lc.mag) for lc in lcs],
                ujy=[list(lc.ujy) for lc in lcs],
                bandpass=[lc.bandpass for lc in lcs],
                color=[constants.COLOR_DICT[lc.bandpass] for lc in lcs],
                variant=[lc.variant for lc in lcs],
                model=[lc.fit_result.model_name for lc in lcs],
                bp_str=[lc.fit_result.bandpasses_str for lc in lcs],
                v_str=[lc.fit_result.variants_str for lc in lcs],
            )
        )
        resids = []
        model_data = {}
        for key in (
            "mjd",
            "magresid",
            "ujyresid",
            "bandpass",
            "color",
            "variant",
            "model",
            "bp_str",
            "v_str",
        ):
            model_data[key] = []
        resid = ColumnDataSource(data=model_data)
        for lc in self.lightcurves.filter(source="model"):
            if not len(lc.mjd):
                continue
            try:
                data_lc = lc.fit_result.data_lightcurves.get(bandpass=lc.bandpass)
                blc = data_lc.bin()
                if not len(blc["mjd"]):
                    continue
            except ObjectDoesNotExist:
                continue
            resids.append(
                pd.DataFrame(
                    data={
                        "mjd": blc["mjd"],
                        "magresid": (blc["mag"] - np.interp(blc["mjd"], lc.mjd, lc.mag))
                        / blc["dmag"],
                        "fluxresid": (
                            blc["ujy"] - np.interp(blc["mjd"], lc.mjd, lc.ujy)
                        )
                        / blc["dujy"],
                        "bandpass": lc.bandpass,
                        "color": constants.COLOR_DICT[lc.bandpass],
                        "variant": lc.variant,
                        "model": lc.fit_result.model_name,
                        "bp_str": lc.fit_result.bandpasses_str,
                        "v_str": lc.fit_result.variants_str,
                    }
                )
            )
        if resids:
            resid = ColumnDataSource(pd.concat(resids))
        return model, resid

    # ACCESSING LOWER LEVEL SCRIPTS
    def get_closest_galaxy(self, box_size=2.5 / 60):
        """get_closest_galaxy.

        Parameters
        ----------
        box_size :
            box_size
        """
        qset = Galaxy.objects.box_search(
            ra=self.ra, dec=self.dec, box_size=box_size
        ).exclude(ra=self.ra, dec=self.dec)
        # Removing catalog entries for the supernova itself
        sn_entries = []
        for g in qset:
            if not (
                g.ned_entries.exclude(prefname__contains=self.TNS_name).exists()
                or g.simbad_entries.exclude(main_id__contains=self.TNS_name).exists()
                or g.ps1_entries.exists()
                or g.glade_entries.exists()
            ):
                sn_entries.append(g.pk)
        qset = qset.exclude(pk__in=sn_entries)
        if not qset.count():
            return
        param_dict = qset.get_normalized_axes()
        sc = SkyCoord(*np.array(qset.values_list("ra", "dec")).T, unit="deg")
        sep = sc.separation(SkyCoord(self.ra, self.dec, unit="deg")).value
        dra = self.ra - sc.ra.value
        ddec = self.dec - sc.dec.value
        dphi = param_dict["phi"] - np.arctan2(ddec, dra) * 180 / np.pi
        dphi[param_dict["phi"] == 0] = 90
        rad = np.zeros(qset.count())
        # calculate distance normalized to maj/min axes away.
        for i, (s, M, a, p, dr, dd, dp) in enumerate(
            zip(
                sep,
                param_dict["major"],
                param_dict["axis_ratio"],
                param_dict["phi"],
                dra,
                ddec,
                dphi,
            )
        ):
            dmajor, dminor = None, None
            # if no axis_ratio, assume round
            if not a:
                a = 1
            if M:
                dmajor = np.cos(dp * np.pi / 180) * s / M
                dminor = np.sin(dp * np.pi / 180) * s / (a * M)
            if dmajor and dminor:
                rad[i] = np.sqrt(dmajor**2 + dminor**2)
            # not all galaxy entries have dimensions
            # if nothing, assume round with maj = 0.1"
            else:
                rad[i] = (
                    np.sqrt((dr * np.cos(self.dec * np.pi / 180)) ** 2 + dd**2)
                    * 3600
                    * 10
                )
        self.galaxy = qset[int(np.argmin(rad))]
        self.save()

    def make_images(self, mef="sn", obs_list="all", force=False):
        if obs_list == "all":
            obs_list = self.observations.all()
        else:
            obs_list = self.observations.filter(name__in=[obs_list])
        for obs in tqdm(obs_list):
            obs.make_image(mef=mef, force=force)

    def photometry(
        self,
        obs_list="all",
        sub_type="all",
        color_dict=constants.HODGKIN_COLOR_CORRECTION,
        thresh=constants.TPHOT_RAD,
        force=False,
        verbose=False,
    ):
        if obs_list == "all":
            obs_list = self.observations.filter(status="science")
        else:
            obs_list = self.observations.filter(name__in=[obs_list])
        if sub_type == "all":
            sub_types = ["nosub", "refsub", "rotsub"]
        else:
            sub_types = (sub_type,)
        for obs in tqdm(obs_list):
            for sub_type in sub_types:
                try:
                    obs.sn_image.photometry(
                        sub_type=sub_type,
                        color_dict=color_dict,
                        thresh=thresh,
                        force=force,
                        verbose=verbose,
                    )
                except:
                    continue

    def fit(
        self,
        bandpasses=[],
        variants="all",
        exclude=[],
        model_name="all",
        calibration=6,
        redlaw="F19",
        force=False,
        clean=True,
        priors={},
        mcmc=False,
        mcmc_priors={},
        use_uninspected=False,
        return_result=False,
        spline_tmin=15,
        model_min_snr=0.2,
        fail_loudly=True,
        verbose=False,
    ):
        """fit.

        Parameters
        ----------
        bandpasses :
            bandpasses
        variants :
            variants
        exclude :
            exclude
        model_name :
            model_name
        force :
            force
        clean :
            clean
        priors :
            priors
        mcmc :
            mcmc
        mcmc_priors :
            mcmc_priors
        use_uninspected :
            use_uninspected
        return_result :
            return_result
        spline_tmin :
            spline_tmin
        fail_loudly :
            fail_loudly
        """
        # Looping over bandpasses if pset
        if bandpasses == "pset":
            pset = utils.pset(list(self.detected_bandpasses))
            for bps in pset[1:]:
                self.fit(
                    bandpasses=bps,
                    variants=variants,
                    exclude=exclude,
                    model_name=model_name,
                    calibration=calibration,
                    redlaw=redlaw,
                    force=force,
                    clean=clean,
                    priors=priors,
                    mcmc=mcmc,
                    mcmc_priors=mcmc_priors,
                    use_uninspected=use_uninspected,
                    return_result=return_result,
                    verbose=verbose,
                    spline_tmin=spline_tmin,
                    model_min_snr=model_min_snr,
                    fail_loudly=fail_loudly,
                )
            return
        elif isinstance(bandpasses, str):
            bandpasses = bandpasses.split("-")

        # Looping over sensible variants if left as None
        if variants == "all":
            ztf_variants = ["none"]
            nir_variants = ["none"]
            if "ztf" in "-".join(bandpasses):
                ztf_variants = [
                    "magpsf",
                ]  # "magap", "magap_big"]
            if not {"Z", "Y", "J", "H", "K"}.isdisjoint(set(bandpasses)):
                nir_variants = [
                    "tphot",
                    "rot",
                    "0D",
                    "1D2",
                    "1D3",
                    "1D4",
                    "2D",
                    "dehvils",
                ]
            for ztfv in ztf_variants:
                for nirv in nir_variants:
                    vs = []
                    if ztfv != "none":
                        vs.append(ztfv)
                    vs.append(nirv)
                    self.fit(
                        bandpasses=bandpasses,
                        variants=vs,
                        exclude=exclude,
                        model_name=model_name,
                        calibration=calibration,
                        redlaw=redlaw,
                        force=force,
                        clean=clean,
                        priors=priors,
                        mcmc=mcmc,
                        mcmc_priors=mcmc_priors,
                        use_uninspected=use_uninspected,
                        return_result=return_result,
                        verbose=verbose,
                        spline_tmin=spline_tmin,
                        model_min_snr=model_min_snr,
                        fail_loudly=fail_loudly,
                    )
            return
        elif isinstance(variants, str):
            variants = variants.split("-")

        all_lcs, no_detections = self.get_usable_bandpasses(
            bandpasses=bandpasses,
            variants=variants,
            exclude=exclude,
            use_uninspected=use_uninspected,
        )
        if no_detections.exists() or all_lcs.count() != len(bandpasses):
            print(
                f"No detections in {no_detections.values_list('bandpass', flat=True)}"
            )
            return

        results = []
        if not all_lcs.exists():
            print("There are no detections in the requested bandpasses")
            return
        if model_name == "all":
            model_name = constants.MODEL_NAMES
        elif isinstance(model_name, str):
            model_name = (model_name,)
        for m in model_name:
            if "snpy" in m:
                results.append(
                    snpy_fit(
                        all_lcs=all_lcs,
                        calibration=calibration,
                        redlaw=redlaw,
                        model_name=m,
                        model_min_snr=model_min_snr,
                        force=force,
                        clean=clean,
                        priors=priors,
                        mcmc=mcmc,
                        mcmc_priors=mcmc_priors,
                        return_result=return_result,
                        fail_loudly=fail_loudly,
                    )
                )
            elif "salt" in m:
                results.append(
                    salt_fit(
                        all_lcs=all_lcs,
                        redlaw=redlaw,
                        return_result=return_result,
                        model_min_snr=model_min_snr,
                        force=force,
                        priors=priors,
                        verbose=verbose,
                    )
                )
            elif "bayesn" in m:
                results.append(
                    bayesn_fit(
                        all_lcs=all_lcs,
                        model_name=m,
                        return_result=return_result,
                        force=force,
                        verbose=verbose,
                        redlaw=redlaw,
                    )
                )
            elif m == "spline":
                for lc in all_lcs:
                    spline_fit(
                        lc, tmin=spline_tmin, model_min_snr=model_min_snr, force=force
                    )
        if return_result:
            return results

    def get_tmax(self):
        self.tmax = {}
        for bp in (
            self.lightcurves.filter(source="model")
            .values_list("bandpass", flat=True)
            .distinct()
        ):
            tmax_qset = self.lightcurves.filter(
                source="model", tmax__isnull=False, bandpass=bp
            )
            if tmax_qset.exists():
                self.tmax[bp] = np.median(tmax_qset.values_list("tmax", flat=True))
        self.save()

    def process(
        self,
        make_images=True,
        phot=True,
        fit=True,
        sub_obs_list="all",
        force_sub=False,
        phot_obs_list="all",
        phot_sub_type="all",
        thresh=10,
        force_phot=False,
        model_name="all",
        exclude=[],
        fit_bandpasses="all-nosub",
        force_fit=True,
    ):
        if make_images:
            self.make_images(obs_list=sub_obs_list, mef="sn", force=force_sub)
        if phot:
            self.photometry(
                obs_list=phot_obs_list,
                sub_type=phot_sub_type,
                thresh=thresh,
                force=force_phot,
            )
        if fit:
            self.fit(
                model_name=model_name,
                bandpasses=fit_bandpasses,
                exclude=exclude,
                force=force_fit,
            )

    # STATIC METHODS
    @staticmethod
    @logger.catch
    def sync_w_tns(m=31):
        """
        Search TNS for targets submitted within the past m minutes.
        If a target is not in the database, add it in.
        """
        timestamp = datetime.utcnow() - timedelta(minutes=m)
        search_obj = [
            ("ra", ""),
            ("dec", ""),
            ("radius", ""),
            ("units", ""),
            ("objname", ""),
            ("objname_exact_match", 0),
            ("internal_name", ""),
            ("internal_name_exact_match", 0),
            ("objid", ""),
            ("public_timestamp", str(timestamp)),
        ]
        data = utils.TNS_query("search", search_obj)
        new, updated = [], []
        for obj in tqdm(data):
            TNS_name = obj["objname"][2:]
            if Target.objects.filter(TNS_name=TNS_name).count() == 0:
                new.append(TNS_name)
            else:
                updated.append(TNS_name)
        return new, updated

    @staticmethod
    def add_SCAT_to_db(sheet_index=1):
        # 0 based indexing
        sheet_instance = utils.google_sheets_scat("SCAT Targets", sheet_index)
        # 1 based indexing
        obs_names = sheet_instance.col_values(1)
        quality = sheet_instance.col_values(2)
        exp_num = sheet_instance.col_values(4)
        for obs, qual, num in zip(obs_names, quality, exp_num):
            if qual != "host":
                continue
            TNS_name = obs.split("_")[0]
            if len(re.findall("[0-9]{2}", TNS_name)) == 2:
                TNS_name = TNS_name[2:]
            t = Target.quick_get(TNS_name)
            if t.galaxy is None:
                new_gal = Galaxy.objects.create(ra=t.ra, dec=t.dec)
                t.galaxy = new_gal
                t.save()

            glob_path = glob.glob(
                f"{constants.SCAT_DIR}/data/{num[:6]}/spec_{num}*_SNIFS.dat"
            )
            if len(glob_path):
                file_path = glob_path[0]
            else:
                continue
            with open(file_path, "r") as f:
                head = f.readlines()[:15]
            data = pd.read_csv(
                file_path,
                delim_whitespace=True,
                skiprows=15,
                names=["wl", "flux", "err"],
                dtype=float,
            ).dropna()
            sp, _ = SnifsSpectrum.objects.get_or_create(
                target=t,
                path=file_path,
                name=num,
                exp_num="_".join(file_path.split("/")[-1].split("_")[3:5]),
            )
            y, m, d = head[6].split()[3].split("T")[0].split("-")
            sp.date = date(int(y), int(m), int(d))
            sp.program = "SCAT"
            # sp.exp_time = int(head[7].split()[3].strip('.0s'))
            sp.mjd = float(head[5].split()[3])

            sp.TNS_name = t.TNS_name
            sp.airmass = float(head[8].split()[3])
            sp.wl = data["wl"].to_list()
            sp.flux = data["flux"].to_list()
            sp.err = data["err"].to_list()
            sp.channels = head[3].split()[3]
            if "B" in sp.channels:
                sp.fluxcal_rms_b = float(head[9].split()[3])
            if "R" in sp.channels:
                sp.fluxcal_rms_r = float(head[10].split()[3])
            sp.save()
            with open(sp.path, "r") as f:
                for line in f.readlines():
                    if line.split()[1] == "RA":
                        ra_str = line.split()[3]
                    elif line.split()[1] == "DEC":
                        dec_str = line.split()[3]
                        break
                ra, dec = utils.HMS2deg(ra_str, dec_str)
            t.galaxy.snifs_entries.get_or_create(
                spectrum=sp,
                defaults=dict(
                    ra=ra,
                    dec=dec,
                    date=date.today(),
                    reduction_date=date.fromtimestamp(os.path.getmtime(sp.path)),
                ),
            )

    @staticmethod
    def import_data(path, delimiter="\t"):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with open(path, "r") as f:
            for line in tqdm(f.readlines()):
                line = line.strip("\n")
                if line.split(delimiter)[0] != "target":
                    continue
                defaults = {}
                for field, val in zip(Target._meta.fields, line.split(delimiter)):
                    if val == "None":
                        defaults[field.name] = None
                    elif field.name == "galaxy":
                        try:
                            defaults[field.name] = Galaxy.objects.get(pk=val)
                        except Galaxy.DoesNotExist:
                            print(
                                f"The database does not contain a galaxy with pk={val}. Making a dummy Galaxy with that name."
                            )
                            defaults[field.name] = Galaxy.objects.create(
                                pk=val, ra=0, dec=0
                            )
                    elif isinstance(field, models.fields.json.JSONField):
                        defaults[field.name] = json.loads(val)
                    elif isinstance(field, models.fields.DateTimeField):
                        defaults[field.name] = datetime.fromisoformat(val)
                    elif isinstance(filed, models.fields.FloatField):
                        defaults[field.name] = float(val)
                    else:
                        defaults[field.name] = str(val)
                TNS_name = defaults.pop("TNS_name")
                sn_type = TransientType.objects.get(pk=defaults.pop("sn_type"))
                Target.objects.update_or_create(
                    TNS_name=TNS_name, sn_type=sn_type, defaults=defaults
                )

    @staticmethod
    def demographics(sn_type):
        """
        Return dict of sub_types of "sn_type", how many we have observations of, and how many of those have galaxies.
        """
        demo = {}
        for t in TransientType.objects.get(name=sn_type).children.all():
            demo[t.name] = {}
            for target in t.target_set.all():
                count_str = str(target.observations.filter(bandpass="J").count())
                if count_str not in demo[t.name].keys():
                    demo[t.name][count_str] = [0, 0]
                demo[t.name][str(target.observations.filter(bandpass="J").count())][
                    0
                ] += 1
                if target.galaxy and target.galaxy.z:
                    demo[t.name][str(target.observations.filter(bandpass="J").count())][
                        1
                    ] += 1
        return demo

    @staticmethod
    def write_good_and_bad_z():
        q = Target.objects.get_by_type("Ia").number_of_observations(logic="gte")
        with open(f"{constants.MEDIA_DIR}/Ia_with_obs_with_z.txt", "w") as f:
            for obj in q.with_good_host_z():
                f.write(f"{obj.TNS_name} {obj.galaxy.z} {obj.galaxy.z_err}\n")
        with open(f"{constants.MEDIA_DIR}/Ia_with_obs_needs_z.txt", "w") as f:
            for obj in q.needs_host_z():
                f.write(f"{obj.TNS_name}\n")

    @staticmethod
    def write_qset_lc(qset, clear=True):
        if clear:
            for path in glob.glob(f"{BASE_DIR}/*.dat"):
                os.remove(path)
        for obj in qset:
            lc_str = obj.lightcurve.get_lc_as_bytes().decode("utf-8")
            with open(f"{BASE_DIR}/lc/{obj.TNS_name}.dat", "w") as f:
                f.write(lc_str)
        os.system("tar -cvzf lc.tar.gz lc")
        os.system("chgrp bad_user lc.tar.gz")

    @staticmethod
    def write_subaru_list():
        with open("host_gal.dat", "w") as f:
            q = (
                Target.objects.get_by_type("Ia")
                .number_of_observations(logic="gte")
                .needs_host_z()
                .ra_range()
            )
            for obj in q:
                c = obj.HMS_coords()
                f.write(f"{obj.TNS_name} {c[0]} {c[1]}\n")


