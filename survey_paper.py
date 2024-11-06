import numpy as np
import pandas as pd
import pingouin as pg
from scipy.odr import ODR, Model, RealData, polynomial
from tqdm import tqdm

import constants
import utils
from utils import print_verb
from fitting.models import FitResults
from snippets import (
    compare_1D3_2D,
    compare_phot_with_DEHVILS,
    compare_snifs_leda,
    prep_data_trend,
    sdss_weightedCC_systematics,
)
from targets.models import Target
from data.models import Observation

models = ("ebv", "max", "salt")


def compare_multiple_sets(
    data_dict=None,
    cuts="standard",
    optical_only=False,
    sigma=-1,
    max_bandpass="J",
    redlaw="F19",
    verbose=False,
    fr_qsets_only=False,
):
    if data_dict is None:
        data_dict = {"fr_qsets": {}, "junk": {}, "stats": {}, "p": {}, "outliers": {}}
    qset = Target.objects.get_by_type("Ia").number_of_observations().with_good_host_z()
    variants = []
    acceptable = ["g", "?", "b", "n"]
    optional_args = {"bandpasses": ["ztfg", "ztfr", "c", "o", "asg"]}
    excluded_args = {
        "bandpasses": ["Y", "H"],
        "variants": ["tphot", "dehvils", "rot", "ref", "1D3-2D", "1D2", "1D4"],
    }
    if optical_only:
        bandpasses = []
        excluded_args["bandpasses"] += ["J"]
        excluded_args["variants"] += ["2D", "0D", "1D3"]
        max_bandpass = "V"
    else:
        bandpasses = ["J"]
        optional_args["variants"] = ["2D", "0D", "1D3"]
    if cuts == "standard":
        cuts = constants.STANDARD_CUTS
    print_verb(
        verbose,
        f"Grabbing {'optical only '*optical_only}fit results with cuts={cuts} and redlaw={redlaw}",
    )
    for key, model_name, calibration in zip(
        ("salt", "ebv", "max"),
        ("salt3-nir", "snpy_ebv_model2", "snpy_max_model"),
        (None, 6, None),
    ):
        if optical_only:
            key = key + "_no_J"
        data_dict["fr_qsets"][key], data_dict["junk"][key] = qset.get_fitresults_qset(
            model_name=model_name,
            bandpasses=bandpasses,
            variants=variants,
            calibration=calibration,
            redlaw=redlaw,
            acceptable_status=acceptable,
            max_bandpass=max_bandpass,
            cuts=cuts,
            optional_args=optional_args,
            excluded_args=excluded_args,
        )
        if not fr_qsets_only:
            data_dict["stats"][key], data_dict["p"][key] = data_dict["fr_qsets"][
                key
            ].get_dispersion(sigma=sigma, max_bandpass=max_bandpass)
    return data_dict


def compare_set_intersection(
    data_dict,
    compare=("salt", "ebv"),
    suffix="common",
    match_optical=False,
    default_max_bandpass="J",
    max_color_1="V",
    max_color_2="r",
    verbose=False,
    fr_qsets_only=False,
):
    import copy

    fr_qsets = data_dict["fr_qsets"]
    junk = data_dict["junk"]
    print_verb(verbose, f"Filtering targets common to {compare}")
    for key in compare:
        fr_qsets[key + "_" + suffix] = fr_qsets[key].order_by("target__TNS_name")
        for other_key in compare:
            fr_qsets[key + "_" + suffix] = fr_qsets[key + "_" + suffix].filter(
                target__TNS_name__in=data_dict["p"][other_key]["TNS_name"]
            )
    if match_optical:
        print_verb(verbose, f"Filtering targets with same optical bandpasses")
        no_match = []
        for key in compare:
            for fr in fr_qsets[key + "_" + suffix]:
                bp_str = fr.bandpasses_str.replace("J-", "")
                for other_key in compare:
                    if bp_str != fr_qsets[other_key + "_" + suffix].get(
                        target__TNS_name=fr.target.TNS_name
                    ).bandpasses_str.replace("J-", ""):
                        no_match.append(fr.target.TNS_name)
        for key in compare:
            junk[key + "_" + suffix] = copy.deepcopy(junk[key])
            junk[key + "_" + suffix]["different_bp_strs"] = Target.objects.filter(
                TNS_name__in=no_match
            )
            fr_qsets[key + "_" + suffix] = fr_qsets[key + "_" + suffix].exclude(
                target__TNS_name__in=no_match
            )
    if fr_qsets_only:
        return data_dict
    for key in compare:
        print_verb(verbose, f"Calculating dispersions of common targets in {key}")
        max_bandpass = default_max_bandpass
        if "no_J" in key:
            max_bandpass = "V"
        (
            data_dict["stats"][key + "_" + suffix],
            data_dict["p"][key + "_" + suffix],
        ) = fr_qsets[key + "_" + suffix].get_dispersion(
            sigma=-1,
            max_bandpass=max_bandpass,
            max_color_1=max_color_1,
            max_color_2=max_color_2,
        )
    return data_dict


def get_data_dict(force=False, unity_force=False, verbose=False):
    import os

    from django.db.models import Q

    if os.path.exists(f"{constants.HSF_DIR}/ado/data_dict.pickle") and not force:
        import pickle

        with open(f"{constants.HSF_DIR}/ado/data_dict.pickle", "rb") as handle:
            return pickle.load(handle)

    data_dict = {"fr_qsets": {}, "junk": {}, "stats": {}, "p": {}}
    try:
        # no cut versions
        print_verb(verbose, "ebv, max, salt, no cuts")
        data_dict = compare_multiple_sets(data_dict, cuts={}, verbose=verbose)
        print_verb(verbose, "ebv, max, salt, no cuts, optical only")
        data_dict = compare_multiple_sets(
            data_dict, cuts={}, optical_only=True, verbose=verbose
        )
        for d in data_dict.values():
            for key in list(d.keys()):  # get around dict size changing
                d[key + "_no_cut"] = d.pop(key)

        # cut versions
        print_verb(verbose, "ebv, max, salt, standard cuts")
        data_dict = compare_multiple_sets(data_dict, cuts="standard", verbose=verbose)
        print_verb(verbose, "ebv, max, salt, standard cuts, optical only")
        data_dict = compare_multiple_sets(
            data_dict, cuts="standard", optical_only=True, verbose=verbose
        )

        # common targets
        print_verb(verbose, "comparing ebv, max, salt set intersections")
        data_dict = compare_set_intersection(
            data_dict, compare=models, suffix="common", verbose=verbose
        )
        for model in models:
            print_verb(verbose, f"comparing {model} NIR vs optical only intersections")
            data_dict = compare_set_intersection(
                data_dict,
                compare=(model, f"{model}_no_J"),
                suffix="common_NIRvO",
                match_optical=True,
                verbose=verbose,
            )

        # DEHVILS comparison fr_qsets
        data_dict = get_DEHVILS_fr_qsets(data_dict)[0]
        data_dict = get_DEHVILS_comparison(data_dict)

        # Detecting Outliers
        outliers = data_dict["outliers"] = {"div_model": {}, "UNITY": {}}
        print_verb(verbose, "getting divergent model outliers")
        outliers["div_model"]["main"], outliers["div_model"]["main_dists"] = (
            get_divergent_inferences_outliers(data_dict["p"])
        )

        for model in models:
            (
                outliers["div_model"][f"{model}_NIRvO"],
                outliers["div_model"][f"{model}_NIRvO_dists"],
            ) = get_divergent_inferences_NIRvO_outliers(
                data_dict["p"],
                compare=(f"{model}_common_NIRvO", f"{model}_no_J_common_NIRvO"),
            )
        for qset_name, unity_dir in zip(
            ("salt", "salt_common_NIRvO", "salt_no_J_common_NIRvO"),
            ("salt", "salt_J", "salt_no_J"),
        ):
            if unity_force or not os.path.exists(
                f"{constants.HSF_DIR}/unity_runs/{unity_dir}"
            ):
                os.makedirs(f"{constants.HSF_DIR}/unity_runs", exist_ok=True)
                data_dict["fr_qsets"][qset_name].run_unity(
                    storage_dir="f{constants.HSF_DIR}/unity_runs/{unity_dir}"
                )
        for unity_dir in ("salt", "salt_J", "salt_no_J"):
            outliers["UNITY"][unity_dir] = get_unity_run_placeholders(
                unity_dir=unity_dir
            )[1]
        data_dict["outliers"]["main"] = np.append(
            data_dict["outliers"]["UNITY"]["salt"],
            data_dict["outliers"]["div_model"]["main"],
        )

        # Final
        print_verb(verbose, "Calculating final dispersions")
        for suffix in ("", "_common_NIRvO", "_no_J_common_NIRvO"):
            data_dict = cut_outliers(data_dict, suffix=suffix)
        print_verb(verbose, "Calculating final dispersions for common sets")
        data_dict = compare_set_intersection(
            data_dict,
            compare=("ebv_final", "salt_final", "max_final"),
            suffix="common",
        )
        return data_dict
    except Exception as e:
        print(str(e))
        return data_dict


def get_placeholders(data_dict=None, verbose=False):
    try:
        if data_dict is None:
            data_dict = get_data_dict(verbose=verbose)
        placeholders = {}
        placeholders["misc"] = {
            "ATLASEdgeBuffer": 40,
            "ATLASAxisRatioLim": constants.ATLAS_AXIS_RATIO_LIM,
            "ATLASSkyLim": constants.ATLAS_SKY_LIM,
            "Phot1D3BadOffset": 0.02,
            # "Phot1D3BadStd": 0.826,
            "Phot1D3BadStd": 0.83,
            "Phot1D3Offset": 0.01,
            "Phot1D3Std": 0.087,
            "Phot2DOffset": 0.008,
            "Phot2DStd": 0.070,
            "PhotRotOffset": 0.030,
            "PhotRotStd": 0.075,
            "coAlpha": 0.155,
            "coBeta": 3.300,
            "coYJHAlpha": 0.138,
            "coYJHBeta": 3.702,
            "YJHAlpha": 0.111,
            "YJHBeta": 2.475,
        }
        print_verb(verbose, "Getting demographic placeholders")
        placeholders["demographics"] = get_demographic_placeholders()
        print_verb(verbose, "Getting redshift placeholders")
        placeholders["redshifts"] = get_redshift_placeholders()
        print_verb(verbose, "Getting 1D3 vs 2D mix-model placeholders")
        placeholders["1D3v2D"] = get_1D3_vs_2D_placeholders()[0]
        print_verb(verbose, "Getting csp comparison placeholders")
        placeholders["CSP"] = get_csp_placeholders()
        print_verb(verbose, "Getting DEHVILS comparison placeholders")
        placeholders["DEHVILS"] = get_DEHVILS_comparison_placeholders(data_dict)[0]
        print_verb(verbose, "Getting first cut placeholders")
        placeholders["first_cuts"], running = get_p1_cut_placeholders(data_dict["junk"])
        print_verb(verbose, "Getting second cut placeholders")
        placeholders["second_cuts"] = get_p2_cut_placeholders(running, data_dict)
        print_verb(verbose, "Getting divergent model outliers and placeholders")
        placeholders["div_inferences"] = get_divergent_inferences_placeholders(
            data_dict["stats"], data_dict["outliers"]["div_model"]["main_dists"]
        )
        for model in models:
            placeholders[f"{model}_NIRvO_div_model"] = (
                get_divergent_inferences_NIRvO_placeholders(
                    data_dict["stats"],
                    data_dict["outliers"]["div_model"][f"{model}_NIRvO_dists"],
                    compare=(f"{model}_common_NIRvO", f"{model}_no_J_common_NIRvO"),
                    std_cutoff=5,
                )
            )
        print_verb(verbose, "Getting shape placeholders")
        placeholders["shape"] = get_shape_placeholders(
            data_dict["p"], outlier_names=data_dict["outliers"]["div_model"]["main"]
        )[0]
        print_verb(verbose, "Getting color placeholders")
        placeholders["color"] = get_color_placeholders(
            data_dict["p"], outlier_names=data_dict["outliers"]["div_model"]["main"]
        )[0]
        print_verb(verbose, "Getting unity placeholders")
        placeholders["UNITY"] = get_unity_placeholders(data_dict)
        print_verb(verbose, "Getting stats placeholders without outliers")
        for suffix in (
            "_final",
            "_common_NIRvO_final",
            "_no_J_common_NIRvO_final",
            "_final_common",
        ):
            placeholders["final" + suffix] = get_third_cut_placeholders(
                data_dict["p"], data_dict["stats"], suffix=suffix
            )
        print_verb(verbose, "Getting bootstrapped NIR vs Optical stats placeholders")
        placeholders["NIRvO_stats"] = get_nir_optical_placeholders()

        m = data_dict["fr_qsets"]["max_final"].values_list(
            "target__TNS_name", flat=True
        )
        e = data_dict["fr_qsets"]["ebv_final"].values_list(
            "target__TNS_name", flat=True
        )
        s = data_dict["fr_qsets"]["salt_final"].values_list(
            "target__TNS_name", flat=True
        )
        placeholders["NumUnionFinal"] = len(set(np.append(m, np.append(e, s))))
        print_verb(verbose, "Getting residual trend placeholders")
        placeholders["residual_trend"] = get_residual_trend_placeholders(
            data_dict["p"], data_dict["stats"]
        )
        return placeholders
    except Exception as e:
        print_verb(verbose, "Oopsies, dumping out")
        print(str(e))
        return placeholders


def write_placeholders(placeholders, force=True):
    if force:
        with open(
            "/data/users/ado/papers_git/HSF_survey_paper/placeholders.txt", "w"
        ) as f:
            for key, val in placeholders.items():
                if isinstance(val, dict):
                    for key2, val2 in val.items():
                        f.write(f"{key2} {val2}\n")
                else:
                    f.write(f"{key} {val}\n")
        return
    for val in placeholders.values():
        if isinstance(val, dict):
            for key, val2 in val.items():
                placeholders[key] = val2
    with open("/data/users/ado/papers_git/HSF_survey_paper/placeholders.txt", "r") as f:
        lines = f.readlines()
    keys = [line.split()[0] for line in lines]
    with open("/data/users/ado/papers_git/HSF_survey_paper/placeholders.txt", "w") as f:
        for line in lines:
            if line.split()[0] not in placeholders:
                f.write(line)
            else:
                f.write(f"{line.split()[0]} {placeholders[line.split()[0]]}\n")
        for key, val in placeholders.items():
            if key not in keys:
                f.write(f"{key} {val}\n")


def get_demographic_placeholders():
    qset = Target.objects.number_of_observations()
    demographics, placeholders = {}, {}
    for obj in qset:
        if obj.sn_type.name not in demographics:
            demographics[obj.sn_type.name] = 0
        demographics[obj.sn_type.name] += 1
    for key in demographics:
        if key in ("reference", "CALSPEC"):
            continue
        num = demographics[key]
        for char in " -/[]":
            key = key.replace(char, "")
        placeholders[key + "Count"] = num
    return placeholders


def get_redshift_placeholders():
    from django.db.models import Q

    from galaxies.models import Galaxy

    _, z, _, _, r, helio_rv, leda_v, _, _ = sdss_weightedCC_systematics()
    sdss = (z * constants.C + helio_rv - leda_v)[~np.isnan(leda_v)][
        r[~np.isnan(leda_v)] > 5
    ]
    dz = compare_snifs_leda()[0]
    observed_set = Target.objects.filter(
        Q(focas_spectra__isnull=False) | Q(snifs_spectra__isnull=False)
    )
    placeholders = {
        "SNIFS+LEDA": Galaxy.objects.filter(
            leda_v__isnull=False, snifs_entries__isnull=False
        )
        .distinct()
        .count(),
        "FOCAS+LEDA": Galaxy.objects.filter(
            leda_v__isnull=False, focas_entries__isnull=False
        )
        .distinct()
        .count(),
        "NumSDSSSpectra": len(z[~np.isnan(leda_v)]),
        "AveDiffSDSS+LEDA": int(np.round(np.average(sdss))),
        "StdSDSS+LEDA": int(np.round(np.std(sdss))),
        "GoodSNIFS+LEDA": Galaxy.objects.filter(
            leda_v__isnull=False, snifs_entries__z_flag="s"
        )
        .distinct()
        .count(),
        "AveDiffSNIFS+LEDA": int(np.round(np.average(dz[np.abs(dz) < 150]))),
        "StdSNIFS+LEDA": int(np.round(np.std(dz[np.abs(dz) < 150]))),
        "BadAveDiffSNIFS+LEDA": int(np.round(np.average(dz[np.abs(dz) < 1e3]))),
        "BadStdSNIFS+LEDA": int(np.round(np.std(dz[np.abs(dz) < 1e3]))),
        "NumGalaxyObserved": observed_set.distinct().count(),
        "NumNewGalaxy": observed_set.filter(
            galaxy__z_flag__startswith="s", galaxy__leda_v__isnull=True
        )
        .distinct()
        .count(),
        "NumRedundantGalaxy": observed_set.exclude(galaxy__leda_v__isnull=True)
        .distinct()
        .count(),
    }
    return placeholders


def get_csp_placeholders():
    from csp import get_tripp_params

    _, _, _, _, params, errors, params_no_mass, errors_no_mass = get_tripp_params()
    placeholders = {}
    differences_mass = []
    differences_no_mass = []
    for pname, pkey, value, error in zip(
        ("P0", "P1", "P2", "Beta", "SigmaInt", "VPec"),
        ("P0", "P1", "P2", "beta", "sig_int", "v_pec"),
        (-18.633, -0.37, 0.61, 0.36, 0.11, 336),
        (0.062, 0.12, 0.32, 0.1, 0, 0),
    ):
        placeholders[f"CSPTripp{pname}"] = f"{np.round(params[pkey], 3):.3f}"
        placeholders[f"CSPTrippErr{pname}"] = f"{np.round(errors[pkey], 3):.3f}"[2:]
        placeholders[f"CSPTripp{pname}NoMass"] = (
            f"{np.round(params_no_mass[pkey], 3):.3f}"
        )
        placeholders[f"CSPTrippErr{pname}NoMass"] = (
            f"{np.round(errors_no_mass[pkey], 3):.3f}"[2:]
        )
        differences_mass.append(
            np.abs(params[pkey] - value) / np.sqrt(errors[pkey] ** 2 + error**2)
        )
        differences_no_mass.append(
            np.abs(params_no_mass[pkey] - value)
            / np.sqrt(errors_no_mass[pkey] ** 2 + error**2)
        )
    placeholders["CSPTrippVPec"] = int(params["v_pec"])
    placeholders["CSPTrippErrVPec"] = int(errors["v_pec"])
    placeholders["CSPTrippVPecNoMass"] = int(params_no_mass["v_pec"])
    placeholders["CSPTrippErrVPecNoMass"] = int(errors_no_mass["v_pec"])
    placeholders["CSPTrippAlpha"] = f"{np.round(params['alpha'], 3):.3f}"
    placeholders["CSPTrippErrAlpha"] = f"{np.round(errors['alpha'], 3):.3f}"[2:]
    differences_mass.append(
        np.abs(params["alpha"] - (-0.056)) / np.sqrt(errors["alpha"] ** 2 + 0.029**2)
    )
    placeholders["CSPTrippDiff"] = f"{np.round(np.average(differences_mass), 3):.3f}"
    placeholders["CSPTrippDiffNoMass"] = (
        f"{np.round(np.average(differences_no_mass), 3):.3f}"
    )
    return placeholders


def get_1D3_vs_2D_placeholders(dm=None, fit=None):
    from snippets import background_subtraction, compare_1D3_2D

    if dm is None:
        dm = compare_1D3_2D()[2]
    idx = np.where((np.array(dm["1D3"]) < 0.5) & (np.array(dm["1D3"]) > -0.5))
    alpha = 3
    beta = 0.3
    input_dict = {
        "N": len(dm["1D3"]),
        "mag_diff": dm["1D3"],
        "alpha": alpha,
        "beta": beta,
    }
    if fit is None:
        fit = background_subtraction(input_dict)
    placeholders = {
        "1D3v2DTightCount": len(idx[0]),
        "1D3v2DTotalCount": len(dm["1D3"]),
        "1D3v2DMixRatioAlpha": alpha,
        "1D3v2DMixRatioBeta": beta,
        "1D3v2DInPercent": f"{np.round(np.median(fit['theta']) * 100, 1):3.1f}",
        "1D3v2DInPercentErr": f"{np.round(np.std(fit['theta']) * 100, 1):3.1f}",
        "1D3v2DOutPercent": f"{np.round((1 - np.median(fit['theta'])) * 100, 1):3.1f}",
        "1D3v2DMuIn": f"{np.round(np.median(fit['mu_in']), 2):.2f}",
        "1D3v2DErrMuIn": f"{np.round(np.std(fit['mu_in']), 3):.3f}",
        "1D3v2DSigmaIn": f"{np.round(np.median(fit['sigma_in']), 2):.2f}",
        "1D3v2DErrSigmaIn": f"{np.round(np.std(fit['sigma_in']), 3):.3f}",
        "1D3v2DMuOut": f"{np.round(np.median(fit['mu_out']), 2):.2f}",
        "1D3v2DErrMuOut": f"{np.round(np.std(fit['mu_out']), 3):.3f}",
        "1D3v2DSigmaOut": f"{np.round(np.median(fit['sigma_out']), 2):.2f}",
        "1D3v2DErrSigmaOut": f"{np.round(np.std(fit['sigma_out']), 3):.3f}",
    }
    return placeholders, fit, dm


def get_p1_cut_placeholders(junk):
    placeholders = {}
    running = Target.objects.number_of_observations()
    placeholders["NumObserved"] = running.count()
    placeholders["NumObservations"] = (
        Observation.objects.filter(path__contains="/")
        .exclude(program__contains="J")
        .count()
        // 2
    )
    for key in ("NumObserved", "NumObservations"):
        if len(str(placeholders[key])) > 3:
            placeholders[key] = f"{placeholders[key][:-3]},{placeholders[key][-3:]}"

    unknown = Target.objects.get_by_type("?").number_of_observations()
    running = running.exclude(pk__in=unknown)
    placeholders["NumUnknown"] = unknown.count()
    placeholders["NumClassified"] = running.count()

    ias = Target.objects.get_by_type("Ia").number_of_observations()
    non_ias = running.exclude(pk__in=ias)
    running = ias
    placeholders["NumBad"] = non_ias.count()
    placeholders["NumIas"] = running.count()

    good_z = running.with_good_host_z()
    bad_z = running.exclude(pk__in=good_z)
    running = good_z
    placeholders["NumBadZ"] = bad_z.count()
    placeholders["NumGoodZ"] = running.count()

    mwebv = running.filter(mwebv__gt=constants.STANDARD_CUTS["mwebv"] / 0.86)
    running = running.exclude(pk__in=mwebv)
    placeholders["MWEBVCut"] = constants.STANDARD_CUTS["mwebv"]
    placeholders["NumMWEBVCut"] = mwebv.count()
    placeholders["NumAfterMWEBVCut"] = running.count()

    min_obs_names = []
    for obj in running:
        total = 0
        for lc in obj.lightcurves.exclude(source__in=("model", "UKIRT")):
            total += lc.detections().count() / (1 + 3 * (lc.bandpass in "oc"))
            if total >= constants.STANDARD_CUTS["min_obs_num"]:
                break
        for variant in ("2D", "0D", "1D3"):
            J_lc = obj.lightcurves.filter(source="UKIRT", bandpass="J", variant=variant)
            for lc in J_lc:
                total += lc.detections().count()
                break  # avoid double counting multiple variants
        if total < constants.STANDARD_CUTS["min_obs_num"]:
            min_obs_names.append(obj.TNS_name)
    running = running.exclude(pk__in=min_obs_names)
    placeholders["MinObsNum"] = constants.STANDARD_CUTS["min_obs_num"]
    placeholders["MinObsCut"] = len(min_obs_names)
    placeholders["NumAfterMinObsNum"] = running.count()

    need_subaru = junk["salt_no_cut"]["no_z"].filter(pk__in=running)
    running = running.exclude(pk__in=need_subaru)
    placeholders["NeedSubaruReduction"] = need_subaru.count()
    placeholders["NumAfterNoSubaru"] = running.count()

    failed_phot = junk["salt_no_cut"]["missing_bps"].filter(pk__in=running)
    running = running.exclude(pk__in=junk["salt_no_cut"]["missing_bps"])
    placeholders["NumFailedPhotometry"] = failed_phot.count()
    placeholders["NumAfterFirstCut"] = running.count()
    return placeholders, running


def get_p2_cut_placeholders(running, data_dict):
    fr_qsets = data_dict["fr_qsets"]
    p = data_dict["p"]
    stats = data_dict["stats"]
    junk = data_dict["junk"]
    placeholders = {
        "x1Cut": constants.STANDARD_SALT_CUTS["x1"],
        "Sigmax1Cut": constants.STANDARD_SALT_CUTS["e_x1"],
        "cCut": constants.STANDARD_SALT_CUTS["c"],
        "SigmacCut": constants.STANDARD_SALT_CUTS["e_c"],
        "STFloor": constants.STANDARD_CUTS["st"][0],
        "STCeil": constants.STANDARD_CUTS["st"][1],
        "SigmaSTCut": constants.STANDARD_CUTS["e_st"],
        "EBVHostCut": constants.STANDARD_CUTS["EBVhost"],
        "Phase1First": constants.ACCEPTABLE_PHASE_COVERAGES[0][0],
        "Phase1Range": constants.ACCEPTABLE_PHASE_COVERAGES[0][2],
        "Phase2First": constants.ACCEPTABLE_PHASE_COVERAGES[1][0],
        "Phase2Range": constants.ACCEPTABLE_PHASE_COVERAGES[1][2],
        "PhaseLast": constants.ACCEPTABLE_PHASE_COVERAGES[0][1],
    }

    ebv_params_and_keys = (
        ("st", "e_st", "EBVhost", "phase", "chi2_reduced_data_model"),
        (
            "EBVNumSTCut",
            "EBVNumSigmaSTCut",
            "EBVNumEBVHostCut",
            "EBVNumPhaseCut",
            "EBVNumChi2Cut",
        ),
        (
            "EBVNumAfterST",
            "EBVNumAfterSigmaST",
            "EBVNumAfterEBVHost",
            "EBVAfterPhase",
            "EBVAfterChi2",
        ),
    )
    max_params_and_keys = (
        ("BP", "st", "e_st", "phase", "chi2_reduced_data_model"),
        (
            "MaxNumBPCut",
            "MaxNumSTCut",
            "MaxNumSigmaSTCut",
            "MaxNumPhaseCut",
            "MaxNumChi2Cut",
        ),
        (
            "MaxNumAfterBP",
            "MaxNumAfterST",
            "MaxNumAfterSigmaST",
            "MaxAfterPhase",
            "MaxAfterChi2",
        ),
    )
    salt_params_and_keys = (
        ("x1", "e_x1", "c", "e_c", "phase", "chi2_reduced_data_model"),
        (
            "Numx1Cut",
            "NumSigmax1Cut",
            "NumcCut",
            "NumSigmacCut",
            "SALTNumPhaseCut",
            "SALTNumChi2Cut",
        ),
        (
            "NumAfterx1",
            "NumAfterSigmax1",
            "NumAfterc",
            "NumAfterSigmac",
            "SALTAfterPhase",
            "SALTAfterChi2",
        ),
    )
    for suffix in ("", "_no_J"):
        second_cut_placeholders = {}
        for cap_name, params_and_keys in zip(
            ("EBV", "Max", "SALT"),
            (ebv_params_and_keys, max_params_and_keys, salt_params_and_keys),
        ):
            model = cap_name.lower()
            failed_fit = junk[model]["no_successful_fit"]
            model_running = running.exclude(pk__in=failed_fit)
            second_cut_placeholders[f"{cap_name}NumNoFit"] = failed_fit.count()
            second_cut_placeholders[f"NumSuccessful{cap_name}"] = model_running.count()
            for param, cut_key, after_key in zip(*params_and_keys):
                cut_set = Target.objects.filter(TNS_name="empty_qset")
                if param in junk[model + suffix]["cut"]:
                    cut_set = junk[model + suffix]["cut"][param].filter(
                        pk__in=model_running
                    )
                    # using cut_set here hangs indefinitely
                    # something about circular references probably
                    model_running = model_running.exclude(
                        pk__in=junk[model + suffix]["cut"][param]
                    )
                elif param == "BP":
                    cut_set = junk["max" + suffix]["missing_bps"].exclude(
                        pk__in=junk["ebv" + suffix]["missing_bps"]
                    )
                    model_running = model_running.exclude(
                        pk__in=junk[model + suffix]["missing_bps"].exclude(
                            pk__in=junk["ebv" + suffix]["missing_bps"]
                        )
                    )
                second_cut_placeholders[cut_key] = cut_set.count()
                second_cut_placeholders[after_key] = model_running.count()

            if suffix == "_no_J":
                ph_suffix = "NoJ"
                second_cut_placeholders[f"{cap_name}AfterBPMatch"] = fr_qsets[
                    f"{model}_no_J_common_NIRvO"
                ].count()
                second_cut_placeholders[f"{cap_name}NumBPMatch"] = (
                    model_running.count()
                    - fr_qsets[f"{model}_no_J_common_NIRvO"].count()
                )
            else:
                ph_suffix = ""
            for key in list(second_cut_placeholders.keys()):
                placeholders[key + ph_suffix] = second_cut_placeholders[key]
    return placeholders


def get_shape_placeholders(p, outlier_names=[]):
    good_idx = [True for i in p["ebv_common"]["TNS_name"]]
    names = p["ebv_common"]["TNS_name"]
    for n in outlier_names:
        if n not in names:
            continue
        good_idx[np.where(names == n)[0][0]] = False
    shape_ebv = p["ebv_common"]["st"] - 1
    shape_max = p["max_common"]["st"] - 1
    shape_salt = p["salt_common"]["x1"]
    e_shape_ebv = p["ebv_common"]["e_st"]
    e_shape_max = p["max_common"]["e_st"]
    e_shape_salt = p["salt_common"]["e_x1"]
    snpy_corr = pg.corr(shape_ebv, shape_max)["r"][0]
    shape_snpy = np.average(
        np.array([shape_ebv, shape_max]),
        weights=(1 / e_shape_ebv**2, 1 / e_shape_max**2),
        axis=0,
    )
    e_shape_snpy = np.sqrt(
        e_shape_ebv**2 + e_shape_max**2 - 2 * e_shape_ebv * e_shape_max * snpy_corr
    )
    shape_data = RealData(
        shape_snpy[good_idx],
        shape_salt[good_idx],
        sx=e_shape_snpy[good_idx],
        sy=e_shape_salt[good_idx],
    )
    shape_odr = {}
    shape_out = {}
    n = len(shape_snpy)
    placeholders = {}
    for poly, i, color in zip(
        ("Linear", "Quadratic", "Cubic"), range(1, 4), ("red", "blue", "green")
    ):
        shape_odr[poly] = ODR(shape_data, polynomial(i))
        shape_out[poly] = shape_odr[poly].run()
        placeholders[f"ShapeBIC{poly}"] = (
            f"{np.round(n * np.log(shape_out[poly].sum_square / n) + (i + 1) * np.log(n), 1):3.1f}"
        )

    placeholders["ShapeConstant"] = f"{np.round(shape_out['Cubic'].beta[0], 2):.2f}"
    placeholders["ShapeLinear"] = f"{np.round(shape_out['Cubic'].beta[1], 2):1.2f}"
    placeholders["ShapeQuad"] = f"{np.round(shape_out['Cubic'].beta[2], 2):1.2f}"
    placeholders["ShapeCubic"] = f"{np.round(shape_out['Cubic'].beta[3], 2):1.2f}"
    placeholders["ShapeConstantErr"] = (
        f"{np.round(shape_out['Cubic'].sd_beta[0], 2):1.2f}"[2:]
    )
    placeholders["ShapeLinearErr"] = (
        f"{np.round(shape_out['Cubic'].sd_beta[1], 2):.2f}"[2:]
    )
    placeholders["ShapeQuadErr"] = f"{np.round(shape_out['Cubic'].sd_beta[2], 2):.2f}"[
        2:
    ]
    placeholders["ShapeCubicErr"] = f"{np.round(shape_out['Cubic'].sd_beta[3], 2):.2f}"[
        2:
    ]
    return placeholders, shape_out, shape_snpy, shape_salt


def get_color_placeholders(p, outlier_names=[]):
    good_idx = [True for i in p["ebv_common"]["TNS_name"]]
    names = p["ebv_common"]["TNS_name"]
    for n in outlier_names:
        if n not in names:
            continue
        good_idx[np.where(names == n)[0][0]] = False
    color_ebv = p["ebv_common"]["EBVhost"]
    color_max = p["max_common"]["Vmax"] - p["max_common"]["Jmax"]
    color_salt = p["salt_common"]["c"]
    e_color_ebv = p["ebv_common"]["e_EBVhost"]
    max_corr = pg.corr(p["max_common"]["Vmax"], p["max_common"]["Jmax"])["r"][0]
    e_color_max = np.sqrt(
        p["max_common"]["e_Vmax"] ** 2
        + p["max_common"]["e_Jmax"] ** 2
        - 2 * p["max_common"]["e_Vmax"] * p["max_common"]["e_Jmax"] * max_corr
    )
    e_color_salt = p["salt_common"]["e_c"]

    color_ebv_data = RealData(
        color_ebv[good_idx],
        color_salt[good_idx],
        sx=e_color_ebv[good_idx],
        sy=e_color_salt[good_idx],
    )
    color_max_data = RealData(
        color_max[good_idx],
        color_salt[good_idx],
        sx=e_color_max[good_idx],
        sy=e_color_salt[good_idx],
    )
    color_ebv_odr, color_ebv_out, color_max_odr, color_max_out = [{} for i in range(4)]
    n = len(color_ebv)
    placeholders = {}
    for poly, i, color in zip(
        ("Linear", "Quadratic", "Cubic"), range(1, 4), ("black", "red", "green")
    ):
        color_ebv_odr[poly] = ODR(color_ebv_data, polynomial(i))
        color_ebv_out[poly] = color_ebv_odr[poly].run()
        color_max_odr[poly] = ODR(color_max_data, polynomial(i))
        color_max_out[poly] = color_max_odr[poly].run()
        placeholders[f"ColorEBVBIC{poly}"] = (
            f"{np.round(n * np.log(color_ebv_out[poly].sum_square / n) + (i + 1) * np.log(n), 1):3.1f}"
        )
        placeholders[f"ColorMaxBIC{poly}"] = (
            f"{np.round(n * np.log(color_max_out[poly].sum_square / n) + (i + 1) * np.log(n), 1):3.1f}"
        )
    placeholders["EBVColorConstant"] = (
        f"{np.round(color_ebv_out['Linear'].beta[0], 2):.2f}"
    )
    placeholders["EBVColorConstantErr"] = (
        f"{np.round(color_ebv_out['Linear'].sd_beta[0], 2):.2f}"[2:]
    )
    placeholders["EBVColorLinear"] = (
        f"{np.round(color_ebv_out['Linear'].beta[1], 2):.2f}"
    )
    placeholders["EBVColorLinearErr"] = (
        f"{np.round(color_ebv_out['Linear'].sd_beta[1], 2):.2f}"[2:]
    )
    placeholders["MaxColorConstant"] = (
        f"{np.round(color_max_out['Cubic'].beta[0], 2):.2f}"
    )
    placeholders["MaxColorLinear"] = (
        f"{np.round(color_max_out['Cubic'].beta[1], 2):.2f}"
    )
    placeholders["MaxColorQuadratic"] = (
        f"{np.round(color_max_out['Cubic'].beta[2], 2):.2f}"
    )
    placeholders["MaxColorCubic"] = f"{np.round(color_max_out['Cubic'].beta[3], 2):.2f}"
    placeholders["MaxColorConstantErr"] = (
        f"{np.round(color_max_out['Cubic'].sd_beta[0], 2):.2f}"[2:]
    )
    placeholders["MaxColorLinearErr"] = (
        f"{np.round(color_max_out['Cubic'].sd_beta[1], 2):.2f}"[2:]
    )
    placeholders["MaxColorQuadraticErr"] = (
        f"{np.round(color_max_out['Cubic'].sd_beta[2], 2):.2f}"[2:]
    )
    placeholders["MaxColorCubicErr"] = (
        f"{np.round(color_max_out['Cubic'].sd_beta[3], 2):.2f}"[2:]
    )
    return placeholders, color_ebv_out, color_max_out, color_ebv, color_max, color_salt


def get_divergent_inferences_outliers(p, std_cutoff=5, outlier_names=[]):
    dt = np.std(
        (p["ebv_common"]["Tmax"], p["max_common"]["Tmax"], p["salt_common"]["t0"]),
        axis=0,
    )
    _, shape_out, shape_snpy, shape_salt = get_shape_placeholders(
        p, outlier_names=outlier_names
    )
    (
        _,
        color_ebv_out,
        color_max_out,
        color_ebv,
        color_max,
        color_salt,
    ) = get_color_placeholders(p, outlier_names=outlier_names)
    dshape = np.std(
        (np.poly1d(shape_out["Cubic"].beta[::-1])(shape_snpy), shape_salt), axis=0
    )
    dcolor = np.std(
        (
            np.poly1d(color_ebv_out["Linear"].beta[::-1])(color_ebv),
            np.poly1d(color_max_out["Cubic"].beta[::-1])(color_max),
            color_salt,
        ),
        axis=0,
    )
    dists = utils.mahalanobis_distances(
        np.array([dt, dshape, dcolor]).T, cov="bootstrap"
    )

    new_outlier_names = p["salt_common"]["TNS_name"][
        np.where(dists > std_cutoff * np.std(dists))
    ]
    return new_outlier_names, dists


def get_divergent_inferences_placeholders(stats, dists, std_cutoff=5):
    N = sum(dists > std_cutoff * np.std(dists))
    placeholders = {
        "NumAllCuts": stats["ebv_common"]["N"],
        "NumModelOutl": sum(dists > std_cutoff * np.std(dists)),
        "EBVNumAfterOutl": stats["ebv"]["N"] - N,
        "MaxNumAfterOutl": stats["max"]["N"] - N,
        "SALTNumAfterOutl": stats["salt"]["N"] - N,
        "ModelStdCutoff": std_cutoff,
    }
    return placeholders


def get_divergent_inferences_NIRvO_outliers(
    p,
    std_cutoff=5,
    compare=("ebv_common_NIRvO", "ebv_no_J_common_NIRvO"),
):
    if compare[0].startswith("ebv"):
        t = "Tmax"
        shape = "st"
        color = "EBVhost"
    elif compare[0].startswith("max"):
        t = "Tmax"
        shape = "st"
    elif compare[0].startswith("salt"):
        t = "t0"
        shape = "x1"
        color = "c"
    t_J = p[compare[0]][t]
    t_no_J = p[compare[1]][t]
    shape_J = p[compare[0]][shape]
    shape_no_J = p[compare[1]][shape]
    if compare[0].startswith("max"):
        color_J = p[compare[0]]["Vmax"] - p[compare[0]]["rmax"]
        color_no_J = p[compare[1]]["Vmax"] - p[compare[1]]["rmax"]
    else:
        color_J = p[compare[0]][color]
        color_no_J = p[compare[1]][color]
    dt = t_J - t_no_J
    dshape = shape_J - shape_no_J
    dcolor = color_J - color_no_J
    dists = utils.mahalanobis_distances(
        np.array([dt, dshape, dcolor]).T, origin=True, cov="bootstrap"
    )

    outlier_names = p[compare[0]]["TNS_name"][
        np.where(dists > std_cutoff * np.std(dists))
    ]
    return outlier_names, dists


def get_divergent_inferences_NIRvO_placeholders(
    stats, dists, std_cutoff=5, compare=("ebv_common_NIRvO", "ebv_no_J_common_NIRvO")
):
    placeholders = {
        f"{compare[0].split('_')[0]}NumAllCutsNIRvO": stats[compare[0]]["N"],
        f"{compare[0].split('_')[0]}NumModelOutlNIRvO": sum(
            dists > std_cutoff * np.std(dists)
        ),
        f"{compare[0].split('_')[0]}NumAfterOutlNIRvO": stats[compare[0]]["N"]
        - sum(dists > std_cutoff * np.std(dists)),
    }
    return placeholders


def get_unity_run_placeholders(unity_dir="salt"):
    import gzip
    import pickle

    unity = pd.read_csv(
        f"{constants.HSF_DIR}/unity_runs/{unity_dir}/params.out",
        delimiter="\s+",
        index_col="param_names",
    )
    with gzip.open(
        f"{constants.HSF_DIR}/unity_runs/{unity_dir}/inputs_UKIRT.pickle",
        "rb",
    ) as handle:
        input_data = pickle.load(handle)[0]
    sn_inputs = pd.read_csv(
        f"{constants.HSF_DIR}/unity_runs/{unity_dir}/sn_input.txt",
        delimiter="\s+",
    )
    outl = unity["mean"][unity.index.str.startswith("outl_loglike_by_SN")]
    inl = unity["mean"][unity.index.str.startswith("inl_loglike_by_SN")]
    likely_outl_idx = np.where(inl.values - outl.values < 0)
    # Not sure why, but UNITY might lose some targets.
    # This ensures that the indexing is accurate
    UNITY_outlier_names = np.array(
        [path.split("/")[-3] for path in input_data["snpaths"]]
    )[likely_outl_idx]
    placeholders = {
        f"{unity_dir}UNITYOutlFrac": f"{np.round(unity.T['outl_frac']['mean'], 3):.3f}",
        f"{unity_dir}UNITYOutlFracSD": f"{np.round(unity.T['outl_frac']['sd'], 3):.3f}",
        f"{unity_dir}UNITYOutlNum": len(UNITY_outlier_names),
    }
    return placeholders, UNITY_outlier_names


def get_unity_placeholders(data_dict):
    placeholders = {}
    for unity_dir in ("salt", "salt_J", "salt_no_J"):
        run_placeholders = get_unity_run_placeholders(unity_dir=unity_dir)[0]
        for key, val in run_placeholders.items():
            placeholders[key] = val
        if unity_dir == "salt":
            div_dir = "main"
        else:
            div_dir = "salt_NIRvO"
        for TNS_name in data_dict["outliers"]["UNITY"][unity_dir]:
            if TNS_name in data_dict["outliers"]["div_model"][div_dir]:
                placeholders[f"{unity_dir}UNITYOutlNum"] -= 1
    for cap_name, model in zip(("EBV", "Max", "SALT"), models):
        placeholders[f"{cap_name}UNITYOutlNumFinal"] = data_dict["junk"][
            f"{model}_final"
        ]["cut"]["outliers_UNITY"].count()
        placeholders[f"{cap_name}UNITYOutlNumNoJFinal"] = data_dict["junk"][
            f"{model}_no_J_common_NIRvO_final"
        ]["cut"]["outliers_UNITY"].count()
    return placeholders


def cut_outliers(data_dict, suffix, force=False):
    import copy

    from django.db.models import Q

    fr_qsets = data_dict["fr_qsets"]
    junk = data_dict["junk"]
    outliers = data_dict["outliers"]
    suffix_f = suffix + "_final"
    max_bandpass = "J"
    if "_no_J" in suffix:
        max_bandpass = "V"
    NIRvO = suffix.endswith("NIRvO")
    if suffix == "_J":
        suffix = ""
    placeholders = {}
    for cap_name, model in zip(("EBV", "Max", "SALT"), models):
        if model + suffix_f in fr_qsets and not force:
            continue
        if NIRvO:
            div_model_outliers = outliers["div_model"][model + "_NIRvO"]
            UNITY_outliers = np.array(
                [
                    name
                    for name in np.append(
                        outliers["UNITY"]["salt_J"], outliers["UNITY"]["salt_no_J"]
                    )
                    if name not in div_model_outliers
                    and name
                    in fr_qsets[model + suffix].values_list(
                        "target__TNS_name", flat=True
                    )
                ]
            )
        else:
            div_model_outliers = outliers["div_model"]["main"]
            UNITY_outliers = np.array(
                [
                    name
                    for name in outliers["UNITY"]["salt"]
                    if name not in div_model_outliers
                    and name
                    in fr_qsets[model + suffix].values_list(
                        "target__TNS_name", flat=True
                    )
                ]
            )
        fr_qsets[model + suffix_f] = (
            fr_qsets[model + suffix]
            .exclude(target__TNS_name__in=np.append(div_model_outliers, UNITY_outliers))
            .order_by("target__TNS_name")
        )
        data_dict["stats"][model + suffix_f], data_dict["p"][model + suffix_f] = (
            fr_qsets[model + suffix_f].get_dispersion(
                sigma=-1, max_bandpass=max_bandpass
            )
        )
        junk[model + suffix_f] = copy.deepcopy(junk[model + suffix])
        junk[model + suffix_f]["cut"]["outliers_div_model"] = Target.objects.filter(
            TNS_name__in=div_model_outliers
        )
        junk[model + suffix_f]["cut"]["outliers_UNITY"] = Target.objects.filter(
            TNS_name__in=UNITY_outliers
        )
        junk[model + suffix_f]["cut"]["any"] = Target.objects.filter(
            Q(pk__in=junk[model + suffix]["cut"]["any"])
            | Q(TNS_name__in=np.append(div_model_outliers, UNITY_outliers))
        )
    return data_dict


def get_third_cut_placeholders(
    p,
    stats,
    suffix="",
):
    placeholders = {}
    for cap_name, model in zip(("EBV", "Max", "SALT"), models):
        placeholders[f"{cap_name}Num"] = stats[model + suffix]["N"]
        boot = utils.bootstrap_fn(
            np.array(
                [p[f"{model}{suffix}"]["resid_DM"], p[f"{model}{suffix}"]["e_DM"]]
            ),
            [utils.NMAD, utils.RMS, utils.WRMS, utils.sigma_int],
        )
        boot["SigmaInt"] = boot.pop("sigma_int")
        for estimator in ("NMAD", "RMS", "WRMS", "SigmaInt"):
            placeholders[f"{cap_name}{estimator}"] = (
                f"{np.round(np.average(boot[estimator]), 3):.3f}"
            )
            placeholders[f"{cap_name}{estimator}Err"] = (
                f"{np.round(np.std(boot[estimator]), 3):.3f}"[2:]
            )
    if suffix == "_no_J_common_NIRvO_final":
        ph_suffix = "NoJFinal"
    elif suffix == "_common_NIRvO_final":
        ph_suffix = "JFinal"
    elif suffix == "_final_common":
        ph_suffix = "Common"
    else:
        ph_suffix = "Final"
    for key in list(placeholders.keys()):
        placeholders[key + ph_suffix] = placeholders.pop(key)
    return placeholders


def get_residual_trend_placeholders(p, stats):
    placeholders = {}
    x, y, dy = [{} for i in range(3)]
    x["diff"], y["diff"], dy["diff"] = prep_data_trend(
        p, stats, "ebv_final", True, "salt_final"
    )
    for key, sample in zip(
        ("EBV", "Max", "SALT", "Diff"), ("ebv_final", "max_final", "salt_final", "diff")
    ):
        if sample not in x:
            x[sample] = p[sample]["z_cmb"]
            y[sample] = p[sample]["resid_DM"]
            dy[sample] = p[sample]["e_DM"]
        res = utils.line_fit(x[sample], y[sample], dy[sample], 0, plots=0)
        placeholders[f"HR{key}Slope"] = f"{np.round(res['slope'][0], 1):2.1f}"
        placeholders[f"HR{key}SlopeErr"] = f"{np.round(res['slope'][1], 1):.1f}"[2:]
        placeholders[f"HR{key}Const"] = f"{np.round(res['intercept'][0], 2):3.2f}"
        placeholders[f"HR{key}ConstErr"] = f"{np.round(res['intercept'][1], 2):.2f}"[2:]
    return placeholders


def bootstrap_nir_vs_optical(p, iter_num=5000):
    vals, actual = {}, {}
    for key in tqdm(("salt", "ebv", "max"), position=0, desc="model"):
        vals[key] = {"OJ": {}, "O": {}}
        actual[key] = {"OJ": {}, "O": {}}

        for bp, p_bp in tqdm(
            zip(
                ("OJ", "O"),
                (p[f"{key}_common_NIRvO_final"], p[f"{key}_no_J_common_NIRvO_final"]),
            ),
            position=1,
            desc="bp",
        ):
            N = len(p_bp["DM"])
            resid = p_bp["resid_DM"]
            errs = p_bp["e_DM"]
            actual[key][bp] = {
                "RMS": utils.RMS(resid),
                "WRMS": utils.WRMS(resid, errs),
                "NMAD": utils.NMAD(resid),
                "sigma_int": utils.sigma_int(resid, errs),
            }
            for estimator in ("RMS", "WRMS", "sigma_int", "NMAD"):
                vals[key][bp][estimator] = np.zeros(iter_num)
            for i in tqdm(range(iter_num), position=2, desc="bootstrap"):
                idx = np.random.randint(N, size=(N))
                vals[key][bp]["RMS"][i] = utils.RMS(resid[idx])
                vals[key][bp]["WRMS"][i] = utils.WRMS(resid[idx], errs[idx])
                vals[key][bp]["NMAD"][i] = utils.NMAD(resid[idx])
                vals[key][bp]["sigma_int"][i] = utils.sigma_int(resid[idx], errs[idx])
    return vals, actual


def get_nir_optical_placeholders(vals_dict=None, actual_dict=None):
    if vals_dict is None and actual_dict is None:
        import pickle

        with open(f"{constants.HSF_DIR}/ado/vals_actual.pickle", "rb") as handle:
            vals_dict, actual_dict = pickle.load(handle)
    placeholders = {}
    for key in vals_dict:
        if "salt" in key.lower():
            model = "SALT"
        elif "ebv" in key.lower():
            model = "EBV"
        elif "max" in key.lower():
            model = "Max"
        for estimator in ("RMS", "sigma_int", "WRMS", "NMAD"):
            placeholders[f"{model}{estimator}JFinal"] = (
                f"{np.round(np.average(vals_dict[key]['OJ'][estimator]), 3):.3f}"
            )
            placeholders[f"{model}{estimator}NoJFinal"] = (
                f"{np.round(np.average(vals_dict[key]['O'][estimator]), 3):.3f}"
            )
            placeholders[f"{model}{estimator}ErrJFinal"] = (
                f"{np.round(np.std(vals_dict[key]['OJ'][estimator]), 3):.3f}"[2:]
            )
            placeholders[f"{model}{estimator}ErrNoJFinal"] = (
                f"{np.round(np.std(vals_dict[key]['O'][estimator]), 3):.3f}"[2:]
            )

            # num = sum(vals_dict[key][estimator] < actual_dict[key]["J"][estimator])
            # placeholders[f"Num{model}{estimator}"] = num
            # placeholders[f"p{model}{estimator}"] = np.round(
            #     (1 + num) / len(vals_dict[key][estimator]), 3
            # )
    return placeholders


def get_DEHVILS_fr_qsets(
    data_dict=None,
    x1_cut=(-3, 3),
    e_x1_cut=1,
    e_t0_cut=2,
    st_cut=(0.65, 1.4),
    mwebv_cut=0.2,
    fit_prob_cut=0.01,
    redlaw="F19",
    ebv_cal=6,
):
    from glob import glob

    if data_dict is None:
        data_dict = {"fr_qsets": {}, "junk": {}}
        fr_qsets = data_dict["fr_qsets"]
    elif not (
        isinstance(data_dict, dict) and "fr_qsets" in data_dict and "junk" in data_dict
    ):
        raise KeyError(
            "data_dict must be None or a dict with keys 'fr_qsets' and 'junk'"
        )
    fr_qsets = data_dict["fr_qsets"]
    junk = data_dict["junk"]
    TNS_names, dehvils_zs, dehvils_e_zs = [[] for _ in range(3)]
    for path in glob(f"{constants.HSF_DIR}/ado/DEHVILSDR1/*.snana.dat"):
        TNS_names.append(path.split("UKIRT_20")[1].split(".")[0])
        with open(path, "r") as f:
            for line in f.readlines():
                if line.startswith("REDSHIFT_HELIO"):
                    dehvils_zs.append(float(line.split()[1]))
                    dehvils_e_zs.append(float(line.split()[3]))
                    break
    # DEHVILS_fit_prob_cut = ("20jdo", "20jfc", "20jht", "20jio", "20jsa", "20kav", "20kbw", "20kcr", "20kku", "20kpx", "20kru", "20kyx", "20kzn", "20lil", "20mbf", "20mby", "20mdd", "20naj", "20nbo", "20ned", "20nef", "20nst", "20nta", "20qne", "20rlj", "20sme", "20tfc", "20tkp", "20uea", "20uek", "20unl", "20wcj", "20wtq", "20ysl", "21bbz", "21fof", "21glz", "21mim", "21zfq", "21zfs",)
    # low_precision_z = ("20nbo", "20vnr", "21fof", "21lug", "20wtq",)  # redshift not spec
    # optical_only_x1_cut = ( "20jio", "20kcr", "20ned", "20nta", "20naj", "20qne", "20uea", "21glz", "21zfq",)  # |x_1| >= 3 with ATLAs c+o fit.
    underluminous = ("20jsa", "20rlj", "20unl", "21mim")
    bt_like = ("20naj", "20sme", "20mbf", "20tkp")  # 06bt-like
    peculiar = ("20kzn",)
    qset = (
        Target.objects.filter(TNS_name__in=TNS_names)
        .filter(sn_type__name="SN Ia")
        .exclude(TNS_name__in=underluminous)
        .exclude(TNS_name__in=bt_like)
        .exclude(TNS_name__in=peculiar)
        # .exclude(TNS_name__in=optical_only_x1_cut)
        # .exclude(TNS_name__in=low_precision_z)
    )
    acceptable = ["g", "b", "n", "?"]
    names = ("SALT", "EBV", "Max")
    model_names = ("salt3-nir", "snpy_ebv_model2", "snpy_max_model")
    calibrations = (None, ebv_cal, None)
    for name, model_name, calibration in zip(names, model_names, calibrations):
        if name == "SALT":
            cuts = {
                "mwebv": mwebv_cut / 0.86,
                "x1": x1_cut,
                "e_x1": e_x1_cut,
                "e_t0": e_t0_cut,
            }
        else:
            cuts = {
                "mwebv": 2 / 0.86,
                "st": st_cut,
                "e_st": e_x1_cut / 8,
                "e_Tmax": e_t0_cut,
            }
        # For comparing the models with co, coYJH, YJH using DEHVILS photometry
        for bps, variants, excl_bps, max_bandpass, max_color_1, max_color_2 in zip(
            ("co", "coYJH", "YJH"),
            ([], ["dehvils"], ["dehvils"]),
            (["Y", "J", "H"], [], ["c", "o"]),
            ("V", "J", "J"),
            ("V", "V", "Y"),
            ("r", "r", "J"),
        ):
            fr_qsets[f"{name}{bps}"], junk[f"{name}{bps}"] = qset.get_fitresults_qset(
                bandpasses=[bp for bp in bps],
                variants=variants,
                model_name=model_name,
                calibration=calibration,
                redlaw=redlaw,
                optional_args={"bandpasses": []},
                excluded_args={"bandpasses": excl_bps},
                acceptable_status=acceptable,
                cuts=cuts,
                max_bandpass=max_bandpass,
                max_color_1=max_color_1,
                max_color_2=max_color_2,
            )

        # For comparing the modesl with coJ using DEHVILS and HSF photometry
        fr_qsets[f"{name}DEHVILS"], junk[f"{name}DEHVILS"] = qset.get_fitresults_qset(
            bandpasses=["c", "o", "J"],
            variants=["dehvils"],
            model_name=model_name,
            optional_args={"bandpasses": []},
            acceptable_status=acceptable,
            cuts=cuts,
            calibration=calibration,
            redlaw=redlaw,
        )
        fr_qsets[f"{name}HSF"], junk[f"{name}HSF"] = qset.get_fitresults_qset(
            bandpasses=["c", "o", "J"],
            variants=[],
            model_name=model_name,
            optional_args={"bandpasses": [], "variants": ["2D", "0D", "1D3"]},
            acceptable_status=acceptable,
            cuts=cuts,
            calibration=calibration,
            redlaw=redlaw,
        )
        junk[f"{name}DEHVILS"]["not_in_HSF"] = Target.objects.filter(
            TNS_name__in=fr_qsets[f"{name}DEHVILS"]
            .values_list("target", flat=True)
            .exclude(
                target__TNS_name__in=fr_qsets[f"{name}HSF"].values_list(
                    "target", flat=True
                )
            )
        )
        junk[f"{name}HSF"]["not_in_DEHVILS"] = Target.objects.filter(
            TNS_name__in=fr_qsets[f"{name}HSF"]
            .values_list("target", flat=True)
            .exclude(
                target__TNS_name__in=fr_qsets[f"{name}DEHVILS"].values_list(
                    "target", flat=True
                )
            )
        )
        fr_qsets[f"{name}DEHVILS"] = (
            fr_qsets[f"{name}DEHVILS"]
            .filter(
                target__TNS_name__in=fr_qsets[f"{name}HSF"].values_list(
                    "target", flat=True
                )
            )
            .order_by("target__TNS_name")
        )
        fr_qsets[f"{name}HSF"] = (
            fr_qsets[f"{name}HSF"]
            .filter(
                target__TNS_name__in=fr_qsets[f"{name}DEHVILS"].values_list(
                    "target", flat=True
                )
            )
            .order_by("target__TNS_name")
        )
    return data_dict, TNS_names, dehvils_zs, dehvils_e_zs


def compare_DEHVILS_samples(data_dict=None, redlaws=("F99", "F19")):
    from scipy.special import gammainc

    if data_dict is None:
        data_dict = {"fr_qsets": {}, "junk": {}, "p": {}}
        for redlaw in redlaws:
            fr_qsets, junk = get_DEHVILS_fr_qsets(redlaw=redlaw)
            for key in list(fr_qsets.keys()):
                data_dict["fr_qsets"][key + "_" + redlaw] = fr_qsets.pop(key)
                data_dict["junk"][key + "_" + redlaw] = junk.pop(key)
        for key in data_dict["fr_qsets"]:
            # required for compare_set_intersection
            data_dict["p"][key] = {
                "TNS_name": np.array(
                    data_dict["fr_qsets"][key].values_list(
                        "target__TNS_name", flat=True
                    )
                )
            }
    for bps in ("co", "coYJH", "YJH"):
        default_max_bandpass = "J"
        if bps == "co":
            default_max_bandpass = "V"
        compare_list = []
        for name in ("SALT", "EBV", "Max"):
            for redlaw in redlaws:
                compare_list.append(f"{name}{bps}_{redlaw}")
        data_dict = compare_set_intersection(
            data_dict,
            compare=compare_list,
            default_max_bandpass=default_max_bandpass,
            fr_qsets_only=True,
        )
    fit_prob, rchi2 = {}, {}
    chi2_keys = ("total", "total_data", "total_data_model")
    for qset_key in data_dict["fr_qsets"]:
        fit_prob[qset_key] = {}
        rchi2[qset_key] = {}
        for chi2_key in chi2_keys:
            fit_prob[qset_key][chi2_key] = []
            rchi2[qset_key][chi2_key] = []
            for fr in data_dict["fr_qsets"][qset_key]:
                if chi2_key not in fr.chi2:
                    fr.chi2 = fr.get_chi2()
                    fr.save()
                fit_prob[qset_key][chi2_key].append(
                    1 - gammainc(fr.chi2["ndof"] / 2, fr.chi2[chi2_key] / 2)
                )
                rchi2[qset_key][chi2_key].append(
                    fr.chi2[chi2_key.replace("total", "reduced")]
                )
    return data_dict, fit_prob, rchi2


def get_DEHVILS_comparison(
    data_dict=None,
    alpha_beta="UNITY",
    redlaw="F19",
    N=47,
    ebv_cal=6,
    cut="rchi2",
):
    from scipy.special import gammainc
    from django.db.models import Q

    if alpha_beta not in ("dehvils", "solve", "UNITY"):
        raise ValueError(
            """alpha_beta should be in 'dehvils', 'solve', 'UNITY'
        dehvils is to use DEHVILS alpha and beta values
        solve is to solve for the alpha and beta values that minimize dispersion
        UNITY is to use UNITY alpha and beta values
        """
        )
    names = ("SALT", "EBV", "Max")
    model_names = ("salt3-nir", "snpy_ebv_model2", "snpy_max_model")
    calibrations = (None, ebv_cal, None)
    bps = ("co", "coYJH", "YJH")
    if data_dict is None:
        data_dict = get_DEHVILS_fr_qsets(redlaw=redlaw)[0]
    elif not (
        isinstance(data_dict, dict) and "fr_qsets" in data_dict and "junk" in data_dict
    ):
        raise KeyError(
            "data_dict must be None or a dict with keys 'fr_qsets' and 'junk'"
        )
    else:
        for name in names:
            for bp in bps:
                if f"{name}{bp}" not in data_dict["fr_qsets"]:
                    raise KeyError(
                        f"data_dict does not contain the DEHVILS fr_qsets {name}{bp}. Consider using data_dict=None or getting a data_dict with get_DEHVILS_fr_qsets()"
                    )
    for key in ("stats", "p"):
        if key not in data_dict:
            data_dict[key] = {}
    fr_qsets = data_dict["fr_qsets"]
    junk = data_dict["junk"]
    stats = data_dict["stats"]
    p = data_dict["p"]
    fit_prob, rchi2, good_targets = [{} for _ in range(3)]

    print(f"Cutting based on {cut} values")
    for name, model_name, calibration in zip(names, model_names, calibrations):
        # cuts to compare methods
        for bp in bps:
            key = f"{name}{bp}"
            count = fr_qsets[key].count()
            fit_prob[key] = np.zeros(count)
            rchi2[key] = np.zeros(count)
            for i, fr in enumerate(fr_qsets[key]):
                if "total_data_model" not in fr.chi2:
                    fr.chi2 = fr.get_chi2()
                    fr.save()
                fit_prob[key][i] = 1 - gammainc(
                    fr.chi2["ndof"] / 2, fr.chi2["total_data_model"] / 2
                )
                rchi2[key][i] = fr.chi2["reduced_data_model"]
        fit_prob[f"{name}cut"] = np.sort(fit_prob[name + "coYJH"])[-N]
        rchi2[f"{name}cut"] = np.sort(rchi2[name + "coYJH"])[N - 1]
        for bp in bps:
            key = f"{name}{bp}"
            if "chi2" in cut:
                idx = np.where(rchi2[key] <= rchi2[f"{name}cut"])
            elif cut == "fit_prob":
                idx = np.where(fit_prob[key] >= fit_prob[f"{name}cut"])
            good_targets[key] = np.array(
                fr_qsets[key].values_list("target", flat=True)
            )[idx]

            variants = ""
            if "YJH" in bp:
                variants = "dehvils"
            junk[key]["cut"][cut] = (
                fr_qsets[key]
                .values_list("target", flat=True)
                .exclude(target__TNS_name__in=good_targets[key])
            )
            junk[key]["cut"]["any"] = Target.objects.filter(
                Q(TNS_name__in=junk[key]["cut"][cut])
                | Q(TNS_name__in=junk[key]["cut"]["any"])
            )
            fr_qsets[key] = fr_qsets[key].filter(
                target__TNS_name__in=good_targets[key],
            )
            alpha = "default"
            beta = "default"
            max_bandpass = "J"
            max_color_1 = "V"
            max_color_2 = "r"
            if bp == "co":
                if alpha_beta == "dehvils":
                    alpha = 0.145
                    beta = 2.359
                elif alpha_beta == "UNITY":
                    alpha = 0.168
                    beta = 3.758
                max_bandpass = "V"
            elif bp == "coYJH":
                if alpha_beta == "dehvils":
                    alpha = 0.075
                    beta = 2.903
                elif alpha_beta == "UNITY":
                    alpha = 0.176
                    beta = 2.579
            elif bp == "YJH":
                if alpha_beta == "dehvils":
                    alpha = 0
                    beta = 0
                elif alpha_beta == "UNITY":
                    # alpha = 0.023
                    # beta = 0.141
                    alpha = "solve"
                    beta = "solve"
                max_color_1 = "Y"
                max_color_2 = "J"
            if alpha_beta == "solve":
                alpha, beta = "solve", "solve"
            stats[key], p[key] = fr_qsets[key].get_dispersion(
                sigma=-1,
                alpha=alpha,
                beta=beta,
                max_bandpass=max_bandpass,
                max_color_1=max_color_1,
                max_color_2=max_color_2,
            )

        # cuts to compare photometry
        for survey in ("DEHVILS", "HSF"):
            key = f"{name}{survey}"
            phot_count = fr_qsets[key].count()
            fit_prob[key] = np.zeros(phot_count)
            rchi2[key] = np.zeros(phot_count)
            for i, fr in enumerate(fr_qsets[key]):
                if "total_data_model" not in fr.chi2:
                    fr.chi2 = fr.get_chi2()
                    fr.save()
                fit_prob[key][i] = 1 - gammainc(
                    fr.chi2["ndof"] / 2, fr.chi2["total_data_model"] / 2
                )
                rchi2[key][i] = fr.chi2["reduced_data_model"]
        if "chi2" in cut:
            idx = np.where(
                (rchi2[f"{name}DEHVILS"] <= rchi2[f"{name}cut"])
                & (rchi2[f"{name}HSF"] <= rchi2[f"{name}cut"])
            )
        elif cut == "fit_prob":
            idx = np.where(
                (fit_prob[f"{name}DEHVILS"] >= fit_prob[f"{name}cut"])
                & (fit_prob[f"{name}HSF"] >= fit_prob[f"{name}cut"])
            )
        for survey in ("DEHVILS", "HSF"):
            key = f"{name}{survey}"
            good_targets[key] = np.array(
                fr_qsets[key].values_list("target", flat=True)
            )[idx]

            # for survey in ("DEHVILS", "HSF"):
            junk[key]["cut"][cut] = (
                fr_qsets[key]
                .values_list("target", flat=True)
                .exclude(target__TNS_name__in=good_targets[key])
            )
            junk[key]["cut"]["any"] = Target.objects.filter(
                Q(TNS_name__in=junk[key]["cut"][cut])
                | Q(TNS_name__in=junk[key]["cut"]["any"])
            )
            fr_qsets[key] = fr_qsets[key].filter(
                target__TNS_name__in=good_targets[key],
            )
            alpha = "default"
            beta = "default"
            max_bandpass = "J"
            max_color_1 = "V"
            max_color_2 = "r"
            if alpha_beta == "dehvils":
                alpha = 0.145
                beta = 2.359
            elif alpha_beta == "UNITY":
                if survey == "DEHVILS":
                    alpha = 0.163
                    beta = 3.770
                elif survey == "HSF":
                    alpha = 0.147
                    beta = 3.551
            elif alpha_beta == "solve":
                alpha, beta = "solve", "solve"
            stats[key], p[key] = fr_qsets[key].get_dispersion(
                sigma=-1,
                alpha=alpha,
                beta=beta,
                max_bandpass=max_bandpass,
                max_color_1=max_color_1,
                max_color_2=max_color_2,
            )
    return data_dict


def get_DEHVILS_comparison_placeholders(data_dict):
    placeholders = {}
    varied_method_dict = {}
    fr_qsets = data_dict["fr_qsets"]
    stats = data_dict["stats"]
    p = data_dict["p"]
    # chi2 ratios
    for bps in ("co", "coYJH", "YJH"):
        placeholders[f"NSALT{bps}FITPROB"] = len(
            np.where(np.array(p[f"SALT{bps}"]["fitprob_data_model"]) > 0.01)[0]
        )
        ebv_qset = fr_qsets[f"EBV{bps}"].filter(
            target__TNS_name__in=fr_qsets[f"SALT{bps}"].values_list(
                "target__TNS_name", flat=True
            )
        )
        salt_qset = fr_qsets[f"SALT{bps}"].filter(
            target__TNS_name__in=fr_qsets[f"EBV{bps}"].values_list(
                "target__TNS_name", flat=True
            )
        )
        ebv = np.array([fr.chi2["reduced_data_model"] for fr in ebv_qset])
        ebv_no_model = np.array([fr.chi2["reduced_data"] for fr in ebv_qset])
        salt = np.array([fr.chi2["reduced_data_model"] for fr in salt_qset])
        salt_no_model = np.array([fr.chi2["reduced_data"] for fr in salt_qset])
        placeholders[f"MedianEBVToSALTChi2Ratio{bps}"] = np.round(
            np.median(ebv / salt), 2
        )
        placeholders[f"MedianEBVToSALTChi2RatioNoModel{bps}"] = np.round(
            np.median(ebv_no_model / salt_no_model), 2
        )
    for key in p:
        if not (key.endswith("co") or key.endswith("YJH")):
            continue
        print("bootstapping dispersion:", key)
        resids = p[key]["resid_DM"]
        varied_method_dict[key] = utils.bootstrap_fn(
            np.array([resids]), fns=[utils.NMAD, utils.STD]
        )
        for estimator, val in varied_method_dict[key].items():
            placeholders[f"Dvs{key}{estimator}"] = f"{np.round(np.median(val), 3):.3f}"
            placeholders[f"Dvs{key}{estimator}Err"] = f"{np.round(np.std(val), 3):.3f}"[
                2:
            ]
        placeholders[f"Dvs{key}Count"] = stats[key]["N"]
    # placeholders["SALTFitProbCut"] = f"{np.round(fit_prob['SALTcut'], 2):3.2f}"
    # placeholders["EBVFitProbCut"] = f"{np.round(fit_prob['EBVcut'], 2):3.2f}"
    # placeholders["MaxFitProbCut"] = f"{np.round(fit_prob['Maxcut'], 2):3.2f}"
    placeholders["SALTChi2Cut"] = (
        f"{np.round(max(p['SALTcoYJH']['rchi2_data_model']), 2):3.2f}"
    )
    placeholders["EBVChi2Cut"] = (
        f"{np.round(max(p['EBVcoYJH']['rchi2_data_model']), 2):3.2f}"
    )
    placeholders["MaxChi2Cut"] = (
        f"{np.round(max(p['MaxcoYJH']['rchi2_data_model']), 2):3.2f}"
    )
    print("Model Filters N NMAD (mag) STD (mag)")
    print("DEHVILS co 47 0.177(029) 0.221(043)")
    print("DEHVILS coYJH 47 0.132(025) 0.175(034)")
    print("DEHVILS YJH 47 0.139(026) 0.172(027)")
    for name, model_name in zip(
        ("EBV", "Max", "SALT"), ("EBV_model2", "max_model", "SALT3-NIR")
    ):
        for bp in ("co", "coYJH", "YJH"):
            s = [f"{model_name} {bp} "]
            s.append(str(stats[f"{name}{bp}"]["N"]) + " ")
            s.append(placeholders[f"Dvs{name}{bp}NMAD"])
            s.append("(" + placeholders[f"Dvs{name}{bp}NMADErr"] + ") ")
            s.append(placeholders[f"Dvs{name}{bp}STD"])
            s.append("(" + placeholders[f"Dvs{name}{bp}STDErr"] + ")")
            print("".join(s))

    varied_phot_dict = {}
    for key in p:
        if "DEHVILS" not in key and "HSF" not in key:
            continue
        print("bootstrapping dispersion:", key)
        resids = p[key]["resid_DM"]
        varied_phot_dict[key] = utils.bootstrap_fn(
            np.array([resids]), fns=[utils.NMAD, utils.STD]
        )
        for estimator, val in varied_phot_dict[key].items():
            placeholders[f"Dvs{key}{estimator}"] = f"{np.round(np.median(val), 3):.3f}"
            placeholders[f"Dvs{key}{estimator}Err"] = f"{np.round(np.std(val), 3):.3f}"[
                2:
            ]
        placeholders[f"Dvs{key}Count"] = stats[key]["N"]
    for survey in ("HSF", "DEHVILS"):
        for name in ("SALT", "EBV", "Max"):
            s = [f"{survey} {name} "]
            s.append(str(stats[f"{name}{survey}"]["N"]) + " ")
            s.append(placeholders[f"Dvs{name}{survey}NMAD"])
            s.append("(" + placeholders[f"Dvs{name}{survey}NMADErr"] + ") ")
            s.append(placeholders[f"Dvs{name}{survey}STD"])
            s.append("(" + placeholders[f"Dvs{name}{survey}STDErr"] + ")")
            print("".join(s))
    return placeholders, varied_method_dict, varied_phot_dict


def write_paper(
    refresh_placeholders=False,
    make_plots=False,
    data_dict=None,
    dists=None,
    outlier_names=[],
):
    with open("/data/users/ado/papers_git/HSF_survey_paper/src.tex", "rb") as f:
        text = f.read()
    placeholders = {}
    if refresh_placeholders:
        write_placeholders()
    with open("/data/users/ado/papers_git/HSF_survey_paper/placeholders.txt", "r") as f:
        for line in f.readlines():
            placeholders["PLACEHOLDER{" + line.split()[0] + "}"] = line.split()[1]
    text = text.decode("UTF-8")
    for key in placeholders:
        text = text.replace(key, placeholders[key]).replace("+ -", "-")
    text = text.encode()
    with open("/data/users/ado/papers_git/HSF_survey_paper/main.tex", "wb") as f:
        f.write(text)
    if make_plots:
        import plotting

        if data_dict is None:
            data_dict = get_data_dict()
        plotting.tns_hist()  # fig 2
        plotting.plot_1D3_2D()  # fig 4
        plotting.compare_z()  # fig 5
        plotting.compare_snifs_leda()  # fig 6
        for stat in ("NMAD", "STD"):
            plotting.hsf_dehvils_varied_method(stat=stat)  # fig 7, 8
            plotting.hsf_dehvils_varied_phot(stat=stat)  # fig 9, 10
        plotting.snpy_salt_shape(
            data_dict["p"], data_dict["outliers"]["main"]
        )  # fig 11
        plotting.snpy_salt_color(
            data_dict["p"], data_dict["outliers"]["main"]
        )  # fig 12
        plotting.divergent_inferences(
            data_dict["p"],
            data_dict["outliers"]["div_model"]["main_dists"],
            data_dict["outliers"]["div_model"]["main"],
        )  # fig 13
        for name, label in zip(
            ("ebv", "max", "salt"), ("SNPY_EBV", "SNPY_MAX", "SALT")
        ):
            plotting.hubble_diagram(  # fig 14, 15, 16
                data_dict["p"][f"{name}_final"],
                sigma_int=data_dict["stats"][f"{name}_final"]["sigma_int"],
                label=label,
            )
        plotting.residual_differences(  # fig 17
            data_dict["p"], data_dict["stats"], "ebv_final", True, "salt_final"
        )
        plotting.trend_vs_params()  # fig 18
        plotting.UNITY_true_vs_observed()  # fig 19
        plotting.eddington_bias_vs_redshift()  # fig 20
        plotting.beta_vs_redshift()  # fig 21
        plotting.mahalanobis_J_O(data_dict["p"])  # fig 22
        plotting.nir_vs_optical()  # fig 23
        # plotting.mollweide()
        plotting.plot_prior_sensitivity_analysis_mixing_ratio()  # Fig A.1
        plotting.plot_prior_sensitivity_analysis_inlier()  # Fig A.2


def appendix_phot_table():
    from data.models import CalspecSpectrum

    for cs in CalspecSpectrum.objects.filter(
        TNS_name__in=[
            "C26202",
            "HS2027+0651",
            "NGC2506-G31",
            "SDSS132811",
            "SDSSJ151421",
            "SF1615+001A",
            "SNAP-2",
            "VB8",
            "WD1657+343",
            "WD0947+857",
            "WD1026+453",
        ]
    ):
        mag = []
        for filt in ("ATgr_total", "ATri_total", "sdss_g", "ztfg_total", "ztfr_total"):
            mag.append(cs.synthetic_flux(filt, mag_sys="ab")[2])
        print(r"\hline")
        print(
            " & ".join(
                [
                    cs.TNS_name,
                ]
                + [str(np.round(m, 2)) for m in mag]
            ),
            r"\\",
        )
    print(r"\hline")


def appendix_calibration_table():
    path = "ado/calibrations.pickle"
    if os.path.exists(path):
        with open(path, "rb") as f:
            res, names, resids, resid_names = pickle.load(f)
    else:
        from snippets import compare_different_calibrations

        res, names, resids, resid_names = compare_different_calibrations()
    for i in range(16):
        cal_list = []
        for param, N in zip(
            (
                "total_model",
                "J_model",
                "c_model",
                "o_model",
                "ztfg_model",
                "ztfr_model",
            ),
            ("ndof", "J_N", "c_N", "o_N", "ztfg_N", "ztfr_N"),
        ):
            arr = res["chi2"][param][i] / res["chi2"][N][i]
            arr = arr[np.where(~np.isnan(arr))]
            cal_list.append(str(np.round(np.median(arr), 3)))
        print(i, " & ", " & ".join(cal_list))


if __name__ == "__main__":
    write_paper()


def dispersion_vs_cuts(fr_qsets, p, iter_num=5000, force=False):
    import os

    if (
        os.path.exists(f"{constants.HSF_DIR}/ado/dispersion_vs_cuts.pickle")
        and not force
    ):
        import pickle

        with open(f"{constants.HSF_DIR}/ado/dispersion_vs_cuts.pickle", "rb") as handle:
            vals_dict = pickle.load(handle)
        return vals_dict
    estimators = ("RMS", "WRMS", "sigma_int", "NMAD")
    N = len(p["salt_final"]["DM"])
    parameters = ("x1", "e_x1", "e_t0", "mwebv", "rchi2")
    vals_dict = {}
    for param in parameters:
        vals_dict[param] = {}
        if param in ("x1", "e_x1", "e_t0", "rchi2"):
            x = np.abs(p["salt_final"][param])
        elif param == "mwebv":
            x = np.array([fr.target.mwebv * 0.86 for fr in fr_qsets["salt_final"]])
        vals_dict[param]["xvals"] = x[np.argsort(x)]
        for estimator in estimators:
            vals_dict[param][estimator] = np.zeros((N, iter_num))
        for i in tqdm(range(N)):
            idx = np.argsort(x)[: i + 1]
            v = utils.bootstrap_fn(
                np.array(
                    [p["salt_final"]["resid_DM"][idx], p["salt_final"]["e_DM"][idx]]
                ),
                fns=[utils.RMS, utils.WRMS, utils.sigma_int, utils.NMAD],
                iter_num=iter_num,
            )
            for estimator in estimators:
                vals_dict[param][estimator][i] = v[estimator]
    return vals_dict


def old_get_cut_placeholders(qset, fr_qsets, p, stats, junk):
    num_observed = Target.objects.number_of_observations().count()
    num_ias = Target.objects.get_by_type("Ia").number_of_observations().count()
    num_unknown = Target.objects.get_by_type("?").number_of_observations().count()
    min_obs_num = 0
    if "min_obs_num" in junk["salt"]["cut"]:
        min_obs_num = junk["salt"]["cut"]["min_obs_num"].count()
    mwebv_clip = qset.filter(mwebv__gt=0.3 / 0.86)
    mwebv = mwebv_clip.count()
    placeholders = {
        "NumObserved": num_observed,
        "NumIas": num_ias,
        "NumUnknown": num_unknown,
        "NumClassified": num_observed - num_unknown,
        "NumBad": num_observed - num_unknown - num_ias,
        "NumGoodZ": qset.count(),
        "NumBadZ": num_ias - qset.count(),
        "MWEBVCut": constants.STANDARD_CUTS["mwebv"],
        "NumMWEBVCut": mwebv,
        "NumAfterMWEBVCut": qset.count() - mwebv,
        "MinObsNum": constants.STANDARD_CUTS["min_obs_num"],
        "MinObsCut": min_obs_num,
        "NumAfterMinObsNum": qset.count() - mwebv - min_obs_num,
        "NeedSubaruReduction": junk["salt_no_cut"]["no_z"]
        .exclude(pk__in=mwebv_clip)
        .count(),
        "NumAfterNoSubaru": qset.count()
        - mwebv
        - min_obs_num
        - junk["salt_no_cut"]["no_z"].exclude(pk__in=mwebv_clip).count(),
        "NumFailedPhotometry": junk["salt_no_cut"]["missing_bps"]
        .exclude(pk__in=mwebv_clip)
        .count(),
        "NumAfterFirstCut": qset.count()
        - mwebv
        - min_obs_num
        - junk["salt_no_cut"]["no_z"].exclude(pk__in=mwebv_clip).count()
        - junk["salt_no_cut"]["missing_bps"].exclude(pk__in=mwebv_clip).count(),
        "x1Cut": constants.STANDARD_SALT_CUTS["x1"],
        "Sigmax1Cut": constants.STANDARD_SALT_CUTS["e_x1"],
        "cCut": constants.STANDARD_SALT_CUTS["c"],
        "SigmacCut": constants.STANDARD_SALT_CUTS["e_c"],
        "STFloor": constants.STANDARD_CUTS["st"][0],
        "STCeil": constants.STANDARD_CUTS["st"][1],
        "SigmaSTCut": constants.STANDARD_CUTS["e_st"],
        "EBVHostCut": constants.STANDARD_CUTS["EBVhost"],
        "Phase1First": constants.ACCEPTABLE_PHASE_COVERAGES[0][0],
        "Phase1Range": constants.ACCEPTABLE_PHASE_COVERAGES[0][2],
        "Phase2First": constants.ACCEPTABLE_PHASE_COVERAGES[1][0],
        "Phase2Range": constants.ACCEPTABLE_PHASE_COVERAGES[1][2],
        "PhaseLast": constants.ACCEPTABLE_PHASE_COVERAGES[0][1],
    }
    for suffix in ("", "_no_J"):
        cuts = {"salt": {}, "ebv": {}, "max": {}}
        for param in ("x1", "e_x1", "c", "e_c", "phase", "chi2_reduced_model"):
            cuts["salt"][param] = 0
            if param in junk["salt" + suffix]["cut"]:
                cuts["salt"][param] += junk["salt" + suffix]["cut"][param].count()
        for param in ("st", "e_st", "phase", "chi2_reduced_model"):
            cuts["ebv"][param] = 0
            cuts["max"][param] = 0
            if param in junk["ebv" + suffix]["cut"]:
                cuts["ebv"][param] += junk["ebv" + suffix]["cut"][param].count()
            if param in junk["max" + suffix]["cut"]:
                cuts["max"][param] += junk["max" + suffix]["cut"][param].count()
        cuts["ebv"]["EBVhost"] = 0
        if "EBVhost" in junk["ebv" + suffix]["cut"]:
            cuts["ebv"]["EBVhost"] += junk["ebv" + suffix]["cut"]["EBVhost"].count()
        # hacky because Target_qset.get_fit_results_qset makes junk, but doesn't
        # interact with the fitting bandpasses chosen and FitResults_qset.get_dispersion
        # doesn't interact with junk
        cuts["max"]["BP"] = (
            junk["max" + suffix]["missing_bps"]
            .exclude(pk__in=junk["ebv" + suffix]["missing_bps"])
            .count()
        )

        second_cut_placeholders = {
            "EBVNumNoFit": junk["ebv"]["no_successful_fit"].count(),
            "MaxNumNoFit": junk["max"]["no_successful_fit"].count(),
            "SALTNumNoFit": junk["salt"]["no_successful_fit"].count(),
            "NumSuccessfulEBV": int(placeholders["NumAfterFirstCut"])
            - junk["ebv"]["no_successful_fit"].count(),
            "NumSuccessfulMax": int(placeholders["NumAfterFirstCut"])
            - junk["max"]["no_successful_fit"].count(),
            "NumSuccessfulSALT": int(placeholders["NumAfterFirstCut"])
            - junk["salt"]["no_successful_fit"].count(),
            "Numx1Cut": cuts["salt"]["x1"],
            "NumSigmax1Cut": cuts["salt"]["e_x1"],
            "NumcCut": cuts["salt"]["c"],
            "NumSigmacCut": cuts["salt"]["e_c"],
            "SALTNumPhaseCut": cuts["salt"]["phase"],
            "SALTNumChi2Cut": cuts["salt"]["chi2_reduced_model"],
            "EBVNumSTCut": cuts["ebv"]["st"],
            "EBVNumSigmaSTCut": cuts["ebv"]["e_st"],
            "EBVNumEBVHostCut": cuts["ebv"]["EBVhost"],
            "EBVNumPhaseCut": cuts["ebv"]["phase"],
            "EBVNumChi2Cut": cuts["ebv"]["chi2_reduced_model"],
            "MaxNumBPCut": cuts["max"]["BP"],
            "MaxNumSTCut": cuts["max"]["st"],
            "MaxNumSigmaSTCut": cuts["max"]["e_st"],
            "MaxNumPhaseCut": cuts["max"]["phase"],
            "MaxNumChi2Cut": cuts["max"]["chi2_reduced_model"],
        }

        running_ebv_total = int(second_cut_placeholders["NumSuccessfulEBV"])
        running_max_total = int(second_cut_placeholders["NumSuccessfulMax"])
        running_salt_total = int(second_cut_placeholders["NumSuccessfulSALT"])
        for param, after_key in zip(
            ("st", "e_st", "EBVhost", "phase", "chi2_reduced_model"),
            (
                "EBVNumAfterST",
                "EBVNumAfterSigmaST",
                "EBVNumAfterEBVHost",
                "EBVAfterPhase",
                "EBVAfterChi2",
            ),
        ):
            running_ebv_total -= cuts["ebv"][param]
            second_cut_placeholders[after_key] = running_ebv_total
        for param, after_key in zip(
            ("BP", "st", "e_st", "phase", "chi2_reduced_model"),
            (
                "MaxNumAfterBP",
                "MaxNumAfterST",
                "MaxNumAfterSigmaST",
                "MaxAfterPhase",
                "MaxAfterChi2",
            ),
        ):
            running_max_total -= cuts["max"][param]
            second_cut_placeholders[after_key] = running_max_total
        for param, after_key in zip(
            ("x1", "e_x1", "c", "e_c", "phase", "chi2_reduced_model"),
            (
                "NumAfterx1",
                "NumAfterSigmax1",
                "NumAfterc",
                "NumAfterSigmac",
                "SALTAfterPhase",
                "SALTAfterChi2",
            ),
        ):
            running_salt_total -= cuts["salt"][param]
            second_cut_placeholders[after_key] = running_salt_total

        if suffix == "_no_J":
            ph_suffix = "NoJ"
            second_cut_placeholders["EBVAfterBPMatch"] = fr_qsets[
                "ebv_no_J_common_NIRvO"
            ].count()
            second_cut_placeholders["EBVNumBPMatch"] = (
                running_ebv_total - fr_qsets["ebv_no_J_common_NIRvO"].count()
            )
            second_cut_placeholders["MaxAfterBPMatch"] = fr_qsets[
                "max_no_J_common_NIRvO"
            ].count()
            second_cut_placeholders["MaxNumBPMatch"] = (
                running_max_total - fr_qsets["max_no_J_common_NIRvO"].count()
            )
            second_cut_placeholders["SALTAfterBPMatch"] = fr_qsets[
                "salt_no_J_common_NIRvO"
            ].count()
            second_cut_placeholders["SALTNumBPMatch"] = (
                running_salt_total - fr_qsets["salt_no_J_common_NIRvO"].count()
            )
        else:
            ph_suffix = ""
        for key in list(second_cut_placeholders.keys()):
            placeholders[key + ph_suffix] = second_cut_placeholders[key]
    return placeholders


def old_compare_nir_optical_sets(
    data_dict=None, cuts="standard", sigma=-1, redlaw="F19", verbose=False
):
    qset = Target.objects.get_by_type("Ia").number_of_observations().with_good_host_z()
    bandpasses = ["J"]
    variants = []
    acceptable = ["g", "?", "b", "n"]
    optional_args = {
        "bandpasses": ["ztfg", "ztfr", "c", "o"],
        "variants": ["2D", "0D", "1D3"],
    }
    excluded_args = {
        "bandpasses": ["Y", "H"],
        "variants": ["tphot", "dehvils", "rot", "ref", "1D3-2D"],
    }
    if cuts == "standard":
        cuts = constants.STANDARD_CUTS
    print_verb(verbose, "Grabbing default fit results")
    if data_dict is None:
        (
            fr_qsets,
            junk,
            stats,
            p,
        ) = [{} for i in range(4)]
    else:
        fr_qsets, junk, stats, p = data_dict
    for key, model_name, calibration in zip(
        ("salt", "ebv", "max"),
        ("salt3-nir", "snpy_ebv_model2", "snpy_max_model"),
        (None, 6, None),
    ):
        if key not in fr_qsets:
            print(f"{model_name} with J")
            fr_qsets[key + "_J"], junk[key + "_J"] = qset.get_fitresults_qset(
                model_name=model_name,
                bandpasses=bandpasses,
                variants=variants,
                calibration=calibration,
                redlaw=redlaw,
                acceptable_status=acceptable,
                cuts=cuts,
                optional_args=optional_args,
                excluded_args=excluded_args,
            )
        else:
            import copy

            fr_qsets[key + "_J"] = copy.deepcopy(fr_qsets[key])
            junk[key + "_J"] = copy.deepcopy(junk[key])
        print_verb(verbose, f"{model_name} without J")
        fr_qsets[key + "_no_J"], junk[key + "_no_J"] = qset.filter(
            TNS_name__in=fr_qsets[key + "_J"].values_list("target__TNS_name", flat=True)
        ).get_fitresults_qset(
            model_name=model_name,
            bandpasses=[],
            variants=[],
            calibration=calibration,
            redlaw=redlaw,
            acceptable_status=acceptable,
            max_bandpass="V",
            cuts=cuts,
            optional_args={"bandpasses": ["ztfg", "ztfr", "c", "o"]},
            excluded_args={"bandpasses": ["Y", "J", "H"]},
        )
    # force matching
    for key in ("salt", "ebv", "max"):
        fr_qsets[key + "_J"] = (
            fr_qsets[key + "_J"]
            .filter(
                target__TNS_name__in=fr_qsets[key + "_no_J"].values_list(
                    "target__TNS_name", flat=True
                )
            )
            .order_by("target__TNS_name")
        )
        fr_qsets[key + "_no_J"] = (
            fr_qsets[key + "_no_J"]
            .filter(
                target__TNS_name__in=fr_qsets[key + "_J"].values_list(
                    "target__TNS_name", flat=True
                )
            )
            .order_by("target__TNS_name")
        )
        junk[key + "_J"]["different_bp_strs"] = []
        junk[key + "_no_J"]["different_bp_strs"] = []
        for fr1, fr2 in zip(fr_qsets[key + "_J"], fr_qsets[key + "_no_J"]):
            if fr1.bandpasses_str.replace("J-", "") != fr2.bandpasses_str:
                junk[key + "_J"]["different_bp_strs"].append(fr1.pk)
                junk[key + "_J"]["different_bp_strs"].append(fr2.pk)
                junk[key + "_no_J"]["different_bp_strs"].append(fr1.pk)
                junk[key + "_no_J"]["different_bp_strs"].append(fr2.pk)
        fr_qsets[key + "_J"] = fr_qsets[key + "_J"].exclude(
            pk__in=junk[key + "_J"]["different_bp_strs"]
        )
        fr_qsets[key + "_no_J"] = fr_qsets[key + "_no_J"].exclude(
            pk__in=junk[key + "_no_J"]["different_bp_strs"]
        )

    print_verb(verbose, "Calculating dispersions")
    for key, fr_qset in fr_qsets.items():
        if key in stats:
            continue
        if "max" in key and "_no_J" in key:
            stats[key], p[key] = fr_qset.get_dispersion(sigma=sigma, max_bandpass="V")
        else:
            stats[key], p[key] = fr_qset.get_dispersion(sigma=sigma, max_bandpass="J")
        if "different_bp_strs" in junk[key]:
            junk[key]["different_bp_strs"] = FitResults.objects.filter(
                pk__in=junk[key]["different_bp_strs"]
            )

    return fr_qsets, junk, stats, p


def old_get_divergent_inferences_placeholders(
    fr_qsets, stats, p, std_cutoff=5, outlier_names=[]
):
    dt = np.std(
        (p["ebv_common"]["Tmax"], p["max_common"]["Tmax"], p["salt_common"]["t0"]),
        axis=0,
    )
    _, shape_out, shape_snpy, shape_salt = get_shape_placeholders(
        p, outlier_names=outlier_names
    )
    (
        _,
        color_ebv_out,
        color_max_out,
        color_ebv,
        color_max,
        color_salt,
    ) = get_color_placeholders(p, outlier_names=outlier_names)
    dshape = np.std(
        (np.poly1d(shape_out["Linear"].beta[::-1])(shape_snpy), shape_salt), axis=0
    )
    dcolor = np.std(
        (
            np.poly1d(color_ebv_out["Linear"].beta[::-1])(color_ebv),
            np.poly1d(color_max_out["Cubic"].beta[::-1])(color_max),
            color_salt,
        ),
        axis=0,
    )
    dists = utils.mahalanobis_distances(
        np.array([dt, dshape, dcolor]).T, cov="bootstrap"
    )

    new_outlier_names = p["salt_common"]["TNS_name"][
        np.where(dists > std_cutoff * np.std(dists))
    ]
    placeholders = {
        "NumAllCuts": stats["ebv_common"]["N"],
        "NumModelOutl": sum(dists > std_cutoff * np.std(dists)),
        "EBVNumAfterOutl": stats["ebv"]["N"] - sum(dists > std_cutoff * np.std(dists)),
        "MaxNumAfterOutl": stats["max"]["N"] - sum(dists > std_cutoff * np.std(dists)),
        "SALTNumAfterOutl": stats["salt"]["N"]
        - sum(dists > std_cutoff * np.std(dists)),
        "ModelStdCutoff": std_cutoff,
    }
    return placeholders, new_outlier_names, dists


def old_get_divergent_inferences_nir_vs_optical_placeholders(
    fr_qsets,
    stats,
    p,
    compare=("ebv_common_NIRvO", "ebv_no_J_common_NIRvO"),
    std_cutoff=5,
):
    placeholders = {}

    # for key in compare:
    #     if key.starswith('ebv') or key.startswith('max'):
    if compare[0].startswith("ebv"):
        t = "Tmax"
        shape = "st"
        color = "EBVhost"
    elif compare[0].startswith("max"):
        t = "Tmax"
        shape = "st"
    elif compare[0].startswith("salt"):
        t = "t0"
        shape = "x1"
        color = "c"
    t_J = p[compare[0]][t]
    t_no_J = p[compare[1]][t]
    shape_J = p[compare[0]][shape]
    shape_no_J = p[compare[1]][shape]
    if compare[0].startswith("max"):
        color_J = p[compare[0]]["Vmax"] - p[compare[0]]["rmax"]
        color_no_J = p[compare[1]]["Vmax"] - p[compare[1]]["rmax"]
    else:
        color_J = p[compare[0]][color]
        color_no_J = p[compare[1]][color]
    dt = t_J - t_no_J
    dshape = shape_J - shape_no_J
    dcolor = color_J - color_no_J
    dists = utils.mahalanobis_distances(
        np.array([dt, dshape, dcolor]).T, origin=True, cov="bootstrap"
    )

    outlier_names = p[compare[0]]["TNS_name"][
        np.where(dists > std_cutoff * np.std(dists))
    ]
    placeholders[f"{compare[0].split('_')[0]}NumAllCutsNIRvO"] = stats[compare[0]]["N"]
    placeholders[f"{compare[0].split('_')[0]}NumModelOutlNIRvO"] = sum(
        dists > std_cutoff * np.std(dists)
    )
    placeholders[f"{compare[0].split('_')[0]}NumAfterOutlNIRvO"] = stats[compare[0]][
        "N"
    ] - sum(dists > std_cutoff * np.std(dists))
    return placeholders, outlier_names


def old_compare_DEHVILS_methods(
    cut_method=3, alpha_beta="UNITY", redlaw="F19", N=47, ebv_cal=6
):
    if cut_method not in (1, 2, 3, 4, 5, None):
        raise ValueError(
            """cut_method should be 1, 2, 3, 4, 5, or None
        1 is to find the rchi2 that gives N targets in SALTcoYJH and apply that cut to each list.
        2 is to do 1 and then only use the common targets.
        3 is to order each list by rchi2 and pick the first N targets in each.
        4 is to do 3 and then only use the common targets
        5 is to order each list by rchi2 and pick the first N targets common to each.
        None is to not cut based on chi2
        """
        )
    if cut_method is not None:
        try:
            int(N)
        except Exception as e:
            print("N should be an integer unless cut_method is None")
            raise e
    if alpha_beta not in ("dehvils", "solve", "UNITY"):
        raise ValueError(
            """alpha_beta should be in 'dehvils', 'solve', 'UNITY'
        dehvils is to use DEHVILS alpha and beta values
        solve is to solve for the alpha and beta values that minimize dispersion
        UNITY is to use UNITY alpha and beta values
        """
        )
    names = ("SALT", "EBV", "Max")
    model_names = ("salt3-nir", "snpy_ebv_model2", "snpy_max_model")
    calibrations = (None, ebv_cal, None)
    bps = ("co", "coYJH", "YJH")
    fr_qsets, _ = get_DEHVILS_fr_qsets()

    def get_dispersions(fr_qsets, stats, p, alpha_beta):
        """Done once here and once more after filtering by chi2"""

        for name, model_name in zip(
            ("SALT", "EBV", "Max"), ("salt3-nir", "snpy_ebv_model2", "snpy_max_model")
        ):
            for bp in ("co", "coYJH", "YJH"):
                # for other_name in ("SALT", "EBV", "Max"):
                #     if other_name == name:
                #         continue  # avoids weird pointer issue that freezes things
                #     fr_qsets[f"{name}{bp}"] = fr_qsets[f"{name}{bp}"].filter(
                #         target__TNS_name__in=fr_qsets[f"{other_name}{bp}"].values_list(
                #             "target__TNS_name", flat=True
                #         )
                #     )
                alpha = "default"
                beta = "default"
                max_bandpass = "J"
                max_color_1 = "V"
                max_color_2 = "r"
                if bp == "co":
                    if alpha_beta == "dehvils":
                        alpha = 0.145
                        beta = 2.359
                    elif alpha_beta == "UNITY":
                        alpha = 0.1548
                        beta = 3.3004
                    max_bandpass = "V"
                elif bp == "coYJH":
                    if alpha_beta == "dehvils":
                        alpha = 0.075
                        beta = 2.903
                    elif alpha_beta == "UNITY":
                        alpha = 0.1375
                        beta = 3.7016
                elif bp == "YJH":
                    if alpha_beta == "dehvils":
                        alpha = 0
                        beta = 0
                    elif alpha_beta == "UNITY":
                        alpha = 0.11107
                        beta = 2.47493
                    max_color_1 = "Y"
                    max_color_2 = "J"
                if alpha_beta == "solve":
                    alpha, beta = "solve", "solve"
                stats[f"{name}{bp}"], p[f"{name}{bp}"] = fr_qsets[
                    f"{name}{bp}"
                ].get_dispersion(
                    sigma=-1,
                    alpha=alpha,
                    beta=beta,
                    max_bandpass=max_bandpass,
                    max_color_1=max_color_1,
                    max_color_2=max_color_2,
                )
        return stats, p

    stats, p = get_dispersions(fr_qsets, stats, p, alpha_beta)

    chi2cut = max([max(p[f"{name}coYJH"]["rchi2"]) for name in names])
    if cut_method is not None:
        print("Cutting based on chi2 values")
    if cut_method in (1, 2):
        chi2cut = np.sort(p["SALTcoYJH"]["rchi2"])[N - 1]
        for bp in bps:
            for name, model_name, calibration in zip(names, model_names, calibrations):
                sub_redlaw = default_redlaw(name, redlaw)
                variants = ""
                if "YJH" in bp:
                    variants = "dehvils"
                fr_qsets[f"{name}{bp}"] = FitResults.objects.filter(
                    success=True,
                    model_name=model_name,
                    bandpasses_str="-".join(sorted(bp)),
                    variants_str=variants,
                    calibration=calibration,
                    redlaw=sub_redlaw,
                    target__TNS_name__in=p[f"{name}{bp}"]["TNS_name"][
                        np.where(p[f"{name}{bp}"]["rchi2"] <= chi2cut)
                    ],
                )
    elif cut_method in (3, 4, 5):
        chi2cut = 0
        chi2_cuts = {}
        good_names = {}
        for bp in bps:
            for name in names:
                good_names[f"{name}{bp}"] = p[f"{name}{bp}"]["TNS_name"][
                    np.argsort(p[f"{name}{bp}"]["rchi2"])[:N]
                ]
                if bp == "coYJH":
                    chi2_cuts[name] = np.sort(p[f"{name}{bp}"]["rchi2"])[47]
            if cut_method == 5:
                # Getting the first N tagets common to all three samples.
                for i in range(len(good_names[f"SALT{bp}"])):
                    good_names["common{bp}"] = []
                    for sn_name in good_names[f"SALT{bp}"][: i + 1]:
                        if (
                            sn_name in good_names[f"EBV{bp}"][: i + 1]
                            and sn_name in good_names[f"Max{bp}"][: i + 1]
                        ):
                            good_names[f"common{bp}"].append(sn_name)
                    if len(good_names[f"common{bp}"]) >= N:
                        for name in names:
                            chi2cut = max(
                                chi2cut,
                                fr_qsets[f"{name}{bp}"]
                                .get(target__TNS_name=sn_name)
                                .chi2["reduced_model"],
                            )
                        break
            for name, model_name, calibration in zip(names, model_names, calibrations):
                sub_redlaw = default_redlaw(name, redlaw)
                good_names_filter = good_names[f"{name}{bp}"]
                if cut_method == 5:
                    good_names_filter = good_names[f"common{bp}"]
                variants = ""
                if "YJH" in bp:
                    variants = "dehvils"
                fr_qsets[f"{name}{bp}"] = FitResults.objects.filter(
                    success=True,
                    model_name=model_name,
                    bandpasses_str="-".join(sorted(bp)),
                    variants_str=variants,
                    calibration=calibration,
                    redlaw=sub_redlaw,
                    target__TNS_name__in=good_names_filter,
                )
    if cut_method in (2, 4):
        for bp in bps:
            for name in names:
                for other_name in names:
                    if other_name == name:
                        continue  # avoids weird pointer issue that freezes things
                    fr_qsets[f"{name}{bp}"] = fr_qsets[f"{name}{bp}"].filter(
                        target__TNS_name__in=fr_qsets[f"{other_name}{bp}"].values_list(
                            "target__TNS_name", flat=True
                        )
                    )

    if cut_method is not None:
        print("Getting dispersions on chi2 filtered sets")
        stats, p = get_dispersions(fr_qsets, stats, p, alpha_beta)

    placeholders = {}
    varied_method_dict = {}
    for key in p:
        print("bootstapping dispersion: ", key)
        resids = p[key]["resid_DM"]
        varied_method_dict[key] = utils.bootstrap_fn(
            np.array([resids]), fns=[utils.NMAD, utils.STD]
        )
        for estimator, val in varied_method_dict[key].items():
            placeholders[f"Dvs{key}{estimator}"] = f"{np.round(np.median(val), 3):.3f}"
            placeholders[f"Dvs{key}{estimator}Err"] = f"{np.round(np.std(val), 3):.3f}"[
                2:
            ]
        placeholders[f"Dvs{key}Count"] = stats[key]["N"]
    # arrs_dict = {}
    # for new_name in ("SALT", "EBV"):
    #     for key in (f"{new_name}HSF", f"{new_name}DEHVILS"):
    #         arrs_dict[key] = np.array([p[key]["resid_DM"], p[key]["e_DM"]])
    # varied_phot_dict = utils.collated_bootstrap_fn(
    #     arrs_dict, fns=[utils.NMAD, utils.STD]
    # )
    # for new_name in ("SALT", "EBV"):
    #     for key in (f"{new_name}HSF", f"{new_name}DEHVILS"):
    #         for estimator, val in varied_phot_dict[key].items():
    #             placeholders[f"Dvs{key}{estimator}"] = np.round(np.median(val), 3)
    #             placeholders[f"Dvs{key}{estimator}Err"] = np.round(np.std(val), 3)
    placeholders["Chi2Cut"] = f"{np.round(chi2cut, 2):3.2f}"
    print("Model Filters N NMAD (mag) STD (mag)")
    print("DEHVILS co 47 0.177(029) 0.221(043)")
    print("DEHVILS coYJH 47 0.132(025) 0.175(034)")
    print("DEHVILS YJH 47 0.139(026) 0.172(027)")
    for name, model_name in zip(
        ("EBV", "Max", "SALT"), ("EBV_model2", "max_model", "SALT3-NIR")
    ):
        for bp in ("co", "coYJH", "YJH"):
            s = [f"{model_name} {bp} "]
            s.append(str(stats[f"{name}{bp}"]["N"]) + " ")
            s.append(placeholders[f"Dvs{name}{bp}NMAD"])
            s.append("(" + placeholders[f"Dvs{name}{bp}NMADErr"] + ") ")
            s.append(placeholders[f"Dvs{name}{bp}STD"])
            s.append("(" + placeholders[f"Dvs{name}{bp}STDErr"] + ")")
            print("".join(s))
    if cut_method == 3:
        return placeholders, fr_qsets, stats, p, varied_method_dict, chi2_cuts
    return placeholders, fr_qsets, stats, p, varied_method_dict


def old_compare_DEHVILS_phot(cut_method=2, redlaw="F19", N=47):
    import glob

    if cut_method not in (1, 2, 3, 4, 5):
        raise ValueError(
            """cut_method should be 1, 2, or 3
        1 is to find the rchi2 that gives 47 targets in SALTcoYJH and apply that cut to each list.
        2 is to do 1 and then only use the common targets.
        3 is to order each list by rchi2 and pick the first 47 targets in each.
        4 is to do 3 and then only use the common targets
        5 is to order each list by rchi2 and pick the first 47 targets common to each.
        """
        )
    TNS_names = []
    for path in glob.glob(f"{constants.HSF_DIR}/ado/DEHVILSDR1/*.snana.dat"):
        TNS_names.append(path.split("UKIRT_20")[1].split(".")[0])
    bad = ["20kcr", "20ned", "20nta", "20qne", "20uea", "21zfq"]
    bad += ["20naj", "20sme", "20mbf", "20tkp"]  # 06bt-like
    qset = Target.objects.filter(TNS_name__in=TNS_names).exclude(TNS_name__in=bad)
    acceptable = ["g", "b", "n", "?"]
    fr_qsets, stats, p = [{} for i in range(3)]
    for name, model_name, calibration in zip(
        ("SALT", "EBV", "Max"),
        ("salt3-nir", "snpy_ebv_model2", "snpy_max_model"),
        (6, 0, 0),
    ):
        fr_qsets[f"{name}DEHVILS"], _ = qset.get_fitresults_qset(
            bandpasses=["c", "o", "J"],
            variants=["dehvils"],
            model_name=model_name,
            optional_args={"bandpasses": []},
            acceptable_status=acceptable,
            cuts={"mwebv": 2 / 0.86},
            calibration=calibration,
            redlaw=redlaw,
        )
        fr_qsets[f"{name}HSF"], _ = qset.get_fitresults_qset(
            bandpasses=["c", "o", "J"],
            variants=[],
            model_name=model_name,
            optional_args={"bandpasses": [], "variants": ["2D", "0D", "1D3"]},
            acceptable_status=acceptable,
            cuts={"mwebv": 2 / 0.86},
            calibration=calibration,
            redlaw=redlaw,
        )
        stats[f"{name}DEHVILS"], p[f"{name}DEHVILS"] = fr_qsets[
            f"{name}DEHVILS"
        ].get_dispersion(
            sigma=-1,
        )
        stats[f"{name}HSF"], p[f"{name}HSF"] = fr_qsets[f"{name}HSF"].get_dispersion(
            sigma=-1,
        )
    if cut_method in (1, 2):
        chi2cut = 5.98
        for key in fr_qsets:
            fr_qsets[key] = fr_qsets[key].filter(
                target__TNS_name__in=p[key]["TNS_name"][
                    np.where(p[key]["rchi2"] <= chi2cut)
                ]
            )
    elif cut_method in (3, 4, 5):
        for name in ("SALT", "EBV", "Max"):
            # Getting the first 47 tagets common to DEHVILS and HSF samples
            dehvils_names = p[f"{name}DEHVILS"]["TNS_name"][
                np.argsort(p[f"{name}DEHVILS"]["rchi2"])
            ]
            hsf_names = p[f"{name}HSF"]["TNS_name"][
                np.argsort(p[f"{name}HSF"]["rchi2"])
            ]
            chi2cut = 0
            good_names = {
                f"{name}DEHVILS": dehvils_names[:47],
                f"{name}HSF": hsf_names[:47],
            }
            if cut_method == 5:
                for i in range(len(dehvils_names)):
                    good_names["common"] = []
                    for n in dehvils_names[: i + 1]:
                        if n in hsf_names[: i + 1]:
                            good_names["common"].append(n)
                    if len(good_names["common"]) >= 47:
                        for phot_source in ("DEHVILS", "HSF"):
                            chi2cut = max(
                                chi2cut,
                                fr_qsets[f"{name}{phot_source}"]
                                .get(target__TNS_name=n)
                                .chi2["reduced_model"],
                            )
                        break
            for model_name in ("salt3-nir", "snpy_ebv_model2", "snpy_max_model"):
                good_names_filter = {
                    "DEHVILS": good_names[name + "DEHVILS"],
                    "HSF": good_names[name + "HSF"],
                }
                if cut_method == 5:
                    good_names_filter = {
                        "DEHVILS": good_names["common"],
                        "HSF": good_names["common"],
                    }
                for phot_source in ("DEHVILS", "HSF"):
                    fr_qsets[f"{name}{phot_source}"] = fr_qsets[
                        f"{name}{phot_source}"
                    ].filter(target__TNS_name__in=good_names_filter[phot_source])
    if cut_method in (2, 4):
        for name in ("SALT", "EBV", "Max"):
            for phot_source, other_phot_source in zip(
                ("DEHVILS", "HSF"), ("HSF", "DEHVILS")
            ):
                fr_qsets[f"{name}{phot_source}"] = fr_qsets[
                    f"{name}{phot_source}"
                ].filter(
                    target__TNS_name__in=fr_qsets[
                        f"{name}{other_phot_source}"
                    ].values_list("target__TNS_name", flat=True)
                )
    for key in fr_qsets:
        alpha, beta = "default", "default"
        if "SALT" in key:
            if "DEHVILS" in key:
                alpha = 0.1087
                beta = 3.2068
            elif "HSF" in key:
                alpha = 0.0427
                beta = 2.8559
        stats[key], p[key] = fr_qsets[key].get_dispersion(
            alpha=alpha, beta=beta, sigma=-1
        )

    placeholders = {}
    varied_phot_dict = {}
    for key in p:
        resids = p[key]["resid_DM"]
        varied_phot_dict[key] = utils.bootstrap_fn(
            np.array([resids]), fns=[utils.NMAD, utils.STD]
        )
        for estimator, val in varied_phot_dict[key].items():
            placeholders[f"Dvs{key}{estimator}"] = f"{np.round(np.median(val), 3):.3f}"
            placeholders[f"Dvs{key}{estimator}Err"] = f"{np.round(np.std(val), 3):.3f}"[
                2:
            ]
        placeholders[f"Dvs{key}Count"] = stats[key]["N"]
    # arrs_dict = {}
    # for new_name in ("SALT", "EBV"):
    #     for key in (f"{new_name}HSF", f"{new_name}DEHVILS"):
    #         arrs_dict[key] = np.array([p[key]["resid_DM"], p[key]["e_DM"]])
    # varied_phot_dict = utils.collated_bootstrap_fn(
    #     arrs_dict, fns=[utils.NMAD, utils.STD]
    # )
    # for new_name in ("SALT", "EBV"):
    #     for key in (f"{new_name}HSF", f"{new_name}DEHVILS"):
    #         for estimator, val in varied_phot_dict[key].items():
    #             placeholders[f"Dvs{key}{estimator}"] = np.round(np.median(val), 3)
    #             placeholders[f"Dvs{key}{estimator}Err"] = np.round(np.std(val), 3)
    return placeholders, fr_qsets, stats, p, varied_phot_dict
