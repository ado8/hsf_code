import copy
import os
import shutil
import pickle
from datetime import datetime
from itertools import chain, combinations
import re

import astropy.units as u
import constants
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
import sncosmo
import snpy
import utils
from astropy.coordinates import SkyCoord
from astropy.table import Table
from django.core.exceptions import ObjectDoesNotExist
from django.db import models
from django.db.models import Q
from scipy import interpolate
from scipy.optimize import minimize
from tqdm import tqdm
from utils import logger, print_verb
from django.core.exceptions import ObjectDoesNotExist
from scipy.special import gammainc

# Create your models here.


def snpy_fit(
    all_lcs,
    model_name="snpy_ebv_model2",
    calibration=6,
    model_min_snr=0.2,
    force=False,
    clean=False,
    priors={},
    mcmc=False,
    mcmc_priors={},
    return_result=False,
    fail_loudly=False,
    redlaw="F19",
    verbose=False,
    **kwargs,
):
    (
        target,
        bps,
        vs,
        bandpasses_str,
        variants_str,
        bpv_str,
        path_head,
    ) = get_dir_parameters(all_lcs)
    if calibration is not None and model_name == "snpy_ebv_model2":
        file_head = f"{model_name}.{calibration}.{redlaw}"
    else:
        file_head = f"{model_name}.{redlaw}"

    # laziness checks
    quit_early = False
    if not force and has_results(
        target=target,
        bandpasses_str=bandpasses_str,
        variants_str=variants_str,
        model_name=model_name,
        calibration=calibration,
        redlaw=redlaw,
        force=force,
    ):
        print("Already see fit results, consider running again with force=True")
        quit_early = True
    for lc in all_lcs:
        if not lc.detections().exists():
            print(f"{lc} has no detections")
            quit_early = True
    if quit_early:
        return

    def make_failed_fit_result():
        target.fit_results.update_or_create(
            model_name=model_name,
            bandpasses_str=bandpasses_str,
            variants_str=variants_str,
            calibration=calibration,
            redlaw=redlaw,
            defaults={"success": False, "priors": priors},
        )

    print(f"{target.TNS_name}: {model_name} {bpv_str} {calibration} {redlaw}")
    # snpy part
    for key in priors:
        if "bps" in key or "bandpasses" in key:
            priors[key.replace("bps", "bands").replace("bandpasses", "bands")] = (
                priors.pop(key)
            )
    if priors.get("bands") is not None and set(priors.get("bands")).isdisjoint(
        set(bps)
    ):
        raise ValueError(
            f"Bandpasses to be fit are {bps}. Instead of using priors['bands'] adjust the bandpasses."
        )
    try:
        gal_z, EBVgal = [None for _ in range(2)]
        if "z" in priors:
            gal_z = priors["z"]
        if "EBVgal" in priors:
            EBVgal = priors["EBVgal"]
        sn = _snpy_prep(
            target,
            bpv_str,
            bandpasses_str,
            variants_str,
            model_name,
            redlaw,
            gal_z,
            EBVgal,
        )
    except RuntimeError as e:
        make_failed_fit_result()
        print(e)
        if fail_loudly:
            raise e
        return

    # fitting
    fixed_priors, iter_fit_priors = [{} for _ in range(2)]
    if len(sn.data.keys()) == 1 and "EBVhost" not in priors:
        print_verb(
            verbose,
            f"len(sn.data.keys()) == {len(sn.data.keys())} and EBVhost not in priors {priors}",
        )
        print_verb(verbose, "Setting EBVhost to 0")
        fixed_priors["EBVhost"] = 0
    if model_name == "snpy_ebv_model2":
        if calibration is None:
            calibration = 0
        # calibrations 0-9 use st, 10-15 use dm15
        if calibration < 10:
            stype = "st"
            fixed_priors["calibration"] = calibration
        elif calibration in np.arange(10, 16):
            stype = "dm15"
            fixed_priors["calibration"] = calibration - 10
        sn.choose_model("EBV_model2", stype=stype)
    elif model_name == "snpy_max_model":
        sn.choose_model("max_model", stype="st")
        calibration = None

    # Check for iterative fitting
    iters = -1
    success = False
    for key in priors:
        if re.search("_[0-9]", key) is not None:
            iters = max(int(key.split("_")[-1]), iters)
        elif key in sn.parameters:  # fixed parameters across all iterations
            fixed_priors[key] = priors[key]
    for i in range(iters + 1):
        print_verb(verbose, f"iteration {i}")
        iter_fit_priors[i] = copy.deepcopy(fixed_priors)
        for key in priors:
            if key.endswith(f"_{i}"):
                stripped_key = key.strip(f"_{i}")
                iter_fit_priors[i][stripped_key] = priors[key]
        if iter_fit_priors[i].get("bands") is None:
            iter_fit_priors[i]["bands"] = [constants.FILT_SNPY_NAMES[i] for i in bps]
        iter_bands = iter_fit_priors[i].pop("bands")
        sub_iter_bands = copy.deepcopy(iter_bands)
        print_verb(verbose, f"bands={iter_bands} {iter_fit_priors[i]}")
        while len(sub_iter_bands) > 1 and not success:
            try:
                sn.fit(sub_iter_bands, **iter_fit_priors[i])
                print_verb(
                    verbose,
                    f"Fit successful with bands={sub_iter_bands} {iter_fit_priors[i]}",
                )
                success = True
            except RuntimeError as e:
                if str(e).startswith("All weights for filter"):
                    sub_iter_bands.remove(str(e).split()[4])
                    print_verb(
                        verbose, f"Trying again with only {iter_fit_priors[i]['bands']}"
                    )
                elif fail_loudly:
                    make_failed_fit_result()
                    raise e
            except (TypeError, ValueError) as e:
                make_failed_fit_result()
                print_verb(verbose, "Failed")
                print_verb(verbose, str(e))
                if fail_loudly:
                    raise e
                break
        if success and len(sub_iter_bands) != len(iter_bands):
            print_verb(
                verbose, f"Trying to fit with {iter_bands} and {iter_fit_priors[i]}"
            )
            try:
                sn.fit(iter_bands, **iter_fit_priors[i])
                print_verb(verbose, "Success!")
            except (RuntimeError, IndexError, TypeError) as e:
                success = False
                print_verb(verbose, "Failed")
                print_verb(verbose, str(e))
                make_failed_fit_result()
                if fail_loudly:
                    raise e
                return
        if not success:
            make_failed_fit_result()
            return
    try:
        print_verb(f"Final iteration: {fixed_priors}")
        sn.fit(**fixed_priors)
    except (RuntimeError, IndexError, TypeError) as e:
        print_verb(verbose, "Failed")
        print_verb(verbose, str(e))
        make_failed_fit_result()
        if fail_loudly:
            raise e
        return

    # for kwarg in kwargs:
    #     if hasattr(sn, kwarg):
    #         try:  # test to see if number
    #             getattr(sn, kwarg) + 1.0
    #             setattr(sn, kwarg, kwargs[kwarg])
    #         except TypeError:
    #             continue

    if mcmc:
        mcmc_sn = copy.deepcopy(sn)
        mcmc_sn.fitMCMC(plot_triangle=True, **mcmc_priors)
        error_ave = np.average(list(mcmc_sn.errors.values()))
        if error_ave < 0.5:  # successful
            sn = mcmc_sn
            figs = [plt.figure(n) for n in plt.get_fignums()]
            figs[1].savefig(f"{path_head}/{file_head}.corner.png")
    # wrapping up
    if calibration is None:
        sn.plot(outfile=f"{path_head}/{file_head}.png")
        sn.save(f"{path_head}/{file_head}.snpy")
    else:
        sn.plot(outfile=f"{path_head}/{file_head}.png")
        sn.save(f"{path_head}/{file_head}.snpy")

    res = _snpy_results_to_db(
        target,
        sn=sn,
        model_name=model_name,
        bandpasses_str=bandpasses_str,
        variants_str=variants_str,
        calibration=calibration,
        redlaw=redlaw,
        priors=priors,
    )
    _snpy_create_model_lcs(
        sn=sn,
        fit_result=res,
        all_lcs=all_lcs,
        model_min_snr=model_min_snr,
        clean=clean,
    )
    res.chi2 = res.get_chi2()
    res.save()
    if return_result:
        return res


def _snpy_prep(
    target,
    bpv_str,
    bandpasses_str,
    variants_str,
    model_name,
    redlaw="F19",
    gal_z=None,
    EBVgal=None,
):
    # write lc
    file_path = (
        f"{constants.DATA_DIR}/20{target.TNS_name[:2]}/{target.TNS_name}/fits/"
        f"{bpv_str}/snpy.input"
    )
    with open(file_path, "w") as f:
        f.write(
            target.write_for_snpy(
                bandpasses=bandpasses_str.split("-"),
                variants=variants_str.split("-"),
                gal_z=gal_z,
            )
        )
    sn = snpy.import_lc(file_path)
    sn.redlaw = redlaw
    # Sometimes fails to connect to IRSA and get EBVgal
    if not sn.EBVgal:
        sn.EBVgal = 0.86 * target.mwebv
    if EBVgal is not None:
        try:
            sn.EBVgal = float(EBVgal)
        except ValueError:
            raise
    return sn


def _snpy_results_to_db(
    target,
    sn,
    model_name,
    bandpasses_str,
    variants_str,
    calibration=None,
    redlaw=None,
    priors={},
):
    # store fit results
    res, created = FitResults.objects.get_or_create(
        target=target,
        model_name=model_name,
        bandpasses_str=bandpasses_str,
        variants_str=variants_str,
        calibration=calibration,
        redlaw=redlaw,
    )
    res.success = True
    res.priors = priors
    res.params = sn.parameters
    res.stat_errors = sn.errors
    res.sys_errors = sn.systematics()
    res.errors = {}
    for (n, stat), sys_errs in zip(sn.errors.items(), sn.systematics().values()):
        res.errors[n] = np.sqrt(stat**2 + sys_errs**2)
    for param in sn.model.C:
        res.covariance[param] = sn.model.C[param]
    try:
        res.chi2 = res.get_chi2()
    except Exception as e:
        pass
    res.last_updated = datetime.now()
    res.save()
    return res


def _snpy_create_model_lcs(
    sn,
    fit_result,
    all_lcs,
    model_min_snr=0.2,
    clean=True,
):
    # create model bands
    target = fit_result.target
    pwd = os.getcwd()
    new_path = os.path.join(fit_result.fits_path, str(fit_result.pk))
    os.makedirs(new_path, exist_ok=True)
    os.chdir(new_path)
    sn.dump_lc(tmin=-constants.PRE_SN, tmax=constants.POST_SN)
    for lc in all_lcs:
        if len(lc.mjd) == 0:
            continue
        suffix = ""
        path = (
            f"{fit_result.fits_path}/{fit_result.pk}/{target.TNS_name}_lc_{constants.FILT_SNPY_NAMES[lc.bandpass]}"
            f"{suffix}_model.dat"
        )
        if not os.path.exists(path):
            continue
        model = pd.read_csv(
            path,
            delim_whitespace=True,
            header=None,
            skiprows=3,
            names=["mjd", "mag", "dmag"],
        )
        zp = np.round(np.median(np.nan_to_num(lc.mag + 2.5 * np.log10(lc.ujy))), 1)
        model = model.replace(",", "", regex=True)
        model["mjd"] = model["mjd"].astype(float)
        ujy = 10 ** (0.4 * (zp - model["mag"]))
        dujy = ujy * (10 ** (model["dmag"] / 2.5) - 1)
        model_lc = target.lightcurves.get_or_create(
            source="model",
            bandpass=lc.bandpass,
            eff_wl=constants.EFF_WL_DICT[lc.bandpass],
            fit_result=fit_result,
        )[0]
        model_lc.detections().delete()
        snr_mask = np.where(ujy / dujy > model_min_snr)
        for mjd, m, dm, uu, du in zip(
            *[
                np.array(arr)[snr_mask]
                for arr in (model["mjd"], model["mag"], model["dmag"], ujy, dujy)
            ]
        ):
            model_lc.model_detections.create(
                lc=model_lc, mjd=mjd, mag=m, dmag=dm, ujy=uu, dujy=du
            )
        if clean:
            shutil.rmtree(new_path)
    os.chdir(pwd)


def salt_fit(
    all_lcs,
    model_name="salt3-nir",
    redlaw="F19",
    return_result=True,
    model_min_snr=0.2,
    priors={},
    force=False,
    verbose=False,
    show=False,
    **kwargs,
):
    from fitting.RubinsNM import miniNM_new
    from fitting.SALT3 import (
        add_model_cov,
        chi2_fn,
        get_delta_mu_instruments,
        get_deriv,
        trim_data,
        write_results,
    )

    (
        target,
        bps,
        vs,
        bandpasses_str,
        variants_str,
        bpv_str,
        path_head,
    ) = get_dir_parameters(all_lcs)
    target = all_lcs.first().target

    # laziness check
    quit_early = False
    if not force and has_results(
        target=target,
        bandpasses_str=bandpasses_str,
        variants_str=variants_str,
        model_name=model_name,
        calibration=None,
        redlaw=redlaw,
        force=force,
    ):
        print("Already see fit results, consider running again with force=True")
        quit_early = True
    for lc in all_lcs:
        if not lc.detections().exists():
            print(f"{lc} has no detections")
            quit_early = True
    if quit_early:
        return

    print(f"{target.TNS_name}: {model_name} {bpv_str} {redlaw}")
    file_head = f"{model_name}.{redlaw}"
    # initializing data structures
    all_lc_data = {"covmat": np.zeros([0, 0]), "model_uncertainties": 0}
    for key in ("lc2fl", "date", "flux"):
        all_lc_data[key] = np.array([])

    other_data = {
        "bandpasslambs": {},
        "bandpasseval": {},
        "magsysflux": {},
        "efflamb": {},
        "MagSys|Instrument|Band": {},
        "MWEBV": 0.86 * target.mwebv,
        "model_name": model_name,
        "MW_redlaw": redlaw,
    }
    if "z" in priors:
        other_data["z_heliocentric"] = priors["z"]
    if target.galaxy is not None and target.galaxy.z is not None:
        other_data["z_heliocentric"] = target.galaxy.z
    else:
        print("No redshift data. Assuming fiducial z=0.04")
        other_data["z_heliocentric"] = 0.04
    for kwarg in kwargs:
        if kwarg in other_data:
            other_data[kwarg] = kwargs[kwarg]
    # add in data from each lightcurve
    for lc in all_lcs:
        instr = lc.source
        if instr == "UKIRT":
            magsys = "VEGAHST"
            zp = 24
        else:
            magsys = "AB"
            zp = 23.9
        all_lc_data, other_data = _salt_add_bp(
            lc,
            all_lc_data,
            other_data,
            instr=instr,
            magsys=magsys,
            zp=zp,
            verbose=verbose,
        )
    all_lc_data["weightmat"] = np.linalg.inv(all_lc_data["covmat"])

    # guess time of max
    if "daymax_guess" in priors:
        other_data["daymax_guess"] = priors["daymax_guess"]
    else:
        other_data["daymax_guess"] = np.average(
            all_lc_data["date"], weights=all_lc_data["flux"] ** 2
        )
    # max_SNR = -1
    # for i in range(len(all_lc_data["flux"])):
    #     this_SNR = all_lc_data["flux"][i] / np.sqrt(
    #         all_lc_data["covmat"][i, i] + (all_lc_data["flux"].max() * 0.05) ** 2
    #     )
    #     if this_SNR > max_SNR:
    #         max_SNR = this_SNR
    #         other_data["daymax_guess"] = all_lc_data["date"][i]
    daymax_current = other_data["daymax_guess"]
    daymax_previous = daymax_current - 10.0

    # Iteratively fit for x0, then t0 and x0, then t0, x0, x1, and c
    n_iter = 0
    while n_iter < 3 and np.abs(daymax_current - daymax_previous) > 0.1:
        daymax_previous = daymax_current
        trimmed_lc_data = trim_data(
            t0=daymax_current,
            all_lc_data=all_lc_data,
            other_data=other_data,
            verbose=verbose,
        )
        P, F, cov_mat = miniNM_new(
            ministart=[daymax_current, 10.0, 0.0, 0.0],
            miniscale=[0.0, 1.0, 0.0, 0.0],
            chi2fn=chi2_fn,
            passdata=[trimmed_lc_data, other_data],
            verbose=verbose,
            compute_Cmat=False,
        )
        P, F, cov_mat = miniNM_new(
            ministart=P,
            miniscale=[1.0, 1.0, 0.0, 0.0],
            chi2fn=chi2_fn,
            passdata=[trimmed_lc_data, other_data],
            verbose=verbose,
            compute_Cmat=False,
        )
        miniscale = [1.0, 1.0, 1.0, 0.1]
        for i, param in enumerate(("t0", "x0", "x1", "c")):
            if param in kwargs:
                P[i] = kwargs[param]
                miniscale[i] = 0
        P, F, cov_mat = miniNM_new(
            ministart=P,
            miniscale=miniscale,
            chi2fn=chi2_fn,
            passdata=[trimmed_lc_data, other_data],
            verbose=verbose,
            compute_Cmat=False,
        )
        for add_model_cov_iter in range(3):
            miniscale = [1.0, 1.0, 1.0, 0.1]
            for i, param in enumerate(("t0", "x0", "x1", "c")):
                if param in priors:
                    P[i] = kwargs[param]
                    miniscale[i] = 0
            trimmed_lc_data = trim_data(
                t0=daymax_current,
                all_lc_data=all_lc_data,
                other_data=other_data,
                verbose=verbose,
            )
            trimmed_lc_data, other_data = add_model_cov(P, trimmed_lc_data, other_data)
            success = True
            try:
                P, F, cov_mat = miniNM_new(
                    ministart=P,
                    miniscale=miniscale,
                    chi2fn=chi2_fn,
                    passdata=[trimmed_lc_data, other_data],
                    verbose=verbose,
                )
            except Exception as e:
                print("Fit failed")
                print(str(e))
                success = False
                cov_mat = np.diag(np.ones(len(P)))
        daymax_current = P[0]
        n_iter += 1
    if (
        P[1] > constants.SALT_FAIL_LIMITS["x0_mag"]
        or np.abs(P[2]) > constants.SALT_FAIL_LIMITS["x1"]
        or np.abs(P[3]) > constants.SALT_FAIL_LIMITS["c"]
    ):
        success = False

    # pickling here in case saving to db throws errors
    with open(f"{path_head}/{file_head}.pickle", "wb") as handle:
        pickle.dump((P, F, cov_mat, all_lc_data, trimmed_lc_data, other_data), handle)
    # uncomment to return fit results without hitting the db
    # return P, F, cov_mat, all_lc_data, trimmed_lc_data, other_data

    write_results(
        file_path=path_head + f"/{file_head}.results.dat",
        chifile_path=path_head + f"/{file_head}.chifile.txt",
        P=P,
        F=F,
        Cmat=cov_mat,
        trimmed_lc_data=trimmed_lc_data,
        other_data=other_data,
    )
    get_deriv(
        P,
        trimmed_lc_data,
        other_data,
        file_path=path_head + f"/{file_head}.deriv.dat",
        verbose=verbose,
    )
    get_delta_mu_instruments(
        P,
        trimmed_lc_data,
        other_data,
        file_path=path_head + f"/{file_head}.dmu_instr.dat",
        verbose=verbose,
    )

    # Save results to db
    res, created = FitResults.objects.get_or_create(
        target=target,
        model_name=model_name,
        bandpasses_str=bandpasses_str,
        variants_str=variants_str,
        redlaw=redlaw,
    )
    _salt_create_model_lcs(
        target=target,
        P=P,
        bps=set(trimmed_lc_data["lc2fl"]),
        other_data=other_data,
        fit_result=res,
        min_phase=-15,
        max_phase=45,
        model_min_snr=model_min_snr,
    )
    res.plot(outdir=path_head, title=target.TNS_name, show=show)
    res.success = success
    res.priors = priors
    res.params = {}
    res.stat_errors = {}
    res.covariance = {}
    errors = np.nan_to_num(np.sqrt(np.diag(cov_mat)), nan=-1)
    for i, param_name in enumerate(("t0", "x0_mag", "x1", "c")):
        res.params[param_name] = P[i]
        res.stat_errors[param_name] = errors[i]
        if not success:
            continue
        res.covariance[param_name] = {}
        for j, cov_param_name in enumerate(("t0", "x0_mag", "x1", "c")):
            res.covariance[param_name][cov_param_name] = cov_mat[i][j]
    res.status = "?"
    if success:
        res.chi2 = res.get_chi2()
    if 0 in res.params.values() or -1 in errors:
        res.success = False
        res.status = "b"
    res.errors = res.stat_errors
    res.save()
    if return_result:
        return P, F, cov_mat, all_lc_data, trimmed_lc_data, other_data


def _salt_add_bp(
    lc,
    all_lc_data,
    other_data,
    instr="UKIRT",
    magsys="VEGAHST",
    zp=24,
    band_clip=0.001,
    verbose=False,
):
    from fitting.Spectra import Spectra

    blc = lc.bin()
    N = len(blc["mjd"])
    cov_mat = None
    path = None
    TNS_name = lc.target.TNS_name
    if lc.variant in ("0D", "1D2", "1D3", "1D4", "2D"):
        path = f"{constants.DATA_DIR}/20{TNS_name[:2]}/{TNS_name}/ukirt/photometry/{lc.bandpass}_{lc.variant}/results.txt"
    if path is None or not os.path.exists(path):
        cov_mat = np.diag(blc["dujy"] ** 2)
    else:
        non_det_idx = []
        with open(path, "r") as f:
            read_cov_mat = False
            for line in f.readlines():
                if line.startswith("SN_A") and float(line.split()[1]) < 0:
                    non_det_idx.append(int(line.split()[0].replace("SN_A", "")) - 1)
                if read_cov_mat:
                    row = np.array(line.split(), dtype=np.float64)
                    if cov_mat is None:
                        cov_mat = np.zeros([len(row), len(row)])
                        i = 0
                    cov_mat[i] = row
                    i += 1
                    if i >= len(cov_mat):
                        break
                elif line.startswith("Cmat"):
                    read_cov_mat = True
            for bad_idx in non_det_idx[::-1]:
                cov_mat = np.array(
                    [
                        [
                            cell
                            for cell_idx, cell in enumerate(row)
                            if cell_idx != bad_idx
                        ]
                        for row_idx, row in enumerate(cov_mat)
                        if row_idx != bad_idx
                    ]
                )

    all_lc_data["lc2fl"] = np.append(all_lc_data["lc2fl"], np.array([lc.bandpass] * N))
    all_lc_data["date"] = np.append(all_lc_data["date"], blc["mjd"])
    all_lc_data["flux"] = np.append(all_lc_data["flux"], blc["ujy"])
    all_lc_data["covmat"] = utils.twod_block_stack(all_lc_data["covmat"], cov_mat)
    other_data["MagSys|Instrument|Band"][
        lc.bandpass
    ] = f"{magsys}|{instr}|{lc.bandpass}"
    band_object = Spectra(
        instrument=instr,
        band=lc.bandpass,
        magsys=magsys,
        radialpos=None,
        verbose=verbose,
    )
    tmp_lambs = np.arange(2000.0, 25000.0, 1)
    for alpha in (0, 0.0001):
        key = lc.bandpass + (alpha != 0) * "_dalpha"

        def bandpassfn(x, alpha):
            return band_object.transmission_fn(x) * np.exp(alpha * x)

        inds = np.where(
            bandpassfn(tmp_lambs, alpha)
            > bandpassfn(tmp_lambs, alpha).max() * band_clip
        )[0]
        other_data["bandpasslambs"][key] = tmp_lambs[inds[0] : inds[-1] + 1]
        other_data["bandpasseval"][key] = bandpassfn(
            other_data["bandpasslambs"][key], alpha
        )
        other_data["magsysflux"][key] = 10.0 ** (-0.4 * zp) * np.sum(
            other_data["bandpasslambs"][key]
            * other_data["bandpasseval"][key]
            * band_object.ref_fn(other_data["bandpasslambs"][key])
        )
        other_data["efflamb"][key] = np.sum(
            other_data["bandpasslambs"][key] ** 2.0 * other_data["bandpasseval"][key]
        ) / np.sum(other_data["bandpasslambs"][key] * other_data["bandpasseval"][key])
    return all_lc_data, other_data


def _salt_create_model_lcs(
    target,
    P,
    bps,
    other_data,
    fit_result,
    min_phase=-15,
    max_phase=45,
    model_min_snr=0.2,
):
    from fitting.SALT3 import model_for_plotting

    model_mjds, model_fluxes, model_dfluxes = [{} for i in range(3)]
    for bp in bps:
        mjds, fluxes, dfluxes = model_for_plotting(
            P=P,
            lc2fl=bp,
            mindate=P[0] + min_phase * (1.0 + other_data["z_heliocentric"]),
            maxdate=P[0] + max_phase * (1.0 + other_data["z_heliocentric"]),
            other_data=other_data,
        )
        idx = np.where(fluxes / dfluxes > model_min_snr)
        model_mjds[bp] = mjds = mjds[idx]
        model_fluxes[bp] = fluxes = fluxes[idx]
        model_dfluxes[bp] = dfluxes = dfluxes[idx]
        model_lc = target.lightcurves.get_or_create(
            source="model",
            bandpass=bp,
            eff_wl=constants.EFF_WL_DICT[bp],
            fit_result=fit_result,
        )[0]
        model_lc.detections().delete()
        if bp in "ZYJHK":
            zp = 24
        else:
            zp = 23.9
        mags = zp - 2.5 * np.log10(fluxes)
        dmags = 2.5 * np.log10(1 + dfluxes / fluxes)
        for mjd, m, dm, uu, du in zip(mjds, mags, dmags, fluxes, dfluxes):
            model_lc.model_detections.create(
                lc=model_lc, mjd=mjd, mag=m, dmag=dm, ujy=uu, dujy=du
            )
    return model_mjds, model_fluxes, model_dfluxes


def _sncosmo_register_bandpass(bandpass, filt="total", name="HSF-like"):
    salt_name = bandpass
    if "ANDI" not in bandpass:
        salt_name = constants.FILT_SNPY_NAMES[bandpass]
    filter_fn = np.loadtxt(
        f"{constants.STATIC_DIR}/filters/{salt_name}_{filt}.dat", unpack=True
    )
    if name == "HSF-like":
        bp_name = bandpass
    elif name == "Jones-like":
        d = {
            "Z": "UKIRT-Z/z",
            "Y": "UKIRT-Y/y",
            "J": "UKIRT-J/j",
            "H": "UKIRT-H/h",
            "c": "ATLAS-cyan/c",
            "o": "ATLAS-orange/a",
            "ztfg": "ZTF-g/X",
            "ztfr": "ZTF-r/Y",
            "ztfi": "ZTF-i/Z",
        }
        bp_name = d[bandpass]
    salt_bp = sncosmo.Bandpass(*filter_fn, name=bp_name)
    sncosmo.registry.register(salt_bp, bp_name, force=True)


def _salt_model(model_name="salt3-nir", MW_redlaw="F19"):
    try:
        source = sncosmo.SALT3Source(modeldir=f"{constants.STATIC_DIR}/{model_name}")
    except:
        source = model_name

    if MW_redlaw == "F99":
        MW_redlaw = sncosmo.F99Dust()
    elif MW_redlaw == "F19":
        MW_redlaw = sncosmo.F19Dust()
    elif MW_redlaw == "CCM89":
        MW_redlaw = sncosmo.CCM89Dust()
    elif MW_redlaw == "O94":
        MW_redlaw = sncosmo.O94Dust()

    model = sncosmo.Model(
        source=source,
        effects=[MW_redlaw],
        effect_names=["MW"],
        effect_frames=["obs"],
    )
    for bp in ("c", "o", "ztfg", "ztfr", "Z", "Y", "J", "H", "K"):
        _sncosmo_register_bandpass(bp)
    return model


def _quick_salt_fit(target, bandpasses_str, variants_str):
    if variants_str != "":
        variants_str = "_" + variants_str
    data = Table.read(
        f"{constants.DATA_DIR}/20{target.TNS_name[:2]}/{target.TNS_name}/fits/{bandpasses_str}{variants_str}/salt.input",
        format="ascii",
    )
    for bp in constants.COLOR_DICT:
        if bp == "asg":
            continue
        _sncosmo_register_bandpass(bp, "total")
    model = _salt_model()
    mjdpk = data["mjd"][data["flux"] == data["flux"].max()][0]
    bounds = {"t0": (mjdpk - 10, mjdpk + 10)}
    fitparams = ["t0", "x0", "x1", "c"]
    if target.galaxy and target.galaxy.z:
        model.set(z=target.galaxy.z)
    else:
        fitparams.append("z")
        bounds["z"] = (0.0, 0.1)
    model.set(MWebv=0.86 * target.mwebv)
    sndata = Table(
        [
            data["mjd"],
            data["filter"],
            data["flux"],
            data["fluxerr"],
            data["zp"],
            data["zpsys"],
        ],
        names=["mjd", "band", "flux", "fluxerr", "zp", "zpsys"],
        meta={"t0": mjdpk},
    )
    result, fitted_model = sncosmo.mcmc_lc(sndata, model, fitparams, bounds)
    return result, fitted_model, data


def _quick_snana_write(target, bandpasses_str, variants_str, out_path="snana.dat"):
    if variants_str != "":
        variants_str = "_" + variants_str
    data = Table.read(
        f"{constants.DATA_DIR}/20{target.TNS_name[:2]}/{target.TNS_name}/fits/{bandpasses_str}{variants_str}/salt.input",
        format="ascii",
    )
    for bp in constants.COLOR_DICT:
        if bp == "asg":
            continue
        _sncosmo_register_bandpass(bp, "total")
    model = _salt_model()
    mjdpk = data["mjd"][data["flux"] == data["flux"].max()][0]
    bounds = {"t0": (mjdpk - 10, mjdpk + 10)}
    fitparams = ["t0", "x0", "x1", "c"]
    if target.galaxy and target.galaxy.z:
        model.set(z=target.galaxy.z)
    else:
        fitparams.append("z")
        bounds["z"] = (0.0, 0.1)
    model.set(MWebv=0.86 * target.mwebv)
    sndata = Table(
        [
            data["mjd"],
            data["filter"],
            data["flux"],
            data["fluxerr"],
            data["zp"],
            data["zpsys"],
        ],
        names=["mjd", "band", "flux", "fluxerr", "zp", "zpsys"],
        meta={
            "t0": mjdpk,
            "ra": target.ra,
            "dec": target.dec,
            "survey": "HSF",
            "FILTERS": "ztfg,ztfr,c,o,J",
            "MWEBV": target.mwebv * 0.86,
            "REDSHIFT_HELIO": target.galaxy.z,
            "REDSHIFT_CMB": utils.convert_z(target.galaxy.z, target.ra, target.dec),
        },
    )
    sncosmo.write_lc(sndata, out_path, format="snana")


def bayesn_fit(
    all_lcs,
    model_name="bayesn_m20",
    return_result=True,
    force=False,
    verbose=False,
    tmax="guess",
    redlaw="F99",
    model=None,
    **kwargs,
):
    from bayesn import SEDmodel

    (
        target,
        bps,
        vs,
        bandpasses_str,
        variants_str,
        bpv_str,
        path_head,
    ) = get_dir_parameters(all_lcs)
    target = all_lcs.first().target

    # laziness check
    quit_early = False
    if not force and has_results(
        target=target,
        bandpasses_str=bandpasses_str,
        variants_str=variants_str,
        model_name=model_name,
        calibration=None,
        redlaw=redlaw,
        force=force,
    ):
        print("Already see fit results, consider running again with force=True")
        quit_early = True
    for lc in all_lcs:
        if not lc.detections().exists():
            print(f"{lc} has no detections")
            quit_early = True
    if quit_early:
        return
    if (
        model is None
        or model.model_name != model_name.split("_")[1].upper() + "_model"
        or model.redlaw_name != redlaw
    ):
        model = SEDmodel(
            load_model=model_name.split("_")[1].upper() + "_model", load_redlaw=redlaw
        )
    file_head = f"{model_name}.{redlaw}"
    t, flux, flux_err, bps_used = [[] for _ in range(4)]
    for lc in all_lcs:
        zp = 23.9
        if lc.bandpass in "ZYJHK":
            zp = 24
        blc = lc.bin()
        for i in range(len(blc["mjd"])):
            # BayeSN uses fluxes calibrated to 27.5 mag.
            t.append(blc["mjd"][i])
            flux.append(blc["ujy"][i] * 10 ** (0.4 * (27.5 - zp)))
            flux_err.append(blc["dujy"][i] * 10 ** (0.4 * (27.5 - zp)))
            bps_used.append(lc.bandpass)
    try:
        z = target.galaxy.z
    except:
        print(f"No galaxy information for {target}, using z=0.04")
        z = 0.04
    if tmax == "guess":
        tmax = np.average(np.array(t), weights=np.array(flux) ** 2)
    samples, sn_prop = model.fit(
        t,
        flux,
        flux_err,
        bps_used,
        z,
        peak_mjd=tmax,
        ebv_mw=target.mwebv * 0.86,
        filt_map=constants.FILT_BAYESN_NAMES,
        mag=False,
    )

    # Save to db
    res, created = FitResults.objects.get_or_create(
        target=target,
        model_name=model_name,
        bandpasses_str=bandpasses_str,
        variants_str=variants_str,
        calibration=None,
        redlaw=redlaw,
    )
    with open(os.path.join(res.fits_path, f"{file_head}.samples.pickle"), "wb") as f:
        pickle.dump((samples, sn_prop), f)
    res.success = True

    # Make model lightcurves
    bayesn_bps = [constants.FILT_BAYESN_NAMES[bp] for bp in bps]
    model_flux = model.get_flux_from_chains(
        np.arange(-constants.PRE_SN, constants.POST_SN),
        bayesn_bps,
        samples,
        z,
        0.86 * target.mwebv,
        mag=False,
        num_samples=1000,
    )
    # tmax = np.median(samples['peak_MJD'])
    model_t = np.arange(tmax - constants.PRE_SN, tmax + constants.POST_SN)
    for i, bp in enumerate(bps):
        model_lc = target.lightcurves.get_or_create(
            source="model",
            bandpass=bp,
            eff_wl=constants.EFF_WL_DICT[bp],
            fit_result=res,
        )[0]
        model_lc.detections().delete()
        if bp in "ZYJHK":
            zp = 24
        else:
            zp = 23.9
        for j, mjd in enumerate(model_t):
            ujy = np.median(model_flux[0, :, i, j]) * 10 ** (0.4 * (zp - 27.5))
            dujy = np.std(model_flux[0, :, i, j]) * 10 ** (0.4 * (zp - 27.5))
            mag = zp - 2.5 * np.log10(ujy)
            dmag = 2.5 * np.log10(1 + dujy / ujy)
            model_lc.model_detections.create(
                lc=model_lc, mjd=mjd, mag=mag, dmag=dmag, ujy=ujy, dujy=dujy
            )

    # Model light curves must be created before chi2
    res.params, res.errors = [{} for _ in range(2)]
    for key in samples:
        res.params[key] = float(np.median(samples[key]))
        res.errors[key] = float(np.std(samples[key]))
    res.last_updated = datetime.now()
    res.chi2 = res.get_chi2()
    res.save()
    if return_result:
        return res, model, samples, sn_prop


def spline_fit(lc, tmin=15, model_min_snr=0.2, force=False):
    target = lc.target

    quit_early = False
    if not force and has_results(
        target=target,
        bandpasses_str=lc.bandpass,
        variants_str=lc.variant,
        model_name="spline",
        calibration=None,
        redlaw=None,
        force=force,
    ):
        print("Already see fit results, consider running again with force=True")
        quit_early = True
    if not lc.detections().exists():
        print(f"{lc} has no detections")
        quit_early = True
    if quit_early:
        return

    blc = lc.bin()
    values = {}
    m1 = blc["mjd"] > target.detection_date - tmin
    m2 = blc["mag"] != 99
    m3 = blc["mag"] > 0
    m = m1 & m2 & m3
    for param in ("mjd", "mag", "dmag", "ujy", "dujy"):
        values[param] = blc[param][m]
    if not len(values["mjd"]):
        return
    t = np.arange(min(values["mjd"]) - 10, max(values["mjd"]) + 10, 1)
    if sum(m) < 2:
        return None
    elif sum(m) < 10:
        k = 1
    else:
        k = 3
    mag = interpolate.splev(
        t,
        interpolate.splrep(
            values["mjd"], values["mag"], 1 / np.array(values["dujy"]), k=k
        ),
        ext=0,
    )
    ujy = interpolate.splev(
        t,
        interpolate.splrep(
            values["mjd"], values["ujy"], 1 / np.array(values["dujy"]), k=k
        ),
        ext=0,
    )
    # rough and dirty errors takes average and if time to nearest epoch is greater than
    # 1 day, multiplies dmag by that difference cubed for the cubic spline
    dmag = np.zeros(len(t))
    dujy = np.zeros(len(t))
    for i, tt in enumerate(t):
        dmag[i] = (
            np.average(values["dmag"]) * max(min(np.abs(tt - values["mjd"])), 1) ** 3
        )
        dujy[i] = (
            np.average(values["dujy"]) * max(min(np.abs(tt - values["mjd"])), 1) ** 3
        )
    snr_mask = np.where(ujy / dujy > model_min_snr)
    res, _ = target.fit_results.update_or_create(
        model_name="spline",
        bandpasses_str=lc.bandpass,
        variants_str=lc.variant,
        calibration=None,
        redlaw=None,
        defaults={"success": True},
    )
    model_lc, new = target.lightcurves.get_or_create(
        source="model",
        fit_result=res,
        bandpass=lc.bandpass,
        eff_wl=constants.EFF_WL_DICT[bp],
    )
    if not new:
        model_lc.detections().delete()
    for mt, mm, mf, mdm, mdf in zip(
        *[np.array(arr)[snr_mask] for arr in (t, mag, ujy, dmag, dujy)]
    ):
        model_lc.model_detections.create(
            lc=model_lc, mjd=mt, mag=mm, ujy=mf, dmag=mdm, dujy=mdf
        )
    res.last_updated = datetime.now()
    res.ch2 = res.get_chi2()
    res.save()


def has_results(
    target,
    bandpasses_str,
    variants_str,
    model_name,
    calibration,
    redlaw,
    force,
):
    fr = target.fit_results.filter(
        bandpasses_str=bandpasses_str,
        variants_str=variants_str,
        model_name=model_name,
        calibration=calibration,
        redlaw=redlaw,
    )
    if fr.exists():
        if (
            force
            and fr.first().last_data(output="date") is not None
            and fr.first().last_updated < fr.first().last_data(output="date")
        ):
            return False
        return True
    return False


def get_dir_parameters(all_lcs):
    target = all_lcs.first().target
    bps = sorted(all_lcs.values_list("bandpass", flat=True))
    vs = sorted(
        [v for v in all_lcs.values_list("variant", flat=True).distinct() if v != "none"]
    )
    bandpasses_str = "-".join(sorted(bps))
    variants_str = "-".join(sorted(vs))
    bpv_str = f"{bandpasses_str}_{variants_str}".strip("_")
    path_head = (
        f"{constants.DATA_DIR}/20{target.TNS_name[:2]}/{target.TNS_name}/fits/"
        f"{bpv_str}"
    )
    os.makedirs(path_head, exist_ok=True)
    bps = bandpasses_str.split("-")
    return (
        target,
        bps,
        vs,
        bandpasses_str,
        variants_str,
        bpv_str,
        path_head,
    )


class FitResultsQuerySet(models.query.QuerySet):
    """
    Methods for analyzing a bunch of fit results at once
    """

    def write_results(self, p=None, stats=None, outpath=None):
        if outpath is None:
            outpath = "results.dat"
        header_str = ""
        model_names = list(set(self.values_list("model_name", flat=True)))
        if len(model_names) == 1:
            header_str += f"# model={model_names[0]}\n"
        cols = np.array(
            [
                "TNS_name",
                "RA",
                "Dec",
                "PGC",
                "galRA",
                "galDec",
                "bandpasses",
                "variant",
                "redlaw",
                "calibration",
            ]
        )
        df = pd.DataFrame(
            self.values_list(
                "target__TNS_name",
                "target__ra",
                "target__dec",
                "target__galaxy__pgc_no",
                "target__galaxy__ra",
                "target__galaxy__dec",
                "bandpasses_str",
                "variants_str",
                "redlaw",
                "calibration",
            ),
            columns=cols,
        )
        if p is not None:
            tmp_p = copy.deepcopy(p)
            for param in ("h", "alpha", "beta"):
                if param in tmp_p:
                    header_str += f"# {param}={tmp_p.pop(param)}\n"
            if "clipped" in tmp_p:
                tmp_p.pop("clipped")
            df = pd.merge(df, pd.DataFrame(tmp_p), on=["TNS_name"])
        if stats is not None:
            for stat in stats:
                header_str += f"# {stat} = {stats[stat]}\n"
        with open(outpath, "w") as f:
            f.write(header_str + df.convert_dtypes().to_string(index=False))

    def get_dispersion(
        self,
        h="resid_average",
        sigma=5,
        max_bandpass="J",
        max_color_1="V",
        max_color_2="r",
        q0=-0.53,
        max_iters=3,  # deprecate
        verbose=True,
        alpha="default",
        beta="default",
        outpath=None,
    ):

        stats, p = [{} for i in range(2)]
        param_names = [
            "DM",
            "e_DM",
            "z_hel",
            "z_cmb",
            "e_z",
            "TNS_name",
            "ra",
            "dec",
            "MWebv",
            "fitprob",
            "fitprob_data",
            "fitprob_data_model",
            "rchi2",
            "rchi2_data",
            "rchi2_data_model",
        ]
        all_models = "".join(self.values_list("model_name", flat=True).distinct())
        calibrations = self.values_list("calibration", flat=True).distinct()
        if "snpy" in all_models:
            param_names += ["Tmax", "e_Tmax", "rchisquare"]
            if (
                not set(np.arange(0, 10)).isdisjoint(calibrations)
                or "snpy_max_model" in all_models
            ):
                param_names += ["st", "e_st"]
            if not set(np.arange(10, 16)).isdisjoint(calibrations):
                param_names += ["dm15", "e_dm15"]
        if "snpy_ebv_model2" in all_models:
            param_names += ["EBVhost", "e_EBVhost"]
        if "snpy_max_model" in all_models:
            param_names += [
                f"{max_bandpass}max",
                f"{max_color_1}max",
                f"{max_color_2}max",
                f"e_{max_bandpass}max",
                f"e_{max_color_1}max",
                f"e_{max_color_2}max",
            ]
        if "salt" in all_models:
            param_names += ["t0", "e_t0", "x0", "e_x0", "x1", "e_x1", "c", "e_c"]
        if "bayesn" in all_models:
            param_names += [
                "AV",
                "Ds",
                "eps",
                "eps_tform",
                "theta",
                "tmax",
                "peak_MJD",
                "delM",
            ]

        for param in param_names:
            p[param] = np.zeros(self.count())
        p["TNS_name"] = np.zeros(self.count(), dtype="<U6")
        solve_for_alpha_beta = False
        if "solve" in (alpha, beta) and "salt" in all_models:
            solve_for_alpha_beta = True
            alpha = "default"
            beta = "default"

        if "snpy_max_model" in all_models:
            max_fr_qset = self.filter(model_name="snpy_max_model")
            (
                max_fit,
                max_input_dict,
                max_cut,
                max_names,
            ) = max_fr_qset.get_max_model_dispersion(
                mag=max_bandpass, color_1=max_color_1, color_2=max_color_2
            )
            max_DM = np.median(max_fit["DM"], axis=1)
            max_e_DM = np.std(max_fit["DM"], axis=1)

        for i, fr in tqdm(enumerate(self), total=self.count()):
            # Universal parameters
            for suffix in ("", "_data", "_data_model"):
                if "reduced" + suffix in fr.chi2:
                    p["rchi2" + suffix][i] = fr.chi2["reduced" + suffix]
                if "fitprob" + suffix in fr.chi2:
                    p["fitprob" + suffix][i] = fr.chi2["fitprob" + suffix]
            for pname, param in zip(
                ("TNS_name", "z_hel", "z_cmb", "e_z", "ra", "dec"),
                (
                    fr.target.TNS_name,
                    fr.target.galaxy.z,
                    utils.convert_z(fr.target.galaxy.z, fr.target.ra, fr.target.dec),
                    fr.target.galaxy.z_err,
                    fr.target.ra,
                    fr.target.dec,
                ),
            ):
                p[pname][i] = param
            # model-specific parameters
            for pname in (
                "DM",
                "e_DM",
                "Tmax",
                "e_Tmax",
                "st",
                "e_st",
                "dm15",
                "e_dm15",
                "t0",
                "e_t0",
                "x0",
                "e_x0",
                "x1",
                "e_x1",
                "c",
                "e_c",
                "EBVhost",
                "e_EBVhost",
                "MWebv",
                f"{max_bandpass}max",
                f"{max_color_1}max",
                f"{max_color_2}max",
                f"e_{max_bandpass}max",
                f"e_{max_color_1}max",
                f"e_{max_color_2}max",
                "AV",
                "Ds",
                "eps",
                "eps_tform",
                "theta",
                "tmax",
                "peak_MJD",
                "delM",
            ):
                if (
                    fr.model_name == "snpy_max_model"
                    and "DM" in pname
                    and fr.target.TNS_name not in max_cut
                ):
                    idx = np.where(max_names == fr.target.TNS_name)[0][0]
                    if pname == "DM":
                        p["DM"][i] = max_DM[idx]
                    elif pname == "e_DM":
                        p["e_DM"][i] = np.sqrt(
                            max_input_dict["e_mag"][idx] ** 2 + max_e_DM[idx] ** 2
                        )
                elif pname in p:
                    p[pname][i] = fr.get(pname, alpha=alpha, beta=beta)
            # if fr.model_name == "snpy_max_model" and fr.target.TNS_name in max_cut:
            #     for pname in ("Tmax", "e_Tmax", "st", "e_st", "MWebv"):
            #         p[pname][i] = fr.get(pname)
            #     for pname in (max_bandpass, max_color_1, max_color_2):
            #         if f"{pname}max" in fr.params:
            #             p[f"{pname}max"][i] = fr.params[f"{pname}max"]
            #             p[f"e_{pname}max"][i] = fr.errors[f"{pname}max"]
            #     continue
            if "rchisquare" in p:
                try:
                    p["rchisquare"][i] = fr.get_snpy().model.rchisquare
                except AttributeError:
                    continue
                except FileNotFoundError:
                    print_verb(verbose, f"snpy file not found for {fr.target.TNS_name}")
                    continue

        if solve_for_alpha_beta:
            salt_idx = np.where(p["x1"] != 0.0)

            def best_alpha_beta(alpha_beta):
                alpha, beta = alpha_beta
                dm = (
                    10.5
                    - constants.FIDUCIAL_M
                    - 2.5 * np.log10(p["x0"][salt_idx])
                    + alpha * p["x1"][salt_idx]
                    - beta * p["c"][salt_idx]
                )
                dist = 10 ** (dm / 5 - 5)  # currently lum dist
                resid = dm - utils.mu_lcdm(
                    p["z_hel"], p["z_cmb"], np.median(constants.C * p["z_cmb"] / dist)
                )
                return utils.NMAD(resid)
                # return np.std(np.log10(constants.C * p["z_cmb"] / dist))

            res = minimize(
                best_alpha_beta,
                [constants.ALPHA["salt3-nir"], constants.BETA["salt3-nir"]],
                method="nelder-mead",
            )
            for i, fr in enumerate(self):
                if "salt" not in fr.model_name:
                    continue
                p["DM"][i], p["e_DM"][i] = fr.salt_dm(alpha=res.x[0], beta=res.x[1])

        # not set up for multiple models at once
        # will mask out everything if snpy and salt in qset
        mask = np.array([True] * self.count())
        all_mask = np.array([True] * self.count())
        clipped = {}
        for pname in p:
            if pname == "TNS_name":
                mask = p[pname] != ""
            elif "fitprob" in pname:
                continue
            else:
                mask = p[pname] != 0 & ~np.isnan(p[pname])
            if sum(mask) != len(p[pname]):
                clipped[pname] = p["TNS_name"][~mask]
            all_mask = all_mask & mask
        if sum(all_mask) == 0:
            print_verb(verbose, "Everything masked out")
            return (
                stats,
                p,
            )
        p["clipped"] = clipped
        p["clipped"]["any"] = p["TNS_name"][~all_mask]
        if sum(all_mask) != len(all_mask):
            for pname in p:
                if pname == "clipped":
                    continue
                # print(pname, p[pname][~mask])
                p[pname] = p[pname][all_mask]
        if solve_for_alpha_beta:
            p["alpha"] = res.x[0]
            p["beta"] = res.x[1]
        p["dist"] = 10 ** (p["DM"] / 5 - 5)  # currently lum dist
        p["h0"] = constants.C * p["z_cmb"] / p["dist"]
        if isinstance(h, str) and h.startswith("resid"):
            p["h"] = np.average(p["h0"])
        elif h == "average":
            p["h"] = np.average(p["h0"], weights=1 / p["e_DM"] ** 2)
        elif h == "median":
            p["h"] = np.median(p["h0"])
        else:
            p["h"] = h
        p["resid_DM"] = p["DM"] - utils.mu_lcdm(p["z_hel"], p["z_cmb"], p["h"], q0)
        if isinstance(h, str) and h.startswith("resid"):
            if h == "resid_average":
                zero_point_fix = np.average(p["resid_DM"], weights=1 / p["e_DM"] ** 2)
            elif h == "resid_median":
                zero_point_fix = np.median(p["resid_DM"])
            p["h"] = p["h"] / 10 ** (0.2 * zero_point_fix)
            p["resid_DM"] -= zero_point_fix
        if sigma > 0:
            _, _, mask, _ = utils.sigmaclip(
                p["resid_DM"], errors=p["e_DM"], sigmalow=sigma, sigmahigh=sigma
            )
            p["clipped"]["sigma_clipped"] = p["TNS_name"][~mask]
            p["clipped"]["any"] = np.append(p["clipped"]["any"], p["TNS_name"][~mask])
            for param in p:
                if param not in ("h", "clipped", "alpha", "beta"):
                    try:
                        p[param] = p[param][mask]
                    except IndexError:
                        print(param, p[param], mask)
                        raise
        stats["N"] = len(p["z_hel"])
        stats["RMS"] = utils.RMS(p["resid_DM"])
        stats["WRMS"] = utils.WRMS(p["resid_DM"], p["e_DM"])
        stats["NMAD"] = utils.NMAD(p["resid_DM"])
        stats["STD"] = np.std(p["resid_DM"])
        stats["STD(logH0)"] = np.std(np.log10(constants.C * p["z_cmb"] / p["dist"]))
        stats["rchisquare"] = sum((np.abs(p["resid_DM"] / p["e_DM"]) ** 2)) / (
            len(p["DM"]) - 4
        )  # fitting t0, x0, x1, c
        stats["sigma_int"] = utils.sigma_int(p["resid_DM"], p["e_DM"])

        if outpath:
            self.write_results(p, outpath=outpath)

        return (
            stats,
            p,
        )

    def get_max_model_dispersion(
        self,
        mag="J",
        color_1="V",
        color_2="r",
        h0=72,
        q0=-0.53,
        max_iters=3,  # deprecate
        verbose=False,  # deprecate
        outpath=None,  # deprecate
    ):
        input_dict = {
            "N": 0,
            "idx": [],
            "st": [],
            "e_st": [],
            "mag": [],
            "e_mag": [],
            "c1": [],
            "e_c1": [],
            "c2": [],
            "e_c2": [],
            "mu_lcdm": [],
        }
        cut = []
        TNS_names = []
        for fr in self:
            if (
                f"{mag}max" not in fr.params
                or f"{color_1}max" not in fr.params
                or f"{color_2}max" not in fr.params
            ):
                cut.append(fr.target.TNS_name)
                continue
            TNS_names.append(fr.target.TNS_name),
            input_dict["idx"].append(input_dict["N"])
            input_dict["N"] += 1
            input_dict["st"].append(fr.params["st"])
            input_dict["e_st"].append(fr.errors["st"])
            input_dict["mag"].append(fr.params[f"{mag}max"])
            input_dict["e_mag"].append(fr.errors[f"{mag}max"])
            input_dict["c1"].append(fr.params[f"{color_1}max"])
            input_dict["e_c1"].append(fr.errors[f"{color_1}max"])
            input_dict["c2"].append(fr.params[f"{color_2}max"])
            input_dict["e_c2"].append(fr.errors[f"{color_2}max"])
            input_dict["mu_lcdm"].append(
                utils.mu_lcdm(
                    fr.target.galaxy.z,
                    utils.convert_z(fr.target.galaxy.z, fr.target.ra, fr.target.dec),
                    h0,
                    q0,
                )
            )

        data = """
        int<lower=1> N;
        vector<lower=0> [N] idx;
        vector<lower=0, upper=4> [N] st;
        vector<lower=0, upper=1> [N] e_st;
        vector<lower=10, upper=24> [N] mag;
        vector<lower=0> [N] e_mag;
        vector<lower=10, upper=24> [N] c1;
        vector<lower=0> [N] e_c1;
        vector<lower=10, upper=24> [N] c2;
        vector<lower=0> [N] e_c2;
        vector<lower=30, upper=40> [N] mu_lcdm;
        """
        transformed_data = """
        vector<lower=-5, upper=5> [N] color;
        vector<lower=0> [N] e_color;
        color = c1 - c2;
        e_color = sqrt(square(e_c1) + square(e_c2)) ; // no correlation yet
        """
        params = """
        real P0 ;
        real P1 ;
        real P2 ;
        real<lower=0> sig_int ;
        real beta ;
        """
        transformed_params = """
        real P0_plus_19 = P0 + 19;
        """
        model = """
        P0_plus_19 ~ normal(0, 5);
        P1 ~ normal(0, 5);
        P2 ~ normal(0, 5);
        beta ~ normal(0, 5);
        sig_int ~ normal(0, 0.5);
        mag ~ normal(P0 + P1*(st - 1) + P2*square(st-1) + beta*(color) + mu_lcdm, sqrt(square(e_mag) + square(sig_int))); // sqrt(square(e_mag) + square(beta*e_color)));
        """
        generated_quantities = """
        vector [N] DM;
        vector [N] resid_DM;
        vector [N] end_idx;
        DM = mag - P0 - P1*(st - 1) - P2*square(st-1) - beta*(color) ;
        resid_DM = mag - P0 - P1*(st - 1) - P2*square(st-1) - beta*(color) - mu_lcdm;
        end_idx = idx;
        """
        fit = utils.stan(
            input_dict=input_dict,
            data=data,
            transformed_data=transformed_data,
            params=params,
            transformed_params=transformed_params,
            model=model,
            generated_quantities=generated_quantities,
            pickle_path=f"{constants.STATIC_DIR}/max_model.stancode",
        )
        return fit, input_dict, np.array(cut), np.array(TNS_names)

    def get_params_and_errors(self):
        params, errors = {}, {}
        for i, fr in enumerate(self):
            for p in fr.params:
                if p not in params:
                    params[p] = np.zeros(self.count())
                    errors[p] = np.zeros(self.count())
                params[p][i] = fr.params[p]
                errors[p][i] = fr.errors[p]
        return params, errors

    def get_salt_derivs(self):
        derivs = {}
        deriv_keys = ("dmu/dP", "dmB/dP", "ds/dP", "dc/dP")
        for param in ("Zeropoint", "Check", "Lambda", "MWEBV", "Redshift"):
            derivs[param] = {}
            if param in ("Check", "MWEBV", "Redshift"):
                for key in deriv_keys:
                    derivs[param][key] = np.zeros(self.count())
                continue
            for bp in constants.COLOR_DICT:
                derivs[param][bp] = {}
                for key in deriv_keys:
                    derivs[param][bp][key] = np.zeros(self.count())
        for i, fr in enumerate(self):
            d = fr.get_salt_deriv()
            for j, row in d.iterrows():
                if row["Parameter"] in ("Zeropoint", "Lambda"):
                    bp = row["MagSys|Instrument|Band"].split("|")[2]
                    for key in deriv_keys:
                        derivs[row["Parameter"]][bp][key][i] = row[key]
                else:
                    for key in deriv_keys:
                        derivs[row["Parameter"]][key][i] = row[key]
        return derivs

    def get_chi2s(self):
        chi2 = {}
        for bp in constants.COLOR_DICT:
            chi2[bp] = np.zeros(self.count())
        for i, fr in enumerate(self):
            for bp in fr.chi2:
                chi2[bp][i] = fr.chi2[bp]
        return chi2

    def check_cuts(self, cuts):
        if not isinstance(cuts, dict):
            return TypeError("cuts should be a dictionary")
        if not self.exists():
            return ValueError("There are no fit results to check")
        fails = []
        good_pks = []
        for fr in self:
            cut_status = fr.check_cuts(cuts)
            if cut_status == "pass":
                good_pks.append(fr.pk)
            else:
                fails.append(cut_status)
        if len(good_pks):
            return self.filter(pk__in=good_pks)
        for param in constants.REVERSE_CUT_ORDER:
            if param in fails:
                return [param]
        return fails

    def judge(self):
        for fr in self:
            fr.plot()
            judgment = input("Judgment: ")
            if judgment == "skip":
                continue
            elif judgment in ("g", "f", "b", "n"):
                fr.status = judgment
                fr.save()
            elif judgment == "":
                fr.status = "g"
                fr.save()
            else:
                break

    def prep_for_unity(self):
        import os

        with open(f"{constants.HSF_DIR}/unity_standalone/UKIRT_v1.txt", "w") as f:
            for fr in self:
                f.write(f"{fr.fits_path}\n")
        for fr in self:
            if not os.path.exists(f"{fr.fits_path}/salt3-nir.{fr.redlaw}.lightfile"):
                fr.make_salt_lightfile()

    def run_unity(
        self, paramfile="default", code_path="default", cosmo_model=1, storage_dir=None
    ):
        import gzip

        from unity_standalone.read_and_sample import (
            read_data,
            add_zbins,
            do_blinding,
            run_stan,
        )
        from unity_standalone.helper_functions import get_params
        from fitting.RubinsNM import save_img

        try:
            import multiprocessing

            multiprocessing.set_start_method("fork")
        except:
            pass
        self.prep_for_unity()
        inputfl = paramfile
        if inputfl == "default":
            inputfl = f"{constants.HSF_DIR}/unity_standalone/paramfiles/paramfile_UKIRT"
        if storage_dir is None:
            storage_dir = os.getcwd()
        os.makedirs(storage_dir, exist_ok=True)

        if inputfl.count("pickle"):
            (the_data, stan_data, params) = pickle.load(gzip.open(inputfl, "rb"))
            shutil.copy(inputfl, os.path.join(storage_dir, inputfl.strip("/")[-1]))
        else:
            params = get_params(inputfl)

            ################################################# And Go! ###################################################
            assert params["iter"] % 4 == 0, "iter should be a multiple of four! " + str(
                params["iter"]
            )

            the_data, stan_data = read_data(params, storage_dir)

        stan_data = add_zbins(stan_data, cosmo_model, storage_dir)
        print("nzadd ", stan_data["nzadd"])

        if stan_data["do_blind"]:
            print("Blinding!")
            the_data, stan_data = do_blinding(the_data, stan_data)
        else:
            print("Not Blinding!")
            assert os.environ["REALLYUNBLIND"] == "1"

        print("Running...")
        fit, av, df = run_stan(
            the_data, stan_data, params, storage_dir=storage_dir, code_path=code_path
        )
        return fit, av, df


class FitResults(models.Model):
    MODEL_CHOICES = [
        ("snpy_max_model", "SNooPy max_model"),
        ("snpy_ebv_model2", "SNooPy ebv_model2"),
        ("salt3", "salt3"),
        ("salt2-extended", "salt2-extended"),
        ("salt3-nir", "salt3-nir"),
        ("dehvils", "Fits from DEHVILS DR1"),
        ("bayesn_m20", "BayeSN Mandel et al. 2022"),
        ("bayesn_t21", "BayeSN Thorp et al 2021"),
        ("bayesn_w22", "BayeSN Ward et al. 2023"),
        ("spline", "Spline fit"),
    ]
    STATUS_CHOICES = [
        ("?", "Uninspected"),
        ("g", "Good"),
        ("f", "Fixable"),
        ("b", "Bad"),
        ("n", "Maybe not a Ia"),
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
        "targets.Target", related_name="fit_results", on_delete=models.CASCADE
    )
    model_name = models.CharField(max_length=20, choices=MODEL_CHOICES)
    status = models.CharField(max_length=1, choices=STATUS_CHOICES, default="?")
    success = models.BooleanField(default=False)
    bandpasses_str = models.CharField(max_length=50, blank=True)
    variants_str = models.CharField(max_length=50, blank=True)
    params = models.JSONField(default=dict)
    errors = models.JSONField(default=dict)
    chi2 = models.JSONField(default=dict)
    last_updated = models.DateTimeField(auto_now=True)
    stat_errors = models.JSONField(default=dict)
    sys_errors = models.JSONField(default=dict)
    covariance = models.JSONField(default=dict)
    ndof = models.FloatField(null=True)
    data_mask = models.JSONField(default=list)
    mean_acceptance_fraction = models.FloatField(null=True)
    calibration = models.PositiveSmallIntegerField(
        choices=CALIBRATION_CHOICES, null=True
    )
    redlaw = models.CharField(max_length=5, choices=REDLAW_CHOICES, null=True)
    priors = models.JSONField(default=dict)

    objects = FitResultsQuerySet.as_manager()

    class Meta:
        constraints = [
            models.UniqueConstraint(
                name="target_fit",
                fields=[
                    "target",
                    "model_name",
                    "bandpasses_str",
                    "variants_str",
                    "calibration",
                    "redlaw",
                ],
            )
        ]
        indexes = [
            models.Index(fields=["target"]),
            models.Index(fields=["model_name"]),
            models.Index(fields=["bandpasses_str"]),
            models.Index(fields=["variants_str"]),
            models.Index(fields=["calibration"]),
            models.Index(fields=["redlaw"]),
        ]

    def __str__(self):
        return f"{self.pk}: {self.target} {self.model_name} {self.bpv_str} {self.calibration} {self.redlaw}"

    @property
    def bpv_str(self):
        return f"{self.bandpasses_str}_{self.variants_str}".strip("_")

    @property
    def fits_path(self):
        return f"{constants.DATA_DIR}/20{self.target.TNS_name[:2]}/{self.target.TNS_name}/fits/{self.bpv_str}"

    @property
    def data_lightcurves(self):
        return self.target.lightcurves.filter(
            bandpass__in=self.bandpasses_str.split("-"),
            variant__in=self.variants_str.split("-")
            + [
                "none",
            ],
        ).exclude(source="model")

    def refit(self, varied_args={}, force=True):
        args = ("model_name", "calibration", "redlaw", "priors")
        in_dict = {}
        for arg in args:
            if arg in varied_args:
                in_dict[arg] = varied_args[arg]
            else:
                in_dict[arg] = getattr(self, arg)
        for i in ("bandpasses", "variants"):
            if f"{i}" in varied_args:
                in_dict[i] = varied_args[i]
            elif f"{i}_str" in varied_args:
                in_dict[i] = varied_args[f"{i}_str"].split("-")
            else:
                in_dict[i] = getattr(self, f"{i}_str").split("-")

        self.target.fit(
            bandpasses=in_dict["bandpasses"],
            variants=in_dict["variants"],
            model_name=in_dict["model_name"],
            calibration=in_dict["calibration"],
            redlaw=in_dict["redlaw"],
            priors=in_dict["priors"],
            force=force,
        )

    def get(
        self,
        param,
        # bandpass=None,
        # ebvhost=0,
        # e_ebvhost=0.06,
        galaxy_mass=1e11,
        alpha="default",
        beta="default",
        verbose=False,
    ):
        err, val = 0, 0
        error = False
        if param.startswith("e_"):
            error = True
            param = param[2:]
        if param in self.params:
            if error:
                return self.errors[param]
            return self.params[param]
        if param == "x0" and "x0_mag" in self.params:
            val = 10 ** (-0.4 * self.params["x0_mag"])
            if error:
                return val * (10 ** (0.4 * self.errors.get("x0_mag", 0)) - 1)
            return val
        if param.replace("_", "").lower() == "mwebv":
            return self.target.mwebv
        if param.lower() == "dm" and "salt" in self.model_name:
            val, err = self.salt_dm(alpha=alpha, beta=beta)
        elif param.lower() == "dm" and "bayesn" in self.model_name:
            val = self.params["mu"]
            err = self.errors["mu"]
        if error:
            return err
        return val
        if self.variants_str != "":
            lcs = lcs.filter(variant__in=self.variants_str.split("-"))
        # if self.model_name == "snpy_max_model":
        #     # For grabbing EBVhost for use with max model
        #     if isinstance(ebvhost, str) or isinstance(e_ebvhost, str):
        #         try:
        #             fr = self.target.fit_results.get(
        #                 model_name=ebvhost,
        #                 bandpasses_str=self.bandpasses_str,
        #                 variants_str=self.variants_str,
        #                 success=True,
        #             )
        #         except FitResults.DoesNotExist:
        #             print_verb(
        #                 verbose, f"{ebvhost} not found for {self.target} {self.bpv_str}"
        #             )
        #             return
        #         if isinstance(ebvhost, str):
        #             ebvhost = fr.get("EBVhost")
        #         if isinstance(e_ebvhost, str):
        #             e_ebvhost = fr.get("e_EBVhost")
        #         if ebvhost is None or e_ebvhost is None:
        #             print(f"something wrong with {fr}, ebvhost is None")
        #             return
        #     val, err = ebvhost, e_ebvhost

    def get_chi2(
        self,
        binning_statistic="median",
        weight_mat_types=("all", "data", "data+model"),
        N=1000,
    ):
        chi2 = {}
        resids, weight_mats, bp_list = self.get_residuals(
            weight_mat_type=weight_mat_types,
            binning_statistic=binning_statistic,
            N=N,
        )
        suffixes = {"all": "", "data": "_data", "data+model": "_data_model"}

        for bp in sorted(set(bp_list)):
            bp_mask = np.where(bp_list == bp)
            chi2[f"{bp}_N"] = len(bp_mask[0])
            for w_type in weight_mat_types:
                suffix = suffixes[w_type]
                W = weight_mats[w_type]
                chi2[f"{bp}{suffix}"] = np.round(
                    np.dot(
                        resids[bp_mask],
                        np.dot(W[bp_mask[0]][:, bp_mask[0]], resids[bp_mask]),
                    ),
                    3,
                )
        ndof = sum(chi2[f"{bp}_N"] for bp in set(bp_list)) - len(self.params)
        for w_type in weight_mat_types:
            suffix = suffixes[w_type]
            W = weight_mats[w_type]
            chi2[f"total{suffix}"] = np.round(np.dot(resids, np.dot(W, resids)), 3)
            if ndof > 0:
                chi2[f"reduced{suffix}"] = np.round(chi2[f"total{suffix}"] / ndof, 3)
                chi2[f"fitprob{suffix}"] = np.round(
                    1 - gammainc(ndof / 2, chi2[f"total{suffix}"] / 2), 3
                )
            else:
                chi2[f"reduced{suffix}"] = -1
                chi2[f"fitprob{suffix}"] = -1
        chi2["ndof"] = ndof
        return chi2

    def get_residuals(
        self,
        weight_mat_type=["data", "data+model", "all"],
        binning_statistic="median",
        N=1000,
    ):
        if not isinstance(weight_mat_type, str) and hasattr(
            weight_mat_type, "__iter__"
        ):
            weight_mats = {}
            for w_type in weight_mat_type:
                resids, weight_mats[w_type], bp_list = self.get_residuals(
                    weight_mat_type=w_type
                )
            return resids, weight_mats, bp_list
        if isinstance(weight_mat_type, str) and weight_mat_type not in (
            "data",
            "data+model",
            "all",
        ):
            raise ValueError(
                """
            weight_mat_type should be one of
                - 'data' (just data variance),
                - 'data+model' (data and model variance),
                - 'all' (data, model, fitting parameter variance)
                """
            )

        bps = self.bandpasses_str.split("-")
        resids, bp_list = [np.array([]) for _ in range(2)]

        # each branch should define a weight_mat, weight_mat_data, weight_mat_data_model,
        # resids, and bp_list
        if "salt" in self.model_name:
            from fitting.SALT3 import model_fn

            P, _, fit_param_cov, all_lc_data, trimmed_lc_data, other_data = (
                self.get_salt_results()
            )
            model_flux = model_fn(P, trimmed_lc_data, other_data)
            resids = trimmed_lc_data["flux"] - model_flux
            bp_list = trimmed_lc_data["lc2fl"]
            if weight_mat_type == "data":
                # for chi2 without model uncertainties need to use all_lc_data
                date_mask = np.where(
                    (all_lc_data["date"] <= max(trimmed_lc_data["date"]))
                    & (all_lc_data["date"] >= min(trimmed_lc_data["date"]))
                )
                weight_mat = all_lc_data["weightmat"][date_mask[0]][:, date_mask[0]]
            elif weight_mat_type == "data+model":
                weight_mat = trimmed_lc_data["weightmat"]
            elif weight_mat_type == "all":
                cov_mat_fit_params = self.get_cov_over_fit_params(N=N)
                weight_mat = np.linalg.inv(
                    cov_mat_fit_params + trimmed_lc_data["covmat"]
                )
        elif "snpy" in self.model_name:
            sn = self.get_snpy()
            P = np.array(list(sn.model.parameters.values()))
            data_var, model_var = [np.array([]) for _ in range(2)]
            mask = {}
            if weight_mat_type == "all":
                fit_param_cov = np.array(
                    [
                        [sn.model.C[param][key] for key in sn.model.C[param]]
                        for param in sn.model.C
                    ]
                )
                if "ebv_model2" in self.model_name:
                    delta_A_obs_poly = {}
                    tmp_ebvhost = (
                        sn.EBVhost - 3 * sn.errors["EBVhost"],
                        sn.EBVhost,
                        sn.EBVhost + 3 * sn.errors["EBVhost"],
                    )
                else:
                    delta_A_obs_poly = None
            for bp in bps:
                snpy_bp = constants.FILT_SNPY_NAMES[bp]
                model_mag, model_dmag, mask[bp] = sn.model(
                    snpy_bp, sn.data[snpy_bp].MJD
                )
                # Note: magnitudes blow up at low fluxes, which makes huge weighted residuals.
                # if sn.model.model_in_mags:
                #     data_var = np.append(
                #         data_var, sn.data[snpy_bp].e_mag[mask[bp]] ** 2
                #     )
                #     model_var = np.append(model_var, model_dmag[mask[bp]] ** 2)
                #     resids = np.append(
                #         resids,
                #         sn.data[snpy_bp].magnitude[mask[bp]] - model_mag[mask[bp]],
                #     )
                # else:
                zp = sn.data[snpy_bp].filter.zp
                data_flux = sn.data[snpy_bp].flux[mask[bp]]
                model_flux = 10 ** (0.4 * (zp - model_mag[mask[bp]]))
                data_var = np.append(
                    data_var,
                    (data_flux * (10 ** (sn.data[snpy_bp].e_mag[mask[bp]] / 2.5) - 1))
                    ** 2,
                )
                model_var = np.append(
                    model_var,
                    (model_flux * (10 ** (model_dmag[mask[bp]] / 2.5) - 1)) ** 2,
                )
                resids = np.append(resids, data_flux - model_flux)
                bp_list = np.append(bp_list, [bp] * sum(mask[bp]))
                if self.model_name == "snpy_ebv_model2" and weight_mat_type == "all":
                    delta_A_obs_poly[bp] = self.get_delta_A_obs_poly(
                        sn, tmp_ebvhost, snpy_bp, mask[bp], ebv_norm_idx=1, deg=2
                    )
            if weight_mat_type == "data":
                weight_mat = np.linalg.inv(np.diag(data_var))
            elif weight_mat_type == "data+model":
                weight_mat = np.linalg.inv(np.diag(data_var + model_var))
            elif weight_mat_type == "all":
                cov_mat_fit_params = self.get_cov_over_fit_params(
                    delta_A_obs_poly=delta_A_obs_poly, N=N
                )
                weight_mat = np.linalg.inv(
                    cov_mat_fit_params + np.diag(data_var) + np.diag(model_var)
                )
        else:
            # flux space, diagonal only, weight_mat NYI
            bp_list = []
            data_var, model_var = [np.array([]) for _ in range(2)]
            for data_lc in self.data_lightcurves:
                try:
                    model_lc = self.model_lightcurves.get(bandpass=data_lc.bandpass)
                except ObjectDoesNotExist:
                    continue
                blc = data_lc.bin(statistic=binning_statistic)
                mask = (blc["mjd"] > model_lc.mjd[0]) & (blc["mjd"] < model_lc.mjd[-1])
                if sum(mask) == 0:
                    continue
                data_flux, data_dflux, data_mjd = (
                    blc["ujy"][mask],
                    blc["dujy"][mask],
                    blc["mjd"][mask],
                )
                resids = np.append(
                    resids, data_flux - np.interp(data_mjd, model_lc.mjd, model_lc.ujy)
                )
                data_var = np.append(data_var, data_dflux**2)
                model_var = np.append(
                    model_var, np.interp(data_mjd, model_lc.mjd, model_lc.dujy) ** 2
                )
                bp_list = np.append(bp_list, [data_lc.bandpass] * sum(mask))
            if weight_mat_type == "data":
                weight_mat = np.linalg.inv(np.diag(data_var))
            elif weight_mat_type in ("data+model", "all"):
                weight_mat = weight_mat_data_model = np.linalg.inv(
                    np.diag(data_var + model_var)
                )
        return resids, weight_mat, bp_list

    def last_data(self, output="mjd"):
        lcs = self.target.data_lightcurves.all()
        maxes = [max(i.mjd) for i in lcs if i.detections().exists()]
        if not len(maxes):
            return None
        elif output == "mjd":
            return max(maxes)
        elif output in ("date", "datetime"):
            y, m, d = utils.MJD_to_ut(max(maxes))
            hour = 24 * (d % 1)
            minute = 60 * (hour % 1)
            second = 60 * (minute % 1)
            return datetime(
                y,
                m,
                int(d),
                int(hour),
                int(minute),
                int(second),
                tzinfo=pytz.timezone("UTC"),
            )

    def get_snpy(self, force=False):
        path = (
            f"{constants.DATA_DIR}/20{self.target.TNS_name[:2]}/{self.target.TNS_name}/"
            f"fits/{self.bpv_str}/{self.model_name}"
        )
        if self.calibration is not None:
            path += f".{self.calibration}.{self.redlaw}.snpy"
        else:
            path += f".{self.redlaw}.snpy"
        if os.path.exists(path):
            return snpy.get_sn(path)
        elif not force:
            raise FileNotFoundError(
                "snpy file not found. "
                "Run again with force=True to redo the fit and make the file."
            )
        else:
            return self.target.fit(
                bandpasses=self.bandpasses_str.split("-"),
                variants=self.variants_str.split("-"),
                model_name=self.model_name,
                force=True,
            )

    def get_delta_A_obs_poly(self, sn, ebvhosts, snpy_bp, mask, ebv_norm_idx=1, deg=2):
        """
        To avoid setting do_Robs to 1 and slowing things down, estimate
        delta_A_obs_poly as a quadratic function of EBVhost based on the min, max,
        and best fit values.
        """
        from snpy.kcorr import get_SED, redden
        from snpy.filters import fset

        epochs = (sn.data[snpy_bp].MJD[mask] - sn.Tmax) / (1 + sn.z)
        delta_A_obs_poly = np.zeros(len(epochs), dtype=object)
        for epoch_idx, epoch in enumerate(epochs):
            spec_w, spec_f = get_SED(int(epoch), sn.k_version, sn.k_extrapolate)
            tmp_A = np.zeros(len(ebvhosts))
            for ebv_idx, ebvhost in enumerate(ebvhosts):
                try:
                    red_f, redlaw_mask = redden(
                        spec_w,
                        spec_f,
                        sn.EBVgal,
                        ebvhost,
                        sn.z,
                        sn.Rv_gal,
                        sn.model.Rv_host[sn.model.calibration],
                        redlaw=sn.redlaw,
                    )
                    resp_red = fset[snpy_bp].response(
                        spec_w[redlaw_mask],
                        red_f,
                        z=sn.z,
                        photons=1,
                    )
                    tmp_A[ebv_idx] = -2.5 * np.log10(resp_red)
                except TypeError:
                    tmp_A = np.zeros(len(ebvhosts))
                    break
            # subtracting ebv_norm_idx to calculate difference that will be
            # applied later.
            delta_A_obs_poly[epoch_idx] = np.polynomial.Polynomial.fit(
                ebvhosts, tmp_A - tmp_A[ebv_norm_idx], deg=deg
            )
        return delta_A_obs_poly

    def get_salt_results(self):
        path = f"{self.fits_path}/{self.model_name}.{self.redlaw}.pickle"
        if os.path.exists(path):
            with open(path, "rb") as handle:
                P, F, cov_mat, all_lc_data, trimmed_lc_data, other_data = pickle.load(
                    handle
                )
            return P, F, cov_mat, all_lc_data, trimmed_lc_data, other_data

    def get_salt_deriv(self):
        return pd.read_csv(
            f"{constants.DATA_DIR}/20{self.target.TNS_name[:2]}/{self.target.TNS_name}/fits/{self.bpv_str}/{self.model_name}.deriv.dat",
            delim_whitespace=True,
            skiprows=2,
            names=[
                "Parameter",
                "MagSys|Instrument|Band",
                "RestLamb",
                "Phase",
                "dmu/dP",
                "dmB/dP",
                "ds/dP",
                "dc/dP",
            ],
        )

    def get_salt_model(self, verbose=False):
        (
            P,
            _,
            _,
            _,
            _,
            other_data,
        ) = self.get_salt_results()
        model = _salt_model(self.model_name, self.redlaw)
        model.set(
            x0=10.0 ** (-0.4 * P[1]),
            x1=P[2],
            c=P[3],
            t0=P[0],
            z=other_data["z_heliocentric"],
            MWebv=other_data["MWEBV"],  # should already include 0.86 factor
        )
        return model

    def make_salt_lightfile(self):
        with open(f"{self.fits_path}/salt3-nir.{self.redlaw}.lightfile", "w") as f:
            f.write(f"z_heliocentric  {self.target.galaxy.z}\n")
            f.write(
                f"z_cmb {utils.convert_z(self.target.galaxy.z, self.target.ra, self.target.dec, 'hel', 'cmb')}\n"
            )
            f.write(f"RA  {self.target.ra}\n")
            f.write(f"Dec {self.target.dec}\n")
            f.write(f"MWEBV {self.target.mwebv*0.86}\n")
            f.write("Mass  0.00  -1.00  1.00")

    def get_cov_over_fit_params(self, delta_A_obs_poly=None, N=1000):
        if "salt" in self.model_name:
            P, _, fit_param_cov, _, trimmed_lc_data, other_data = (
                self.get_salt_results()
            )
            lc_len = len(trimmed_lc_data["lc2fl"])
        elif "snpy" in self.model_name:
            if self.model_name == "snpy_ebv_model2" and delta_A_obs_poly is None:
                raise TypeError(
                    "delta_A_obs_poly is currently None. The covariance matrix will not include the effects of variation in EBVhost"
                )
            bps = self.bandpasses_str.split("-")
            sn = self.get_snpy()
            fit_param_cov = np.array(
                [
                    [sn.model.C[param][key] for key in sn.model.C[param]]
                    for param in sn.model.C
                ]
            )
            P = np.array([sn.model.parameters[param] for param in sn.model.C])
            mask = {}
            lc_len = 0
            for bp in bps:
                snpy_bp = constants.FILT_SNPY_NAMES[bp]
                _, _, mask[bp] = sn.model(snpy_bp, sn.data[snpy_bp].MJD)
                lc_len += sum(mask[bp])
        else:
            raise TypeError(
                "get_cov_over_fit_params not yet implemented for {self.model_name}"
            )
        sample = np.zeros((N, lc_len))
        fit_param_draws = np.random.multivariate_normal(P, fit_param_cov, size=N)
        if "salt" in self.model_name:
            from fitting.SALT3 import model_fn

            for i in tqdm(range(N)):
                sample[i] = model_fn(fit_param_draws[i], trimmed_lc_data, other_data)
        elif "snpy" in self.model_name:
            for i in range(N):
                for param, val in zip(sn.parameters, fit_param_draws[i]):
                    sn.model.parameters[param] = val
                sn_fluxes = np.array([])
                for bp in bps:
                    snpy_bp = constants.FILT_SNPY_NAMES[bp]
                    try:
                        bp_mags = sn.model(snpy_bp, sn.data[snpy_bp].MJD[mask[bp]])[0]
                    except TypeError:
                        continue
                    if self.model_name == "snpy_ebv_model2":
                        bp_mags += np.array(
                            [poly(sn.model.EBVhost) for poly in delta_A_obs_poly[bp]]
                        )
                    bp_fluxes = 10 ** (0.4 * (sn.data[snpy_bp].filter.zp - bp_mags))
                    sn_fluxes = np.append(sn_fluxes, bp_fluxes)
                sample[i] = sn_fluxes
        return np.cov(sample, rowvar=False)

    def plot(
        self,
        outdir=None,
        show=True,
        one_panel=True,
        title="default",
        fig_axs=None,
        return_fig=False,
        face_color="#333",
        ytype="flux",
        **fig_kwargs,
    ):
        data_lcs = self.data_lightcurves.all().order_by("eff_wl")
        model_lcs = self.model_lightcurves.all().order_by("eff_wl")
        N = data_lcs.count()

        if one_panel:
            row_count = 1
            col_count = 1
        else:
            row_count = int(np.ceil(N / 3))
            col_count = 3
            if row_count == 1:
                col_count = N
        if fig_axs is None:
            fig, axs = plt.subplots(
                nrows=2 * row_count,
                ncols=col_count,
                sharex=True,
                gridspec_kw={
                    "height_ratios": np.array(
                        [[4, 1] for _ in range(row_count)]
                    ).flatten()
                },
                **fig_kwargs,
            )
        else:
            fig, axs = fig_axs
        plot_axs = axs[::2]
        resid_axs = axs[1::2]
        if one_panel:
            plot_axs = np.array([plot_axs])
            resid_axs = np.array([resid_axs])
        for row in range(row_count):
            if ytype == "flux":
                plot_axs[row, 0].set_ylabel("Scaled Flux")
            elif ytype == "mag":
                plot_axs[row, 0].set_ylabel("Magnitude")
            if row + 1 != row_count or one_panel:
                resid_axs[row, -1].set_ylabel("Pull", rotation=270, labelpad=10)
            else:
                resid_axs[row, N - row * col_count - 1].set_ylabel(
                    "Pull", rotation=270, labelpad=10
                )
            for col in range(col_count):
                if row * col_count + col >= N:
                    plot_axs[row, col].set_axis_off()
                    resid_axs[row, col].set_axis_off()
                plot_axs[row, col].xaxis.set_tick_params(
                    which="both",
                    labelbottom=False,
                )
                resid_axs[row, col].xaxis.set_tick_params(
                    which="both",
                    labelbottom=(row + 1 == row_count),
                )
                plot_axs[row, col].yaxis.set_tick_params(
                    which="both",
                    labelleft=(col == 0),
                )
                resid_axs[row, col].yaxis.set_label_position("right")
                resid_axs[row, col].yaxis.tick_right()
                resid_axs[row, col].yaxis.set_tick_params(
                    which="both",
                    labelleft=False,
                    labelright=(
                        (col + 1 == col_count) or (row * col_count + col + 1 == N)
                    ),
                )
        for i, ax in enumerate(resid_axs[-1]):
            ax.set_xlabel("Date (MJD)")
        if one_panel:
            plot_axs = np.array([plot_axs[0][0] for _ in range(N)])
            resid_axs = np.array([resid_axs[0][0] for _ in range(N)])
        else:
            plot_axs = plot_axs.flatten()[:N]
            resid_axs = resid_axs.flatten()[:N]
        # Independent sharey for plotting and resid axes.
        for ax_idx in range(N):
            for other_idx in range(N):
                if ax_idx == other_idx:
                    continue
                plot_axs[ax_idx]._shared_axes["y"].join(
                    plot_axs[ax_idx], plot_axs[other_idx]
                )
                resid_axs[ax_idx]._shared_axes["y"].join(
                    resid_axs[ax_idx], resid_axs[other_idx]
                )
        # model first to set reasonable ylims without outliers
        ymin = np.inf
        ymax = -np.inf
        xmin = np.inf
        xmax = -np.inf
        for data_lc, model_lc, plot_ax, resid_ax in zip(
            data_lcs, model_lcs, plot_axs, resid_axs
        ):
            color = constants.COLOR_DICT[data_lc.bandpass]
            if (color == "white" and face_color is None) or color == face_color:
                color = "black"
            if ytype == "flux":
                y = model_lc.ujy
                dy = model_lc.dujy
            elif ytype == "mag":
                y = model_lc.mag
                dy = model_lc.dmag
            plot_ax.plot(model_lc.mjd, y, color=color)
            plot_ax.fill_between(
                model_lc.mjd,
                y - dy,
                y + dy,
                zorder=-1,
                color=color,
                alpha=0.5,
                linewidth=0,
            )
            ymin = min(ymin, plot_ax.get_ylim()[0])
            ymax = max(ymax, plot_ax.get_ylim()[1])
            xmin = min(xmin, plot_ax.get_xlim()[0])
            xmax = max(xmax, plot_ax.get_xlim()[1])
        yrange = ymax - ymin
        xrange = xmax - xmin
        for i, (data_lc, model_lc, plot_ax, resid_ax) in enumerate(
            zip(data_lcs, model_lcs, plot_axs, resid_axs)
        ):
            color = constants.COLOR_DICT[data_lc.bandpass]
            if (color == "white" and face_color is None) or color == face_color:
                color = "black"
            if data_lc.bandpass in "ZYJHK":
                magsys = "Vega"
                zp = 24
            else:
                magsys = "AB"
                zp = 23.9
            blc = data_lc.bin()
            if ytype == "flux":
                y = blc["ujy"]
                dy = blc["dujy"]
                model_y = model_lc.ujy
                model_dy = model_lc.dujy
                label = f"{data_lc.bandpass}: {magsys} {zp}"
            elif ytype == "mag":
                y = blc["mag"]
                dy = blc["dmag"]
                model_y = model_lc.mag
                model_dy = model_lc.dmag
                label = f"{data_lc.bandpass}: {magsys}"
            ylim_min_mask = y > ymin
            ylim_max_mask = y < ymax
            over_inds = np.where(~ylim_max_mask)
            under_inds = np.where(~ylim_min_mask)
            plot_ax.errorbar(
                blc["mjd"],
                y,
                yerr=dy,
                fmt=".",
                label=label,
                markersize=1,
                color=color,
            )
            for idx in over_inds[0]:
                plot_ax.arrow(
                    blc["mjd"][idx],
                    ymax - yrange / 10,
                    0,
                    yrange / 10,
                    head_width=xrange / 100,
                    head_length=yrange / 100,
                    length_includes_head=True,
                    color=color,
                )
            for idx in under_inds[0]:
                plot_ax.arrow(
                    blc["mjd"][idx],
                    ymin + yrange / 10,
                    0,
                    -yrange / 10,
                    head_width=xrange / 100,
                    head_length=yrange / 100,
                    length_includes_head=True,
                    color=constants.COLOR_DICT[data_lc.bandpass],
                )
            resid_ax.scatter(
                blc["mjd"],
                (y - np.interp(blc["mjd"], model_lc.mjd, model_y))
                / np.sqrt(dy**2 + np.interp(blc["mjd"], model_lc.mjd, model_dy) ** 2),
                color=color,
            )
            plot_ax.set_xlim(xmin, xmax)
            if ytype == "flux":
                plot_ax.set_ylim(ymin, ymax)
            elif ytype == "mag":
                plot_ax.set_ylim(ymax, ymin)
            plot_ax.legend(loc="best")
            plot_ax.axhline(0, color="k")
            resid_ax.axhline(0, color="k")
            plot_ax.set_facecolor(face_color)
            resid_ax.set_facecolor(face_color)
        if title == "default":
            title = str(self)
        plt.suptitle(title)
        plt.tight_layout()
        fig.subplots_adjust(hspace=0, wspace=0)
        if outdir is not None:
            fig.savefig(f"{outdir}/{self.model_name}.png", bbox_inches="tight")
        if show:
            plt.show()
        if return_fig:
            return fig, axs
        plt.close(fig)

    def view_plot(self):
        import PIL

        path = f"{self.fits_path}/{self.model_name}"
        if self.calibration is not None:
            path += f".{self.calibration}"
        path += ".png"
        PIL.Image.open(path).show()

    def summarize(self):
        s = {"DM", "Tmax", "EBVhost", "shape"}
        for param in s:
            s[param] = [
                self.get(param),
                self.get("e_" + param),
                self.systematics[param],
            ]
        return s, self.chi2

    def check_cuts(self, cuts):
        if not isinstance(cuts, dict):
            raise TypeError("cuts should be a dictionary")
        for param in cuts:
            if param.lower().replace("_", "") == "mwebv":
                if 0.86 * self.target.mwebv > cuts[param]:
                    return "mwebv"
            elif param.lower() in ("sep", "separation"):
                if (
                    SkyCoord(self.target.ra, self.target.dec, unit="deg")
                    .separation(
                        SkyCoord(
                            self.target.galaxy.ra, self.target.galaxy.dec, unit="deg"
                        )
                    )
                    .value
                    > cuts[param] / 3600
                ):
                    return "separation"
            elif (
                param == "min_bandpasses"
                and len(self.bandpasses_str.split("-")) < cuts["min_bandpasses"]
            ):
                return "min_bandpasses"

            elif (
                param == "min_obs_num"
                and sum(lc.detections().count() for lc in self.data_lightcurves)
                < cuts["min_obs_num"]
            ):
                return "min_obs_num"
            elif param.startswith("err_") or param.startswith("e_"):
                e_param = param.replace("err_", "").replace("e_", "")
                if e_param in self.errors and self.errors[e_param] > cuts[param]:
                    return param
            elif param in self.params:
                value = self.params[param]
                if hasattr(cuts[param], "__iter__") and len(cuts[param]) == 2:
                    if value < cuts[param][0] or value > cuts[param][1]:
                        return param
                elif np.abs(value) > cuts[param]:
                    return param
            elif (
                "salt" in param.lower()
                and "salt" in self.model_name
                and cuts["salt"] == "standard"
            ):
                std = constants.STANDARD_SALT_CUTS
                if np.abs(self.params["x1"]) > std["x1"]:
                    return "x1"
                if self.errors["x1"] > std["e_x1"]:
                    return "e_x1"
                if np.abs(self.params["c"]) > std["c"]:
                    return "c"
                if self.errors["c"] > std["e_c"]:
                    return "e_c"
            elif "chi2" in param.lower():
                if param == "chi2":
                    chi2_key = "reduced_data_model"
                elif "chi2_" in param:
                    chi2_key = param.replace("chi2_", "")
                if chi2_key not in self.chi2:
                    self.chi2 = self.get_chi2()
                    self.save()
                if isinstance(cuts[param], str) and cuts[param] == "standard":
                    chi2_model = self.model_name
                    if chi2_model not in constants.STANDARD_CHI2_CUTS:
                        chi2_model = "other"
                    chi2_cut_val = constants.STANDARD_CHI2_CUTS[chi2_model]
                if self.chi2[chi2_key] > chi2_cut_val:
                    return param
                continue

        if "phase" in cuts:
            if "salt" in self.model_name:
                tmax = self.params["t0"]
            elif "snpy" in self.model_name:
                tmax = self.params["Tmax"]
            first_phase = (min(min(lc.mjd) for lc in self.data_lightcurves) - tmax) / (
                1 + self.target.galaxy.z
            )
            last_phase = (max(max(lc.mjd) for lc in self.data_lightcurves) - tmax) / (
                1 + self.target.galaxy.z
            )
            if isinstance(cuts["phase"], str) and cuts["phase"] == "standard":
                if not any(
                    first_phase <= phases[0]
                    and last_phase >= phases[1]
                    and last_phase - first_phase >= phases[2]
                    for phases in constants.ACCEPTABLE_PHASE_COVERAGES
                ):
                    return "phase"
            elif hasattr(cuts["phase"], "__iter__") and len(cuts["phase"]) == 2:
                if first_phase > cuts["phase"][0] or last_phase < cuts["phase"][1]:
                    return "phase"

        if "salt3-nir-derivs" in cuts and self.model_name == "salt3-nir":
            deriv = self.get_salt_deriv()
            check = deriv.T[np.where(deriv["Parameter"] == "Check")[0][0]]
            tolerances = constants.DERIV_TOLS
            if cuts["salt3-nir-derivs"] == "strict":
                tolerances = constants.DERIV_TOLS_STRICT
            for key in tolerances:
                if check[key] < tolerances[key][0] or check[key] > tolerances[key][1]:
                    return "derivs"

        return "pass"

    def dm_from_mmax(
        self,
        bandpass,
        ebvhost=0,
        e_ebvhost=0.06,
        galaxy_mass=1e11,
        model="Tripp",
        R_x=None,
    ):
        bpmax = f"{bandpass}max"
        if bpmax not in self.params:
            return None, None
        if model == "Tripp":
            if "Bmax" not in self.params or "Vmax" not in self.params:
                return None, None
            else:
                return dm_from_mmax_tripp(bandpass, galaxy_mass)
        if model == "Reddening":
            if R_x is None:
                return None, None
            else:
                return dm_from_mmax_reddening(
                    bandpass,
                    ebvhost=ebvhost,
                    e_ebvhost=e_ebvhost,
                    galaxy_mass=galaxy_mass,
                    R_x=R_x,
                )

    def dm_from_mmax_tripp(
        self,
        bandpass,
        galaxy_mass=1e11,
    ):
        mag = self.params[f"{bandpass}Max"]
        e_mag = self.errors[f"{bandpass}Max"]
        st = self.params["st"]
        e_st = self.params["e_st"]
        color = self.params["Bmax"] - self.params["Vmax"]
        e_color = np.sqrt(self.errors["Bmax"] ** 2 + self.errors["Vmax"] ** 2)
        # Burns 2018, https://iopscience.iop.org/article/10.3847/1538-4357/aae51c/pdf
        if st > 0.5 and color < 0.5:
            sample = "st > 0.5 and B-V < 0.5"
        elif color < 0.5:
            sample = "B-V < 0.5"
        elif st > 0.5:
            sample = "st > 0.5"
        else:
            sample = "all"
        (
            p0,
            e_p0,
            p1,
            e_p1,
            p2,
            e_p2,
            rxbv,
            e_rxbv,
            rv,
            e_rv,
            alpha,
            e_alpha,
            disp,
            v_pec,
        ) = constants.SNPY_M_TO_DM[sample][bandpass]
        st -= 1
        DM = (
            mag
            - p0
            - p1 * st
            - p2 * st**2
            - rxbv * color
            - alpha * np.log10(galaxy_mass / 1e11)
        )  # paper has it as - 1e11
        # should probably have covariances built in here
        e_DM = np.sqrt(
            disp**2
            + e_mag**2
            + e_p0**2
            + (e_p1 * st) ** 2
            + (p1 * e_st) ** 2
            + (e_p2 * st**2) ** 2
            + (2 * p2 * st * e_st) ** 2
            + (e_alpha * np.log10(galaxy_mass / 1e11)) ** 2
            + (rxbv * e_color) ** 2
            + (e_rxbv * color) ** 2
        )
        return DM, e_DM

    def dm_from_mmax_reddening(
        self, bandpass, ebvhost=0, e_ebvhost=0.06, galaxy_mass=1e11, R_x=None
    ):
        # Burns 2018, https://iopscience.iop.org/article/10.3847/1538-4357/aae51c/pdf
        mag = self.params[f"{bandpass}max"]
        e_mag = self.errors[f"{bandpass}max"]
        st = self.params["st"]
        e_st = self.errors["st"]
        if st > 0.5 and ebvhost < 0.5:
            sample = "st > 0.5 and E(B-V) < 0.5"
        elif ebvhost < 0.5:
            sample = "E(B-V) < 0.5"
        elif st > 0.5:
            sample = "st > 0.5"
        else:
            sample = "all"
        (
            p0,
            e_p0,
            p1,
            e_p1,
            p2,
            e_p2,
            alpha,
            e_alpha,
            sigma_x,
            sigma_cv,
            v_pec,
        ) = constants.SNPY_M_TO_DM_REDDENING[sample][bandpass]
        st -= 1
        DM = (
            mag
            - p0
            - p1 * st
            - p2 * st**2
            - R_x * ebvhost
            - alpha * np.log10(galaxy_mass / 1e11)
        )  # paper has it as - 1e11
        # should probably have covariances built in here
        e_DM = np.sqrt(
            sigma_cv**2
            + e_mag**2
            + e_p0**2
            + (e_p1 * st) ** 2
            + (p1 * e_st) ** 2
            + (e_p2 * st**2) ** 2
            + (2 * p2 * st * e_st) ** 2
            + (e_alpha * np.log10(galaxy_mass / 1e11)) ** 2
            + (R_x * e_ebvhost) ** 2
        )
        return DM, e_DM

    def salt_dm(self, alpha="default", beta="default"):
        from fitting.SALT3 import get_B0

        if alpha == "default":
            alpha = constants.ALPHA[self.model_name]
        if beta == "default":
            beta = constants.BETA[self.model_name]
        # eqn 16 in Kenworthy 2021)
        P = self.get_salt_results()[0]
        m = get_B0(P)["b"]

        DM = (
            m
            - constants.FIDUCIAL_M
            + alpha * self.params.get("x1", 0)
            - beta * self.params.get("c", 0)
        )
        z = 0
        if self.target.galaxy and self.target.galaxy.z:
            z = self.target.galaxy.z
        e_DM = np.sqrt(
            # constants.SIGMA_INT ** 2
            constants.SIGMA_MU_Z**2
            + (2.5 * np.log10(1 + self.errors.get("x0", 0) / self.params.get("x0", 1)))
            ** 2
            + self.errors.get("x0_mag", 0) ** 2
            + (alpha * self.errors.get("x1", 0)) ** 2
            + (beta * self.errors.get("c", 0)) ** 2
            + (constants.SIGMA_LENS * z) ** 2
            + 2 * alpha * beta * self.covariance.get("c", {}).get("x1", 0)
            + 2 * alpha * self.covariance.get("x0", {}).get("x1", 0)
            + 2 * beta * self.covariance.get("x0", {}).get("c", 0)  # m_b,c
        )

        return DM, e_DM

    def salt_model_uncertainties(self, mjds, bps):
        if "salt" not in self.model_name:
            return TypeError("Not a SALT model")

        from fitting.FileRead import read_param
        from fitting.SALT3 import add_model_cov, init_band

        if isinstance(bps, str):
            bps = np.array([bps] * len(mjds))
        tmp_lc2data = dict(
            date=mjds,
            lc2fl=bps,
            covmat=np.diag(np.ones(len(mjds), dtype=np.float64)) * 1e-10,
            model_uncertainties=0,
        )

        if self.model_name == "salt3-nir":
            P, _, _, _, _, other_data = self.get_salt_results()
            tmp_lc2data, _ = add_model_cov(P, tmp_lc2data, other_data)
            return np.sqrt(np.diag(tmp_lc2data["covmat"]))

        P = [
            self.params["t0"],
            -2.5 * np.log10(self.params["x0"]),
            self.params["x1"],
            self.params["c"],
        ]
        other_data = {
            "z_heliocentric": self.target.galaxy.z,
            "MWEBV": self.target.mwebv,
            "bandpassfn": {},
            "bandpasslambs": {},
            "bandpasseval": {},
            "magsysflux": {},
            "efflamb": {},
            "MagSys|Instrument|Band": {},
        }
        for bp in bps:
            other_data["MagSys|Instrument|Band"][bp] = (
                read_param(bp, "@MAGSYS", verbose=False)
                + "|"
                + read_param(bp, "@INSTRUMENT", verbose=False)
                + "|"
                + read_param(bp, "@BAND", verbose=False)
            )
            other_data = init_band(other_data=other_data, lc2fl=bp, alpha=0.0)

        tmp_lc2data, _ = add_model_cov(P, tmp_lc2data, other_data)
        return np.sqrt(np.diag(tmp_lc2data["covmat"]))
