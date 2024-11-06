import numpy as np
import pandas as pd
from astropy.io import fits
import snpy
from datetime import datetime

import constants
import utils
from targets.models import Target

bad_list = [  # From CSP
    "05gj",  # Ia-CSM
    "05hk",  # Iax
    "06bt",  # 06bt
    "06dd",  # normal, but could not fit, see Stritzinger 2010.
    "06ot",  # 06bt
    "07if",  # SC
    "08J",  # Ia-CSM
    "08ae",  # Iax
    "08ha",  # Iax
    "09J",  # Iax
    "09dc",  # SC
    "10ae",  # Iax
]

needs_help_to_fit = [
    "05hj",  # Fit without V. (normal)
    "05ke",  # Fit without u or V first, then without V. (91bg)
    "06mr",  #  Fit without Ydw, with Tmax=1050, st=1, then fit without Ydw (91bg)
    "07al",  # Fit with redlaw='O94' first, then with F19. (91bg)
    "07ax",  # Fit with 'BVriY' Tmax=1188, st=0.5, then refit with all but Ydw. (91bg)
    "07N",  # Fit without u (91bg)
    "08bd",  # Fit without Ydw with Tmax=1530, results are sensitive to Tmax guess, then fit without Ydw. (91bg)
    "08bi",  # Fit without u (91bg)
    "08cd",  # I could not fix it (normal)
    "09F",  # Fit 'Bgri' first, then all (91bg)
    "09al",  # Fit without J (normal)
]


def clean_CSP_input_files(read_path, write_path):
    with open(read_path, "r") as f:
        lines = f.readlines()
    with open(write_path, "w") as f:
        for i, line in enumerate(lines):
            if line.startswith("filter") and i == len(lines):
                # Doesn't seem to occur, but skips filters at the end with no data.
                continue
            elif line.startswith("filter") and lines[i + 1].startswith("filter"):
                # Skips filters with no data everywhere else
                continue
            f.write(line)


def make_fit_result(path, calibration=None, model_name="snpy_max_model", redlaw="F19"):
    # In max_model, with redlaw='F19', will fail for objects in needs_help_to_fit
    TNS_name = path.split("/")[-1].replace("SN20", "").replace("_snpy.dat", "")
    with open(path, "r") as f:
        _, z, ra, dec = f.readlines()[0].split()
    obj = Target.objects.update_or_create(
        TNS_name=TNS_name, defaults={"ra": float(ra), "dec": float(dec)}
    )[0]
    sn = snpy.get_sn(path)
    priors = {}
    stype = "st"
    if "ebv_model2" in model_name:
        if calibration is None:
            priors["calibration"] = 0
        elif calibration in np.arange(10, 16):
            stype = "dm15"
            priors["calibration"] = calibration - 10
        sn.choose_model("EBV_model2", stype=stype)
    elif "max_model" in model_name:
        calibration = None
        sn.choose_model("max_model", stype=stype)
    good_bps = list(sn.data.keys())
    success = False
    sn.redlaw = redlaw
    while len(good_bps) > 1 and not success:
        try:
            sn.fit(good_bps, **priors)
            print(f"Fit successful with {good_bps}")
            success = True
        except RuntimeError as e:
            if str(e).startswith("All weights for filter"):
                good_bps.remove(str(e).split()[4])
                print(f"Trying again with only {good_bps}")
        except (TypeError, ValueError) as e:
            print("Failed")
            print(str(e))
            break
    if success and len(good_bps) != len(sn.data):
        print("Trying to fit with all bandpasses")
        try:
            sn.fit(list(sn.data.keys()), **priors)
            print("Success!")
        except (RuntimeError, IndexError, TypeError) as e:
            success = False
            print("Failed")
            print(str(e))
            return
    res, created = obj.fit_results.get_or_create(
        model_name=model_name,
        bandpasses_str="-".join(sorted(sn.data.keys())),
        variants_str="CSP",
        calibration=calibration,
        redlaw=redlaw,
    )
    res.success = True
    res.params = sn.parameters
    res.stat_errors = sn.errors
    res.sys_errors = sn.systematics()
    res.errors = {}
    for (n, stat), sys_errs in zip(sn.errors.items(), sn.systematics().values()):
        res.errors[n] = np.sqrt(stat**2 + sys_errs**2)
    try:
        res.chi2 = res.get_chi2()
    except Exception as e:
        pass
    res.last_updated = datetime.now()
    res.save()
    return res


def initialize():
    csp = pd.read_csv("ado/CSPDR3/tab1.dat", delimiter="\t")
    neill09_sne = pd.read_csv("ado/neil09_sne.dat")
    neill09_gal = pd.read_csv("ado/neil09_masses.dat")
    sw_input = fits.open("ado/sw_input.fits")
    sw_output = fits.open("ado/sw_output.fits")
    my_z, csp_z = [np.zeros(len(csp["SN"])) for i in range(2)]
    names = np.zeros(len(csp["SN"]), dtype="<U8")
    k_mag, k_dmag = [
        {
            "2MASS": np.zeros(len(csp["SN"])),
            "neill09": np.zeros(len(csp["SN"])),
            "chang15": np.zeros(len(csp["SN"])),
        }
        for i in range(2)
    ]
    input_dict = {
        "N": 0,
        "N_unknown": 0,
        "st": [],
        "e_st": [],
        "mag": [],
        "e_mag": [],
        "c1": [],
        "e_c1": [],
        "c2": [],
        "e_c2": [],
        "mu_lcdm": [],
        "z_helio": [],
        "tm_k_mag": [],
        "catalog_logm": [],
        "unknown_mass_idx": [],
    }
    return (
        csp,
        neill09_sne,
        neill09_gal,
        sw_input,
        sw_output,
        my_z,
        csp_z,
        names,
        k_mag,
        k_dmag,
        input_dict,
    )


def add_redshift_info(i, obj, row, my_z, csp_z, names):
    if obj.galaxy and obj.galaxy.z:
        my_z[i] = obj.galaxy.z
    else:
        my_z[i] = -1
    csp_z[i] = float(row["zhelio"])
    names[i] = obj.TNS_name


def add_2mass(i, obj, row, k_mag, k_dmag):
    if obj.galaxy is None or not obj.galaxy.ned_entries.exists():
        return
    ned_set = obj.galaxy.ned_entries.exclude(prefname__startswith="SN ")
    n = ned_set.first()
    phot_set = n.photometry.filter(
        observed_passband="K_s",
        refcode="20032MASX.C.......:",
        qualifiers__endswith="arcsec integration area.",
    )
    if phot_set.exists():
        k_mag["2MASS"][i] = phot_set[0].photometry_measurement
        k_dmag["2MASS"][i] = str(phot_set[0].uncertainty).replace("+/-", "")


def add_neill09(i, obj, row, k_mag, k_dmag, neill09_sne, neill09_gal):
    name_idx = np.where(neill09_sne["TNS_name"] == row["SN"])
    if len(name_idx[0]):
        host_name = neill09_sne["Host"][name_idx[0][0]]
        host_idx = np.where(neill09_gal["name"] == host_name)
        if len(host_idx[0]):
            k_mag["neill09"][i] = neill09_gal["m"][host_idx[0][0]]
            k_dmag["neill09"][i] = np.average(
                [neill09_gal["m-"][host_idx[0][0]], neill09_gal["m-"][host_idx[0][0]]]
            )


def add_chang15(i, obj, row, k_mag, k_dmag, sw_input, sw_output, threshold=1):
    ra, dec = obj.galaxy.ra, obj.galaxy.dec
    dist = np.sqrt(
        (ra - sw_input[1].data["ra"]) ** 2 * np.cos(dec * np.pi / 180) ** 2
        + (dec - sw_input[1].data["dec"]) ** 2
    )
    if min(dist) * 3600 < threshold:
        k_mag["chang15"][i] = sw_output[1].data["lmass50_all"][np.argmin(dist)]
        k_dmag["chang15"][i] = np.average(
            [
                sw_output[1].data["lmass16_all"][np.argmin(dist)],
                sw_output[1].data["lmass84_all"][np.argmin(dist)],
            ]
        )


def add_to_input_dict(
    i,
    obj,
    row,
    input_dict,
    TNS_names,
    fr,
    k_mag,
    k_dmag,
    mag="J",
    color_1="B",
    color_2="V",
):
    input_dict["N"] += 1
    TNS_names.append(obj.TNS_name)
    input_dict["st"].append(fr.params["st"])
    input_dict["e_st"].append(fr.errors["st"])
    input_dict["mag"].append(fr.params[f"{mag}max"])
    input_dict["e_mag"].append(fr.errors[f"{mag}max"])
    input_dict["c1"].append(fr.params[f"{color_1}max"])
    input_dict["e_c1"].append(fr.errors[f"{color_1}max"])
    input_dict["c2"].append(fr.params[f"{color_2}max"])
    input_dict["e_c2"].append(fr.errors[f"{color_2}max"])
    input_dict["z_helio"].append(float(row["zhelio"]))
    input_dict["mu_lcdm"].append(
        utils.mu_lcdm(
            float(row["zhelio"]),
            utils.convert_z(float(row["zhelio"]), obj.ra, obj.dec),
            72,
            -0.53,
        )
    )
    input_dict["tm_k_mag"].append(k_mag["2MASS"][i])
    if k_mag["neill09"][i]:
        input_dict["catalog_logm"].append(k_mag["neill09"][i])
    elif k_mag["chang15"][i]:
        input_dict["catalog_logm"].append(k_mag["chang15"][i])
    else:
        input_dict["catalog_logm"].append(0)
        if k_mag["2MASS"][i] == 0:
            input_dict["N_unknown"] += 1
            input_dict["unknown_mass_idx"].append(input_dict["N"])


def fit(input_dict):
    data = """
    int<lower=1> N;
    int<lower=0> N_unknown;
    array[N_unknown] int<lower=1, upper=N> unknown_mass_idx;
    vector<lower=0, upper=4> [N] st;
    vector<lower=0, upper=1> [N] e_st;
    vector<lower=10, upper=24> [N] mag;
    vector<lower=0> [N] e_mag;
    vector<lower=10, upper=24> [N] c1;
    vector<lower=0> [N] e_c1;
    vector<lower=10, upper=24> [N] c2;
    vector<lower=0> [N] e_c2;
    vector<lower=30, upper=40> [N] mu_lcdm;
    vector<lower=0, upper=1> [N] z_helio;
    vector<lower=0, upper=15> [N] tm_k_mag;
    vector<lower=0, upper=13> [N] catalog_logm;
    """
    transformed_data = """
    vector<lower=-5, upper=5> [N] color;
    vector<lower=0> [N] e_color;
    color = c1 - c2;
    e_color = sqrt(square(e_c1) + square(e_c2)) ; // no correlation yet
    """
    params = """
    real<lower=-20, upper=-17> P0 ;
    real<lower=-5, upper=5> P1 ;
    real<lower=-5, upper=5> P2 ;
    real alpha ;
    real<lower=0> beta ;
    real<lower=0> v_pec ;
    real<lower=0> sig_int ;
    vector<lower=9, upper=10.5> [N_unknown] unknown_logm ;
    """
    transformed_params = """
    vector [N] logm ;
    vector [N] abs_mag ;
    for (i in 1:N) {
        if (tm_k_mag[i] != 0)
            logm[i] = (-0.4*(tm_k_mag[i] - mag[i] + P0 + P1*(st[i] - 1) + P2*square(st[i]-1) + beta*color[i] - 11*alpha) + 1.04)/(1+0.4*alpha);
        else if (catalog_logm[i] != 0)
            logm[i] = catalog_logm[i];
    }
    for (i in 1:N_unknown) {
        logm[unknown_mass_idx[i]] = unknown_logm[i] ;
    }
    abs_mag = P0 + P1*(st - 1) + P2*square(st-1) + beta*color + alpha*(logm - 11);
    """
    model = """
    for (i in 1:N) {
        target += normal_lpdf(mag[i] | mu_lcdm[i] + abs_mag[i], sqrt(square(e_mag[i]) + square(sig_int) + square(v_pec*2.17147*(1+z_helio[i])/(z_helio[i]*(1+0.5*z_helio[i])))));
    }
    for (i in 1:N_unknown) {
        target += uniform_lpdf(unknown_logm[i] | 9, 11.5);
    }
    """
    generated_quantities = """
    vector [N] resid_DM;
    resid_DM = mag - abs_mag - mu_lcdm;
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
    return fit


def fit_no_mass(input_dict):
    data = """
    int<lower=1> N;
    int<lower=0> N_unknown;
    array[N_unknown] int<lower=1, upper=N> unknown_mass_idx;
    vector<lower=0, upper=4> [N] st;
    vector<lower=0, upper=1> [N] e_st;
    vector<lower=10, upper=24> [N] mag;
    vector<lower=0> [N] e_mag;
    vector<lower=10, upper=24> [N] c1;
    vector<lower=0> [N] e_c1;
    vector<lower=10, upper=24> [N] c2;
    vector<lower=0> [N] e_c2;
    vector<lower=30, upper=40> [N] mu_lcdm;
    vector<lower=0, upper=1> [N] z_helio;
    vector<lower=0, upper=15> [N] tm_k_mag;
    vector<lower=0, upper=13> [N] catalog_logm;
    """
    transformed_data = """
    vector<lower=-5, upper=5> [N] color;
    vector<lower=0> [N] e_color;
    color = c1 - c2;
    e_color = sqrt(square(e_c1) + square(e_c2)) ; // no correlation yet
    """
    params = """
    real<lower=-20, upper=-17> P0 ;
    real<lower=-5, upper=5> P1 ;
    real<lower=-5, upper=5> P2 ;
    real alpha ;
    real<lower=0> beta ;
    real<lower=0> v_pec ;
    real<lower=0> sig_int ;
    vector<lower=9, upper=10.5> [N_unknown] unknown_logm ;
    """
    transformed_params = """
    // vector [N] logm ;
    vector [N] abs_mag ;
    // for (i in 1:N) {
    //     if (tm_k_mag[i] != 0)
    //         logm[i] = (-0.4*(tm_k_mag[i] - mag[i] + P0 + P1*(st[i] - 1) + P2*square(st[i]-1) + beta*color[i] - 11*alpha) + 1.04)/(1+0.4*alpha);
    //     else if (catalog_logm[i] != 0)
    //         logm[i] = catalog_logm[i];
    // }
    // for (i in 1:N_unknown) {
    //     logm[unknown_mass_idx[i]] = unknown_logm[i] ;
    // }
    abs_mag = P0 + P1*(st - 1) + P2*square(st-1) + beta*color; // + alpha*(logm - 11);
    """
    model = """
    for (i in 1:N) {
        target += normal_lpdf(mag[i] | mu_lcdm[i] + abs_mag[i], sqrt(square(e_mag[i]) + square(sig_int) + square(v_pec*2.17147*(1+z_helio[i])/(z_helio[i]*(1+0.5*z_helio[i])))));
    }
    // for (i in 1:N_unknown) {
    //     target += uniform_lpdf(unknown_logm[i] | 9, 11.5);
    // }
    """
    generated_quantities = """
    vector [N] resid_DM;
    resid_DM = mag - abs_mag - mu_lcdm;
    """
    fit = utils.stan(
        input_dict=input_dict,
        data=data,
        transformed_data=transformed_data,
        params=params,
        transformed_params=transformed_params,
        model=model,
        generated_quantities=generated_quantities,
        pickle_path=f"{constants.STATIC_DIR}/max_model_no_mass.stancode",
    )
    return fit


def get_tripp_params(mag="J", color_1="B", color_2="V"):
    (
        csp,
        neill09_sne,
        neill09_gal,
        sw_input,
        sw_output,
        my_z,
        csp_z,
        names,
        k_mag,
        k_dmag,
        input_dict,
    ) = initialize()
    cut, TNS_names, unknown_mass_idx = [[] for _ in range(3)]
    for i, row in csp.iterrows():
        if row["SN"][2:] in bad_list:
            continue
        try:
            obj = Target.quick_get(row["SN"][2:])
        except:
            continue
        fr_set = obj.fit_results.filter(
            variants_str="CSP", model_name="snpy_max_model", redlaw="F19", success=True
        )
        if not fr_set.exists():
            cut.append(obj.TNS_name)
            continue
        fr = fr_set.first()
        if (
            f"{mag}max" not in fr.params
            or f"{color_1}max" not in fr.params
            or f"{color_2}max" not in fr.params
        ):
            continue
        # if fr.params["Bmax"] - fr.params["Vmax"] >= 0.5:
        #     continue
        # if fr.params["st"] <= 0.5:
        #     continue
        add_redshift_info(i, obj, row, my_z, csp_z, names)
        add_2mass(i, obj, row, k_mag, k_dmag)
        add_neill09(i, obj, row, k_mag, k_dmag, neill09_sne, neill09_gal)
        add_chang15(i, obj, row, k_mag, k_dmag, sw_input, sw_output, threshold=20)
        mass_idx = add_to_input_dict(
            i,
            obj,
            row,
            input_dict,
            TNS_names,
            fr,
            k_mag,
            k_dmag,
            mag=mag,
            color_1=color_1,
            color_2=color_2,
        )
        if mass_idx is not None:
            unknown_mass_idx.append(mass_idx)
    res = fit(input_dict)
    res_no_mass = fit_no_mass(input_dict)
    params = {}
    errors = {}
    params_no_mass = {}
    errors_no_mass = {}
    for param in ("P0", "P1", "P2", "beta", "sig_int"):
        params[param] = np.median(res[param])
        errors[param] = np.std(res[param])
        params_no_mass[param] = np.median(res_no_mass[param])
        errors_no_mass[param] = np.std(res_no_mass[param])
    params["v_pec"] = np.median(res["v_pec"]) * constants.C
    errors["v_pec"] = np.std(res["v_pec"]) * constants.C
    params_no_mass["v_pec"] = np.median(res["v_pec"]) * constants.C
    errors_no_mass["v_pec"] = np.std(res["v_pec"]) * constants.C
    params["alpha"] = np.median(res["alpha"])
    errors["alpha"] = np.std(res["alpha"])
    return (
        input_dict,
        k_mag,
        k_dmag,
        res,
        params,
        errors,
        params_no_mass,
        errors_no_mass,
    )
