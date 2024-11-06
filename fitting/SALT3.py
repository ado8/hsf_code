import subprocess
import sys
from copy import deepcopy

import constants
import matplotlib.pyplot as plt
import numpy as np
import sncosmo
from scipy.interpolate import interp1d
from utils import print_verb, twod_block_stack

import fitting.Spectra as Spectra
from fitting.FileRead import format_file, read_param, writecol
from fitting.RubinsNM import miniNM_new, save_img


def sncosmo_model(model_name="salt3-nir", MW_redlaw="F99"):
    if MW_redlaw == "old_F99":
        return old_F99model
    if MW_redlaw == "F99":
        return F99model
    elif MW_redlaw == "F19":
        return F19model
    elif MW_redlaw == "CCM89":
        return CCM89model
    elif MW_redlaw == "O94":
        return O94model


def model_fn(
    P,
    trimmed_lc_data,
    other_data,
    verbose=False,
):
    """model_fn.

    Parameters
    ----------
    P :
        P from DavidsNM
    trimmed_lc_data :
        trimmed_lc_data
    other_data :
        other_data
    """
    the_model = np.zeros(len(trimmed_lc_data["lc2fl"]), dtype=np.float64)
    model = sncosmo_model(
        model_name=other_data["model_name"], MW_redlaw=other_data["MW_redlaw"]
    )
    model.set(
        x0=10.0 ** (-0.4 * P[1]),
        x1=P[2],
        c=P[3],
        t0=P[0],
        z=other_data["z_heliocentric"],
        MWebv=other_data["MWEBV"],
    )  # , MWr_v = 3.1)

    for lc2fl in set(trimmed_lc_data["lc2fl"]):
        inds = np.where(trimmed_lc_data["lc2fl"] == lc2fl)

        try:
            tmp_flux = model.flux(
                trimmed_lc_data["date"][inds], other_data["bandpasslambs"][lc2fl]
            )
        except:
            print_verb(verbose, "Couldn't get model flux for ", lc2fl)
            sys.exit(1)

        tmp_flux *= (
            other_data["bandpasslambs"][lc2fl] * other_data["bandpasseval"][lc2fl]
        )

        the_model[inds] = np.sum(tmp_flux, axis=1) / other_data["magsysflux"][lc2fl]

    return the_model


def get_B0(P, model_name="salt3-nir", MW_redlaw="F99"):
    """get_B0.
    uses a global "model" param

    Parameters
    ----------
    P :
        P from DavidsNM
    """
    model = sncosmo_model(model_name=model_name, MW_redlaw=MW_redlaw)
    model.set(
        x0=10.0 ** (-0.4 * P[1]), x1=P[2], c=P[3], t0=P[0], z=0.0, MWebv=0.0
    )  # , MWr_v = 3.1)

    rest_filters = {}
    for filt in ["ux", "b", "v", "r"]:
        rest_filters[filt] = model.bandmag("bessell" + filt, "vega", P[0]) + 0.03
    return rest_filters


def chi2_fn(
    P,
    pass_data,
    simul_chi2=False,
    robust_chi2=False,
):
    """chi2_fn.

    Parameters
    ----------
    P :
        P from DavidsNM
    pass_data :
        pass_data
    """
    trimmed_lc_data, other_data = pass_data[0]
    the_model = model_fn(P, trimmed_lc_data, other_data)

    resid = trimmed_lc_data["flux"] - the_model
    chi2 = np.dot(resid, np.dot(trimmed_lc_data["weightmat"], resid))
    if simul_chi2:
        trimmed_lc_data = add_model_cov(P, deepcopy(trimmed_lc_data))

        (NA, log_det_w) = np.linalg.slogdet(trimmed_lc_data["weightmat"])
        log_det_c = -log_det_w
    else:
        log_det_c = 0.0

    chi2 = np.dot(resid, np.dot(trimmed_lc_data["weightmat"], resid)) + log_det_c

    if robust_chi2:
        assert simul_chi2 == 0

        chi2_outl = 0.0
        for i in range(len(resid)):
            chi2_outl += -2.0 * np.logaddexp(
                -0.5 * resid[i] ** 2.0 / trimmed_lc_data["covmat"][i, i] + np.log(0.99),
                -0.5 * resid[i] ** 2.0 / (25.0 * trimmed_lc_data["covmat"][i, i])
                + np.log(0.01),
            )

        LL_total = np.logaddexp(
            -0.5 * chi2 + np.log(0.99), -0.5 * chi2_outl + np.log(0.01)
        )
        chi2 = -2.0 * LL_total

    return chi2


def residfn(P, pass_data):
    # assert simul_chi2 == 0
    # assert robust_chi2 == 0
    trimmed_lc_data, other_data = pass_data[0]
    the_model = model_fn(P, trimmed_lc_data, other_data)

    resid = trimmed_lc_data["flux"] - the_model

    Lmat = np.linalg.cholesky(trimmed_lc_data["weightmat"])
    return np.dot(resid, Lmat)


def add_model_cov(P, trimmed_lc_data, other_data):
    """add_model_cov.

    Parameters
    ----------
    P :
        P
    trimmed_lc_data :
        trimmed_lc_data
    other_data :
        other_data
    """
    assert trimmed_lc_data["model_uncertainties"] == 0
    trimmed_lc_data["model_uncertainties"] = 1

    source.set(x1=P[2], c=P[3])

    rest_frame_bands = {}
    for lc2fl in trimmed_lc_data["lc2fl"]:
        rest_frame_bands[lc2fl] = sncosmo.Bandpass(
            other_data["bandpasslambs"][lc2fl] / (1.0 + other_data["z_heliocentric"]),
            other_data["bandpasseval"][lc2fl],
        )

    the_model = model_fn(P, trimmed_lc_data, other_data)
    r_cov = source.bandflux_rcov(
        band=np.array([rest_frame_bands[item] for item in trimmed_lc_data["lc2fl"]]),
        phase=(trimmed_lc_data["date"] - P[0]) / (1.0 + other_data["z_heliocentric"]),
    )

    r_cov = np.clip(r_cov, -1, 1)
    """
    r_cov = np.zeros([len(the_model)]*2, dtype=np.float64)
    for lc2fl in trimmed_lc_data["lc2fl"]:
        inds = np.where(trimmed_lc_data["lc2fl"] == lc2fl)

        tmp_r_cov = source.bandflux_rvar_single(rest_frame_bands[lc2fl], phase = (trimmed_lc_data["date"][inds] - P[0])/(1. + other_data["z_heliocentric"]))[1]
        print(tmp_r_cov)
        assert len(inds[0]) == len(tmp_r_cov)

        for i, indi in enumerate(inds[0]):
            for j, indj in enumerate(inds[0]):
                r_cov[indi, indj] = tmp_r_cov[i,j]
    """

    cov_mat = np.outer(the_model, the_model) * r_cov

    # save_img(cov_mat, "cov_mat.fits")
    # save_img(r_cov, "r_cov.fits")
    trimmed_lc_data["covmat"] += cov_mat
    trimmed_lc_data["weightmat"] = np.linalg.inv(trimmed_lc_data["covmat"])

    return trimmed_lc_data, other_data


def model_for_plotting(P, lc2fl, mindate, maxdate, other_data):
    """model_for_plotting.

    Parameters
    ----------
    P :
        P
    lc2fl :
        lc2fl
    mindate :
        mindate
    maxdate :
        maxdate
    other_data :
        other_data
    """
    plot_dates = np.arange(mindate, maxdate + 0.5, 0.5)
    tmp_lc2data = dict(
        date=plot_dates,
        lc2fl=np.array([lc2fl] * len(plot_dates)),
        covmat=np.diag(np.ones(len(plot_dates), dtype=np.float64)) * 1e-10,
        model_uncertainties=0,
    )
    tmp_lc2data, tmp_other_data = add_model_cov(P, tmp_lc2data, other_data)
    plot_model = model_fn(P, tmp_lc2data, tmp_other_data)

    return (
        plot_dates,
        plot_model,
        np.sqrt(np.diag(tmp_lc2data["covmat"])),
    )


# in utils, remove after debuggin
def twod_block_stack(m1, m2):
    assert m1.shape[0] == m1.shape[1]
    assert m2.shape[0] == m2.shape[1]

    m3 = np.zeros([len(m1) + len(m2)] * 2, dtype=np.float64)
    m3[: len(m1), : len(m1)] = m1
    m3[len(m1) :, len(m1) :] = m2

    return m3


def init_band(other_data, lc2fl, alpha, verbose=False):
    """init_band.

    Parameters
    ----------
    other_data :
        other_data
    lc2fl :
        lc2fl
    alpha :
        alpha
    """
    X_FOCAL_PLANE = read_param(lc2fl, "@X_FOCAL_PLANE", verbose=verbose)
    Y_FOCAL_PLANE = read_param(lc2fl, "@Y_FOCAL_PLANE", verbose=verbose)

    if X_FOCAL_PLANE is not None:
        radialpos = np.sqrt(X_FOCAL_PLANE**2.0 + Y_FOCAL_PLANE**2.0)
    else:
        radialpos = None

    band_object = Spectra.Spectra(
        instrument=read_param(lc2fl, "@INSTRUMENT", verbose=verbose),
        band=read_param(lc2fl, "@BAND", verbose=verbose),
        magsys=read_param(lc2fl, "@MAGSYS", verbose=verbose),
        radialpos=radialpos,
    )

    lc2_key = lc2fl + (alpha != 0) * "_dalpha"
    other_data["bandpassfn"][lc2_key] = lambda x: band_object.transmission_fn(
        x
    ) * np.exp(alpha * x)

    tmp_lambs = np.arange(2000.0, 25000.0, d_lamb)
    inds = np.where(
        other_data["bandpassfn"][lc2_key](tmp_lambs)
        > other_data["bandpassfn"][lc2_key](tmp_lambs).max() * band_clip
    )[0]
    other_data["bandpasslambs"][lc2_key] = tmp_lambs[inds[0] : inds[-1] + 1]
    other_data["bandpasseval"][lc2_key] = other_data["bandpassfn"][lc2_key](
        other_data["bandpasslambs"][lc2_key]
    )
    other_data["magsysflux"][lc2_key] = 10.0 ** (-0.4 * global_zp) * np.sum(
        other_data["bandpasslambs"][lc2_key]
        * other_data["bandpasseval"][lc2_key]
        * band_object.ref_fn(other_data["bandpasslambs"][lc2_key])
    )
    other_data["efflamb"][lc2_key] = np.sum(
        other_data["bandpasslambs"][lc2_key] ** 2.0
        * other_data["bandpasseval"][lc2_key]
    ) / np.sum(
        other_data["bandpasslambs"][lc2_key] * other_data["bandpasseval"][lc2_key]
    )
    return other_data


def init_data(verbose=False):
    """init_data.
    Create holders for data
    Uses Spectra.get_lc2data to read sysargs (probably paths) into a Spectra
    populate holders

    Returns
    -------
    all_lc_data:
        dict with keys for list:lc2fl: np.array:date, np.array:flux, np.array(2D):covmat
    other_data:
        dict with keys for fn:bandpassfn, np.array:bandpasslambdas, np.array:bandpasseval, float:magsysflux, float:efflamb, str:Magsys|Instrument|Band, float:daymax_guess, float:z_heliocentric, float:MWebv
    """
    other_data = {
        "bandpassfn": {},
        "bandpasslambs": {},
        "bandpasseval": {},
        "magsysflux": {},
        "efflamb": {},
        "MagSys|Instrument|Band": {},
    }
    all_lc_data = {
        "lc2fl": [],
        "date": [],
        "flux": [],
        "covmat": np.zeros([0, 0], dtype=np.float64),
    }  # all_lc_data is all data; trimmed_lc_data is data in phase range and possibly with model uncertainties

    for lc2fl in sys.argv[1:]:
        tmp_lc_data = Spectra.get_lc2data(
            lc2fl, global_zp=global_zp, single_offset=False
        )  # single_offset = False will return the actual inverse variance, not an approximation of it
        # get_lc2data
        # reads off str for instrument, band, magsys, with read_param
        # make "radius" from X/Y_FOCAL_PLANE, or None
        # gets floats date, flux, flux_err, flux_zp with readcol "ffff"
        # if not 4 cols, reads date, mag, mag_err with readcol "fff"
        # reads weight_fl path with read_param @WEIGHTMAT
        # makes weights, offset with read_weightmat
        # not sure where the weight matrix comes from...
        # returns dict with str:instrument, str:band, str:magsys, float:radius, np.array:flux, np.array:date, np.array(1or2D):weight, float:offset

        all_lc_data["lc2fl"] += [lc2fl] * len(tmp_lc_data["flux"])
        all_lc_data["date"].extend(tmp_lc_data["date"])
        all_lc_data["flux"].extend(tmp_lc_data["flux"])
        if len(tmp_lc_data["weight"].shape) == 1:
            print_verb(verbose, "1d weight found")
            tmp_lc_data["weight"] = np.diag(tmp_lc_data["weight"])
        else:
            print_verb(verbose, "2d weight found")

        all_lc_data["covmat"] = twod_block_stack(
            all_lc_data["covmat"], np.linalg.inv(tmp_lc_data["weight"])
        )

        other_data["MagSys|Instrument|Band"][lc2fl] = (
            read_param(lc2fl, "@MAGSYS", verbose=verbose)
            + "|"
            + read_param(lc2fl, "@INSTRUMENT", verbose=verbose)
            + "|"
            + read_param(lc2fl, "@BAND", verbose=verbose)
        )

        # bandpassfn is the transmission fn * exp(alpha*lambda)
        # bandpasslambs is wavelengths
        # bandpasseval is the throughput at lambda
        # efflamb is the effective center
        other_data = init_band(other_data=other_data, lc2fl=lc2fl, alpha=0.0)
        other_data = init_band(other_data=other_data, lc2fl=lc2fl, alpha=d_alpha)

        plt.plot(
            other_data["bandpasslambs"][lc2fl],
            other_data["bandpasseval"][lc2fl],
            ".",
            label=lc2fl,
        )

    for key in all_lc_data:
        all_lc_data[key] = np.array(all_lc_data[key])

    all_lc_data["model_uncertainties"] = 0

    max_SNR = -1

    for i in range(len(all_lc_data["flux"])):
        this_SNR = all_lc_data["flux"][i] / np.sqrt(
            all_lc_data["covmat"][i, i] + (all_lc_data["flux"].max() * 0.05) ** 2.0
        )
        if this_SNR > max_SNR:
            max_SNR = this_SNR
            other_data["daymax_guess"] = all_lc_data["date"][i]

    plt.legend(loc="best")
    plt.savefig("read_bandpasses.pdf", bbox_inches="tight")
    plt.close()

    print_verb(verbose, "all_lc_data", all_lc_data)

    other_data["z_heliocentric"] = read_param("lightfile", "z_heliocentric")
    other_data["MWEBV"] = read_param("lightfile", "MWEBV")

    print_verb(verbose, "other_data", other_data)

    for lc2fl in set(all_lc_data["lc2fl"]):
        inds = np.where(all_lc_data["lc2fl"] == lc2fl)

        plt.errorbar(
            all_lc_data["date"][inds],
            all_lc_data["flux"][inds],
            yerr=np.sqrt(np.diag(all_lc_data["covmat"])[inds]),
            fmt="o",
            color=eff_lamb_to_rgb(
                other_data["efflamb"][lc2fl] / (1.0 + other_data["z_heliocentric"])
            ),
            markersize=1,
        )
    plt.title(subprocess.getoutput("pwd").split("/")[-1])
    plt.savefig("lcplot.pdf", bbox_inches="tight")
    plt.close()
    return all_lc_data, other_data


def do_actual_trimming(trimmed_lc_data, inds):
    """do_actual_trimming.
    slice up the trimmed_lc_data dict by inds
    Handle 2d data in covmat and weightmat

    Parameters
    ----------
    trimmed_lc_data :
        trimmed_lc_data
    inds :
        inds
    """
    trimmed_lc_data["lc2fl"] = trimmed_lc_data["lc2fl"][inds]
    trimmed_lc_data["flux"] = trimmed_lc_data["flux"][inds]
    trimmed_lc_data["date"] = trimmed_lc_data["date"][inds]
    trimmed_lc_data["covmat"] = trimmed_lc_data["covmat"][inds[0]][:, inds[0]]
    trimmed_lc_data["weightmat"] = np.linalg.inv(trimmed_lc_data["covmat"])

    return trimmed_lc_data


def trim_data(
    t0,
    all_lc_data,
    other_data,
    min_phase=-15,
    max_phase=45,
    min_wave=3000.0,
    max_wave=20000.0,
    verbose=False,
):  # Phase and wavelength cuts
    """trim_data.

    Parameters
    ----------
    t0 :
        t0
    all_lc_data :
        all_lc_data
    other_data :
        other_data
    """
    print_verb(verbose, "trimming around ", t0)

    trimmed_lc_data = deepcopy(all_lc_data)

    phase = (all_lc_data["date"] - t0) / (1.0 + other_data["z_heliocentric"])
    rest_wave = np.array(
        [
            other_data["efflamb"][lc2fl] / (1.0 + other_data["z_heliocentric"])
            for lc2fl in all_lc_data["lc2fl"]
        ]
    )
    inds = np.where(
        (phase >= min_phase)
        * (phase <= max_phase)
        * (rest_wave >= min_wave)
        * (rest_wave <= max_wave)
    )

    trimmed_lc_data = do_actual_trimming(trimmed_lc_data, inds)

    return trimmed_lc_data


def write_line(f, towrite):
    """write_line.

    Parameters
    ----------
    f :
        f
    towrite :
        towrite
    """
    f.write(" ".join([str(item) for item in towrite]) + "\n")


def compute_single_deriv(
    P,
    f,
    trimmed_lc_data,
    other_data,
    update_lc_not_other,
    key_to_update,
    d_step,
    lc2fl_for_alpha="None",
    verbose=False,
):
    """compute_single_deriv.

    Parameters
    ----------
    P :
        P
    f :
        f
    trimmed_lc_data :
        trimmed_lc_data
    other_data :
        other_data
    update_lc_not_other :
        update_lc_not_other
    key_to_update :
        key_to_update
    d_step :
        d_step
    lc2fl_for_alpha :
        lc2fl_for_alpha
    """
    tmp_other_data = deepcopy(other_data)
    tmp_lc_data = deepcopy(trimmed_lc_data)

    if lc2fl_for_alpha != "None":
        for key in [
            # "bandpassfn",
            "bandpasslambs",
            "bandpasseval",
            "magsysflux",
            "efflamb",
        ]:
            tmp_other_data[key][lc2fl_for_alpha] = other_data[key][
                lc2fl_for_alpha + "_dalpha"
            ]
    else:
        if update_lc_not_other:
            inds = np.where(tmp_lc_data["lc2fl"] == key_to_update)
            tmp_lc_data["flux"][inds] *= 10 ** (-0.4 * d_step)
            tmp_lc_data["covmat"][inds[0]][:, inds[0]] *= 10 ** (-0.8 * d_step)
            tmp_lc_data["weightmat"] = np.linalg.inv(tmp_lc_data["covmat"])
        else:
            if type(key_to_update) == list:
                tmp_other_data[key_to_update[0]][key_to_update[1]] += d_step
            else:
                tmp_other_data[key_to_update] += d_step

    P1, NA, NA = miniNM_new(
        ministart=P,
        miniscale=[1.0, 1.0, 1.0, 0.1],
        chi2fn=chi2_fn,
        passdata=[tmp_lc_data, tmp_other_data],
        verbose=verbose,
        compute_Cmat=False,
    )
    dP = (P1 - P) / d_step

    return dP


def get_deriv(
    P,
    trimmed_lc_data,
    other_data,
    file_path,
    verbose=False,
):
    """get_deriv.

    Parameters
    ----------
    P :
        P
    trimmed_lc_data :
        trimmed_lc_data
    other_data :
        other_data
    """
    f = open(file_path, "w")
    f.write(
        """#Format:
#Parameter MagSys|Instrument|Band RestLamb Phase dmu/dP dmB/dP ds/dP dc/dP
"""
    )

    d_step = 0.01
    total_dP = 0.0

    for lc2fl in set(trimmed_lc_data["lc2fl"]):
        dP = compute_single_deriv(
            P=P,
            f=f,
            trimmed_lc_data=trimmed_lc_data,
            other_data=other_data,
            update_lc_not_other=1,
            key_to_update=lc2fl,
            d_step=d_step,
            verbose=verbose,
        )

        towrite = [
            "Zeropoint",
            other_data["MagSys|Instrument|Band"][lc2fl],
            other_data["efflamb"][lc2fl] / (1.0 + other_data["z_heliocentric"]),
            "All",
            dP[1] + alpha * dP[2] - beta * dP[3],
            dP[1],
            dP[2],
            dP[3],
        ]
        write_line(f, towrite)
        f.flush()

        total_dP += dP
    towrite = [
        "Check",
        "All|All|All",
        "All",
        "All",
    ]
    if hasattr(total_dP, "__iter__"):
        towrite += [
            total_dP[1] + alpha * total_dP[2] - beta * total_dP[3],
            total_dP[1],
            total_dP[2],
            total_dP[3],
        ]
    else:
        towrite += ["fail" for _ in range(4)]
    write_line(f, towrite)

    """
    d_step = 20
    for lc2fl in set(trimmed_lc_data["lc2fl"]):
        dP = compute_single_deriv(P = P, f = f, trimmed_lc_data = trimmed_lc_data, other_data = other_data,
                                  update_lc_not_other = 0, key_to_update = ["bandpasslambs", lc2fl], d_step = d_step)

        towrite = ["Lambda", other_data["MagSys|Instrument|Band"][lc2fl], other_data["efflamb"][lc2fl]/(1. + other_data["z_heliocentric"]),
                   "All", dP[1] + alpha*dP[2] - beta*dP[3], dP[1], dP[2], dP[3]]
        write_line(f, towrite)
    """
    for lc2fl in set(trimmed_lc_data["lc2fl"]):
        this_d_step = (
            other_data["efflamb"][lc2fl + "_dalpha"] - other_data["efflamb"][lc2fl]
        ) / (1.0 + other_data["z_heliocentric"])

        print_verb(verbose, "lc2fl, this_d_step", lc2fl, this_d_step)

        dP = compute_single_deriv(
            P=P,
            f=f,
            trimmed_lc_data=trimmed_lc_data,
            other_data=other_data,
            update_lc_not_other=0,
            key_to_update=["bandpasslambs", lc2fl],
            d_step=this_d_step,
            lc2fl_for_alpha=lc2fl,
            verbose=verbose,
        )

        towrite = [
            "Lambda",
            other_data["MagSys|Instrument|Band"][lc2fl],
            other_data["efflamb"][lc2fl] / (1.0 + other_data["z_heliocentric"]),
            "All",
            dP[1] + alpha * dP[2] - beta * dP[3],
            dP[1],
            dP[2],
            dP[3],
        ]
        write_line(f, towrite)
        f.flush()

    d_step = 0.01
    dP = compute_single_deriv(
        P=P,
        f=f,
        trimmed_lc_data=trimmed_lc_data,
        other_data=other_data,
        update_lc_not_other=0,
        key_to_update="MWEBV",
        d_step=d_step,
        verbose=verbose,
    )

    towrite = [
        "MWEBV",
        "All|All|All",
        "All",
        "All",
        dP[1] + alpha * dP[2] - beta * dP[3],
        dP[1],
        dP[2],
        dP[3],
    ]
    write_line(f, towrite)
    f.flush()

    d_step = 0.01
    dP = compute_single_deriv(
        P=P,
        f=f,
        trimmed_lc_data=trimmed_lc_data,
        other_data=other_data,
        update_lc_not_other=0,
        key_to_update="z_heliocentric",
        d_step=d_step,
        verbose=verbose,
    )

    towrite = [
        "Redshift",
        "All|All|All",
        "All",
        "All",
        dP[1] + alpha * dP[2] - beta * dP[3],
        dP[1],
        dP[2],
        dP[3],
    ]
    write_line(f, towrite)

    print_verb(verbose, "total_dP", total_dP)
    f.close()

    format_file(file_path)


def check_if_enough_data_for_delta_mu(trimmed_lc_data, other_data, verbose=False):
    all_instrument_filts = []
    for lc2fl_name in trimmed_lc_data["lc2fl"]:
        all_instrument_filts.append(other_data["MagSys|Instrument|Band"][lc2fl_name])

    unique_instrument_filts = list(set(all_instrument_filts))
    print_verb(verbose, "unique_instrument_filts", unique_instrument_filts)

    unique_instrument_names_counts = {}
    for item in unique_instrument_filts:
        instr = item.split("|")[1]
        if instr in unique_instrument_names_counts:
            unique_instrument_names_counts[instr] += 1
        else:
            unique_instrument_names_counts[instr] = 1

    print_verb(
        verbose, "unique_instrument_names_counts", unique_instrument_names_counts
    )

    instruments_with_more_than_one = [
        instr
        for instr in unique_instrument_names_counts
        if (unique_instrument_names_counts[instr] > 1)
    ]
    return instruments_with_more_than_one


def get_delta_mu_instruments(
    P,
    trimmed_lc_data,
    other_data,
    file_path,
    verbose=False,
):
    """get_delta_mu_instruments.
    Writes to result_salt2.dat for now.
    Value appears to be the distance modulus plus some arbitrary constant.

    Parameters
    ----------
    P :
        P
    trimmed_lc_data :
        trimmed_lc_data
    other_data :
        other_data
    """
    for key in trimmed_lc_data:
        print_verb(verbose, "trimmed_lc_data", key)
    for key in other_data:
        print_verb(verbose, "other_data", key)

    print_verb(verbose, "trimmed_lc_data flux", trimmed_lc_data["flux"])
    print_verb(verbose, "trimmed_lc_data lc2fl", trimmed_lc_data["lc2fl"])
    print_verb(verbose, "MagSys|Instrument|Band", other_data["MagSys|Instrument|Band"])
    instruments_with_more_than_one = check_if_enough_data_for_delta_mu(
        trimmed_lc_data, other_data
    )
    if len(instruments_with_more_than_one) < 2:
        print_verb(verbose, "Not enough unique instruments/bands for a comparison!")
        return 0
    else:
        print_verb(verbose, "Running delta mus")

    f = open(file_path, "w")
    f.write("\nCOMPARISON_OF_INSTRUMENTS\n")
    f.write("Instrument|All\t" + str(P[1] + alpha * P[2] - beta * P[3]) + "\n")

    for instr in instruments_with_more_than_one:
        one_instrument_lc_data = deepcopy(trimmed_lc_data)
        instrs_from_lc2fl = np.array(
            [
                other_data["MagSys|Instrument|Band"][item].split("|")[1]
                for item in trimmed_lc_data["lc2fl"]
            ]
        )

        print_verb(verbose, "instrs_from_lc2fl", instrs_from_lc2fl)

        inds = np.where(instr == instrs_from_lc2fl)

        one_instrument_lc_data = do_actual_trimming(one_instrument_lc_data, inds)

        print_verb(verbose, "Running", instr)
        one_instrument_P, NA, Cmat = miniNM_new(
            ministart=P * 1.0,
            miniscale=[0.0, 1.0, 0.0, 0.1],
            chi2fn=chi2_fn,
            passdata=[one_instrument_lc_data, other_data],
            verbose=verbose,
        )
        f.write(
            "Instrument|"
            + instr
            + "\t"
            + str(
                one_instrument_P[1]
                + alpha * one_instrument_P[2]
                - beta * one_instrument_P[3]
            )
            + "\t"
            + str(np.sqrt(np.dot([1, -beta], np.dot(Cmat, [1, -beta]))))
            + "\n"
        )
    f.close()


def get_NIR_x0(
    P,
    trimmed_lc_data,
    other_data,
    verbose=False,
):

    NIR_lc2fls = []

    for lc2fl in set(trimmed_lc_data["lc2fl"]):
        this_efflamb = other_data["efflamb"][lc2fl] / (
            1.0 + other_data["z_heliocentric"]
        )

        if this_efflamb > 10000.0:
            NIR_lc2fls.append(lc2fl)

    print_verb(verbose, "NIR_lc2fls", NIR_lc2fls)

    NIR_lc_data = deepcopy(trimmed_lc_data)

    inds = np.where(
        [
            NIR_lc2fls.count(trimmed_lc_data["lc2fl"][i]) > 0
            for i in range(len(trimmed_lc_data["lc2fl"]))
        ]
    )

    if len(inds[0]) > 0:
        NIR_lc_data = do_actual_trimming(NIR_lc_data, inds)

        NIR_P = P * 1.0
        NIR_P[3] = 0.0

        NIR_P, NA, Cmat = miniNM_new(
            ministart=NIR_P,
            miniscale=[0.0, 1.0, 0.0, 0.0],
            chi2fn=chi2_fn,
            passdata=[NIR_lc_data, other_data],
            verbose=False,
        )

        assert len(Cmat) == 1

        # f = open("results_salt.dat")
        # f.write("\n")
        # write_line(
        #     f,
        #     [
        #         "NIR-2.5log10X0(Daymax_fixed,X1_fixed,c_fixed0)",
        #         NIR_P[1],
        #         np.sqrt(Cmat[0, 0]),
        #     ],
        # )
        # f.close()
        return NIR_P[1], np.sqrt(Cmat[0, 0])


def get_optical_x0(
    P,
    trimmed_lc_data,
    other_data,
    verbose=False,
):

    optical_lc2fls = []

    for lc2fl in set(trimmed_lc_data["lc2fl"]):
        this_efflamb = other_data["efflamb"][lc2fl] / (
            1.0 + other_data["z_heliocentric"]
        )

        if this_efflamb < 10000.0:
            optical_lc2fls.append(lc2fl)

    print_verb(verbose, "optical_lc2fls", optical_lc2fls)

    optical_lc_data = deepcopy(trimmed_lc_data)

    inds = np.where(
        [
            optical_lc2fls.count(trimmed_lc_data["lc2fl"][i]) > 0
            for i in range(len(trimmed_lc_data["lc2fl"]))
        ]
    )

    if len(inds[0]) > 0:
        optical_lc_data = do_actual_trimming(optical_lc_data, inds)

        optical_P = P * 1.0
        # optical_P[3] = 0.0

        optical_P, NA, Cmat = miniNM_new(
            ministart=optical_P,
            miniscale=[0.0, 1.0, 0.0, 0.0],
            chi2fn=chi2_fn,
            passdata=[optical_lc_data, other_data],
            verbose=False,
        )

        assert len(Cmat) == 1

        # f = open("results_salt.dat")
        # f.write("\n")
        # write_line(
        #     f,
        #     [
        #         "NIR-2.5log10X0(Daymax_fixed,X1_fixed,c_fixed0)",
        #         NIR_P[1],
        #         np.sqrt(Cmat[0, 0]),
        #     ],
        # )
        # f.close()
        return optical_P[1], np.sqrt(Cmat[0, 0])


def write_results(
    file_path,
    chifile_path,
    P,
    F,
    Cmat,
    trimmed_lc_data,
    other_data,
    alpha=0.137,
    beta=3.07,
):
    """write_results.

    Parameters
    ----------
    P :
        P
    F :
        F
    Cmat :
        Cmat
    trimmed_lc_data :
        trimmed_lc_data
    """
    f = open(file_path, "w")
    f.write("Salt3Model\nBEGIN_OF_FITPARAMS Salt3Model\n")

    towrite = ["DayMax", P[0], np.sqrt(Cmat[0, 0])]
    write_line(f, towrite)

    towrite = ["Redshift", other_data["z_heliocentric"], "0", "F"]
    write_line(f, towrite)

    towrite = ["Color", P[3], np.sqrt(Cmat[3, 3])]
    write_line(f, towrite)

    towrite = [
        "X0",
        10 ** (-0.4 * P[1]),
        np.sqrt(Cmat[1, 1]) * 0.4 * np.log(10.0) * 10 ** (-0.4 * P[1]),
    ]
    write_line(f, towrite)

    towrite = ["X1", P[2], np.sqrt(Cmat[2, 2])]
    write_line(f, towrite)

    rest_filters = get_B0(P)
    for filt in ["ux", "b", "v", "r"]:
        towrite = [
            "RestFrameMag_0_" + filt.upper(),
            rest_filters[filt],
            np.sqrt(Cmat[1, 1]) * (filt == "b"),
        ]
        write_line(f, towrite)

    params = ["DayMax", "RestFrameMag_0_B", "X1", "Color"]
    assert len(params) == len(P)
    for i in range(len(params)):
        for j in range(len(params)):
            towrite = ["Cov" + params[i] + params[j], Cmat[i, j], -1]
            write_line(f, towrite)

    towrite = [
        "\ndmu_estimate",
        np.sqrt(np.dot([1, alpha, -beta], np.dot(Cmat[1:, 1:], [1, alpha, -beta]))),
    ]
    write_line(f, towrite)

    if not len(trimmed_lc_data["date"]):
        f.write("AllDatesClipped\n")
    else:
        phases = (trimmed_lc_data["date"] - P[0]) / (1.0 + other_data["z_heliocentric"])
        f.write("FirstPhase  " + str(phases.min()) + "\n")
        f.write("LastPhase  " + str(phases.max()) + "\n")

        inds = np.where(phases < 0)
        negphases = phases[inds]
        if len(negphases) == 0:
            negphases = [1000]
        f.write("LatestNegPhase  " + str(np.max(negphases)) + "\n")

        inds = np.where(phases > 0)
        posphases = phases[inds]
        if len(posphases) == 0:
            posphases = [-1000]
        f.write("EarliestPosPhase  " + str(np.min(posphases)) + "\n")

        nights_in_fit = (
            trimmed_lc_data["date"] - 0.2
        )  # Roughly centered for Hawai'i, Southwest, and Chile
        nights_in_fit = np.around(nights_in_fit)
        nights_in_fit = np.sort(np.unique(nights_in_fit)) + 0.2

        f.write("NightsNegPhase  " + str(sum(nights_in_fit <= P[0])) + "\n")
        f.write("NightsPosPhase  " + str(sum(nights_in_fit > P[0])) + "\n")

    f.write("\nsalt3source   " + str("salt3-nir") + "\n")
    f.write("salt2_version   " + str(salt2_version) + "\n")
    f.close()

    f = open(chifile_path, "w")
    f.write(str(len(trimmed_lc_data["flux"])) + "  " + str(F) + "\n")
    f.close()


def eff_lamb_to_rgb(eff_lamb, verbose=False):
    """eff_lamb_to_rgb.

    Parameters
    ----------
    eff_lamb :
        eff_lamb
    """
    color_table = [
        [0.0, 1.0, 0.0, 1.0],
        [3000.0, 1.0, 0.0, 1.0],
        [4400.0, 0.0, 0.0, 1.0],
        [5000.0, 0.0, 0.7, 0.7],
        [5500.0, 0.0, 0.7, 0.0],
        [6500.0, 1.0, 0.0, 0.0],
        [7500.0, 0.5, 0.0, 0.0],
        [10000.0, 0.0, 0.0, 0.0],
        [50000.0, 0.0, 0.0, 0.0],
    ]
    print_verb(verbose, "color for ", eff_lamb)

    return (
        float(
            interp1d(
                [item[0] for item in color_table],
                [item[1] for item in color_table],
                kind="linear",
            )(eff_lamb)
        ),
        float(
            interp1d(
                [item[0] for item in color_table],
                [item[2] for item in color_table],
                kind="linear",
            )(eff_lamb)
        ),
        float(
            interp1d(
                [item[0] for item in color_table],
                [item[3] for item in color_table],
                kind="linear",
            )(eff_lamb)
        ),
    )


def get_opts():
    if sys.argv.count("-w"):
        ind = sys.argv.index("-w")
        min_wave = float(sys.argv[ind + 1])
        max_wave = float(sys.argv[ind + 2])

        del sys.argv[ind + 2]
        del sys.argv[ind + 1]
        del sys.argv[ind]
    else:
        min_wave = 3000.0
        max_wave = 8800.0

    if sys.argv.count("-p"):
        ind = sys.argv.index("-p")
        min_phase = float(sys.argv[ind + 1])
        max_phase = float(sys.argv[ind + 2])

        del sys.argv[ind + 2]
        del sys.argv[ind + 1]
        del sys.argv[ind]
    else:
        min_phase = -15.0
        max_phase = 45.0

    if sys.argv.count("-sc"):
        ind = sys.argv.index("-sc")
        del sys.argv[ind]

        simul_chi2 = 1
    else:
        simul_chi2 = 0

    if sys.argv.count("-robust"):
        ind = sys.argv.index("-robust")
        del sys.argv[ind]

        robust_chi2 = 1
    else:
        robust_chi2 = 0

    if sys.argv.count("-mi"):
        ind = sys.argv.index("-mi")
        model_iterations = int(sys.argv[ind + 1])

        del sys.argv[ind + 1]
        del sys.argv[ind]

    else:
        model_iterations = 3

    if sys.argv.count("-daymax_guess"):
        ind = sys.argv.index("-daymax_guess")
        daymax_guess = float(sys.argv[ind + 1])

        del sys.argv[ind + 1]
        del sys.argv[ind]

    else:
        daymax_guess = "None"

    if sys.argv.count("-v"):
        ind = sys.argv.index("-v")
        salt2_version = sys.argv[ind + 1]

        del sys.argv[ind + 1]
        del sys.argv[ind]

    else:
        # salt2_version = "salt3-22"
        salt2_version = "salt3-f22"
        # salt2_version = "salt3-22" #"salt3-k21"

    print(
        "min_wave",
        min_wave,
        "max_wave",
        max_wave,
        "min_phase",
        min_phase,
        "max_phase",
        max_phase,
        "simul_chi2",
        simul_chi2,
        "model_iterations",
        model_iterations,
        "salt2_version",
        salt2_version,
        "robust_chi2",
        robust_chi2,
    )
    return (
        min_wave,
        max_wave,
        simul_chi2,
        model_iterations,
        salt2_version,
        min_phase,
        max_phase,
        daymax_guess,
        robust_chi2,
    )


global_zp = 24
d_lamb = 1.0
band_clip = 0.001
min_phase = -15
max_phase = 45.0  # 49.9
min_wave = 3000.0
max_wave = 20000.0

alpha = 0.137
beta = 3.07
salt2_version = "salt3-nir"  # "salt3-k21"
d_alpha = 0.0001  # For derivatives in wavelength: exp(alpha*lambda) warping on bandpass


# if salt2_version.count("salt3"):
#     salt3source = 1
#     source = sncosmo.SALT3Source(
#         modeldir=os.environ["PATHMODEL"] + "/" + salt2_version + "/"
#     )
# else:
#     salt3source = 0
#     source = sncosmo.SALT2Source(
#         modeldir=os.environ["PATHMODEL"] + "/" + salt2_version + "/"
#     )

source = sncosmo.SALT3Source(modeldir=f"{constants.STATIC_DIR}/salt3-nir")
CCM89model = sncosmo.Model(
    source, effects=[sncosmo.CCM89Dust()], effect_names=["MW"], effect_frames=["obs"]
)
O94model = sncosmo.Model(
    source, effects=[sncosmo.O94Dust()], effect_names=["MW"], effect_frames=["obs"]
)
old_F99model = sncosmo.Model(
    source, effects=[sncosmo.old_F99Dust()], effect_names=["MW"], effect_frames=["obs"]
)
F99model = sncosmo.Model(
    source, effects=[sncosmo.F99Dust()], effect_names=["MW"], effect_frames=["obs"]
)
F19model = sncosmo.Model(
    source, effects=[sncosmo.F19Dust()], effect_names=["MW"], effect_frames=["obs"]
)

if __name__ == "__main__":
    assert len(set(sys.argv[1:])) == len(sys.argv[1:]), "Duplicated file!"
    verbose = True
    subprocess.getoutput("touch result_salt2.dat")
    all_lc_data, other_data = init_data()
    (
        min_wave,
        max_wave,
        simul_chi2,
        model_iterations,
        salt2_version,
        min_phase,
        max_phase,
        daymax_guess,
        robust_chi2,
    ) = get_opts()

    if daymax_guess != "None":
        other_data["daymax_guess"] = daymax_guess
    daymax_current = other_data["daymax_guess"]
    daymax_previous = daymax_current - 10.0
    n_iter = 0
    while n_iter < 3 and np.abs(daymax_current - daymax_previous) > 0.1:
        daymax_previous = daymax_current
        trimmed_lc_data = trim_data(
            t0=daymax_current,
            all_lc_data=all_lc_data,
            other_data=other_data,
            verbose=verbose,
        )
        P, F, Cmat = miniNM_new(
            ministart=[other_data["daymax_guess"], 10.0, 0.0, 0.0],
            miniscale=[0.0, 1.0, 0.0, 0.0],
            chi2fn=chi2_fn,
            passdata=[trimmed_lc_data, other_data],
            compute_Cmat=False,
            verbose=verbose,
        )

        P, F, Cmat = miniNM_new(
            ministart=P,
            miniscale=[1.0, 1.0, 0.0, 0.0],
            chi2fn=chi2_fn,
            passdata=[trimmed_lc_data, other_data],
            compute_Cmat=False,
            verbose=verbose,
        )

        P, F, Cmat = miniNM_new(
            ministart=P,
            miniscale=[1.0, 1.0, 1.0, 0.1],
            chi2fn=chi2_fn,
            passdata=[trimmed_lc_data, other_data],
            compute_Cmat=(model_iterations == 0),
            verbose=verbose,
        )
        fail = 0

        for add_model_cov_iter in range(model_iterations):
            trimmed_lc_data = trim_data(
                t0=daymax_current,
                all_lc_data=all_lc_data,
                other_data=other_data,
                verbose=verbose,
            )
            if not simul_chi2:
                trimmed_lc_data = add_model_cov(P, trimmed_lc_data)

            plt.errorbar(
                trimmed_lc_data["date"],
                trimmed_lc_data["flux"],
                yerr=np.sqrt(np.diag(trimmed_lc_data["covmat"])),
                fmt=".",
            )
            plt.savefig("trimmed.pdf")
            plt.close()
            fail = 0

            try:
                P, F, Cmat = miniNM_new(
                    ministart=P,
                    miniscale=[1.0, 1.0, 1.0, 0.1],
                    chi2fn=chi2_fn,
                    passdata=[trimmed_lc_data, other_data],
                    verbose=verbose,
                )

                print("final chi2 from NM", F)

                # P, F, NA = miniLM_new(ministart = P,
                #                      miniscale = [1., 1., 1., 0.1],
                #                      residfn = residfn,
                #                      passdata = [trimmed_lc_data, other_data], verbose = True)

                # print("final chi2 from LM", F)

            except:
                print_verb(verbose, "Fit failed")
                Cmat = np.diag(np.ones(len(P)))
                fail = 1
        print_verb(verbose, "P", "F", P, F)
        print_verb(verbose, "uncs", np.sqrt(np.diag(Cmat)))
        daymax_current = P[0]
        n_iter += 1

    write_results(
        "results.dat",
        "chifile.txt",
        P=P,
        F=F,
        Cmat=Cmat,
        trimmed_lc_data=trimmed_lc_data,
    )
    if fail == 0:
        get_delta_mu_instruments(P, trimmed_lc_data, other_data, verbose=verbose)
        get_deriv(P, trimmed_lc_data, verbose=verbose)

    for lc2fl in set(all_lc_data["lc2fl"]):
        inds = np.where(all_lc_data["lc2fl"] == lc2fl)

        plt.errorbar(
            all_lc_data["date"][inds],
            all_lc_data["flux"][inds],
            yerr=np.sqrt(np.diag(all_lc_data["covmat"])[inds]),
            fmt="o",
            color=eff_lamb_to_rgb(
                other_data["efflamb"][lc2fl] / (1.0 + other_data["z_heliocentric"])
            ),
            markersize=1,
        )

    ylim = plt.ylim()

    for lc2fl in set(trimmed_lc_data["lc2fl"]):
        inds = np.where(trimmed_lc_data["lc2fl"] == lc2fl)
        plt.plot(
            trimmed_lc_data["date"][inds],
            trimmed_lc_data["flux"][inds],
            ".",
            color=eff_lamb_to_rgb(
                other_data["efflamb"][lc2fl] / (1.0 + other_data["z_heliocentric"])
            ),
            label=lc2fl,
        )

        plot_dates, plot_model, plot_uncs = model_for_plotting(
            P=P,
            lc2fl=lc2fl,
            mindate=P[0] + min_phase * (1.0 + other_data["z_heliocentric"]),
            maxdate=P[0] + max_phase * (1.0 + other_data["z_heliocentric"]),
            other_data=other_data,
        )
        plt.plot(
            plot_dates,
            plot_model,
            color=eff_lamb_to_rgb(
                other_data["efflamb"][lc2fl] / (1.0 + other_data["z_heliocentric"])
            ),
        )
        plt.fill_between(
            plot_dates,
            plot_model - plot_uncs,
            plot_model + plot_uncs,
            zorder=-1,
            color=eff_lamb_to_rgb(
                other_data["efflamb"][lc2fl] / (1.0 + other_data["z_heliocentric"])
            ),
            alpha=0.5,
            linewidth=0,
        )

        writecol(
            "lc_Salt2Model_%s.dat"
            % other_data["MagSys|Instrument|Band"][lc2fl].split("|")[2],
            [plot_dates, plot_model, plot_uncs, [global_zp] * len(plot_uncs)],
        )

    plt.ylim(ylim)
    plt.xlabel("Date")
    plt.ylabel("Flux (Zeropoint = " + str(global_zp) + ")")
    plt.title(subprocess.getoutput("pwd").split("/")[-1])
    plt.legend(loc="best")
    plt.axhline(0, color="k")
    plt.savefig("lcplot.pdf", bbox_inches="tight")
    plt.close()
