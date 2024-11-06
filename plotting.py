import matplotlib as mpl
import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
import pingouin as pg

import constants
import utils
from targets.models import Target

file_ext = "pdf"


def tns_hist(
    bins=20, alpha=0.7, xlim_left=-30, xlim_right=50, disc_grp=None, show=False
):
    names = []
    arr = [[], [], []]
    qset = (
        Target.objects.get_by_type("Ia")
        .exclude(classification_date=0)
        .exclude(lightcurves__isnull=True)
    )
    if disc_grp is not None:
        qset = qset.filter(discovering_group=disc_grp)
    for obj in qset:
        if "J" not in obj.peak_values:
            continue
        if "mjd" not in obj.peak_values["J"]:
            continue
        names.append(obj.TNS_name)
        if isinstance(obj.peak_values["J"]["mjd"], list):
            mjd_pk = np.average(
                np.array(obj.peak_values["J"]["mjd"])[
                    np.array(obj.peak_values["J"]["mjd"]) != 0
                ]
            )
        else:
            mjd_pk = obj.peak_values["J"]["mjd"]
        arr[0].append(obj.detection_date)
        arr[1].append(obj.classification_date)
        arr[2].append(mjd_pk)
    arr = np.array(arr)
    m1 = arr[0] - arr[2] < xlim_right
    m2 = arr[1] - arr[2] < xlim_right
    m3 = arr[0] - arr[2] > xlim_left
    m4 = arr[1] - arr[2] > xlim_left
    m = m1 & m2 & m3 & m4
    fig, ax = plt.subplots(1, figsize=(6, 4))
    ax.hist(
        arr[0][m] - arr[2][m],
        label="detection",
        bins=bins,
        alpha=alpha,
        density=True,
        hatch="/",
    )
    ax.hist(
        arr[1][m] - arr[2][m],
        label="classification",
        bins=bins,
        alpha=alpha,
        density=True,
        hatch="\\",
    )
    ax.yaxis.set_visible(False)
    plt.xlim(xlim_left, xlim_right)
    plt.xlabel("Days after J band peak")
    plt.legend()
    plt.tight_layout()
    fig.savefig(
        f"/data/users/ado/papers_git/HSF_survey_paper/figures/tns_hist.{file_ext}"
    )
    if show:
        plt.show()
    plt.close()
    return arr, m


def hodgkin_recreation(trust=True):
    import constants
    from photometry import prepare_input_dict, prepare_qset, stan_color_term

    (
        fig,
        [ax0, ax1, ax2],
    ) = plt.subplots(nrows=3, sharex=True, figsize=(8, 8))
    res = {}
    for filtcombo, ax in zip(
        [
            # ["Z", "J", "J", "H"],
            ["Y", "J", "J", "H"],
            ["J", "J", "J", "H"],
            ["H", "H", "J", "H"],
            # ["K", "K", "J", "K"],
        ],
        [ax0, ax1, ax2],
    ):
        qset = prepare_qset(*filtcombo)
        qset = qset.filter(image__observation__name__endswith="1")
        if not qset.exists():
            continue
        input_dict = prepare_input_dict(
            qset, filtcombo[0], filtcombo[1], filtcombo[2], filtcombo[3]
        )
        if trust:
            _, slope, _, _, offset = constants.STAN_COLOR_CORRECTION[filtcombo[0]]
        else:
            res[filtcombo[0]] = stan_color_term(input_dict)
            slope = np.average(res[filtcombo[0]]["slope"])
            offset = np.average(res[filtcombo[0]]["offset"])
        _, old_slope, _, _, old_offset = constants.HODGKIN_COLOR_CORRECTION[
            filtcombo[0]
        ]
        ax.scatter(
            input_dict["m1"] - input_dict["m2"],
            input_dict["zp"] + input_dict["mi"] - input_dict["m0"],
            alpha=0.1,
            color="black",
            marker=".",
        )
        ax.plot(
            np.linspace(0, 1, 30),
            np.linspace(0, 1, 30) * old_slope + old_offset,
            color="orange",
        )
        ax.plot(
            np.linspace(0, 1, 30), np.linspace(0, 1, 30) * slope + offset, color="red"
        )
        ax.hlines(0, 0, 1, color="black")
        ax.set_xlim(0, 1)
        ax.set_xlabel("$J_{2MASS} - H_{2MASS}$")
        if filtcombo[0] == "Z":
            ax.set_ylim(-0.3, 1.2)
            ax.set_ylabel("$Z_{WFCAM} - J_{2MASS}$ (mag)")
        elif filtcombo[0] == "Y":
            ax.set_ylim(-0.4, 0.8)
            ax.set_ylabel("$Y_{WFCAM} - J_{2MASS}$ (mag)")
        elif filtcombo[0] == "J":
            ax.set_ylim(-0.4, 0.4)
            ax.set_ylabel("$J_{WFCAM} - J_{2MASS}$ (mag)")
        elif filtcombo[0] == "H":
            ax.set_ylim(-0.4, 0.4)
            ax.set_ylabel("$H_{WFCAM} - H_{2MASS}$ (mag)")
    fig.subplots_adjust(hspace=0)
    fig.savefig(f"/data/users/ado/papers_git/HSF_survey_paper/figures/hess.{file_ext}")
    plt.show()


def mollweide(show=False):
    import pandas as pd

    # Ias
    ra_ia, dec_ia = np.array(
        Target.objects.get_by_type("Ia")
        .number_of_observations()
        .values_list("ra", "dec")
    ).T
    fig, ax = utils.plot_mollweide()
    x_ia, y_ia = utils.mollweide(ra_ia, dec_ia)
    ax.scatter(
        x_ia,
        y_ia,
        color="red",
        label=f"(N={len(x_ia)}) Spectroscopically classified SNe Ia",
        # marker=".",
    )
    # Unclassified
    ra_q, dec_q = np.array(
        Target.objects.get_by_type("?")
        .number_of_observations()
        .values_list("ra", "dec")
    ).T
    x_q, y_q = utils.mollweide(ra_q, dec_q)
    ax.scatter(
        x_q,
        y_q,
        color="red",
        label=f"(N={len(x_q)}) Unclassified transients",
        marker=".",
    )
    # CSP DR3
    p = pd.read_csv(f"{constants.HSF_DIR}/ado/CSPDR3/tab1.dat", delimiter="\t")
    ra_csp = np.zeros(len(p["RA"]))
    dec_csp = np.zeros(len(p["DEC"]))
    for i, row in p.iterrows():
        ra_csp[i], dec_csp[i] = utils.HMS2deg(row["RA"], row["DEC"])
    x_csp, y_csp = utils.mollweide(ra_csp, dec_csp)
    ax.scatter(
        x_csp,
        y_csp,
        color="blue",
        label=f"(N={len(x_csp)}) CSP DR3 SNe Ia",
        # marker=".",
    )
    # ax.legend()
    fig.savefig(
        f"/data/users/ado/papers_git/HSF_survey_paper/figures/mollweide.{file_ext}"
    )
    fig.savefig(f"{constants.MEDIA_DIR}/mollweide.{file_ext}")
    if show:
        plt.show()
    plt.close()


def demographics(show=False):
    from targets.models import Target

    demographics = {}
    for obj in Target.objects.number_of_observations():
        if obj.sn_type.name not in demographics:
            demographics[obj.sn_type.name] = 0
        demographics[obj.sn_type.name] += 1
    demographics.pop("CALSPEC")
    demographics.pop("reference")
    sub_demo = {"SN Ia": {}, "Unclassified": 0, "Other": {}}
    for key, val in demographics.items():
        if key.startswith("SN Ia"):
            if key not in sub_demo["SN Ia"]:
                sub_demo["SN Ia"][key] = 0
            sub_demo["SN Ia"][key] += val
        elif key == "?":
            sub_demo["Unclassified"] += val
        else:
            if key not in sub_demo["Other"]:
                sub_demo["Other"][key] = 0
            sub_demo["Other"][key] += val
    sub_demo["SN Ia"]["Total"] = sum(sub_demo["SN Ia"].values())
    sub_demo["Other"]["Total"] = sum(sub_demo["Other"].values())
    ia_mask = np.argsort(list(sub_demo["SN Ia"].values()))[::-1]
    ia_labels = np.array(list(sub_demo["SN Ia"].keys()))[ia_mask]
    ia_values = np.array(list(sub_demo["SN Ia"].values()))[ia_mask]
    other_mask = np.argsort(list(sub_demo["Other"].values()))[::-1]
    other_labels = np.array(list(sub_demo["Other"].keys()))[other_mask]
    other_values = np.array(list(sub_demo["Other"].values()))[other_mask]

    fig, ax = plt.subplots(figsize=(5, 10))
    width = 1
    multiplier = -1
    for key, value in zip(
        list(ia_labels) + [None, "Unclassified", None] + list(other_labels),
        list(ia_values) + [None, sub_demo["Unclassified"], None] + list(other_values),
    ):
        multiplier += 1
        if value is None:
            continue
        ax.barh(width * multiplier, value, width, label=key)
        if value > 300:
            ha = "right"
            offset = -5
        else:
            ha = "left"
            offset = 5
        ax.text(
            value + offset,
            width * multiplier,
            key + ": " + str(value),
            horizontalalignment=ha,
            verticalalignment="center",
        )
    ax.set_yticks([])
    ax.invert_yaxis()
    if show:
        plt.show()
    fig.savefig(
        f"/data/users/ado/papers_git/HSF_survey_paper/figures/demographics.{file_ext}"
    )


def compare_z(show=False):
    from targets.models import Target

    fig, [[top_hist, blank], [scatter, right_hist]] = plt.subplots(
        nrows=2, ncols=2, gridspec_kw={"height_ratios": [2, 5], "width_ratios": [5, 2]}
    )
    color_dict = {"SNIFS": "cyan", "FOCAS": "orange", "Literature": "black"}
    d = {}
    qset = Target.objects.get_by_type("Ia").number_of_observations()
    d["SNIFS"] = qset.filter(galaxy__snifs_entries__z_flag="s").distinct()
    d["FOCAS"] = qset.filter(galaxy__focas_entries__z_flag__in=["s", "?"]).distinct()
    d["Literature"] = (
        qset.filter(galaxy__z_flag__startswith="sp")
        .exclude(galaxy__snifs_entries__z_flag="s")
        .exclude(galaxy__focas_entries__z_flag__in=["s", "?"])
    )
    for key, marker, hatch in zip(d, ("x", "o", "."), ("/", "\\", "")):
        z = []
        mag = []
        for obj in d[key]:
            try:
                gmag = obj.galaxy.ps1_entries.first().gkronmag
                if gmag is not None and obj.galaxy.z is not None:
                    mag.append(gmag)
                    z.append(obj.galaxy.z)
            except:
                continue
        scatter.scatter(
            z, mag, label=key, color=color_dict[key], alpha=0.5, marker=marker
        )
        top_hist.hist(
            z, label=key, color=color_dict[key], alpha=0.5, density=True, hatch=hatch
        )
        right_hist.hist(
            mag,
            label=key,
            color=color_dict[key],
            alpha=0.5,
            orientation="horizontal",
            density=True,
            hatch=hatch,
        )

    scatter.set_xlim(-0.01, 0.12)
    top_hist.set_xlim(scatter.get_xlim())
    right_hist.set_ylim(scatter.get_ylim())
    top_hist.set_yticks([])
    top_hist.tick_params(axis="x", labelbottom=False)
    right_hist.set_xticks([])
    right_hist.tick_params(axis="y", labelleft=False)

    scatter.legend()
    scatter.set_ylabel("PanSTARRS-1 g Kron mag")
    scatter.set_xlabel("Heliocentric Redshift")

    blank.set_xticks([])
    blank.set_yticks([])
    blank.spines["top"].set_visible(False)
    blank.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0, wspace=0)
    fig.savefig(
        f"/data/users/ado/papers_git/HSF_survey_paper/figures/z_distribution.{file_ext}"
    )
    if show:
        plt.show()
    plt.close()


def compare_snifs_leda(h_alpha=6564.6, line_window=10, show=False):
    import astropy.units as u
    import matplotlib.pyplot as plt
    import numpy as np
    from specutils import SpectralRegion
    from specutils.analysis import equivalent_width

    import constants
    from galaxies.models import SnifsEntry

    fig, ax = plt.subplots()
    z, r, leda_v, se = np.array(
        SnifsEntry.objects.filter(
            galaxy__leda_v__isnull=False, z__isnull=False, z_flag__in=["s", "?"]
        ).values_list("z", "r", "galaxy__leda_v", "pk")
    ).T
    dz = constants.C * z - leda_v
    ew = []
    for s in se:
        sn = SnifsEntry.objects.get(pk=s)
        cs = sn.spectrum.get_continuum_normed_spec1d()
        ew.append(
            equivalent_width(
                cs,
                regions=SpectralRegion(
                    (h_alpha - line_window) * (1 + sn.z) * u.AA,
                    (h_alpha + line_window) * (1 + sn.z) * u.AA,
                ),
            ).value
        )
    ax.hist(dz, bins=10, alpha=0.8)
    ax.set_ylabel("N")
    ax1 = ax.twinx()
    ax1.scatter(dz, np.array(ew) / min(ew), color="black")
    ax1.set_ylabel(r"Normalized H$_\alpha$ Equivalent Width")
    ax.set_xlabel("SNIFS Velocity - HyperLEDA Velocity (km/s)")
    plt.tight_layout()
    fig.savefig(
        f"/data/users/ado/papers_git/HSF_survey_paper/figures/snifs_abs_cal.{file_ext}"
    )
    if show:
        plt.show()
    plt.close()


def dt_scatter_and_hist(p=None, show=False, std_cutoff=3):
    if p is None:
        from survey_paper import compare_multiple_sets

        (
            _,
            _,
            _,
            _,
            p,
        ) = compare_multiple_sets(cuts="standard")
    t = np.arctan(1 / np.sqrt(2))
    rot_z = [[np.cos(t), np.sin(t), 0], [-np.sin(t), np.cos(t), 0], [0, 0, 1]]
    rot_y = [
        [np.cos(np.pi / 4), 0, np.sin(np.pi / 4)],
        [0, 1, 0],
        [-np.sin(np.pi / 4), 0, np.cos(np.pi / 4)],
    ]
    M = np.matmul(rot_z, rot_y)
    a, b, c = (
        p["ebv_common"]["Tmax"],
        p["max_common"]["Tmax"],
        p["salt_common"]["t0"],
    )
    fig, [ax1, ax2] = plt.subplots(figsize=(8, 5), ncols=2)
    ax1.scatter(
        np.sqrt(2 / 3) * (-0.5 * a + b - 0.5 * c),
        np.sqrt(2) / 2 * (-a + c),
        color="black",
    )
    dt = np.sqrt(2 / 3 * (-0.5 * a + b - 0.5 * c) ** 2 + 0.5 * (-a + c) ** 2)
    ax1.set_xlabel("v (days)")
    ax1.set_ylabel("w (days)")
    xlim = ax1.get_xlim()
    ylim = ax1.get_ylim()
    for coord, model in zip(
        ([1, 0, 0], [0, 1, 0], [0, 0, 1]),
        ("SNooPy EBV_model2", "SNooPy max_model", "SALT3-NIR"),
    ):
        new_coord = np.matmul(M, coord)
        ax1.plot([0, new_coord[1] * 100], [0, new_coord[2] * 100], label=model)
    ax1.add_patch(
        plt.Circle((0, 0), std_cutoff * np.std(dt), facecolor="None", edgecolor="red")
    )
    ax1.set_aspect("equal")
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax1.legend()

    ax2.hist(dt, bins=20)
    ax2.axvline(std_cutoff * np.std(dt), color="red")
    ax2.set_yscale("log")
    ax2.set_xlabel("Days")
    plt.tight_layout()
    fig.savefig(
        f"/data/users/ado/papers_git/HSF_survey_paper/figures/tmax_comparison.{file_ext}"
    )
    if show:
        plt.show()
    plt.close()


def x1_scatter_and_hist(compare_multiple_sets=None, limit=1.5, show=False):
    if compare_multiple_sets is not None:
        (
            qset,
            fr_qset_J,
            qset_J,
            junk_J,
            common_qsets,
            stats,
            p,
            common_to_all,
            stats_all,
            p_all,
        ) = compare_multiple_sets
    else:
        from snippets import compare_multiple_sets

        (
            qset,
            fr_qset_J,
            qset_J,
            junk_J,
            common_qsets,
            stats,
            p,
            common_to_all,
            stats_all,
            p_all,
        ) = compare_multiple_sets(cuts="standard")
    t = np.arctan(1 / np.sqrt(2))
    rot_z = [[np.cos(t), np.sin(t), 0], [-np.sin(t), np.cos(t), 0], [0, 0, 1]]
    rot_y = [
        [np.cos(np.pi / 4), 0, np.sin(np.pi / 4)],
        [0, 1, 0],
        [-np.sin(np.pi / 4), 0, np.cos(np.pi / 4)],
    ]
    M = np.matmul(rot_z, rot_y)
    a, b, c = (
        p_all["snpy_ebv_model2"]["st"],
        p_all["snpy_max_model"]["st"],
        p_all["J"]["x1"],
    )
    fig, [ax1, ax2] = plt.subplots(figsize=(8, 5), ncols=2)
    ax1.scatter(
        np.sqrt(2 / 3) * (-0.5 * a + b - 0.5 * c),
        np.sqrt(2) / 2 * (-a + c),
        color="black",
    )
    ax1.set_xlabel("v (days)")
    ax1.set_ylabel("w (days)")
    xlim = ax1.get_xlim()
    ylim = ax1.get_ylim()
    for coord, model in zip(
        ([1, 0, 0], [0, 1, 0], [0, 0, 1]),
        ("SNooPy EBV_model2", "SNooPy max_model", "SALT3-NIR"),
    ):
        new_coord = np.matmul(M, coord)
        ax1.plot([0, new_coord[1] * 100], [0, new_coord[2] * 100], label=model)
    ax1.add_patch(plt.Circle((0, 0), limit, facecolor="None", edgecolor="red"))
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax1.set_aspect("equal")
    ax1.legend()

    ax2.hist(
        np.sqrt(2 / 3 * (-0.5 * a + b - 0.5 * c) ** 2 + 1 / 2 * (-a + c) ** 2), bins=30
    )
    ax2.axvline(limit, color="red")
    ax2.set_yscale("log")
    ax2.set_xlabel("Days")
    plt.tight_layout()
    fig.savefig(
        f"/data/users/ado/papers_git/HSF_survey_paper/figures/x1_comparison.{file_ext}"
    )
    if show:
        plt.show()
    plt.close()


def nir_vs_optical(vals_dict=None, actual_dict=None, show=False):
    if vals_dict is None and actual_dict is None:
        import pickle

        with open(f"{constants.HSF_DIR}/ado/vals_actual.pickle", "rb") as handle:
            vals_dict, actual_dict = pickle.load(handle)
    fig, [axs_ebv, axs_max, axs_salt] = plt.subplots(
        figsize=(9, 6),
        nrows=3,
        ncols=4,
        sharey=True,
    )
    # colors = ("red", "blue", "green" "orange")
    for axs, model in zip((axs_ebv, axs_max, axs_salt), ("ebv", "max", "salt")):
        if model == "salt":
            model_name = "SALT"
        elif model == "ebv":
            model_name = "SNPY_EBV"
        elif model == "max":
            model_name = "SNPY_Max"
        vals = vals_dict[model]
        for ax, estimator in zip(axs, ("RMS", "WRMS", "NMAD", "sigma_int")):
            if estimator == "RMS":
                ax.set_ylabel(model_name)
                ax.tick_params(labelleft=False)
            diff = np.array(vals["OJ"][estimator] - vals["O"][estimator])
            ax.hist(diff, bins=30)
            ax.axvline(0, color="black")
            ax.axvline(np.average(diff), color="red")
            ax.axvline(np.average(diff) - np.std(diff), color="red", ls="--", alpha=0.5)
            ax.axvline(np.average(diff) + np.std(diff), color="red", ls="--", alpha=0.5)
            ax.set_xlabel(r"$\Delta$" + estimator + " (mag)")
            if estimator == "sigma_int":
                ax.set_xlabel(r"$\Delta\sigma_{int}$ (mag)")
            # axLine, axLabel = ax.get_legend_handles_labels()
            if model != "max":
                ax.set_xlim(-0.08, 0.08)
    # fig.legend(
    #     handles=axLine,
    #     labels=axLabel,
    #     ncol=2,
    #     loc="upper center",
    #     bbox_to_anchor=(0.5, 0.95),
    # )
    plt.tight_layout()
    fig.subplots_adjust(wspace=0)
    fig.savefig(
        f"/data/users/ado/papers_git/HSF_survey_paper/figures/NIR_vs_optical_tests.{file_ext}"
    )
    if show:
        plt.show()


def plot_1D3_2D(proj_sep=None, ave_dm=None, dm=None, show=False):
    if proj_sep is None and ave_dm is None and dm is None:
        from snippets import compare_1D3_2D

        (
            m,
            f,
            dm,
            df,
            m_std,
            f_std,
            sep,
            g_pk,
            z,
            ave_dm,
            short_sep,
            short_z,
            proj_sep,
        ) = compare_1D3_2D()
    fig, [ax1, ax2] = plt.subplots(
        figsize=(6, 4), ncols=2, sharey=True, gridspec_kw={"width_ratios": [3, 1]}
    )
    ax1.scatter(proj_sep, ave_dm["1D3"])
    ax1.set_xlabel(r"Projected Separation (h$^{-1}$ kpc)")
    ax1.set_ylabel(r"m$_{ref} - $m$_{0}$ (mag)")
    ax2.hist(ave_dm["1D3"], orientation="horizontal", bins=30)
    ax2.set_xlabel("N")
    ax2.set_xscale("log")
    ax1.axhline(0, color="black")
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    fig.savefig(
        f"/data/users/ado/papers_git/HSF_survey_paper/figures/1D3_vs_2D.{file_ext}"
    )
    if show:
        plt.show()
    plt.close()


def mahalanobis_J_O(p, cov_ebv=None, cov_salt=None, cov_max=None, show=False):
    salt_diffs, ebv_diffs, max_diffs = {}, {}, {}
    salt_diffs["x0"] = -2.5 * np.log(
        p["salt_common_NIRvO"]["x0"] / p["salt_no_J_common_NIRvO"]["x0"]
    )
    for param in ("t0", "x1", "c"):
        salt_diffs[param] = (
            p["salt_common_NIRvO"][param] - p["salt_no_J_common_NIRvO"][param]
        )
    salt_dists = utils.mahalanobis_distances(
        np.array([arr for arr in salt_diffs.values()]).T,
        origin=True,
        cov=cov_salt,
    )
    for param in ("Tmax", "st", "EBVhost"):
        ebv_diffs[param] = (
            p["ebv_common_NIRvO"][param] - p["ebv_no_J_common_NIRvO"][param]
        )
    ebv_dists = utils.mahalanobis_distances(
        np.array([arr for arr in ebv_diffs.values()]).T,
        origin=True,
        cov=cov_ebv,
    )
    max_diffs["V-r"] = (
        p["max_common_NIRvO"]["Vmax"]
        - p["max_no_J_common_NIRvO"]["Vmax"]
        - p["max_common_NIRvO"]["rmax"]
        + p["max_no_J_common_NIRvO"]["rmax"]
    )
    for param in ("Tmax", "st"):
        max_diffs[param] = (
            p["max_common_NIRvO"][param] - p["max_no_J_common_NIRvO"][param]
        )
    max_dists = utils.mahalanobis_distances(
        np.array([arr for arr in max_diffs.values()]).T,
        origin=True,
        cov=cov_max,
    )

    salt_dist_limit = 5 * np.std(salt_dists)
    ebv_dist_limit = 5 * np.std(ebv_dists)
    max_dist_limit = 5 * np.std(max_dists)
    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(5, 8), nrows=3, sharex=True)

    ebv_mask_bad = np.where(ebv_dists > ebv_dist_limit)
    ebv_mask_good = np.where(ebv_dists < ebv_dist_limit)
    ax1.hist(ebv_dists[ebv_mask_good], bins=15)
    ax1.hist(ebv_dists[ebv_mask_bad], color="red", width=0.5)
    ax1.set_ylabel("SNPY_EBV")
    ax1.axvline(ebv_dist_limit, color="red")

    max_mask_bad = np.where(max_dists > max_dist_limit)
    max_mask_good = np.where(max_dists < max_dist_limit)
    ax2.hist(max_dists[max_mask_good], bins=15)
    ax2.hist(max_dists[max_mask_bad], color="red", width=0.5)
    ax2.set_ylabel("SNPY_Max")
    ax2.axvline(max_dist_limit, color="red")

    salt_mask_bad = np.where(salt_dists > salt_dist_limit)
    salt_mask_good = np.where(salt_dists < salt_dist_limit)
    ax3.hist(salt_dists[salt_mask_good], bins=15)
    ax3.hist(salt_dists[salt_mask_bad], color="red", width=0.5)
    ax3.set_ylabel("SALT")
    ax3.axvline(salt_dist_limit, color="red")

    ax3.set_xlabel("Mahalanobis Distance (unitless)")
    ax1.set_yscale("log")
    ax2.set_yscale("log")
    ax3.set_yscale("log")
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    fig.savefig(
        f"/data/users/ado/papers_git/HSF_survey_paper/figures/mahalanobis_J_O.{file_ext}"
    )
    if show:
        plt.show()


def snpy_salt_shape(p, outlier_names=[], show=False):
    import pingouin as pg
    from scipy.odr import ODR, Model, RealData, polynomial

    if p is None:
        from survey_paper import get_data_dict

        p = get_data_dict()["p"]

    outlier_idx = []
    names = p["ebv_common"]["TNS_name"]
    for n in outlier_names:
        if n not in names:
            continue
        outlier_idx.append(np.where(names == n)[0][0])
    outlier_idx = np.array(outlier_idx)

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

    fig, (ax1, ax2) = plt.subplots(figsize=(9, 5), ncols=2)
    ax1.errorbar(
        shape_ebv, shape_max, xerr=e_shape_ebv, yerr=e_shape_max, ls="none", alpha=0.75
    )
    ax1.errorbar(
        shape_ebv[outlier_idx],
        shape_max[outlier_idx],
        xerr=e_shape_ebv[outlier_idx],
        yerr=e_shape_max[outlier_idx],
        ls="none",
        color="red",
    )
    ax1.plot(
        np.linspace(-0.4, 0.3, 10),
        np.linspace(-0.4, 0.3, 10),
        color="black",
        ls="dotted",
    )
    ax2.errorbar(
        shape_snpy,
        shape_salt,
        xerr=e_shape_snpy,
        yerr=e_shape_salt,
        ls="none",
        alpha=0.75,
    )
    ax2.errorbar(
        shape_snpy[outlier_idx],
        shape_salt[outlier_idx],
        xerr=e_shape_snpy[outlier_idx],
        yerr=e_shape_salt[outlier_idx],
        ls="none",
        color="red",
    )
    shape_data = RealData(shape_snpy, shape_salt, sx=e_shape_snpy, sy=e_shape_salt)
    shape_odr = {}
    shape_out = {}
    for poly, i, color, ls in zip(
        ("Linear", "Quadratic", "Cubic"),
        range(1, 4),
        ("black", "orange", "magenta"),
        ("dotted", "solid", "dashed"),
    ):
        shape_odr[poly] = ODR(shape_data, polynomial(i))
        shape_out[poly] = shape_odr[poly].run()
        ax2.plot(
            np.linspace(-0.4, 0.3, 20),
            np.poly1d(shape_out[poly].beta[::-1])(np.linspace(-0.4, 0.3, 20)),
            color=color,
            ls=ls,
        )
    ax1.set_xlabel(r"$s_{BV} - 1$" + " (EBV_model2)")
    ax1.set_ylabel(r"$s_{BV} - 1$" + " (max_model)")
    ax2.set_xlabel(r"$s_{BV} - 1$" + " (Average)")
    ax2.set_ylabel(r"$x_1$" + " (SALT3-NIR)")
    plt.tight_layout()
    fig.savefig(
        f"/data/users/ado/papers_git/HSF_survey_paper/figures/SNPY_SALT_shape_match.{file_ext}"
    )
    if show:
        plt.show()


def snpy_salt_color(p, outlier_names=[], show=False):
    import pingouin as pg
    from scipy.odr import ODR, Model, RealData, polynomial

    if p is None:
        from survey_paper import get_data_dict

        p = get_data_dict()["p"]

    outlier_idx = []
    names = p["ebv_common"]["TNS_name"]
    for n in outlier_names:
        if n not in names:
            continue
        outlier_idx.append(np.where(names == n)[0][0])
    outlier_idx = np.array(outlier_idx)
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

    fig, (ax1, ax2) = plt.subplots(figsize=(9, 5), ncols=2)
    ax1.errorbar(
        color_ebv,
        color_salt,
        xerr=e_color_ebv,
        yerr=e_color_salt,
        ls="none",
        alpha=0.75,
    )
    ax2.errorbar(
        color_max,
        color_salt,
        xerr=e_color_max,
        yerr=e_color_salt,
        ls="none",
        alpha=0.75,
    )
    ax1.errorbar(
        color_ebv[outlier_idx],
        color_salt[outlier_idx],
        xerr=e_color_ebv[outlier_idx],
        yerr=e_color_salt[outlier_idx],
        ls="none",
        color="red",
    )
    ax2.errorbar(
        color_max[outlier_idx],
        color_salt[outlier_idx],
        xerr=e_color_max[outlier_idx],
        yerr=e_color_salt[outlier_idx],
        ls="none",
        color="red",
    )
    color_ebv_data = RealData(color_ebv, color_salt, sx=e_color_ebv, sy=e_color_salt)
    color_max_data = RealData(color_max, color_salt, sx=e_color_max, sy=e_color_salt)
    color_ebv_odr, color_ebv_out, color_max_odr, color_max_out = [{} for i in range(4)]
    for poly, i, color, ls in zip(
        ("Linear", "Quadratic", "Cubic"),
        range(1, 4),
        ("black", "orange", "magenta"),
        ("dotted", "solid", "dashed"),
    ):
        color_ebv_odr[poly] = ODR(color_ebv_data, polynomial(i))
        color_ebv_out[poly] = color_ebv_odr[poly].run()
        color_max_odr[poly] = ODR(color_max_data, polynomial(i))
        color_max_out[poly] = color_max_odr[poly].run()
        ax1.plot(
            np.linspace(-0.2, 0.3, 20),
            np.poly1d(color_ebv_out[poly].beta[::-1])(np.linspace(-0.2, 0.3, 20)),
            color=color,
            ls=ls,
        )
        ax2.plot(
            np.linspace(-1, 0.1, 20),
            np.poly1d(color_max_out[poly].beta[::-1])(np.linspace(-1, 0.1, 20)),
            color=color,
            ls=ls,
        )
    ax1.set_xlabel(r"$E(B-V)_\mathrm{host}$" + " (mag) (EBV_model2)")
    ax2.set_xlabel(r"$m_V - m_r$" + " (mag) (max_model)")
    ax1.set_ylabel(r"$c$" + " (SALT3-NIR)")
    ax2.set_ylabel(r"$c$" + " (SALT3-NIR)")
    plt.tight_layout()
    fig.savefig(
        f"/data/users/ado/papers_git/HSF_survey_paper/figures/SNPY_SALT_color_match.{file_ext}"
    )
    if show:
        plt.show()


def divergent_inferences(p, dists, outlier_names=[], std_cutoff=5, show=False):
    good_idx = [True for i in p["ebv_common"]["TNS_name"]]
    names = p["ebv_common"]["TNS_name"]
    for n in outlier_names:
        if n not in names:
            continue
        good_idx[np.where(names == n)[0][0]] = False
    good_idx = np.array(good_idx)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(dists[good_idx])
    ax.axvline(std_cutoff * np.std(dists), color="red")
    ax.hist(dists[~good_idx], color="red")
    ax.set_xlabel("Mahalanobis Distance (unitless)")
    plt.tight_layout()
    fig.savefig(
        f"/data/users/ado/papers_git/HSF_survey_paper/figures/mahalanobis_outliers.{file_ext}"
    )
    if show:
        plt.show()


def hsf_dehvils_varied_method(bootstrap_dict=None, stat="STD", show=False):
    if bootstrap_dict is None:
        import pickle

        with open(
            f"{constants.HSF_DIR}/ado/varied_method_dict.pickle",
            "rb",
        ) as handle:
            bootstrap_dict = pickle.load(handle)
    dehvils_values = {
        "STDYJH": (0.172, 0.027),
        "NMADYJH": (0.139, 0.026),
        "STDcoYJH": (0.175, 0.034),
        "NMADcoYJH": (0.132, 0.025),
        "STDco": (0.221, 0.043),
        "NMADco": (0.177, 0.029),
    }
    fig, [[ax1, ax4, ax7], [ax2, ax5, ax8], [ax3, ax6, ax9]] = plt.subplots(
        ncols=3,
        nrows=3,
        sharex=True,
        figsize=(9, 6),
    )
    for ax, sample in zip(
        [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9],
        [
            "EBVco",
            "Maxco",
            "SALTco",
            "EBVcoYJH",
            "MaxcoYJH",
            "SALTcoYJH",
            "EBVYJH",
            "MaxYJH",
            "SALTYJH",
        ],
    ):
        ax.hist(bootstrap_dict[sample][stat], density=True, bins=20)
        ax.set_yticks([])
        if "coYJH" in sample:
            dval, derr = dehvils_values[f"{stat}coYJH"]
        elif "co" in sample:
            dval, derr = dehvils_values[f"{stat}co"]
        else:
            dval, derr = dehvils_values[f"{stat}YJH"]
        ax.axvline(dval, color="black")
        ax.axvline(dval - derr, ls="dashed", color="black")
        ax.axvline(dval + derr, ls="dashed", color="black")
        if ax not in (ax3, ax6, ax9):
            ax.tick_params(labelbottom=False)
    ax5.set_yticks([])
    ax1.set_ylabel("EBV_model2")
    ax2.set_ylabel("max_model")
    ax3.set_ylabel("SALT3-NIR")
    ax1.set_title("co")
    ax4.set_title("coYJH")
    ax7.set_title("YJH")
    ax3.set_xlabel(stat + " (mag)")
    ax6.set_xlabel(stat + " (mag)")
    ax9.set_xlabel(stat + " (mag)")
    plt.suptitle(stat)
    plt.subplots_adjust(wspace=0, hspace=0)
    if show:
        plt.show()
    fig.savefig(
        f"/data/users/ado/papers_git/HSF_survey_paper/figures/varied_method_{stat}.{file_ext}"
    )


def hsf_dehvils_varied_phot(bootstrap_dict=None, stat="STD", show=False):
    if bootstrap_dict is None:
        import pickle

        with open(f"{constants.HSF_DIR}/ado/varied_phot_dict.pickle", "rb") as handle:
            bootstrap_dict = pickle.load(handle)
    fig, [[ax1, ax2], [ax3, ax4], [ax5, ax6]] = plt.subplots(
        ncols=2, nrows=3, figsize=(9, 6)
    )
    ax_xmin, ax_xmax, diff_ax_xmin, diff_ax_xmax = 1, 0, 1, -1
    for ax, diff_ax, sample in zip(
        [ax1, ax3, ax5],
        [ax2, ax4, ax6],
        ["EBV", "Max", "SALT"],
    ):
        ax.hist(
            bootstrap_dict[sample + "HSF"][stat],
            label="HSF",
            alpha=0.5,
            density=True,
            color="blue",
            bins=15,
            hatch="/",
        )
        ax.hist(
            bootstrap_dict[sample + "DEHVILS"][stat],
            label="DEHVILS",
            alpha=0.5,
            density=True,
            color="red",
            bins=15,
            hatch="\\",
        )
        diff_ax.hist(
            bootstrap_dict[sample + "HSF"][stat]
            - bootstrap_dict[sample + "DEHVILS"][stat],
            density=True,
            color="purple",
            bins=15,
            hatch="x",
        )
        ax.set_yticks([])
        diff_ax.set_yticks([])
        ax_xmin = min(ax_xmin, ax.get_xlim()[0])
        ax_xmax = max(ax_xmax, ax.get_xlim()[1])
        diff_ax_xmin = min(diff_ax_xmin, diff_ax.get_xlim()[0])
        diff_ax_xmax = max(diff_ax_xmax, diff_ax.get_xlim()[1])
        if ax != ax5:
            ax.tick_params(labelbottom=False)
        if diff_ax != ax6:
            diff_ax.tick_params(labelbottom=False)
    for ax in (ax1, ax3, ax5):
        ax.set_xlim(ax_xmin, ax_xmax)
    for diff_ax in (ax2, ax4, ax6):
        diff_ax.set_xlim(diff_ax_xmin - 0.05, diff_ax_xmax)
    ax1.set_ylabel("EBV_model2")
    ax1.legend()
    ax3.set_ylabel("max_model")
    ax5.set_ylabel("SALT3-NIR")
    ax5.set_xlabel(stat + " (mag)")
    ax6.set_xlabel(stat + r"$_{HSF}$ - " + stat + "$_{DEHVILS}$ (mag)")
    plt.suptitle(stat)
    plt.subplots_adjust(wspace=0, hspace=0)
    if show:
        plt.show()
    fig.savefig(
        f"/data/users/ado/papers_git/HSF_survey_paper/figures/varied_phot_{stat}.{file_ext}"
    )


def plot_dispersion_vs_cuts(vals_dict=None, show=False):
    if vals_dict is None:
        import pickle

        with open(f"{constants.HSF_DIR}/ado/dispersion_vs_cuts.pickle", "rb") as f:
            vals_dict = pickle.load(f)
    estimators = ("RMS", "WRMS", "sigma_int", "NMAD")
    dehvils_cuts = {
        "x1": 3,
        "e_x1": 1,
        "e_t0": 2,
        "mwebv": 0.2,
        "f": 326,
        "rchi2_model": 5.5,
    }
    parameters = ("x1", "e_x1", "e_t0", "mwebv", "rchi2_model")
    param_names = (
        r"$|x_1|$ cut",
        r"$\sigma_{x_1}$ cut",
        r"$\sigma_{t_0}$ cut",
        r"$E(B-V)_{MW}$ cut",
        r"$\chi^2 / DoF$ cut",
    )
    fig, axs = plt.subplots(figsize=(12, 6), ncols=len(parameters), sharey=True)
    for ax, param, param_name in zip(axs, parameters, param_names):
        x = vals_dict[param]["xvals"]
        for estimator in estimators:
            estimator_name = estimator
            if estimator == "sigma_int":
                estimator_name = r"$\sigma_{int}$"
            ave = np.average(vals_dict[param][estimator], axis=1)
            sd = np.std(vals_dict[param][estimator], axis=1)
            ax.plot(x, ave, label=estimator_name)
            ax.fill_between(x, ave - sd, ave + sd, zorder=-1, alpha=0.5, linewidth=0)
            ax.set_xlabel(param_name)
            if param in dehvils_cuts:
                ax.axvline(dehvils_cuts[param], color="black")
                ax.set_xlim(0, min(x[-1], 3 * dehvils_cuts[param]))
    axs[0].legend()
    fig.subplots_adjust(wspace=0)
    plt.ylim(0.07, 0.3)
    fig.savefig(
        f"/data/users/ado/papers_git/HSF_survey_paper/figures/dispersion_vs_cuts.{file_ext}"
    )
    if show:
        plt.show()


def hubble_diagram(
    p_sample,
    p_other=None,
    pec_v=250,
    sigma_int=0.1,
    title="",
    label="SALT",
    figsize=(8, 6),
    show=False,
    dark=False,
):
    from constants import C

    facecolor = "white"
    linecolor = "black"
    plotcolor = "blue"
    offcolor = "red"
    utils.light_plot()
    if dark:
        facecolor = "black"
        linecolor = "white"
        plotcolor = "cyan"
        offcolor = "orange"
        utils.dark_plot()

    fig, [ax1, ax2] = plt.subplots(
        nrows=2,
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
        figsize=figsize,
        facecolor=facecolor,
    )
    ax1.set_facecolor(facecolor)
    ax2.set_facecolor(facecolor)
    if p_other is not None:
        mask = np.ones(len(p_other["TNS_name"]), dtype=bool)
        for i, name in enumerate(p_other["TNS_name"]):
            if name in p_sample["TNS_name"]:
                mask[i] = False
        ax1.errorbar(
            p_other["z_cmb"][mask],
            p_other["DM"][mask],
            yerr=p_other["e_DM"][mask],
            ls="none",
            color=offcolor,
            label="Cut Targets",
        )
        ax2.errorbar(
            p_other["z_cmb"][mask],
            p_other["resid_DM"][mask],
            yerr=p_other["e_DM"][mask],
            ls="none",
            color=offcolor,
        )
    ax1.errorbar(
        p_sample["z_cmb"],
        p_sample["DM"],
        yerr=p_sample["e_DM"],
        ls="none",
        color=plotcolor,
        label=label,
    )
    ax2.errorbar(
        p_sample["z_cmb"],
        p_sample["resid_DM"],
        yerr=p_sample["e_DM"],
        ls="none",
        color=plotcolor,
    )
    hub_z = np.linspace(0, 0.12, 100)
    hub_DM = utils.mu_lcdm(hub_z, hub_z, p_sample["h"])
    ax1.plot(hub_z, hub_DM, color=linecolor, alpha=0.75)
    ax1ylim = ax1.get_ylim()
    ax1ylim = (26.1, ax1ylim[1])
    ax2ylim = ax2.get_ylim()

    pv = pec_v / constants.C * 5 / np.log(10) * (1 + hub_z) / (hub_z * (1 + hub_z / 2))
    sigma = np.sqrt(pv**2 + sigma_int**2)
    ax1.plot(hub_z, hub_DM - sigma, ls="dashed", color=linecolor, alpha=0.5)
    ax1.plot(hub_z, hub_DM + sigma, ls="dashed", color=linecolor, alpha=0.5)
    ax2.plot(hub_z, sigma, ls="dashed", color=linecolor, alpha=0.5)
    ax2.plot(hub_z, -sigma, ls="dashed", color=linecolor, alpha=0.5)

    ax2.axhline(0, color=linecolor)
    ax1.set_ylim(ax1ylim)
    ax2.set_ylim(ax2ylim)
    xlim = ax1.get_xlim()
    if xlim[0] < 0:
        ax1.set_xlim(0, xlim[1])
    ax1.set_ylabel(r"$\mu$ (mag)")
    ax2.set_ylabel(r"$\Delta\mu$ (mag)")
    ax2.set_xlabel(r"$z_{CMB}$")
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    if title != "":
        plt.title(title)
    if show:
        plt.show()
    fig.savefig(
        f"/data/users/ado/papers_git/HSF_survey_paper/figures/hubble_{label}.{file_ext}"
    )


def residual_differences(
    p,
    stats,
    sample_1,
    diff,
    sample_2,
    label_1="SNPY\_EBV",
    label_2="SALT",
    plot_xor=False,
    odr=False,
    mcmc=True,
    mcmc_center=0,
    rolling_fn=None,
    show=False,
    return_results=True,
    legend=False,
    cuts={"c": (-0.3, 0.3)},
    figsize=(5, 4),
):
    mask_1 = np.ones(len(p[sample_1]["TNS_name"]), dtype=bool)
    if sample_2 is not None:
        mask_2 = np.ones(len(p[sample_2]["TNS_name"]), dtype=bool)
    y = p[sample_1]["resid_DM"]
    dy = p[sample_1]["e_DM"]
    if diff:
        import pingouin as pg

        for i, name in enumerate(p[sample_1]["TNS_name"]):
            if name not in p[sample_2]["TNS_name"]:
                mask_1[i] = False
                continue
            for cut in cuts:
                if cut in p[sample_1]:
                    if (
                        p[sample_1][cut][i] < cuts[cut][0]
                        or p[sample_1][cut][i] > cuts[cut][1]
                    ):
                        mask_1[i] = False
                if cut in p[sample_2]:
                    idx = np.where(p[sample_2]["TNS_name"] == name)[0][0]
                    if (
                        p[sample_2][cut][idx] < cuts[cut][0]
                        or p[sample_2][cut][idx] > cuts[cut][1]
                    ):
                        mask_1[i] = False
        for i, name in enumerate(p[sample_2]["TNS_name"]):
            if name not in p[sample_1]["TNS_name"]:
                mask_2[i] = False
                continue
            for cut in cuts:
                if cut in p[sample_1]:
                    idx = np.where(p[sample_1]["TNS_name"] == name)[0][0]
                    if (
                        p[sample_1][cut][idx] < cuts[cut][0]
                        or p[sample_1][cut][idx] > cuts[cut][1]
                    ):
                        mask_2[i] = False
                if cut in p[sample_2]:
                    if (
                        p[sample_2][cut][i] < cuts[cut][0]
                        or p[sample_2][cut][i] > cuts[cut][1]
                    ):
                        mask_2[i] = False
        y = y[mask_1] - p[sample_2]["resid_DM"][mask_2]
        dy = np.sqrt(
            dy[mask_1] ** 2
            + p[sample_2]["e_DM"][mask_2] ** 2
            - 2
            * dy[mask_1]
            * p[sample_2]["e_DM"][mask_2]
            * pg.corr(dy[mask_1], p[sample_2]["e_DM"][mask_2]).r[0]
        )
    # dx = np.nan_to_num(p[sample_1]["e_z"][mask_1], nan=0.0002)
    x = p[sample_1]["z_cmb"][mask_1]

    fig, ax = plt.subplots(figsize=figsize)
    ax.errorbar(x, y, yerr=np.sqrt(dy**2), ls="none", color="black", alpha=0.7)
    z = np.linspace(0, 0.12, 5)
    ax.axhline(0, color="black")
    if plot_xor:
        ax.errorbar(
            p[sample_1]["z_cmb"][~mask_1],
            p[sample_1]["resid_DM"][~mask_1],
            yerr=p[sample_1]["e_DM"][~mask_1],
            color="red",
            ls="none",
        )
        ax.errorbar(
            p[sample_2]["z_cmb"][~mask_2],
            -p[sample_2]["resid_DM"][~mask_2],
            yerr=p[sample_2]["e_DM"][~mask_2],
            color="orange",
            ls="none",
        )
    ax.set_xlabel(r"$z_{CMB}$")
    ax.set_ylabel(
        r"$\Delta\mu_{%s} - \Delta\mu_{%s}$ (mag)" % (label_1, label_2)
    )  # braces overloaded
    ret_list = []
    if odr:
        odr_object = utils.odr_polyfit(
            x, y, np.ones(len(x)) * 0.0001, np.sqrt(dy**2), degree=1
        )
        odr_object.pprint()
        ax.plot(
            z, odr_object.beta[0] * z + odr_object.beta[1], label="ODR", color="red"
        )
        ret_list.append(odr_object)
    if mcmc:
        res, samples = utils.line_fit(
            x,
            y,
            np.sqrt(dy**2),
            mcmc_center,
            plots=0,
            return_samples=True,
        )
        xfit = np.linspace(0, 0.13, 100)
        yfit = samples.T[0] + samples.T[1] * xfit[..., np.newaxis]
        mu = yfit.mean(-1)
        sig = yfit.std(-1)
        ax.fill_between(xfit, mu - 2 * sig, mu + 2 * sig, color="red", alpha=0.2)
        ax.fill_between(xfit, mu - sig, mu + sig, color="red", alpha=0.5)
        # ax.plot(xfit, mu, label="MCMC", color="red")
        ret_list.append(res)
    if rolling_fn is not None:
        roll, xcenter = utils.rolling_fn(
            x, y, np.median, xcenter=np.linspace(0, 0.13, 50), window=0.01, units="z"
        )
        ax.plot(xcenter, roll, label="Rolling Median", color="green")
        ret_list.append(np.array([xcenter, roll]))

    ax.set_xlim(0, 0.12)

    if legend:
        plt.legend()
    plt.tight_layout()
    fig.savefig(
        f"/data/users/ado/papers_git/HSF_survey_paper/figures/residual_trend.{file_ext}"
    )
    if show:
        plt.show()
    if return_results:
        return ret_list


def trend_vs_params(show=False, figsize=(8, 6)):
    import pickle

    from snippets import prep_data_trend

    with open(f"{constants.HSF_DIR}/ado/data_dict.pickle", "rb") as f:
        data_dict = pickle.load(f)
    fr_qsets = data_dict["fr_qsets"]
    junk = data_dict["junk"]
    stats = data_dict["stats"]
    p = data_dict["p"]
    (
        x,
        y,
        dy,
        st,
        e_st,
        x1,
        e_x1,
        EBVhost,
        e_EBVhost,
        c,
        e_c,
        snpy_resid,
        salt_resid,
        snpy_err,
        salt_err,
    ) = prep_data_trend(
        p,
        stats,
        "ebv_final",
        True,
        "salt_final",
        [
            "st",
            "e_st",
            "x1",
            "e_x1",
            "EBVhost",
            "e_EBVhost",
            "c",
            "e_c",
            "resid_DM",
            "e_DM",
        ],
    )
    fig, [ax1, ax2, ax3] = plt.subplots(nrows=3, ncols=4, figsize=figsize, sharey=True)
    mcmc = {"snpy": [], "salt": [], "diff": []}
    all_samples = {"snpy": [], "salt": [], "diff": []}
    # cmap = plt.cm.viridis
    # norm = matplotlib.colors.Normalize(vmin=0, vmax=max(x))
    # sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm|)
    # sm.set_array([])
    # color = cmap(norm(x))
    for ax, y_val, y_errs, label in zip(
        (ax1, ax2, ax3),
        (snpy_resid, salt_resid, y),
        (snpy_err, salt_err, dy),
        ("snpy", "salt", "diff"),
    ):
        for i, (sub_ax, x_vals, x_errs) in enumerate(
            zip(ax, (st, x1, EBVhost, c), (e_st, e_x1, e_EBVhost, e_c))
        ):
            sub_ax.errorbar(
                x_vals, y_val, xerr=x_errs, yerr=y_errs, ls="none", alpha=0.3
            )
            if y_val is not y:
                sub_ax.tick_params(labelbottom=False)
            if i > 0:
                sub_ax.tick_params(labelleft=False)
            sub_ax.axhline(0, color="black")
            value, samples = utils.line_fit(
                x_vals, y_val, y_errs, 0, plots=0, return_samples=True
            )
            mcmc[label].append(value)
            all_samples[label].append(samples)
            # odr[label].append(
            #     utils.odr_polyfit(x_vals, y_val, x_errs, y_errs, degree=1)
            # )
    for ax, label in zip((ax1, ax2, ax3), ("snpy", "salt", "diff")):
        for i, sub_ax in enumerate(ax):
            xlim = sub_ax.get_xlim()
            fit_x_vals = np.linspace(xlim[0], xlim[1], 10)
            yfit = (
                fit_x_vals[..., np.newaxis] * all_samples[label][i].T[1]
                + all_samples[label][i].T[0]
            )
            mu = yfit.mean(-1)
            sig = yfit.std(-1)

            # fit_y_vals = odr[label][i].beta[0] * fit_x_vals + odr[label][i].beta[1]
            # sigma = 2 * utils.linear_model_uncertainty(
            #     fit_x_vals, cov=odr[label][i].cov_beta
            # )
            sub_ax.plot(fit_x_vals, mu, color="red")
            sub_ax.fill_between(
                fit_x_vals,
                mu - 2 * sig,
                mu + 2 * sig,
                color="red",
                alpha=0.5,
            )
            # sub_ax.fill_between(
            #     fit_x_vals,
            #     fit_y_vals - sigma,
            #     fit_y_vals + sigma,
            #     color="red",
            #     alpha=0.2,
            # )
            sub_ax.set_xlim(xlim)
    ax1[0].set_ylabel(r"$\Delta\mu_{SNPY\_EBV}$ (mag)")
    ax2[0].set_ylabel(r"$\Delta\mu_{SALT}$ (mag)")
    ax3[0].set_ylabel(r"$\Delta\mu_{SNPY\_EBV} - \Delta\mu_{SALT}$ (mag)")
    ax3[0].set_xlabel(r"$s_{BV}$")
    ax3[1].set_xlabel(r"$x_1$")
    ax3[2].set_xlabel(r"$E(B-V)_{host}$ (mag)")
    ax3[3].set_xlabel(r"c")
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    fig.savefig(
        f"/data/users/ado/papers_git/HSF_survey_paper/figures/trend_vs_params.{file_ext}"
    )
    if show:
        plt.show()


def UNITY_true_vs_observed(show=False):
    import gzip
    import pickle

    import pandas as pd

    # subdir = "240220" # pname.endswith uses 1-based indexing
    subdir = "salt"  # pname.endswith uses 0-based indexing
    res = pd.read_csv(
        f"{constants.HSF_DIR}/unity_runs/{subdir}/params.out",
        delim_whitespace=True,
    )
    with gzip.open(
        f"{constants.HSF_DIR}/unity_runs/{subdir}/inputs_UKIRT.pickle", "rb"
    ) as f:
        inputs = pickle.load(f)
    true_mB, e_mB, true_x1, e_x1, true_cB, e_cB = [[] for i in range(6)]
    for i in range(len(res["param_names"])):
        pname = res["param_names"][i]
        if "cov" in pname:
            continue
        if pname.startswith("model_mBx1c[") and pname.endswith("0]"):
            true_mB.append(res["mean"][i])
            e_mB.append(res["sd"][i])
        elif pname.startswith("model_mBx1c[") and pname.endswith("1]"):
            true_x1.append(res["mean"][i])
            e_x1.append(res["sd"][i])
        elif pname.startswith("model_mBx1c[") and pname.endswith("2]"):
            true_cB.append(res["mean"][i])
            e_cB.append(res["sd"][i])
    true_x1 = np.array(true_x1)
    e_x1 = np.array(e_x1)
    true_cB = np.array(true_cB)
    e_cB = np.array(e_cB)
    fig, [ax1, ax2] = plt.subplots(ncols=2, figsize=(8, 5))
    cmap = plt.cm.viridis
    norm = matplotlib.colors.Normalize(vmin=0, vmax=max(inputs[0]["z_CMB_list"]))
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    color = cmap(norm(inputs[0]["z_CMB_list"]))
    m = np.ones(len(true_x1), dtype=bool)
    # z_split = np.median(inputs[0]['z_CMB_list'])
    # m_lo = np.where(inputs[0]['z_CMB_list'] < z_split)
    # m_hi = np.where(inputs[0]['z_CMB_list'] >= z_split)
    ax1.errorbar(
        inputs[0]["x1_list"][m],
        true_x1[m],
        xerr=[np.sqrt(cov[1, 1]) for cov in inputs[0]["mBx1c_cov_list"]],
        yerr=e_x1[m],
        ecolor=color,
        ls="none",
        alpha=0.5,
    )
    ax1.plot([min(true_x1), max(true_x1)], [min(true_x1), max(true_x1)], color="black")
    ax1.set_xlabel(r"Observed $x_1$")
    ax1.set_ylabel(r"True $x_1$")
    ax2.errorbar(
        inputs[0]["c_list"][m],
        true_cB[m],
        xerr=[np.sqrt(cov[2, 2]) for cov in inputs[0]["mBx1c_cov_list"]],
        yerr=e_cB,
        ecolor=color,
        ls="none",
        alpha=0.5,
    )
    ax2.set_xlabel(r"Observed $c$")
    ax2.set_ylabel(r"True $c$")
    ax2.plot([min(true_cB), max(true_cB)], [min(true_cB), max(true_cB)], color="black")
    plt.tight_layout()
    cbar_ax = fig.add_axes([0.105, 0.125, 0.86, 0.025])
    fig.subplots_adjust(bottom=0.3)
    fig.colorbar(sm, cax=cbar_ax, orientation="horizontal", label=r"$z_{CMB}$")

    fig.savefig(
        f"/data/users/ado/papers_git/HSF_survey_paper/figures/UNITY_true_vs_observed.{file_ext}"
    )
    if show:
        plt.show()


def eddington_bias_vs_redshift(
    bin_min_z=0,
    bin_max_z=0.1,
    num_bins=21,
    capsize=5,
    roll_fn=utils.STD,
    show=False,
    figsize=(5, 4),
):
    import gzip
    import pickle

    import pandas as pd
    import pingouin as pg

    xcenter = np.linspace(bin_min_z, bin_max_z, num_bins)
    window = (bin_max_z - bin_min_z) / num_bins
    res = pd.read_csv(
        f"{constants.HSF_DIR}/unity_runs/salt/params.out",
        delim_whitespace=True,
        index_col="param_names",
    )
    with gzip.open(
        f"{constants.HSF_DIR}/unity_runs/salt/inputs_UKIRT.pickle", "rb"
    ) as f:
        inputs = pickle.load(f)
    true_mB, e_mB, true_x1, e_x1, true_cB, e_cB = [[] for i in range(6)]
    for i in np.where(res.index.str.startswith("model_mBx1c["))[0]:
        pname = res.index[i]
        if pname.endswith("0]"):
            true_mB.append(res["mean"][i])
            e_mB.append(res["sd"][i])
        elif pname.endswith("1]"):
            true_x1.append(res["mean"][i])
            e_x1.append(res["sd"][i])
        elif pname.endswith("2]"):
            true_cB.append(res["mean"][i])
            e_cB.append(res["sd"][i])
    true_x1 = np.array(true_x1)
    e_x1 = np.array(e_x1)
    true_cB = np.array(true_cB)
    e_cB = np.array(e_cB)
    input_e_x1 = np.array([np.sqrt(cov[1][1]) for cov in inputs[0]["mBx1c_cov_list"]])
    input_e_c = np.array([np.sqrt(cov[2][2]) for cov in inputs[0]["mBx1c_cov_list"]])

    fig, [ax1, ax2] = plt.subplots(nrows=2, sharex=True, figsize=figsize)
    z = inputs[0]["z_CMB_list"]
    delta_x1 = inputs[0]["x1_list"] - true_x1
    delta_c = inputs[0]["c_list"] - true_cB
    e_delta_x1 = np.sqrt(
        e_x1**2
        + input_e_x1**2
        - 2 * e_x1 * input_e_x1 * pg.corr(e_x1, input_e_x1)["r"][0]
    )
    e_delta_c = np.sqrt(
        e_cB**2 + input_e_c**2 - 2 * e_cB * input_e_c * pg.corr(e_cB, input_e_c)["r"][0]
    )
    x1_roll = utils.rolling_fn(
        z, delta_x1, roll_fn, dy=e_delta_x1, xcenter=xcenter, window=window, units="z"
    )[0]
    c_roll = utils.rolling_fn(
        z, delta_c, roll_fn, dy=e_delta_c, xcenter=xcenter, window=window, units="z"
    )[0]
    x1_bin_med = utils.rolling_fn(
        z, delta_x1, np.median, dy=e_delta_x1, xcenter=xcenter, window=window, units="z"
    )[0]
    c_bin_med = utils.rolling_fn(
        z, delta_c, np.median, dy=e_delta_c, xcenter=xcenter, window=window, units="z"
    )[0]
    ax1.errorbar(z, delta_x1, yerr=e_delta_x1, ls="none", alpha=0.75)
    ax2.errorbar(z, delta_c, yerr=e_delta_c, ls="none", alpha=0.75)
    ax1.axhline(0, color="black")
    ax2.axhline(0, color="black")
    ax1.errorbar(
        xcenter,
        x1_bin_med,
        yerr=x1_roll,
        color="red",
        capsize=capsize,
        marker=".",
        ls="none",
    )
    ax2.errorbar(
        xcenter,
        c_bin_med,
        yerr=c_roll,
        color="red",
        capsize=capsize,
        marker=".",
        ls="none",
    )
    ax1.set_ylabel(r"$\Delta x_1$")
    ax2.set_ylabel(r"$\Delta c$")
    ax2.set_xlabel(r"$z_{CMB}$")
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    fig.savefig(
        f"/data/users/ado/papers_git/HSF_survey_paper/figures/eddington_bias_vs_redshift.{file_ext}"
    )
    if show:
        plt.show()


def beta_vs_redshift(show=False, figsize=(4, 3)):
    import glob

    import pandas as pd

    z_cut = np.array([0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09])
    alpha_low, alpha_bin, beta_low, beta_bin = [
        np.zeros((len(z_cut), 2)) for i in range(4)
    ]
    for path in glob.glob("unity_runs/240223/*/params.out"):
        subdir = path.split("/")[-2]
        if subdir.startswith("high"):
            continue
        z = float(subdir.split("_")[1]) / 100
        params = pd.read_csv(path, delim_whitespace=True)
        idx = np.where(params["param_names"] == "alpha")[0][0]
        try:
            z_idx = np.where(z_cut == z)[0][0]
        except:
            continue
        alpha_low[z_idx, 0] = params["mean"][idx]
        alpha_low[z_idx, 1] = params["sd"][idx]
        beta_low[z_idx, 0] = params["mean"][idx + 1]
        beta_low[z_idx, 1] = params["sd"][idx + 1]
    for path in glob.glob("unity_runs/240227/*/params.out"):
        subdir = path.split("/")[-2]
        z = float(subdir) / 100
        params = pd.read_csv(path, delim_whitespace=True)
        idx = np.where(params["param_names"] == "alpha")[0][0]
        try:
            z_idx = np.where(z_cut == z)[0][0]
        except:
            continue
        alpha_bin[z_idx, 0] = params["mean"][idx]
        alpha_bin[z_idx, 1] = params["sd"][idx]
        beta_bin[z_idx, 0] = params["mean"][idx + 1]
        beta_bin[z_idx, 1] = params["sd"][idx + 1]
    fig, ax = plt.subplots(figsize=figsize)
    ax.errorbar(
        z_cut,
        beta_low[:, 0],
        yerr=beta_low[:, 1],
        marker=".",
        label=r"$\beta$ ($z < z'$)",
    )
    ax.errorbar(
        z_cut,
        beta_bin[:, 0],
        yerr=beta_bin[:, 1],
        marker=".",
        ls="none",
        label=r"$\beta$ ($|z - z'| < 0.005$)",
    )
    ax.set_xlabel(r"$z'_{CMB}$")
    ax.set_ylabel(r"$\beta$")
    plt.legend()
    plt.tight_layout()
    fig.savefig(
        f"/data/users/ado/papers_git/HSF_survey_paper/figures/beta_vs_redshift.{file_ext}"
    )
    if show:
        plt.show()


def mass_step(
    data_dict=None,
    g_masses=None,
    e_g_masses=None,
    figsize=(8, 8),
    xlim=(8.6, 11.7),
    show=False,
    dark=False,
):
    facecolor = "white"
    linecolor = "black"
    plotcolor = "blue"
    offcolor = "red"
    utils.light_plot()
    if dark:
        facecolor = "black"
        linecolor = "white"
        plotcolor = "cyan"
        offcolor = "orange"
        utils.dark_plot()

    samples = ("ebv", "salt")
    if data_dict is None:
        from survey_paper import get_data_dict

        fr_qsets, qsets, junk, stats, p = get_data_dict()
    else:
        fr_qsets, qsets, junk, stats, p = data_dict
    if g_masses is None or e_g_masses is None:
        g_masses, e_g_masses = [{} for i in range(2)]
        for name in samples:
            sub_name = f"{name}_common_NIRvO_final"
            g_masses[name] = np.zeros(stats[sub_name]["N"])
            e_g_masses[name] = np.zeros(stats[sub_name]["N"])
            for i, fr in enumerate(fr_qsets[sub_name]):
                g_masses[name][i], e_g_masses[name][i] = fr.target.galaxy.get_mass(
                    H0=p[sub_name]["h"]
                )
    fig, [[ax1, ax2], [ax3, ax4], [ax5, ax6]] = plt.subplots(
        figsize=figsize, nrows=3, ncols=2, sharex=True, sharey=True
    )
    for ax in (ax1, ax2, ax3, ax4, ax5, ax6):
        ax.set_facecolor(facecolor)
    for ax_triplet, sample, cap_name in zip(
        ((ax1, ax3, ax5), (ax2, ax4, ax6)), samples, ("SNPY_EBV", "SALT")
    ):
        nir_name = f"{sample}_common_NIRvO_final"
        no_nir_name = f"{sample}_no_J_common_NIRvO_final"
        ax_triplet[0].axhline(0, color=linecolor)
        ax_triplet[1].axhline(0, color=linecolor)
        ax_triplet[2].axhline(0, color=linecolor)
        x = g_masses[sample]
        mask = ~np.isnan(x)
        e_x = e_g_masses[sample]
        nir_y = p[nir_name]["resid_DM"]
        no_nir_y = p[no_nir_name]["resid_DM"]
        e_nir_y = p[nir_name]["e_DM"]
        e_no_nir_y = p[no_nir_name]["e_DM"]
        diff_err = np.sqrt(
            e_nir_y**2
            + e_no_nir_y**2
            - 2 * e_nir_y * e_no_nir_y * pg.corr(e_nir_y, e_no_nir_y)["r"][0]
        )
        fit_x = np.linspace(xlim[0], xlim[1], 10)
        for ax, y, e_y in zip(
            ax_triplet,
            (nir_y, no_nir_y, nir_y - no_nir_y),
            (e_nir_y, e_no_nir_y, diff_err),
        ):
            odr_fit = utils.odr_polyfit(x[mask], y[mask], e_x[mask], e_y[mask], 1)
            ax.errorbar(
                x,
                y,
                xerr=e_x,
                yerr=e_y,
                ls="none",
                alpha=0.5,
                color=plotcolor,
            )
            fit_y = fit_x * odr_fit.beta[0] + odr_fit.beta[1]
            ax.plot(fit_x, fit_y, color=offcolor)
            sigma = utils.linear_model_uncertainty(fit_x, cov=odr_fit.cov_beta)
            ax.fill_between(
                fit_x, fit_y - sigma, fit_y + sigma, color=offcolor, alpha=0.5
            )
            # print(cap_name)
            # odr_fit.pprint()
            # print()
        ax_triplet[0].set_title(cap_name)
        ax_triplet[2].set_xlabel(r"$\log_{10}(M_\ast/M_\odot)$")
    ax1.set_ylabel(r"$\Delta \mu_{OJ}$ (mag)")
    ax3.set_ylabel(r"$\Delta \mu_{O}$ (mag)")
    ax5.set_ylabel(r"$\Delta \mu_{OJ} - \Delta \mu_O$ (mag)")
    # ax5.set_ylabel(r"$\Delta \Delta \mu$ (mag)")
    ax1.set_xlim(xlim)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    fig.savefig(f"ado/mass_step.{file_ext}")
    if show:
        plt.show()
    return (
        fig,
        (
            g_masses,
            e_g_masses,
        ),
    )


def mass_step_max(
    data_dict=None,
    g_masses=None,
    e_g_masses=None,
    salt_nir_resid=None,
    salt_e_nir_dm=None,
    salt_optical_resid=None,
    salt_e_optical_dm=None,
    figsize=(8, 8),
    xlim=(8.6, 11.7),
    dark=True,
    show=False,
):
    facecolor = "white"
    linecolor = "black"
    plotcolor = "blue"
    offcolor = "red"
    utils.light_plot()
    if dark:
        facecolor = "black"
        linecolor = "white"
        plotcolor = "cyan"
        offcolor = "orange"
        utils.dark_plot()
    if data_dict is None:
        from survey_paper import get_data_dict

        fr_qsets, qsets, junk, stats, p = get_data_dict()
    else:
        fr_qsets, qsets, junk, stats, p = data_dict
    if g_masses is None or e_g_masses is None:
        g_masses, e_g_masses = [{} for i in range(2)]
        sub_name = "max_common_NIRvO_final"
        g_masses["max"] = np.zeros(stats[sub_name]["N"])
        e_g_masses["max"] = np.zeros(stats[sub_name]["N"])
        for i, fr in enumerate(fr_qsets[sub_name]):
            g_masses["max"][i], e_g_masses["max"][i] = fr.target.galaxy.get_mass(
                H0=p[sub_name]["h"]
            )
        g_masses["salt_max"] = np.zeros(stats["salt_common_NIRvO_final"]["N"])
        e_g_masses["salt_max"] = np.zeros(stats["salt_common_NIRvO_final"]["N"])
        for i, fr in enumerate(fr_qsets["salt_common_NIRvO_final"]):
            (
                g_masses["salt_max"][i],
                e_g_masses["salt_max"][i],
            ) = fr.target.galaxy.get_mass(H0=p["salt_common_NIRvO_final"]["h"])
    if salt_nir_resid is None or salt_e_nir_dm is None:
        from fitting.SALT3 import get_NIR_x0

        salt_nir_resid, salt_e_nir_dm = [
            np.zeros(stats["salt_common_NIRvO_final"]["N"]) for i in range(2)
        ]
        for i, fr in enumerate(fr_qsets["salt_common_NIRvO_final"]):
            (
                P,
                F,
                cov_mat,
                all_lc_data,
                trimmed_lc_data,
                other_data,
            ) = fr.get_david_results()
            nir_mag, nir_err = get_NIR_x0(P, trimmed_lc_data, other_data)
            salt_nir_resid[i] = (
                p["salt_common_NIRvO_final"]["resid_DM"][i]
                - fr.params["x0_mag"]
                + nir_mag
                + constants.BETA["david-salt3-nir"] * fr.params["c"]
            )
            salt_e_nir_dm[i] = np.sqrt(
                p["salt_common_NIRvO_final"]["e_DM"][i] ** 2
                - fr.errors["x0_mag"] ** 2
                + nir_err**2
                - (constants.BETA["david-salt3-nir"] * fr.errors["c"]) ** 2
            )
    if salt_optical_resid is None or salt_e_optical_dm is None:
        from fitting.SALT3 import get_optical_x0

        salt_optical_resid, salt_e_optical_dm = [
            np.zeros(stats["salt_no_J_common_NIRvO_final"]["N"]) for i in range(2)
        ]
        for i, fr in enumerate(fr_qsets["salt_no_J_common_NIRvO_final"]):
            (
                P,
                F,
                cov_mat,
                all_lc_data,
                trimmed_lc_data,
                other_data,
            ) = fr.get_david_results()
            optical_mag, optical_err = get_optical_x0(P, trimmed_lc_data, other_data)
            salt_optical_resid[i] = (
                p["salt_no_J_common_NIRvO_final"]["resid_DM"][i]
                - fr.params["x0_mag"]
                + optical_mag
                # + constants.BETA["david-salt3-nir"] * fr.params["c"]
            )
            salt_e_optical_dm[i] = np.sqrt(
                p["salt_no_J_common_NIRvO_final"]["e_DM"][i] ** 2
                - fr.errors["x0_mag"] ** 2
                + optical_err**2
                # - (constants.BETA["david-salt3-nir"] * fr.errors["c"]) ** 2
            )

    fig, [[ax1, ax2], [ax3, ax4], [ax5, ax6]] = plt.subplots(
        figsize=figsize, nrows=3, ncols=2, sharex=True, sharey=True
    )
    for ax_triplet, g_name, nir_y, e_nir_y, no_nir_y, e_no_nir_y in zip(
        ((ax1, ax3, ax5), (ax2, ax4, ax6)),
        ("max", "salt_max"),
        (p["max_common_NIRvO_final"]["resid_DM"], salt_nir_resid),
        (p["max_common_NIRvO_final"]["e_DM"], salt_e_nir_dm),
        (p["max_no_J_common_NIRvO_final"]["resid_DM"], salt_optical_resid),
        (p["max_no_J_common_NIRvO_final"]["e_DM"], salt_e_optical_dm),
    ):
        x = g_masses[g_name]
        fit_x = np.linspace(xlim[0], xlim[1], 10)
        e_x = e_g_masses[g_name]
        mask = ~np.isnan(x)
        diff_err = np.sqrt(
            e_nir_y**2
            + e_no_nir_y**2
            - 2 * e_nir_y * e_no_nir_y * pg.corr(e_nir_y, e_no_nir_y)["r"][0]
        )
        for ax, y, e_y in zip(
            ax_triplet,
            (nir_y, no_nir_y, nir_y - no_nir_y),
            (e_nir_y, e_no_nir_y, diff_err),
        ):
            ax.set_facecolor(facecolor)
            print(g_name, len(x), len(y), len(e_x), len(e_y))
            ax.errorbar(
                x,
                y,
                xerr=e_x,
                yerr=e_y,
                ls="none",
                alpha=0.5,
                color=plotcolor,
            )
            odr_fit = utils.odr_polyfit(x[mask], y[mask], e_x[mask], e_y[mask], 1)
            fit_y = fit_x * odr_fit.beta[0] + odr_fit.beta[1]
            ax.axhline(0, color=linecolor)
            ax.plot(fit_x, fit_y, color=offcolor)
            sigma = utils.linear_model_uncertainty(fit_x, cov=odr_fit.cov_beta)
            ax.fill_between(
                fit_x, fit_y - sigma, fit_y + sigma, color=offcolor, alpha=0.5
            )
            print(g_name)
            odr_fit.pprint()
            print()

    ax1.set_xlim(xlim)
    ax1.set_ylim(-1.0911943823631272, 1.409853039476859)
    ax1.set_title("SNPY_Max")
    ax2.set_title("SALT_Max")
    ax1.set_ylabel(r"$\Delta \mu_{OJ}$ (mag)")
    ax3.set_ylabel(r"$\Delta \mu_{O}$ (mag)")
    ax5.set_ylabel(r"$\Delta \mu_{OJ} - \Delta \mu_{O}$ (mag)")
    # ax5.set_ylabel(r"$\Delta \Delta \mu$ (mag)")
    ax5.set_xlabel(r"$\log_{10}(M_\ast/M_\odot)$")
    ax6.set_xlabel(r"$\log_{10}(M_\ast/M_\odot)$")
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    fig.savefig(f"ado/mass_step_max.{file_ext}")
    if show:
        plt.show()
    return (
        fig,
        (
            g_masses,
            e_g_masses,
            salt_nir_resid,
            salt_e_nir_dm,
            salt_optical_resid,
            salt_e_optical_dm,
        ),
    )


def shape_standardization(p):
    import matplotlib.animation as animation

    fig, [ax1, ax2] = plt.subplots(ncols=2, sharey=True)
    ax1.set_facecolor("black")
    ax2.set_facecolor("black")
    color = "cyan"
    linecolor = "white"
    utils.dark_plot()
    init_x, e_x, e_y = p["x1"], p["e_x1"], p["e_DM"]
    shape_corr = constants.ALPHA["david-salt3-nir"] * p["x1"]
    color_corr = constants.BETA["david-salt3-nir"] * p["c"]
    init_y = p["resid_DM"] - shape_corr + color_corr

    ax1.axhline(0, color=linecolor)
    ax2.axhline(0, color=linecolor)
    ax1.set_ylabel(r"$\Delta \mu$ (mag)")
    ax1.set_xlabel(r"$x_1$ (Shape)")
    ax2.set_xlabel("N")
    plt.tight_layout()
    plt.subplots_adjust(wspace=0)

    gradient = np.linspace(0, 1, 101)

    def artists(i):
        """Update the scatter plot."""
        x = init_x
        y = init_y + shape_corr * gradient[i]

        # Set x and y data...
        scat = ax1.errorbar(
            x, y, xerr=e_x, yerr=e_y, color=color, alpha=0.75, ls="none"
        )
        hist = ax2.hist(y, color=color, orientation="horizontal")
        # Set sizes...
        # self.scat.set_sizes(300 * abs(data[:, 2]) ** 1.5 + 100)
        # Set colors..
        # self.scat.set_array(data[:, 3])

        # We need to return the updated artist for FuncAnimation to draw..
        # Note that it expects a sequence of artists, thus the trailing comma.
        return (scat[0], scat[2][0], scat[2][1], *hist[2].patches)

    artist_list = []
    for i in range(100):
        artist_list.append(artists(i))

    ani = animation.ArtistAnimation(
        fig,
        artist_list,
        interval=1,
        blit=True,
    )
    plt.show()


def nir_dispersion(show=False):
    fig, ax = plt.subplots()
    ax.scatter([21, 89, 47], [0.15, 0.12, 0.17], label="Template", color="green")

    ax.scatter(
        [165, 144, 37, 338], [0.19, 0.23, 0.17, 0.17], label="SNooPy", color="orange"
    )
    ax.scatter([246, 338], [0.17, 0.13], marker="*", color="orange")

    ax.scatter(
        [12, 116, 89, 354], [0.12, 0.15, 0.10, 0.20], label="Other Fitter", color="blue"
    )
    ax.scatter(
        [
            354,
        ],
        [
            0.15,
        ],
        marker="*",
        color="blue",
    )
    plt.legend()
    if show:
        plt.show()
    return (
        fig,
        (
            g_masses,
            e_g_masses,
            salt_nir_resid,
            salt_e_nir_dm,
            salt_optical_resid,
            salt_e_optical_dm,
        ),
    )


def shape_standardization(p):
    import matplotlib.animation as animation

    fig, [ax1, ax2] = plt.subplots(ncols=2, sharey=True)
    ax1.set_facecolor("black")
    ax2.set_facecolor("black")
    color = "cyan"
    linecolor = "white"
    utils.dark_plot()
    init_x, e_x, e_y = p["x1"], p["e_x1"], p["e_DM"]
    shape_corr = constants.ALPHA["david-salt3-nir"] * p["x1"]
    color_corr = constants.BETA["david-salt3-nir"] * p["c"]
    init_y = p["resid_DM"] - shape_corr + color_corr

    ax1.axhline(0, color=linecolor)
    ax2.axhline(0, color=linecolor)
    ax1.set_ylabel(r"$\Delta \mu$ (mag)")
    ax1.set_xlabel(r"$x_1$ (Shape)")
    ax2.set_xlabel("N")
    plt.tight_layout()
    plt.subplots_adjust(wspace=0)

    gradient = np.linspace(0, 1, 101)

    def artists(i):
        """Update the scatter plot."""
        x = init_x
        y = init_y + shape_corr * gradient[i]

        # Set x and y data...
        scat = ax1.errorbar(
            x, y, xerr=e_x, yerr=e_y, color=color, alpha=0.75, ls="none"
        )
        hist = ax2.hist(y, color=color, orientation="horizontal")
        # Set sizes...
        # self.scat.set_sizes(300 * abs(data[:, 2]) ** 1.5 + 100)
        # Set colors..
        # self.scat.set_array(data[:, 3])

        # We need to return the updated artist for FuncAnimation to draw..
        # Note that it expects a sequence of artists, thus the trailing comma.
        return (scat[0], scat[2][0], scat[2][1], *hist[2].patches)

    artist_list = []
    for i in range(100):
        artist_list.append(artists(i))

    ani = animation.ArtistAnimation(
        fig,
        artist_list,
        interval=1,
        blit=True,
    )
    plt.show()


def plot_prior_sensitivity_analysis_mixing_ratio(
    show=True,
    dm=None,
    ab_range=(-1, 2),
    length=30,
    alpha=3,
    beta=0.3,
    m=0.783,
    s=0.0184,
):
    import os
    import pickle

    path = f"{constants.HSF_DIR}/ado/background_sub_mixing_ratio.pickle"
    if os.path.exists(path):
        with open(path, "rb") as f:
            d = pickle.load(f)
    else:
        from snippets import prior_sensitivity_analysis_mixing_ratio

        d = prior_sensitivity_analysis_mixing_ratio(
            dm, ab_range=ab_range, length=length
        )
    fig = plt.figure(figsize=(6, 5))
    ab = np.logspace(ab_range[0], ab_range[1], length)

    plt.contourf(ab, ab, d["theta"])
    plt.text(3, 0.3, "*", color="red")
    plt.text(1, 1, "*", color="black")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"Prior $\alpha$")
    plt.ylabel(r"Prior $\beta$")
    cbar = plt.colorbar()
    cbar.set_label("Median Mixing Ratio Posterior")
    plt.tight_layout()
    if show:
        plt.show()
    fig.savefig(
        f"/data/users/ado/papers_git/HSF_survey_paper/figures/alpha_beta_sensitivity.{file_ext}"
    )


def plot_prior_sensitivity_analysis_inlier(
    show=True, key="mu_in", dm=None, mu_in_range=(-2, 0), sigma_range=(-1, 1), length=30
):
    import os
    import pickle

    path = f"{constants.HSF_DIR}/ado/background_sub_inlier.pickle"
    if os.path.exists(path):
        with open(path, "rb") as f:
            d = pickle.load(f)
    else:
        from snippets import prior_sensitivity_analysis_inlier

        d = prior_sensitivity_analysis_inlier(
            dm, mu_in_range=mu_in_range, sigma_range=sigma_range, length=length
        )
    mu_in = np.logspace(mu_in_range[0], mu_in_range[1], length)
    sigma = np.logspace(sigma_range[0], sigma_range[1], length)
    fig, ax = plt.subplots()
    masked_array = np.zeros((length, length))
    ave_rhat = np.average([d[param] for param in d if param.startswith("rhat")], axis=0)
    for i in range(30):
        for j in range(30):
            if ave_rhat[j, i] < 1.05:
                masked_array[j, i] = d[key][j, i]
            else:
                masked_array[j, i] = np.nan
    plt.contourf(mu_in, sigma, masked_array)
    plt.text(0.1, 2, "*", color="red", ha="center", va="center")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"Prior $\sigma_\mu$ (mag)")
    plt.ylabel(r"Prior $\sigma_\sigma$ (mag)")
    cbar = plt.colorbar()
    if key == "mu_in":
        cbar.set_label(r"Median Posterior $\mu_{in}$ (mag)")
    elif key == "sigma_in":
        cbar.set_label(r"Median Posterior $\sigma_{in}$ (mag)")
    plt.tight_layout()
    if show:
        plt.show()
    fig.savefig(
        f"/data/users/ado/papers_git/HSF_survey_paper/figures/sigma_sensitivity.{key}.{file_ext}"
    )


def plot_redlaw_chi2_comparison(
    names,
    chi2,
    chi_param="total_model",
    chi_N="ndof",
    redlaws=("O94", "F99", "F19"),
    chi_threshold=1,
    fig=None,
    axs=None,
    show=False,
):
    import math

    n_combos = math.comb(len(redlaws), 2)
    if isinstance(chi_param, str) and isinstance(chi_N, str):
        num_bps = 1
    elif (
        hasattr(chi_param, "__iter__")
        and hasattr(chi_N, "__iter__")
        and len(chi_param) == len(chi_N)
    ):
        num_bps = len(chi_param)
    if fig is None and axs is None:
        fig, axs = plt.subplots(
            nrows=num_bps, ncols=n_combos, figsize=(4 + 3 * n_combos, 5 + num_bps)
        )
    if num_bps > 1:
        for row, (cp, cn) in enumerate(zip(chi_param, chi_N)):
            plot_redlaw_chi2_comparison(
                names,
                chi2,
                chi_param=cp,
                chi_N=cn,
                redlaws=redlaws,
                chi_threshold=chi_threshold,
                fig=fig,
                axs=axs[row],
                show=False,
            )
        plt.tight_layout()
        if show:
            plt.show()
        return
    tmp_c = {}
    for redlaw in redlaws:
        tmp_c[redlaw] = chi2[redlaw][chi_param] / chi2[redlaw][chi_N]
    if n_combos > 1:
        ax = axs[0]
    else:
        ax = axs
    ax.scatter(
        (tmp_c[redlaws[0]] + tmp_c[redlaws[1]]) / 2,
        tmp_c[redlaws[1]] - tmp_c[redlaws[0]],
    )
    ax.axhline(0, alpha=0.2)
    # ax.plot(tmp_c[redlaws[0]], tmp_c[redlaws[0]], alpha=0.1)
    ax.set_xlabel(f"Ave {chi_param}")
    ax.set_ylabel(f"{redlaws[1]} - {redlaws[0]} {chi_param}")
    for i, (c1, c2) in enumerate(zip(tmp_c[redlaws[0]], tmp_c[redlaws[1]])):
        if np.abs(c1 - c2) > chi_threshold:
            ax.text((c1 + c2) / 2, c2 - c1, names[i])
    if n_combos == 3:
        axs[1].scatter(
            (tmp_c[redlaws[0]] + tmp_c[redlaws[1]]) / 2,
            tmp_c[redlaws[2]] - tmp_c[redlaws[0]],
        )
        axs[2].scatter(
            (tmp_c[redlaws[1]] + tmp_c[redlaws[2]]) / 2,
            tmp_c[redlaws[2]] - tmp_c[redlaws[1]],
        )
        axs[1].axhline(0, alpha=0.2)
        axs[2].axhline(0, alpha=0.2)
        # axs[1].plot(tmp_c[redlaws[0]], tmp_c[redlaws[0]], alpha=0.1)
        # axs[2].plot(tmp_c[redlaws[0]], tmp_c[redlaws[0]], alpha=0.1)
        axs[1].set_xlabel(f"Ave {chi_param}")
        axs[1].set_ylabel(f"{redlaws[2]} - {redlaws[0]} {chi_param}")
        axs[2].set_xlabel(f"Ave {chi_param}")
        axs[2].set_ylabel(f"{redlaws[2]} - {redlaws[1]} {chi_param}")
        for i, (c1, c2, c3) in enumerate(
            zip(tmp_c[redlaws[0]], tmp_c[redlaws[1]], tmp_c[redlaws[2]])
        ):
            if np.abs(c1 - c3) > chi_threshold:
                axs[1].text((c1 + c3) / 2, c3 - c1, names[i])
            if np.abs(c2 - c3) > chi_threshold:
                axs[2].text((c2 + c3) / 2, c3 - c2, names[i])
    plt.tight_layout()
    if show:
        plt.show()
