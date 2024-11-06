import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import stan
from astropy.coordinates import SkyCoord
from scipy.special import erf

import constants
import utils

# kessler 2009


def get_data_dict():
    with open(f"{constants.HSF_DIR}/ado/data_dict.pickle", "rb") as handle:
        fr_qsets, qsets, junk, stats, p = pickle.load(handle)
    return fr_qsets, qsets, junk, stats, p


def get_input_dict(sample):
    fr_qsets, qsets, junk, stats, p = get_data_dict()
    input_dict = {
        "n_obs": stats[sample]["N"],
        "z_hel": p[sample]["z_hel"],
        "z_cmb": p[sample]["z_cmb"],
        "ra": p[sample]["ra"],
        "dec": p[sample]["dec"],
        "resid_dm": p[sample]["resid_DM"],
        "e_dm": p[sample]["e_DM"],
        "speed_of_light": constants.C,
    }
    return input_dict


def stan_dipole(
    input_dict, iters=5000, jobs=7,
):
    input_dict["ra"] = np.array([ra if ra > 0 else ra + 360 for ra in input_dict["ra"]])
    input_dict["log_sigma_pv_mean"] = 5.5
    input_dict["log_sigma_pv_width"] = 2
    data = """
            // number of SNe
            int<lower=1> n_obs;
            // CMB Redshift
            vector<lower=0> [n_obs] z_cmb;
            // CMB Redshift error
            vector<lower=0> [n_obs] e_z_cmb;
            // Right Ascension (J2000)
            vector<lower=0, upper=360> [n_obs] ra;
            // Declination (J2000)
            vector<lower=-90, upper=90> [n_obs] dec;
            // Distance Modulus
            vector<lower=25, upper=40> [n_obs] dm;
            // Distance Modulus Error
            vector<lower=0> [n_obs] e_dm;
            // Deceleration constants
            real<lower=-1, upper=1> q0;
            // Hubble constants
            real<lower=60, upper=90> H0;
            // prior on sigma_pv
            real log_sigma_pv_mean ;
            real log_sigma_pv_width ;
            """
    transformed_data = """
            // Radian versions of SN coords
            vector<lower=0, upper=2*pi()> [n_obs] rad_ra ;
            vector<lower=-pi()/2, upper=pi()/2> [n_obs] rad_dec ;
            real speed_of_light ;
            speed_of_light = 299792.458 ;
            rad_ra = ra * pi() / 180 ;
            rad_dec = dec * pi() / 180 ;
            """
    params = """
            // latent parameters
            vector<lower=0.0005, upper=0.17> [n_obs] z_cos ;
            // fudge factor for M and H0
            real f ;
            // Unit Vector of Dipole in xyz coords
            unit_vector[3] dipole_xyz;
            // Velocity of dipole
            real<lower=0, upper=1500> v;
            // Dispersion on uncorrelated pv
            real<lower=0> sigma_pv;
            """
    transformed_params = """
            vector<lower=25, upper=42> [n_obs] expected_dm ;
            vector<lower=-0.005, upper=0.18> [n_obs] expected_z ;
            vector<lower=0, upper=pi()> [n_obs] separation;
            // Radian values for dipole coordinates
            real<lower=-pi(), upper=pi()> d_ra = atan2(dipole_xyz[2], dipole_xyz[1]);
            real<lower=-pi()/2, upper=pi()/2> d_dec = pi()/2 - acos(dipole_xyz[3]);
            for (n in 1:n_obs) {
                separation[n]= acos(
                    sin(rad_dec[n])*sin(d_dec)
                    +cos(rad_dec[n])*cos(d_dec)*cos(rad_ra[n] - d_ra)
                    ) ;
                // Expected Redshift
                expected_z[n] = (1+v/speed_of_light*cos(separation[n]))
                    *(1+z_cos[n]) - 1;
                // Expected distance modulus
                expected_dm[n] = f
                    + 5*log10(speed_of_light*z_cos[n]/H0*(1+(1-q0)/2*z_cos[n])) + 25;
                }
            """
    model = """
        dm ~ normal(expected_dm, e_dm);
        z_cmb ~ normal(expected_z,
            sqrt(square(e_z_cmb) + square(sigma_pv/speed_of_light*(1+expected_z)))
            );
        f ~ normal(0, 1);
        target += normal_lpdf(log(sigma_pv) | log_sigma_pv_mean, log_sigma_pv_width);
    """
    pickle_path = f"{constants.STATIC_DIR}/stan_dipole.pickle"
    fit = utils.stan(
        input_dict=input_dict,
        data=data,
        transformed_data=transformed_data,
        params=params,
        transformed_params=transformed_params,
        model=model,
        iters=iters,
        jobs=jobs,
        pickle_path=pickle_path,
    )
    summary_dict = fit.summary()
    df = pd.DataFrame(
        summary_dict["summary"],
        columns=summary_dict["summary_colnames"],
        index=summary_dict["summary_rownames"],
    )
    dft = df.transpose()
    return fit, df, dft


def stan_dipole_fast(
    input_dict, iters=5000, jobs=7,
):
    input_dict["ra"] = np.array([ra if ra > 0 else ra + 360 for ra in input_dict["ra"]])
    data = """
            // number of SNe
            int<lower=1> n_obs;
            // CMB Redshift
            vector<lower=0> [n_obs] z_cmb;
            // CMB Redshift error
            vector<lower=0> [n_obs] e_z_cmb;
            // Right Ascension (J2000)
            vector<lower=0, upper=360> [n_obs] ra;
            // Declination (J2000)
            vector<lower=-90, upper=90> [n_obs] dec;
            // Distance Modulus
            vector<lower=25, upper=40> [n_obs] dm;
            // Distance Modulus Error
            vector<lower=0> [n_obs] e_dm;
            // Deceleration constants
            real<lower=-1, upper=1> q0;
            // Hubble constants
            real<lower=60, upper=90> H0;
            // Dispersion on uncorrelated pv
            real<lower=0> sigma_pv;
            """
    transformed_data = """
            // Radian versions of SN coords
            vector<lower=0, upper=2*pi()> [n_obs] rad_ra ;
            vector<lower=-pi()/2, upper=pi()/2> [n_obs] rad_dec ;
            real speed_of_light ;
            speed_of_light = 299792.458 ;
            rad_ra = ra * pi() / 180 ;
            rad_dec = dec * pi() / 180 ;
            """
    params = """
            // latent parameters
            vector<lower=0.0005, upper=0.17> [n_obs] z_cos ;
            // fudge factor for M and H0
            real f ;
            // Unit Vector of Dipole in xyz coords
            unit_vector[3] dipole_xyz;
            // Velocity of dipole
            real<lower=0, upper=1500> v;
            """
    transformed_params = """
            vector<lower=25, upper=42> [n_obs] expected_dm ;
            vector<lower=-0.005, upper=0.18> [n_obs] expected_z ;
            vector<lower=0, upper=pi()> [n_obs] separation;
            // Radian values for dipole coordinates
            real<lower=-pi(), upper=pi()> d_ra = atan2(dipole_xyz[2], dipole_xyz[1]);
            real<lower=-pi()/2, upper=pi()/2> d_dec = pi()/2 - acos(dipole_xyz[3]);
            for (n in 1:n_obs) {
                separation[n]= acos(
                    sin(rad_dec[n])*sin(d_dec)
                    +cos(rad_dec[n])*cos(d_dec)*cos(rad_ra[n] - d_ra)
                    ) ;
                // Expected Redshift
                expected_z[n] = (1+v/speed_of_light*cos(separation[n]))
                    *(1+z_cos[n]) - 1;
                // Expected distance modulus
                expected_dm[n] = f
                    + 5*log10(speed_of_light*z_cos[n]/H0*(1+(1-q0)/2*z_cos[n])) + 25;
                }
            """
    model = """
        dm ~ normal(expected_dm, e_dm);
        z_cmb ~ normal(expected_z,
            sqrt(square(e_z_cmb) + square(sigma_pv/speed_of_light))
            );
        f ~ normal(0, 1);
    """
    pickle_path = f"{constants.STATIC_DIR}/stan_dipole_fast.pickle"
    fit = utils.stan(
        input_dict=input_dict,
        data=data,
        transformed_data=transformed_data,
        params=params,
        transformed_params=transformed_params,
        model=model,
        iters=iters,
        jobs=jobs,
        pickle_path=pickle_path,
    )
    summary_dict = fit.summary()
    df = pd.DataFrame(
        summary_dict["summary"],
        columns=summary_dict["summary_colnames"],
        index=summary_dict["summary_rownames"],
    )
    dft = df.transpose()
    return fit, df, dft


def mix_model_stan_dipole(
    input_dict, iters=5000, jobs=7,
):
    input_dict["ra"] = np.array([ra if ra > 0 else ra + 360 for ra in input_dict["ra"]])
    input_dict["outl_dm_uncertainty"] = 1
    # input_dict["outl_z_uncertainty"] = 0.05
    input_dict["outl_frac_exp_tau"] = 20
    input_dict["log_sigma_pv_mean"] = 5.5
    input_dict["log_sigma_pv_width"] = 2
    data = """
            // number of SNe
            int<lower=1> n_obs;
            // CMB Redshift
            vector<lower=0> [n_obs] z_cmb;
            // CMB Redshift error
            vector<lower=0> [n_obs] e_z_cmb;
            // Right Ascension (J2000)
            vector<lower=0, upper=360> [n_obs] ra;
            // Declination (J2000)
            vector<lower=-90, upper=90> [n_obs] dec;
            // Distance Modulus
            vector<lower=25, upper=40> [n_obs] dm;
            // Distance Modulus Error
            vector<lower=0> [n_obs] e_dm;
            // Deceleration constants
            real<lower=-1, upper=1> q0;
            // Hubble constants
            real<lower=60, upper=90> H0;
            // prior on sigma_pv
            real log_sigma_pv_mean ;
            real log_sigma_pv_width ;

            real outl_dm_uncertainty ; // Outlier distribution DM uncertainty.
            // real outl_z_uncertainty ; // Outlier distribution z uncertainty.
            real outl_frac_exp_tau ;  // outl frac Exponential prior
            """
    transformed_data = """
            // Radian versions of SN coords
            vector<lower=0, upper=2*pi()> [n_obs] rad_ra ;
            vector<lower=-pi()/2, upper=pi()/2> [n_obs] rad_dec ;
            real speed_of_light ;
            speed_of_light = 299792.458 ;
            rad_ra = ra * pi() / 180 ;
            rad_dec = dec * pi() / 180 ;
            """
    params = """
            // latent parameters
            vector<lower=0.001, upper=0.17> [n_obs] z_cos ;
            // fudge factor for M and H0
            real f ;
            // Unit Vector of Dipole in xyz coords
            unit_vector[3] dipole_xyz;
            // Velocity of dipole
            real<lower=0, upper=1500> v;
            // Dispersion on uncorrelated peculiar velocity
            real<lower=0, upper=1500> sigma_pv;
            // Mixture model outlier probability
            real<lower=0, upper=1> outl_frac;
            """
    transformed_params = """
            vector<lower=25, upper=42> [n_obs] expected_dm ;
            vector<lower=-0.005, upper=0.18> [n_obs] expected_z ;
            vector<lower=0, upper=pi()> [n_obs] separation;
            // Radian values for dipole coordinates
            real<lower=-pi(), upper=pi()> d_ra = atan2(dipole_xyz[2], dipole_xyz[1]);
            real<lower=-pi()/2, upper=pi()/2> d_dec = pi()/2 - acos(dipole_xyz[3]);
            separation = acos(
                sin(rad_dec)*sin(d_dec)
                +cos(rad_dec)*cos(d_dec) .* cos(rad_ra - d_ra)
                );
            expected_z = (1+v/speed_of_light*cos(separation)) .* (1+z_cos) -1;
            expected_dm = f + 5*log10(speed_of_light/H0)
                + 5*log10(z_cos + (1-q0)/2*square(z_cos))+25;
            """
    model = """
        for (n in 1:n_obs)
            target += log_mix(outl_frac,
                normal_lpdf(dm[n] | expected_dm[n], outl_dm_uncertainty)
                + normal_lpdf(z_cmb[n] | expected_z[n], hypot(e_z_cmb[n], sigma_pv/speed_of_light*(1+expected_z[n]))),
                normal_lpdf(dm[n] | expected_dm[n], e_dm[n])
                + normal_lpdf(z_cmb[n] | expected_z[n], hypot(e_z_cmb[n], sigma_pv/speed_of_light*(1+expected_z[n])))
                );
        outl_frac ~ exponential(outl_frac_exp_tau);
        f ~ normal(0, 1);
        target += normal_lpdf(log(sigma_pv) | log_sigma_pv_mean, log_sigma_pv_width);
    """
    pickle_path = f"{constants.STATIC_DIR}/mix_model_stan_dipole.pickle"
    fit = utils.stan(
        input_dict=input_dict,
        data=data,
        transformed_data=transformed_data,
        params=params,
        transformed_params=transformed_params,
        model=model,
        iters=iters,
        jobs=jobs,
        pickle_path=pickle_path,
    )
    summary_dict = fit.summary()
    df = pd.DataFrame(
        summary_dict["summary"],
        columns=summary_dict["summary_colnames"],
        index=summary_dict["summary_rownames"],
    )
    dft = df.transpose()
    return fit, df, dft


def general_window(k, R, n_samples=200, res=None):
    if n_samples is None and res is None:
        return TypeError("res and n_samples cannot both be None")
    if n_samples is None:
        n_samples = 2 * int(R / res) + 1
    x, y, z = np.meshgrid(*[np.linspace(-R, R, n_samples) for i in range(3)])
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    weights = np.nan_to_num(
        1
        # / r ** 2  # generic weighting
        * (z / r < np.sin(60 * np.pi / 180))  # northern dec limit
        * (z / r > np.sin(-40 * np.pi / 180))  # southern dec limit
        * (r <= R)  # range limit
    )
    FW = np.zeros((len(k), len(k), len(k)))
    # very slow
    for x_idx, k_x in enumerate(k):
        for y_idx, k_y in enumerate(k):
            for z_idx, k_z in enumerate(k):
                FW[y_idx, x_idx, z_idx] = np.sum(
                    weights * np.exp(-complex(0, 1) * (k_x * x + k_y * y + k_z * z))
                )
    FW /= np.sum(weights)
    return FW


def axisymm_window(k, R, dec_min=None, dec_max=None, n_samples=201):
    # exploit axisymmetry to reframe problem as rad and z.
    if k[0] != 0:
        k = np.append(0, np.array(k))
    x, y, z = np.meshgrid(*[np.linspace(-R, R, n_samples) for i in range(3)])
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    weight_cube = np.nan_to_num(
        1
        # / r ** 2  # generic weighting
        * (r <= R)  # range limit
    )
    if dec_min is not None:
        weight_cube *= z / r > np.sin(dec_min * np.pi / 180)
    if dec_max is not None:
        weight_cube *= z / r < np.sin(dec_max * np.pi / 180)
    weights = np.sum(weight_cube, axis=2)
    rad, z = np.meshgrid(*[np.linspace(-R, R, n_samples) for i in range(2)])
    FW = np.zeros((len(k), len(k),), dtype=complex)
    for rad_idx, k_rad in enumerate(k):
        for z_idx, k_z in enumerate(k):
            FW[rad_idx, z_idx] = np.sum(
                weights * np.exp(-complex(0, 1) * (k_rad * rad + k_z * z))
            )
    FW /= np.sum(weights)
    return FW


def tophat_window(k, R, **kwargs):
    return 3 * (np.sin(k * R) - k * R * np.cos(k * R)) / (k * R) ** 3


def gauss_window(k, R, **kwargs):
    return np.exp(-(k ** 2) * R ** 2 / 2)


def vR_full_symm(k, pk, window_fn, R, H=67.3, om=0.316):
    Rh = 100 * R / H
    return np.sqrt(
        (H * (om ** (4 / 7) + (1 + om / 2) * (1 - om) / 70)) ** 2
        / (2 * np.pi ** 2)
        * np.trapz(pk * np.abs(window_fn(k, Rh)) ** 2, k)
    )


def vR_axisymm(
    k, pk, window_fn, R, H=67.3, om=0.316, n_samples=250, dec_min=None, dec_max=None
):
    RH = 100 * R / H
    if k[0] != 0:
        k = np.append(0, np.array(k))
        pk = np.append(0, np.array(pk))
    FW = window_fn(k, RH, n_samples=n_samples, dec_min=dec_min, dec_max=dec_max)
    p_grid, k_grid, rad_grid = [np.zeros(FW.shape) for i in range(3)]
    for a, k1 in enumerate(k):
        for b, k2 in enumerate(k):
            k_val = np.sqrt(k1 ** 2 + k2 ** 2)
            k_grid[a, b] = k_val
            p_grid[a, b] = np.interp(k_val, k, pk)
            rad_grid[a, b] = k1
    const_numerator = (H * (om ** (4 / 7) + (1 + om / 2) * (1 - om) / 70)) ** 2
    const_denom = (2 * np.pi) ** 2
    integrand = np.nan_to_num(np.abs(FW) ** 2 * rad_grid * p_grid / k_grid ** 2)

    return np.sqrt(
        2  # to account for negative k_z values
        * const_numerator
        / const_denom
        * np.trapz(np.trapz(integrand, k), k)
    )


def vR_general(k, pk, window_fn, R, H=67.3, om=0.316, n_samples=100):
    RH = 100 * R / H
    if k[0] != 0:
        k = np.append(0, np.array(k))
        pk = np.append(0, np.array(pk))
    FW = window_fn(k, RH, n_samples=n_samples)
    k_cube, p_cube = [np.zeros(FW.shape) for i in range(2)]
    for a, k1 in enumerate(k):
        for b, k2 in enumerate(k):
            for c, k3 in enumerate(k):
                k_val = np.sqrt(k1 ** 2 + k2 ** 2 + k3 ** 2)
                k_cube[a, b, c] = k_val
                p_cube[a, b, c] = np.interp(k_val, k, pk)
    const_numerator = (H * (om ** (4 / 7) + (1 + om / 2) * (1 - om) / 70)) ** 2
    const_denom = (2 * np.pi) ** 3
    integrand = np.nan_to_num(np.abs(FW) ** 2 * p_cube / k_cube ** 2)
    return np.sqrt(
        8  # to account for all octants
        * const_numerator
        / const_denom
        * np.trapz(np.trapz(np.trapz(integrand, k), k), k)
    )


def get_pk(file_path=f"{constants.HSF_DIR}/ado/powerspectrum.csv"):
    k, pk = np.array(pd.read_csv(file_path, delim_whitespace=True, header=None))
    return k, pk


def initialize():
    k, pk = get_pk()
    Rs = np.logspace(np.log10(15), np.log10(640), 20)
    k_res = (1, 2, 4, 8, 16, 32, 64, 128)
    tophat, axisymm = [np.zeros((len(k_res), len(Rs))) for i in range(2)]
    for i, kr in enumerate(k_res):
        for j, r in enumerate(Rs):
            tophat[i, j] = vR_full_symm(k[::kr], pk[::kr], tophat_window, r)
    return k, Rs, k_res, tophat, axisymm


def plot_k_res_effects():
    k, Rs, k_res, tophat, axisymm = initialize()
    for i, line in enumerate(tophat[:0:-1]):
        plt.plot(Rs, line - tophat[0], label=np.floor(len(k) / k_res[-(i + 1)]))
    plt.yscale("log")
    plt.xscale("log")
    plt.legend()
    plt.ylabel(r"$\sigma_v(R,N_k) - \sigma_v(R,2000)$")
    plt.xlabel(r"$R$ $h^{-1}$ Mpc")
    plt.xlim(np.sqrt(10), 1000)
    plt.tight_layout()
    plt.show()


def plot_sim(H0=72, show=False):
    (x, y, z), (vx, vy, vz), mu_true, mu_obs, pv_true, z_cos, z_obs = fake_data(H0=H0)
    davis = (
        constants.C
        * (z_obs - utils.z_lcdm(H0, mu_obs))
        / (1 + utils.z_lcdm(H0, mu_obs))
    )
    kessler = (
        -constants.C
        * np.log(10)
        / 5
        * (z_obs * (1 + z_obs / 2))
        / (1 + z_obs)
        * (mu_obs - utils.mu_lcdm(z_obs, z_obs, H0=H0))
    )
    fig, [ax1, ax2, ax3] = plt.subplots(ncols=3, figsize=(12, 5))
    ax1.scatter(pv_true, davis, alpha=0.5, marker=".")
    ax1.plot((min(pv_true), max(pv_true)), (min(pv_true), max(pv_true)), color="black")
    ax2.scatter(pv_true, kessler, alpha=0.5, marker=".")
    ax2.plot((min(pv_true), max(pv_true)), (min(pv_true), max(pv_true)), color="black")
    ax3.scatter(davis, kessler, alpha=0.5, marker=".")
    ax3.plot((min(davis), max(davis)), (min(davis), max(davis)), color="black")
    ax1.set_xlabel("Simulated PV [km/s]")
    ax1.set_ylabel("Davis+14 inferred PV [km/s]")
    ax2.set_xlabel("Simulated PV [km/s]")
    ax3.set_ylabel("Kessler+09 inferred PV [km/s]")
    ax3.set_xlabel("Davis+14 inferred PV [km/s]")
    ax3.set_ylabel("Kessler+09 inferred PV [km/s]")
    plt.tight_layout()
    if show:
        plt.show()


def get_pv_from_mu_z(mu, z, H0=72, q0=-0.53):
    ln10 = np.log(10)
    return (
        ln10
        / 5
        * (mu - 5 * np.log10(constants.C * z / H0 * (1 + (1 - q0) / 2 * z)) - 25)
        / ((1 + q0) / 2 - 1 / z - 2 - (1 - q0 ** 2) / 4 * z)
    )


def lin_theory_cov_mat(x, y, z, k=None, pk=None, H0=67.3, f=0.316 ** 0.55):
    from tqdm import tqdm

    if k is None or pk is None:
        k, pk = get_pk()
    N = len(x)
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    ra = np.arctan2(y, x)
    dec = np.arcsin(z / r)
    sc = SkyCoord(ra, dec, unit="radian")
    Gmn = np.zeros((N, N))

    def j0(kA):
        return np.sin(kA) / kA

    def j2(kA):
        return (3 / kA ** 2 - 1) * np.sin(kA) / kA - 3 * np.cos(kA) / kA ** 2

    for m in tqdm(range(N), position=0):
        for n in tqdm(range(N), position=1):
            if n < m:
                Gmn[m, n] = Gmn[n, m]
                continue
            alpha = sc[m].separation(sc[n]).value
            A = np.sqrt(r[m] ** 2 + r[n] ** 2 - 2 * r[m] * r[n] * np.cos(alpha))
            Wmn = (
                1 / 3 * np.cos(alpha) * (j0(k * A) - 2 * j2(k * A))
                + 1 / A ** 2 * j2(k * A) * r[m] * r[n] * np.sin(alpha) ** 2
            )
            Gmn[m, n] = H0 ** 2 * f ** 2 / (2 * np.pi ** 2) * np.trapz(k, pk * Wmn)
    return Gmn


def random_pvs(x, y, z, Gmn, pv_var, pad=5):
    bf_ra = np.random.uniform(0, 2 * np.pi)
    bf_dec = np.arcsin(np.random.uniform(-1, 1))
    bf_vel = np.abs(np.random.normal(0, pad * np.sqrt(pv_var)))
    dipole = SkyCoord(bf_ra, bf_dec, unit="radian")
    sc = SkyCoord(
        np.arctan2(y, x),
        np.arcsin(z / np.sqrt(x ** 2 + y ** 2 + z ** 2)),
        unit="radian",
    )
    for i in range(len(Gmn)):
        Gmn[i, i] = 1
    rad_pvs = np.random.multivariate_normal(np.zeros(len(Gmn)), Gmn)
    rad_pvs += bf_vel * np.cos(dipole.separation(sc).value * np.pi / 180)
    return bf_ra, bf_dec, bf_vel, rad_pvs


def obs_errors(p, z_cos):
    mu_err, z_err = [np.zeros(len(z_cos)) for i in range(2)]
    e_z = np.nan_to_num(p["e_z"], nan=0.00001)
    for i, z in enumerate(z_cos):
        idx = np.argsort(np.abs(p["z_cmb"] - z))
        mu_err[i] = np.median(p["e_DM"][idx[:10]])
        z_err[i] = np.median(e_z[idx[:10]])
    return mu_err, z_err


def fake_data_2(
    N=5000,
    R=500,
    H0=67.3,
    Om=0.316,
    Gmn_path=f"{constants.HSF_DIR}/ado/bf_mock.pickle",
    pv_var=None,
    k=None,
    pk=None,
    p=None,
):
    if Gmn_path is None:
        x, y, z = np.random.uniform(-R, R, size=(3, N * 10))
        d_true = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        idx = np.where(d_true < R)[0][:N]
        # dec = np.arcsin(z/d_true)*180/np.pi
        # idx = np.where((d_true < R) & (dec > -40) & (dec < 60))[0][:N]
        x, y, z, d_true = x[idx], y[idx], z[idx], d_true[idx]

        mu_true = 5 * np.log10(d_true) + 25
        z_cos = utils.z_lcdm(H0, mu_true)
        Gmn = lin_theory_cov_mat(x, y, z, H0=H0, f=Om ** 0.55)
    else:
        with open(Gmn_path, "rb") as f:
            x, y, z, Gmn, mu_true, z_cos = pickle.load(f)
    if k is None or pk is None:
        k, pk = get_pk()
    if pv_var is None:
        pv_var = vR_full_symm(k, pk, tophat_window, R, H0, Om) ** 2
    bf_ra, bf_dec, bf_vel, rad_pvs = random_pvs(x, y, z, Gmn, pv_var)

    if p is None:
        from survey_paper import get_data_dict

        p = get_data_dict()[-1]["ebv_final"]
    mu_err, z_err = obs_errors(p, z_cos)
    mu_obs, z_obs = [np.zeros(N) for i in range(2)]
    z_true = (1 + z_cos) * (1 + rad_pvs / constants.C) - 1
    for i in range(N):
        mu_obs[i] = mu_true[i] + np.random.normal(0, mu_err[i])
        z_obs[i] = z_true[i] + np.random.normal(0, z_err[i])
    return (
        x,
        y,
        z,
        rad_pvs,
        bf_ra,
        bf_dec,
        bf_vel,
        mu_true,
        mu_obs,
        mu_err,
        z_cos,
        z_true,
        z_obs,
        z_err,
    )


def get_vel_intervals(x, vR):
    # x is the fraction of velocities above a certain threshld.
    # i.e. 1 sigma interval around max would be x=0.6827
    velocities = np.linspace(0, 5 * vR, 10000)
    prob_v = (
        np.sqrt(2 / np.pi)
        * (3 / vR ** 2) ** (3 / 2)
        * velocities ** 2
        * np.exp(-3 * velocities ** 2 / (2 * vR ** 2))
    )
    prob_sort = np.sort(prob_v)[::-1]
    total = np.trapz(prob_v, velocities)
    for prob in prob_sort[1:]:
        idx = np.where(prob_v > prob)
        prob_mass = np.trapz(prob_v[idx], velocities[idx])
        if prob_mass / total > x:
            low_v = velocities[idx][0]
            high_v = velocities[idx][-1]
            break
    return low_v, high_v


def get_two_tailed_prob(v, vR):
    from scipy.special import erf

    prob_v = prob_v_sig_v(v, vR)
    mass = cdf_v_sig_v(v, vR)
    if v < np.sqrt(2 / 3) * vR:
        test_v = np.linspace(np.sqrt(2 / 3) * vR, 10000, 10000)
    else:
        test_v = np.linspace(0, np.sqrt(2 / 3) * vR, 10000)
    other_prob_v = prob_v(test_v, vR)
    other_v = test_v[np.argmin(np.abs(other_prob_v - prob_v))]
    other_mass = cdf_v_sig_v(other_v, vR)
    if v < np.sqrt(2 / 3) * vR:
        return mass + 1 - other_mass
    else:
        return 1 - mass + other_mass


def plot_lcdm_prediction():
    fig, ax = plt.subplots(figsize=(7, 5))
    for pickle_path, color in zip(
        (
            f"{constants.HSF_DIR}/ado/R_vel_disp.pickle",
        ),  # "ado/R_vel_disp_fullsymm.pickle"),
        ("red",),  # "blue")
    ):
        with open(pickle_path, "rb") as f:
            Rs, vRs = pickle.load(f)
        ax.plot(Rs, np.sqrt(2 / 3) * vRs, color=color)
        for prob_mass in (0.6827, 0.9545, 0.9973):
            low_v, high_v = [np.zeros(len(Rs)) for i in range(2)]
            for i, v in enumerate(vRs):
                low_v[i], high_v[i] = get_vel_intervals(prob_mass, v)
            ax.fill_between(Rs, low_v, high_v, alpha=0.25, color=color)
    ax.set_xscale("log")
    ax.set_xlim(min(Rs), max(Rs))
    ax.set_ylim(0, max(vRs) * 2)
    ax.set_xlabel(r"$R~h^{-1}$ Mpc")
    ax.set_ylabel(r"Bulk Flow Speed km s$^{-1}$")
    plt.tight_layout()
    fig.savefig(f"{constants.HSF_DIR}/ado/lcdm_pv_prediction.pdf")
    plt.show()


def make_whitford_catalog(
    mu_obs, z_obs, mu_err=0.15, z_err=0.00001, x=None, y=None, z=None, ra=None, dec=None
):
    if x is None and ra is None:
        raise ValueError("Provide either x, y, and z, or ra and dec")
    elif x is not None and y is not None and z is not None:
        ra = np.arctan2(y, x) * 180 / np.pi
        dec = np.arcsin(z / np.sqrt(x ** 2 + y ** 2 + z ** 2)) * 180 / np.pi
    if min(ra) < 0:
        for i in range(len(ra)):
            if ra[i] < 0:
                ra[i] += 360
    if not hasattr(mu_err, "__iter__") and mu_err is not None:
        mu_err = np.ones(len(mu_obs)) * mu_err
    if not hasattr(z_err, "__iter__") and z_err is not None:
        z_err = np.ones(len(z_obs)) * z_err
    with open("/home/ado/Measuring_bulkflows/example_surveymock.dat", "w") as f:
        for i in range(len(mu_obs)):
            f.write(
                f"{ra[i]} {dec[i]} {z_obs[i]} {utils.get_log_dist(mu_obs[i], z_obs[i])} {utils.get_log_dist_err(mu_obs[i], mu_err[i], z_obs[i], z_err[i])} {1e-5}\n"
            )


def parse_whitford_results(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
    ones = np.array([1, 1, 1])
    peery_vels = np.array(lines[0].split(), dtype=float)
    peery_vels[2] *= -1
    peery_cmat = np.array(
        [np.array(lines[i].split(), dtype=float) for i in range(2, 5)]
    )
    peery_vels = np.append(peery_vels, np.sqrt(np.sum(peery_vels ** 2)))
    nusser_vels = np.array(lines[6].split(), dtype=float)
    nusser_vels[2] *= -1
    nusser_cmat = np.array(
        [np.array(lines[i + 6].split(), dtype=float) for i in range(2, 5)]
    )
    nusser_vels = np.append(nusser_vels, np.sqrt(np.sum(nusser_vels ** 2)))
    peery_total_err = np.sqrt(np.matmul(np.matmul(ones, peery_cmat), ones.T))
    peery_err = np.append(np.sqrt(np.diag(peery_cmat)), peery_total_err)
    nusser_total_err = np.sqrt(np.matmul(np.matmul(ones, nusser_cmat), ones.T))
    nusser_err = np.append(np.sqrt(np.diag(nusser_cmat)), nusser_total_err)
    return peery_vels, peery_err, nusser_vels, nusser_err


def parse_bayesian_results(file_path):
    with open(file_path, "rb") as f:
        fit = pickle.load(f)[0]
    bx = fit["v"] * np.cos(fit["d_ra"]) * np.cos(fit["d_dec"])
    by = fit["v"] * np.sin(fit["d_ra"]) * np.cos(fit["d_dec"])
    bz = fit["v"] * np.sin(fit["d_dec"])
    return (
        np.append(np.median([bx, by, bz], axis=1), np.median(fit["v"])),
        np.append(np.std([bx, by, bz], axis=1), np.std(fit["v"])),
    )


def compare_mocks(
    true_d=None,
    my_d=None,
    p_d=None,
    n_d=None,
    pickle_path=None,
    N=500,
    figsize=(10, 8),
    show=False,
):
    import glob

    if pickle_path is not None:
        with open(pickle_path, "rb") as f:
            true_d, my_d, p_d, n_d, odr = pickle.load(f)
    if true_d is None:
        truth = pd.read_csv(
            f"{constants.HSF_DIR}/ado/bf_mocks/N{N}.truths.txt",
            delim_whitespace=True,
            header=None,
            names=["ra", "dec", "vel"],
        )
        true_d = {}
        true_d["x"] = truth["vel"] * np.cos(truth["ra"]) * np.cos(truth["dec"])
        true_d["y"] = truth["vel"] * np.sin(truth["ra"]) * np.cos(truth["dec"])
        true_d["z"] = truth["vel"] * np.sin(truth["dec"])
        true_d["vel"] = truth["vel"]

    if my_d is None:
        my_d = {}
    if n_d is None:
        n_d = {}
    if p_d is None:
        p_d = {}
    for d in (my_d, n_d, p_d):
        if not len(d):
            for key in ("x", "y", "z", "vel"):
                d[key] = np.zeros(len(true_d["x"]))
                d["e_" + key] = np.zeros(len(true_d["x"]))

    if not sum(my_d["vel"]):
        for path in glob.glob(f"{constants.HSF_DIR}/ado/bf_mocks/fit_pickles/N{N}*"):
            if "ebv" in path or "max" in path or "salt" in path:
                continue
            i = int(path.split(".")[1])
            (
                (my_d["x"][i], my_d["y"][i], my_d["z"][i], my_d["vel"][i]),
                (my_d["e_x"][i], my_d["e_y"][i], my_d["e_z"][i], my_d["e_vel"][i]),
            ) = parse_bayesian_results(path)
    if not sum(p_d["vel"]) or not sum(n_d["vel"]):
        for path in glob.glob(
            f"{constants.HSF_DIR}/ado/bf_mocks/whitford_results/N{N}*"
        ):
            i = int(path.split(".")[1])
            (
                (p_d["x"][i], p_d["y"][i], p_d["z"][i], p_d["vel"][i]),
                (p_d["e_x"][i], p_d["e_y"][i], p_d["e_z"][i], p_d["e_vel"][i]),
                (n_d["x"][i], n_d["y"][i], n_d["z"][i], n_d["vel"][i]),
                (n_d["e_x"][i], n_d["e_y"][i], n_d["e_z"][i], n_d["e_vel"][i]),
            ) = parse_whitford_results(path)
    fig, [n_axes, p_axes, my_axes] = plt.subplots(
        nrows=3,
        ncols=5,
        figsize=figsize,
        gridspec_kw={"width_ratios": [3, 3, 3, 2, 3]},
    )
    odr = {"This Work": [], "Peery MVE": [], "Nusser MLE": []}
    for axes, d, estimator in zip(
        (my_axes, p_axes, n_axes),
        (my_d, p_d, n_d),
        ("This Work", "Peery MVE", "Nusser MLE"),
    ):
        odr[estimator] = {}
        for key, ax in zip(
            ("x", "y", "z", "vel"), (axes[0], axes[1], axes[2], axes[4])
        ):
            ax.plot(true_d[key], true_d[key], color="black")
            ax.errorbar(
                true_d[key],
                d[key],
                yerr=d["e_" + key],
                # color="red",
                alpha=0.5,
                ls="none",
            )
            odr[estimator][key] = utils.odr_polyfit(
                true_d[key], d[key], np.ones(len(true_d[key])) * 1e-5, d["e_" + key], 1
            )
            xfit = np.linspace(min(true_d[key]), max(true_d[key]), 100)
            yfit = odr[estimator][key].beta[0] * xfit + odr[estimator][key].beta[1]
            ax.plot(xfit, yfit, color="red")
            sig = utils.linear_model_uncertainty(xfit, cov=odr[estimator][key].cov_beta)
            # ax.fill_between(
            #     xfit, yfit - 2 * sig, yfit + 2 * sig, color="red", alpha=0.2
            # )
            ax.fill_between(xfit, yfit - sig, yfit + sig, color="red", alpha=0.5)
            ax.set_xlim(-710, 710)
            ax.set_ylim(-810, 810)
            if key == "vel":
                ax.set_xlim(0, 750)
                ax.set_ylim(-250, 1100)
            if key not in ("x", "vel"):
                ax.set_yticklabels(["" for _ in ax.get_yticks()])
            if estimator != "This Work":
                ax.set_xticklabels(["" for _ in ax.get_xticks()])
        axes[0].set_ylabel(estimator + "\n" + r"Measured $B_i$ (km s$^{-1}$)")
        axes[3].axis("off")
        # axes[4].set_ylabel(r"Measured |B| (km s$^{-1}$)")
    my_axes[0].set_xlabel(r"True $B_x$ (km s$^{-1}$)")
    my_axes[1].set_xlabel(r"True $B_y$ (km s$^{-1}$)")
    my_axes[2].set_xlabel(r"True $B_z$ (km s$^{-1}$)")
    my_axes[4].set_xlabel(r"True $|B|$ (km s$^{-1}$)")
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    if show:
        plt.show()
    return fig, (true_d, my_d, p_d, n_d, odr)


def prob_v_sig_v(v, sig_v):
    return (
        np.sqrt(2 / np.pi)
        * (3 / sig_v ** 2) ** 1.5
        * v ** 2
        * np.exp(-3 * v ** 2 / (2 * sig_v ** 2))
    )


def cdf_v_sig_v(v, sig_v):
    return (
        1
        / (np.sqrt(6 * np.pi) * sig_v)
        * (
            np.sqrt(6 * np.pi) * sig_v * erf(np.sqrt(3 / 2) * v / sig_v)
            - 6 * v * np.exp(-3 * v ** 2 / (2 * sig_v ** 2))
        )
    )


def get_p_value(sig_v, mu=None, err=None, posterior=None):
    all_v = np.linspace(0, 10000, 10001)
    if mu is not None and err is not None:
        p = np.trapz(
            0.5
            * (1 + erf((all_v - mu) / (err * np.sqrt(2))))
            * prob_v_sig_v(all_v, sig_v),
            all_v,
        )
    else:
        p = np.average(cdf_v_sig_v(posterior, sig_v))
    return 2 * min(p, 1 - p)


def old_mix_model_stan_dipole(
    input_dict, iters=5000, jobs=7,
):
    input_dict["ra"] = np.array([ra if ra > 0 else ra + 360 for ra in input_dict["ra"]])
    input_dict["outl_dm_uncertainty"] = 1
    # input_dict["outl_z_uncertainty"] = 0.05
    input_dict["outl_frac_exp_tau"] = 20
    input_dict["log_sigma_pv_mean"] = 5.5
    input_dict["log_sigma_pv_width"] = 2
    data = """
            // number of SNe
            int<lower=1> n_obs;
            // CMB Redshift
            vector<lower=0> [n_obs] z_cmb;
            // CMB Redshift error
            vector<lower=0> [n_obs] e_z_cmb;
            // Right Ascension (J2000)
            vector<lower=0, upper=360> [n_obs] ra;
            // Declination (J2000)
            vector<lower=-90, upper=90> [n_obs] dec;
            // Distance Modulus
            vector<lower=25, upper=40> [n_obs] dm;
            // Distance Modulus Error
            vector<lower=0> [n_obs] e_dm;
            // Deceleration constants
            real<lower=-1, upper=1> q0;
            // Hubble constants
            real<lower=60, upper=90> H0;
            // prior on sigma_pv
            real log_sigma_pv_mean ;
            real log_sigma_pv_width ;

            real outl_dm_uncertainty ; // Outlier distribution DM uncertainty.
            // real outl_z_uncertainty ; // Outlier distribution z uncertainty.
            real outl_frac_exp_tau ;  // outl frac Exponential prior
            """
    transformed_data = """
            // Radian versions of SN coords
            vector<lower=0, upper=2*pi()> [n_obs] rad_ra ;
            vector<lower=-pi()/2, upper=pi()/2> [n_obs] rad_dec ;
            real speed_of_light ;
            speed_of_light = 299792.458 ;
            rad_ra = ra * pi() / 180 ;
            rad_dec = dec * pi() / 180 ;
            """
    params = """
            // latent parameters
            vector<lower=0.0005, upper=0.17> [n_obs] z_cos ;
            // fudge factor for M and H0
            real f ;
            // Unit Vector of Dipole in xyz coords
            unit_vector[3] dipole_xyz;
            // Velocity of dipole
            real<lower=0, upper=1500> v;
            // Dispersion on uncorrelated peculiar velocity
            real<lower=0> sigma_pv;
            // Mixture model outlier probability
            real<lower=0, upper=1> outl_frac;
            """
    transformed_params = """
            vector<lower=25, upper=42> [n_obs] expected_dm ;
            vector<lower=-0.005, upper=0.18> [n_obs] expected_z ;
            vector [n_obs] outl_loglike;
            vector [n_obs] inl_loglike;
            vector<lower=0, upper=pi()> [n_obs] separation;
            // Radian values for dipole coordinates
            real<lower=-pi(), upper=pi()> d_ra = atan2(dipole_xyz[2], dipole_xyz[1]);
            real<lower=-pi()/2, upper=pi()/2> d_dec = pi()/2 - acos(dipole_xyz[3]);
            for (n in 1:n_obs) {
                separation[n]= acos(
                    sin(rad_dec[n])*sin(d_dec)
                    +cos(rad_dec[n])*cos(d_dec)*cos(rad_ra[n] - d_ra)
                    ) ;
                // Expected Redshift
                expected_z[n] = (1+v/speed_of_light*cos(separation[n]))
                    *(1+z_cos[n]) - 1;
                // Expected distance modulus
                expected_dm[n] = f
                    + 5*log10(speed_of_light*z_cos[n]/H0*(1+(1-q0)/2*z_cos[n])) + 25;
                inl_loglike[n] = log(1-outl_frac) + normal_lpdf(dm[n] | expected_dm[n], e_dm[n]) + normal_lpdf(z_cmb[n] | expected_z[n], e_z_cmb[n]);
                outl_loglike[n] = log(outl_frac) + normal_lpdf(dm[n] | expected_dm[n], outl_dm_uncertainty) + normal_lpdf(z_cmb[n] | expected_z[n], e_z_cmb[n]);
                }
            """
    model = """
        for (n in 1:n_obs)
            target += log_sum_exp(outl_loglike[n], inl_loglike[n]);
        outl_frac ~ exponential(outl_frac_exp_tau);
        f ~ normal(0, 1);
        target += normal_lpdf(log(sigma_pv) | log_sigma_pv_mean, log_sigma_pv_width);
    """
    pickle_path = f"{constants.STATIC_DIR}/mix_model_stan_dipole.pickle"
    fit = utils.stan(
        input_dict=input_dict,
        data=data,
        transformed_data=transformed_data,
        params=params,
        transformed_params=transformed_params,
        model=model,
        iters=iters,
        jobs=jobs,
        pickle_path=pickle_path,
    )
    summary_dict = fit.summary()
    df = pd.DataFrame(
        summary_dict["summary"],
        columns=summary_dict["summary_colnames"],
        index=summary_dict["summary_rownames"],
    )
    dft = df.transpose()
    return fit, df, dft


def old_fake_data(
    N=5000, R=500, sigma_v=300, sigma_mu=0.15, sigma_z=0.00001, H0=72, bulk_flow_fn=None
):
    # ra = 360 * np.random.uniform(size=N)
    # dec = np.arcsin(np.random.uniform(size=2 * N) * 2 - 1) * 180 / np.pi
    # dec = dec[np.where((dec > dec_min) & (dec < dec_max))][:N]
    x, y, z = np.random.uniform(-R, R, size=(3, N))
    vx, vy, vz = np.random.normal(0, sigma_v, size=(3, N))
    if bulk_flow_fn is not None:
        bfx, bfy, bfz = bulk_flow_fn(x, y, z)
        vx += bfx
        vy += bfy
        vz += bfz
    d_true = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    mu_true = 5 * np.log10(d_true) + 25
    mu_obs = mu_true + np.random.normal(0, sigma_mu, size=N)
    pv_true = (x * vx + y * vy + z * vz) / d_true
    z_cos = utils.z_lcdm(H0, mu_true)
    z_true = (1 + pv_true / constants.C) * (1 + z_cos) - 1
    z_obs = z_true + np.random.normal(0, sigma_z, size=N)
    mask = np.where(z_cos > 0.005)
    return (
        (x[mask], y[mask], z[mask]),
        (vx[mask], vy[mask], vz[mask]),
        mu_true[mask],
        mu_obs[mask],
        pv_true[mask],
        z_cos[mask],
        z_true[mask],
        z_obs[mask],
    )


def table_7_1():
    for name in ("ebv.corr", "max.corr", "salt.corr"):
        p_v, p_e, n_v, n_e = parse_whitford_results(
            f"{constants.HSF_DIR}/ado/bf_mocks/MLE_MVE_results.{name}.txt"
        )
        my_v, my_e = parse_bayesian_results(
            f"{constants.HSF_DIR}/ado/bf_mocks/fitdfdft.{name}.pickle"
        )
        for v, e in zip((n_v, p_v, my_v), (n_e, p_e, my_e)):
            print(
                "".join(
                    [f"{np.round(v[i], 1)}({np.round(e[i], 1)}) & " for i in range(4)]
                )
            )
        print("-----------------------")


def table_7_2(sig_v):
    for name in ("ebv", "max", "salt"):
        p_v, p_e, n_v, n_e = parse_whitford_results(
            f"{constants.HSF_DIR}/ado/bf_mocks/MLE_MVE_results.{name}.txt"
        )
        with open(
            f"{constants.HSF_DIR}/ado/bf_mocks/fitdfdft.{name}.pickle", "rb"
        ) as f:
            res = pickle.load(f)
        print(
            np.round(get_p_value(sig_v, n_v[-1], n_e[-1]), 3),
            np.round(get_p_value(sig_v, p_v[-1], p_e[-1]), 3),
            np.round(get_p_value(sig_v, posterior=res[0]["v"]), 3),
        )
