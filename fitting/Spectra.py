import numpy as np
from scipy import interpolate
from utils import print_verb

from fitting.FileRead import clean_lines, read_param, readcol
import constants


def RA_Dec_to_focal_plane(RAs, Decs):
    P = np.array(
        [
            3.64960466e01,
            1.50115249e02,
            2.14859099e02,
            3.33877844e02,
            -4.49495043e00,
            2.21087566e00,
            5.26792496e01,
            -1.77298322e01,
            3.84718562e-02,
        ]
    )

    field = np.array(np.around((RAs - 20.0) / 100.0), dtype=np.int32)

    dx = (RAs - P[field]) * np.cos(Decs * np.pi / 180)
    dy = Decs - P[field + 4]
    dr = np.sqrt(dx**2.0 + dy**2.0) / P[-1]
    return dr


def check_derivs(thepath, verbose=False):
    try:
        f = open(thepath, "r")
    except FileNotFoundError:
        return None
    lines = f.read()
    f.close()

    if lines.count("Lambda") > 1:
        # If anything is in the file at all
        lines = lines.split("\n")

        for line in lines:
            parsed = line.split(None)
            if parsed.count("Check") and parsed.count("All|All|All"):
                print_verb(verbose, parsed)
                d_dzps = [float(item) for item in parsed[4:8]]

                if (
                    abs(np.log(d_dzps[0])) < 0.2
                    and abs(np.log(d_dzps[1])) < 0.2
                    and abs(d_dzps[2]) < 0.2
                    and abs(d_dzps[3]) < 0.2
                ):
                    return True
                else:
                    return False
    else:
        return False


def read_weightmat(weightfl, flux_scale, single_offset, verbose=False):
    print_verb(verbose, "Reading weightmat", weightfl)

    with open(weightfl, "r") as f:
        lines = f.read().split("\n")

    lines = clean_lines(lines)
    lines = [item.split(None) for item in lines]
    if len(lines[0]) == 2:
        del lines[0]

    wmat = np.zeros([len(lines)] * 2, dtype=np.float64)
    for i in range(len(wmat)):
        for j in range(i, len(wmat)):
            wmat[i, j] = float(lines[i][j])
            wmat[j, i] = wmat[i, j]

    cmat = np.linalg.inv(wmat)
    cmat *= np.outer(flux_scale, flux_scale)

    if not single_offset or len(cmat) == 1:
        wmat = np.linalg.inv(cmat)
        return wmat, 0

    offdiags = []
    for i in range(len(cmat)):
        for j in range(i + 1, len(cmat)):
            offdiags.append(np.sqrt(abs(cmat[i, j])))  # abs just in case
    return (
        1.0 / np.diag(cmat - np.mean(offdiags) ** 2.0),
        1.0 / np.mean(offdiags) ** 2.0,
    )
    """
    for i in range(len(cmat)):
        for j in range(len(cmat)):
            print "%.1f" % np.sqrt(cmat[i,j]),
        print
    """


def get_lc2data(lc2fl, global_zp=25.0, single_offset=False, verbose=False):
    lc2data = {}

    lc2data["instrument"] = read_param(lc2fl, "@INSTRUMENT")
    lc2data["band"] = read_param(lc2fl, "@BAND")
    lc2data["magsys"] = read_param(lc2fl, "@MAGSYS")

    xpos = read_param(lc2fl, "@X_FOCAL_PLANE")
    ypos = read_param(lc2fl, "@Y_FOCAL_PLANE")

    if xpos is not None:
        lc2data["radius"] = np.sqrt(xpos**2.0 + ypos**2.0)
    else:
        lc2data["radius"] = None

    [date, flux, flux_err, flux_zp] = readcol(lc2fl, "ffff")

    if len(flux) == 0:
        [date, mag, mag_err] = readcol(lc2fl, "fff")
        flux_zp = np.ones(len(date), dtype=np.float64) * global_zp
        flux = 10.0 ** (0.4 * (flux_zp - mag))
        flux_err = mag_err * flux * np.log(10.0) / 2.5

    flux_scale = 10.0 ** (0.4 * (global_zp - flux_zp))
    flux *= flux_scale
    flux_err *= flux_scale

    lc2data["flux"] = flux
    lc2data["date"] = date

    weight_fl = read_param(lc2fl, "@WEIGHTMAT")

    if weight_fl is not None:
        # If single_offset == True, this will be 1D weights and an offset weight.
        # If single_offset == False, this will be a weight matrix and zero.

        if lc2fl.count("/"):
            weight_prefix = lc2fl[: lc2fl.rfind("/")] + "/"
        else:
            weight_prefix = ""

        weights, offset = read_weightmat(
            weight_prefix + weight_fl,
            flux_scale,
            single_offset,
            verbose=verbose,
        )
        lc2data["weight"] = weights
        lc2data["offset"] = offset
    else:
        lc2data["weight"] = 1.0 / flux_err**2.0
        lc2data["offset"] = 0.0

    return lc2data


def CCM(wave, R_V):  # A(wave)/A_V
    x = 1.0 / (wave / 10000.0)  # um^-1
    y = x - 1.82

    # 0.3 to 1.1
    a = (0.3 <= x) * (x <= 1.1) * 0.574 * (x**1.61)
    b = (0.3 <= x) * (x <= 1.1) * (-0.527 * x**1.61)

    a += (
        (1.1 < x)
        * (x <= 3.3)
        * (
            1.0
            + 0.17699 * y
            - 0.50447 * y**2.0
            - 0.02427 * y**3.0
            + 0.72085 * y**4.0
            + 0.01979 * y**5.0
            - 0.77530 * y**6.0
            + 0.32999 * y**7.0
        )
    )
    b += (
        (1.1 < x)
        * (x <= 3.3)
        * (
            1.41338 * y
            + 2.28305 * y**2.0
            + 1.07233 * y**3.0
            - 5.38434 * y**4.0
            - 0.62251 * y**5.0
            + 5.30260 * y**6.0
            - 2.09002 * y**7.0
        )
    )

    Fa = (x >= 5.9) * (-0.04473 * (x - 5.9) ** 2.0 - 0.009779 * (x - 5.9) ** 3.0)
    Fb = (x >= 5.9) * (0.2130 * (x - 5.9) ** 2.0 + 0.1207 * (x - 5.9) ** 3.0)

    a += (
        (3.3 < x)
        * (x <= 8.0)
        * (1.752 - 0.316 * x - 0.104 / ((x - 4.67) ** 2.0 + 0.341) + Fa)
    )
    b += (
        (3.3 < x)
        * (x <= 8.0)
        * (-3.090 + 1.825 * x + 1.206 / ((x - 4.62) ** 2.0 + 0.263) + Fb)
    )

    a += (
        (8.0 < x)
        * (x <= 10.0)
        * (
            -1.073
            - 0.628 * (x - 8.0)
            + 0.137 * (x - 8.0) ** 2.0
            - 0.070 * (x - 8.0) ** 3.0
        )
    )
    b += (
        (8.0 < x)
        * (x <= 10.0)
        * (
            13.670
            + 4.257 * (x - 8.0)
            - 0.420 * (x - 8.0) ** 2.0
            + 0.374 * (x - 8.0) ** 3.0
        )
    )

    return a + b / R_V


def file_to_function(file_path, fill_value=np.nan, normalize="max", verbose=False):

    f = open(file_path)
    lines = f.read()
    f.close()

    lines = lines.split("\n")
    lines = clean_lines(lines)

    spectrum = []

    for line in lines:
        parsed = line.split(None)
        try:
            spectrum.append([float(parsed[0]), float(parsed[1])])
        except:
            print_verb(verbose, "Skipping Line ", line)

    spectrum = np.array(spectrum, dtype=np.float64)

    if normalize == "max":
        spectrum[:, 1] /= max(spectrum[:, 1])
    elif normalize is not None:
        spectrum[:, 1] *= normalize

    interp_function = interpolate.interp1d(
        spectrum[:, 0],
        spectrum[:, 1],
        kind="linear",
        bounds_error=False,
        fill_value=fill_value,
    )
    return interp_function


def get_mag_offset(file, instrument, band):
    f = open(file)
    lines = f.read().split("\n")
    f.close()

    lines = clean_lines(lines)
    lines = [item.split(None) for item in lines]

    for line in lines:
        if line[0] == instrument and line[1] == band:
            return float(line[2])


def radectoxyz(RAdeg, DECdeg):
    x = np.cos(DECdeg / (180.0 / np.pi)) * np.cos(RAdeg / (180.0 / np.pi))
    y = np.cos(DECdeg / (180.0 / np.pi)) * np.sin(RAdeg / (180.0 / np.pi))
    z = np.sin(DECdeg / (180.0 / np.pi))

    return np.array([x, y, z], dtype=np.float64)


def radecztoxyzMpc(RAdeg, Decdeg, z):
    return 4282.7494 * z * radectoxyz(RAdeg, Decdeg)


def get_dz(RAdeg, DECdeg, verbose=False):

    dzCMB = 371.0e3 / 299792458.0  # NED
    # http://arxiv.org/pdf/astro-ph/9609034
    # CMBcoordsRA = 167.98750000 # J2000 Lineweaver
    # CMBcoordsDEC = -7.22000000
    CMBcoordsRA = 168.01190437  # NED
    CMBcoordsDEC = -6.98296811

    CMBxyz = radectoxyz(CMBcoordsRA, CMBcoordsDEC)
    inputxyz = radectoxyz(RAdeg, DECdeg)

    dz = dzCMB * np.dot(CMBxyz, inputxyz)
    dv = dzCMB * np.dot(CMBxyz, inputxyz) * 299792.458

    print_verb(verbose, "Add this to z_helio to lowest order:")
    print_verb(verbose, dz, dv)

    return dz


def get_zCMB(RAdeg, DECdeg, z_helio, verbose=False):
    dz = -get_dz(RAdeg, DECdeg, verbose=verbose)

    one_plus_z_pec = np.sqrt((1.0 + dz) / (1.0 - dz))
    one_plus_z_CMB = (1 + z_helio) / one_plus_z_pec
    return one_plus_z_CMB - 1.0


def get_zhelio(RAdeg, DECdeg, z_CMB, verbose=False):
    dz = -get_dz(RAdeg, DECdeg, verbose=verbose)

    one_plus_z_pec = np.sqrt((1.0 + dz) / (1.0 - dz))
    one_plus_z_helio = (1 + z_CMB) * one_plus_z_pec
    return one_plus_z_helio - 1.0


class Spectra:
    import os

    def __init__(
        self,
        band,
        instrument,
        obslambdas=None,
        pathmodel=f"{constants.HSF_DIR}/rubin/snfit_fitting",
        radialpos=None,
        magsys=None,
        verbose=False,
    ):
        self.pathmodel = pathmodel
        self.band = band
        self.instrument = instrument
        self.obslambdas = obslambdas
        self.radialpos = radialpos

        self.magsys = magsys
        if magsys is not None:
            if magsys[0] != "@":
                self.magsys = "@" + magsys

        self.get_band(verbose=verbose)

    def get_rad_filter(self, directory_path, filterwheel, verbose=False):
        f = open(directory_path + filterwheel)
        lines = clean_lines(f.read().split("\n"))
        f.close()

        filter_radpos_list = []
        for line in lines:
            parsed = line.split(None)
            if parsed[0] == self.band:
                radialpos = read_param(
                    directory_path + parsed[-1],
                    "@MEASUREMENT_RADIUS",
                    ind=1,
                    verbose=verbose,
                )
                filter_radpos_list.append([directory_path + parsed[-1], radialpos])
        filter_radpos_list.sort()

        filter_list = [item[0] for item in filter_radpos_list]
        radial_list = [item[1] for item in filter_radpos_list]

        if self.radialpos <= radial_list[0]:
            return file_to_function(filter_list[0], fill_value=0.0, verbose=verbose)
        if self.radialpos >= radial_list[-1]:
            return file_to_function(filter_list[-1], fill_value=0.0, verbose=verbose)

        radial_list = np.array(radial_list)

        inds = np.argsort(abs(radial_list - self.radialpos))

        f0 = file_to_function(filter_list[inds[0]], fill_value=0.0, verbose=verbose)
        f1 = file_to_function(filter_list[inds[1]], fill_value=0.0, verbose=verbose)

        x0 = radial_list[inds[0]]
        x1 = radial_list[inds[1]]

        return lambda x: (
            f0(x) * self.radialpos - f1(x) * self.radialpos + f1(x) * x0 - f0(x) * x1
        ) / (x0 - x1)

        """get_band.

        Parameters
        ----------
        verbose :
            verbose
        """

    def get_band(self, verbose=False):
        """get_band.

        Parameters
        ----------
        verbose :
            verbose
        """

        # Get path to directory with filters:
        print_verb(verbose, locals().get("verbose"))
        print_verb(verbose, globals().get("verbose"))
        directory_path = (
            self.pathmodel
            + "/"
            + read_param(
                self.pathmodel + "/fitmodel.card",
                "@" + self.instrument,
                0,
                verbose=verbose,
            )
            + "/"
        )

        print_verb(
            verbose,
            "[instrument, directory_path, band] ",
            [self.instrument, directory_path, self.band],
        )
        instrument_cards = directory_path + "instrument.cards"

        optics_trans_path = read_param(
            instrument_cards, "@OPTICS_TRANS", verbose=verbose
        )
        mirror_reflect_path = read_param(
            instrument_cards, "@MIRROR_REFLECTIVITY", verbose=verbose
        )
        atm_trans_path = read_param(
            instrument_cards, "@ATMOSPHERIC_TRANS", verbose=verbose
        )
        qe_path = read_param(instrument_cards, "@QE", verbose=verbose)

        print_verb(verbose, "self.radialpos", self.radialpos)
        if self.radialpos is None:
            filters_path = directory_path + read_param(
                instrument_cards, "@FILTERS", verbose=verbose
            )
            print_verb(verbose, "filters_path", filters_path)

            try:
                this_filter_path = directory_path + read_param(
                    filters_path,
                    self.band,
                    ind=2,
                    verbose=verbose,
                )
            except:
                this_filter_path = directory_path + read_param(
                    filters_path,
                    self.band,
                    ind=1,
                    verbose=verbose,
                )
            transmission_fn = file_to_function(
                this_filter_path, fill_value=0.0, verbose=verbose
            )

        else:
            print_verb(verbose, "Radial filter found!", self.radialpos)
            filterwheel = read_param(
                instrument_cards, "@RADIALLY_VARIABLE_FILTERS", verbose=verbose
            )
            transmission_fn = self.get_rad_filter(
                directory_path, filterwheel, verbose=verbose
            )

        try:
            print_verb(verbose, "Trying @PSF_CORRECTION_FILTERS")
            psf_correction_filters_path = directory_path + read_param(
                instrument_cards, "@PSF_CORRECTION_FILTERS", verbose=verbose
            )
            this_psf_correction_filter_path = directory_path + read_param(
                psf_correction_filters_path,
                self.band,
                ind=1,
                verbose=verbose,
            )
            psf_correction = file_to_function(
                this_psf_correction_filter_path,
                fill_value=0.0,
                verbose=verbose,
            )
            print_verb(verbose, "PSF Correction found for ", self.band)
        except:
            try:
                print_verb(verbose, "Trying @CHROMATIC_CORRECTIONS")
                psf_correction_filters_path = directory_path + read_param(
                    instrument_cards, "@CHROMATIC_CORRECTIONS", verbose=verbose
                )
                this_psf_correction_filter_path = directory_path + read_param(
                    psf_correction_filters_path,
                    self.band,
                    ind=1,
                    verbose=verbose,
                )
                psf_correction = file_to_function(
                    this_psf_correction_filter_path,
                    fill_value=0.0,
                    verbose=verbose,
                )
                print_verb(verbose, "PSF Correction found for ", self.band)
            except:
                print_verb(verbose, "Setting PSF Correction to 1")
                psf_correction = lambda x: 1.0

        if optics_trans_path != 1 and optics_trans_path is not None:
            optics_fn = file_to_function(
                directory_path + optics_trans_path,
                fill_value=0.0,
                verbose=verbose,
            )
        else:
            print_verb(verbose, "Setting optics to 1")

            def optics_fn(x):
                return 1.0

        if mirror_reflect_path != 1 and mirror_reflect_path is not None:
            mirror_fn = file_to_function(
                directory_path + mirror_reflect_path,
                fill_value=0.0,
                verbose=verbose,
            )
        else:
            print_verb(verbose, "Setting mirror to 1")

            def mirror_fn(x):
                return 1.0

        if atm_trans_path != 1 and atm_trans_path is not None:
            atm_fn = file_to_function(
                directory_path + atm_trans_path,
                fill_value=0.0,
                verbose=verbose,
            )
        else:
            print_verb(verbose, "Setting atmosphere to 1")

            def atm_fn(x):
                return 1.0

        if qe_path != 1 and qe_path is not None:
            qe_fn = file_to_function(
                directory_path + qe_path, fill_value=0.0, verbose=verbose
            )
        else:
            print_verb(verbose, "Setting QE to 1")

            def qe_fn(x):
                return 1.0

        self.transmission_fn = (
            lambda x: transmission_fn(x)
            * psf_correction(x)
            * optics_fn(x)
            * mirror_fn(x)
            * atm_fn(x)
            * qe_fn(x)
        )

        if self.obslambdas is not None:
            self.evaluated = self.transmission_fn(self.obslambdas)

        if self.magsys is not None:
            print_verb(verbose, "Reading magsys...")

            if self.magsys == "@AB":
                self.ref_fn = file_to_function(
                    self.pathmodel + "/MagSys/ab-spec.dat",
                    fill_value=0.0,
                    normalize=1.0,
                    verbose=verbose,
                )
            else:
                magsysfile = read_param(
                    self.pathmodel + "/fitmodel.card", self.magsys, verbose=verbose
                )
                mag_offset = get_mag_offset(
                    self.pathmodel + "/" + magsysfile, self.instrument, self.band
                )

                print_verb(
                    verbose, "mag_offset ", self.instrument, self.band, mag_offset
                )
                self.ref_fn = file_to_function(
                    self.pathmodel
                    + "/"
                    + read_param(
                        self.pathmodel + "/" + magsysfile, "@SPECTRUM", verbose=verbose
                    ),
                    fill_value=0.0,
                    normalize=10.0 ** (0.4 * mag_offset),
                    verbose=verbose,
                )

                # Let's note some signs. BD+17 on the Vega system is ~ 9, so the right sign is -2.5*log10(flux measured with the same zeropoint).
                # The flux of BD+17 here would be about 2.5e-4*Vega, so we would read in the BD+17 spectrum, then apply a normalization of 10.**(0.4*9).

                # What about AB offsets? Suppose SDSS_Mag - AB_Mag is found to be 0.03. Then the reference for SDSS_Mag is AB*1.03:
                # SDSS_Mag - AB_Mag = -2.5*log10(flux/(AB*1.03)) - -2.5*log10(flux/AB) = 0.03, so mag_offset should be +0.03.
