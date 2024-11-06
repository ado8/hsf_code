import glob
import subprocess
import sys

# from astroquery.vizier import Vizier
import astropy.units as u
import numpy as np
import sep
import tqdm
from astropy import wcs
from astropy.io import fits
from astropy.nddata import NDData
from astropy.table import Table
from DavidsNM import miniNM_new, save_img
from photutils.psf import EPSFBuilder, extract_stars
from scipy.interpolate import RectBivariateSpline


def NMAD(vals):
    return 1.4826 * np.median(np.abs(vals - np.median(vals)))


def get_2MASS(RA, Dec):  # replace with utils.refcat calls
    output = subprocess.getoutput(
        "../refcat/refcat %f %f -rad 0.002 -dir ../refcat/00_m_16/ -var J,dJ,H,dH | grep -v dH"
        % (RA, Dec)
    ).split("\n")
    output += subprocess.getoutput(
        "../refcat/refcat %f %f -rad 0.002 -dir ../refcat/16_m_17/ -var J,dJ,H,dH | grep -v dH"
        % (RA, Dec)
    ).split("\n")

    print(output)
    output = [
        item
        for item in output
        if item.replace("0.000", "").replace(" ", "").strip() != ""
    ]

    print("parsed", output)

    if len(output) == 0:
        return [0.0] * 4
    elif len(output) > 1:
        return [0.0] * 4
    else:
        return [float(item) for item in output[0].split(None)]


def get_image_psf(image, psfsize=27):

    f = fits.open(image.path)
    dat = f[1].data * 1.0

    w0 = wcs.WCS(f[1].header)

    filt = f[1].header["SKYSUB"].split(".fit")[0][-1]

    # ADUSCALE = f[1].header["ADUSCALE"]

    arcsecperpix = (
        max(
            np.abs(f[1].header["CD1_1"]),
            np.abs(f[1].header["CD1_2"]),
            np.abs(f[1].header["CD2_1"]),
            np.abs(f[1].header["CD2_2"]),
        )
        * 3600.0
    )  # not right, but close enough here

    print("arcsecperpix", arcsecperpix)

    dat -= sep.Background(dat, bw=128)
    the_NMAD = NMAD(dat)
    print("the_NMAD", the_NMAD)

    objs = sep.extract(dat, thresh=the_NMAD * 2.0)

    print("objs", objs)
    print("objs x", objs["x"])
    print(len(objs), "objects found")

    f_reg = open(image.path.replace(image.path.split("/")[-1], "2M_stars.reg"), "w")
    f_reg.write(
        """# Region file format: DS9 version 4.1
global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1
physical
"""
    )

    good_obj_mask = (
        (objs["peak"] < 10000)
        & (objs["peak"] > the_NMAD * 100)
        & (objs["x"] > 55)
        & (objs["x"] < len(dat[0]) - 55)
        & (objs["y"] > 55)
        & (objs["y"] < len(dat) - 55)
    )

    for i in tqdm.trange(len(good_obj_mask)):

        if good_obj_mask[i]:
            ij = [int(np.around(objs["y"][i])) - 1, int(np.around(objs["x"][i])) - 1]
            pixels_around_peak = dat[ij[0] - 5 : ij[0] + 6, ij[1] - 5 : ij[1] + 6]
            # print("pixels_around_peak", pixels_around_peak, ij, objs["x"][i], objs["y"][i])

            if pixels_around_peak.max() < objs["peak"][i] * 0.5:
                # No maximum nearby!
                good_obj_mask[i] = 0

        for j in range(len(good_obj_mask)):
            if i != j:
                delta_pix = np.sqrt(
                    (objs["x"][i] - objs["x"][j]) ** 2.0
                    + (objs["y"][i] - objs["y"][j]) ** 2.0
                )
                delta_arcsec = delta_pix * arcsecperpix

                if delta_arcsec < 15:
                    # Does j contaminate i?
                    i_at_j = (
                        objs["peak"][i] / (1.0 + (delta_arcsec / 2.0) ** 2.0) ** 2.0
                    )

                    if (
                        i_at_j < objs["peak"][j] * 50
                        and i_at_j / objs["peak"][i] > 0.005
                    ):
                        good_obj_mask[i] = 0

    print("good objects", sum(good_obj_mask))

    for i in range(len(objs["x"])):
        f_reg.write(
            "circle(%f,%f,%i)\n"
            % (objs["x"][i], objs["y"][i], 2 + 8 * good_obj_mask[i])
        )
    f_reg.close()

    # save_img(dat, "dat.fits")
    f.close()

    stars_tbl = Table()
    stars_tbl["x"] = objs["x"][good_obj_mask]
    stars_tbl["y"] = objs["y"][good_obj_mask]

    stamps = extract_stars(NDData(dat), stars_tbl, size=35)
    # print("stamps", stamps[0])

    save_img(
        [stamps[i].data for i in range(len(stamps))],
        image.replace("ukirt_data/", "psf_models/").replace(".fits", "_stamps.fits"),
    )

    epsf_builder = EPSFBuilder(oversampling=1.0, maxiters=15, progress_bar=True)
    epsf, fitted_stars = epsf_builder(stamps)

    print("epsf.data", np.sum(epsf.data))
    the_psf = epsf.data  # Junk around the edges, so cutout

    psf_center = the_psf * 1.0
    psf_center[:5] *= 0.0
    psf_center[-5:] *= 0.0
    psf_center[:, :5] *= 0.0
    psf_center[:, -5:] *= 0.0

    brightest_ij = np.where(psf_center == psf_center.max())

    the_psf = the_psf[
        brightest_ij[0][0] - 13 : brightest_ij[0][0] + 14,
        brightest_ij[1][0] - 13 : brightest_ij[1][0] + 14,
    ]

    print("the_psf", the_psf.sum())

    the_psf /= the_psf.sum()

    print("the_psf", the_psf.sum())

    save_img(
        the_psf,
        image.replace("ukirt_data/", "psf_models/").replace(".fits", "_epsf.fits"),
    )

    subxs = np.arange(len(the_psf), dtype=np.float64) / 1.0
    subxs -= np.median(subxs)

    xs = np.arange(psfsize, dtype=np.float64)
    xs -= np.median(xs)

    ifn = RectBivariateSpline(subxs, subxs, the_psf, kx=2, ky=2)

    evalxs = np.arange(psfsize * 3, dtype=np.float64) / 3.0
    evalxs -= np.mean(evalxs)

    save_img(
        ifn(evalxs, evalxs),
        image.replace("ukirt_data/", "psf_models/").replace(".fits", "_epsfx3.fits"),
    )

    evalxs = np.arange(psfsize * 3 * 9, dtype=np.float64) / 27.0
    evalxs -= np.mean(evalxs)

    tmp_PSF = ifn(evalxs, evalxs) / (9.0**2)
    ePSFsubpix = 0.0

    for i in range(9):
        for j in range(9):
            ePSFsubpix += tmp_PSF[i::9, j::9]

    save_img(
        ePSFsubpix,
        image.replace("ukirt_data/", "psf_models/").replace(
            ".fits", "_epsfx3subpix.fits"
        ),
    )

    f_phot = open(
        image.replace("ukirt_data/", "2M_stars/field_stars_").split(".fits")[0]
        + ".txt",
        "w",
    )
    all_stars = []

    ZP_dict = {"2MASSJ": [], "2MASSdJ": [], "UKInstJ": []}

    def chi2fn(P, alldat, ifn=ifn, xs=xs):
        mod = P[0] * ifn(xs - P[1], xs - P[2])
        resid = alldat[0] - mod
        resid -= np.mean(resid)
        return np.sum(resid**2.0)

    for ind in np.where(good_obj_mask > 0)[0]:
        print("ind", ind)
        stardat = dat[
            int(np.around(objs["y"][ind])) - 13 : int(np.around(objs["y"][ind])) + 14,
            int(np.around(objs["x"][ind])) - 13 : int(np.around(objs["x"][ind])) + 14,
        ]

        P, NA, NA = miniNM_new(
            ministart=[np.sum(stardat), 0.0, 0.0],
            miniscale=[np.sum(stardat) / 10.0, 2.0, 2.0],
            passdata=stardat,
            chi2fn=chi2fn,
            verbose=False,
            compute_Cmat=False,
        )

        mod = P[0] * ifn(xs - P[1], xs - P[2])
        all_stars.append(np.concatenate((stardat, mod, (stardat - mod) / mod.max())))
        # save_img([stardat, mod, stardat - mod], "stardat.fits")
        RA_Dec = w0.all_pix2world([[objs["x"][ind], objs["y"][ind]]], 1)[0]
        print("RA_Dec", RA_Dec)

        TMJ, TMdJ, TMH, TMdH = get_2MASS(RA_Dec[0], RA_Dec[1])

        if TMJ > 1 and TMH > 1 and P[0] > 0:
            if TMJ == TMH:
                print("Warning! J-H = 0")

            if filt == "J":
                ZP_dict["2MASSJ"].append(TMJ - 0.065 * (TMJ - TMH))
                ZP_dict["2MASSdJ"].append(np.sqrt(TMdJ**2.0 + (0.065 * TMdH) ** 2.0))
            elif filt == "H":
                ZP_dict["2MASSJ"].append(TMH + 0.070 * (TMJ - TMH) - 0.030)
                ZP_dict["2MASSdJ"].append(np.sqrt(TMdH**2.0 + (0.070 * TMdJ) ** 2.0))
            elif filt == "Y":
                ZP_dict["2MASSJ"].append(TMJ + 0.500 * (TMJ - TMH) + 0.080)
                ZP_dict["2MASSdJ"].append(
                    np.sqrt((1.5 * TMdJ) ** 2.0 + (0.50 * TMdH) ** 2.0)
                )
            else:
                assert 0, image

            ZP_dict["UKInstJ"].append(-2.5 * np.log10(P[0]))

        towrite = [
            RA_Dec[0],
            RA_Dec[1],
            objs["x"][ind],
            objs["y"][ind],
            P[0],
            TMJ,
            TMdJ,
            TMH,
            TMdH,
        ]
        towrite = [str(item) for item in towrite]

        f_phot.write("  ".join(towrite) + "\n")
    f_phot.close()
    save_img(all_stars, image.replace("ukirt_data/", "2M_stars/all_stars_"))

    """
    print("number of stars for zeropoint", len(ZP_dict["2MASSJ"]))


    for key in ZP_dict:
        ZP_dict[key] = np.array(ZP_dict[key])


    print("ZP_dict", ZP_dict)

    if len(ZP_dict["2MASSJ"]) < 3:
        print("Not enough stars!")
        print(subprocess.getoutput("rm -fv " + image))
        stop_here

    ZP = get_zp(ZP_dict)
    print("Found ZP", ZP, image)
    assert 1 - np.isnan(ZP), image

    new_scale = 10.**(-0.4*(ZP - 24.))
    print("new_scale", new_scale)

    f = fits.open(image, 'update')
    f[0].header["history"] = "ZP " + str(ZP)

    f[0].data *= new_scale
    f[1].data *= new_scale
    f[0].header["ADUSCALE"] *= new_scale

    f.flush()
    f.close()
    """
