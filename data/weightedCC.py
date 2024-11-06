# mpl.use("TkAgg")
import argparse
import copy
import sys
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as interp
from astropy import wcs
from astropy.io import fits
from astropy.table import Table
from IPython import embed
from scipy.interpolate import UnivariateSpline
from scipy.signal import argrelextrema
from tqdm import tqdm

from data.spec import Spectrum

warnings.simplefilter("ignore")

# smoothing factors for the continuum estimate
galsmooth = 125.0
tempsmooth = 125.0


def read_param(flnm, param, default=None, ind=1, verbose=True):
    """read_param.

    Parameters
    ----------
    flnm :
        flnm
    param :
        param
    default :
        default
    ind :
        ind
    verbose :
        verbose
    """
    # This one is commonly used for photometry parameter files.

    fp = open(flnm)
    lines = fp.read()
    fp.close()

    lines = lines.split("\n")
    lines = clean_lines(lines)

    for line in lines:
        parsed = line.split(None)
        if parsed[0] == param:
            if verbose:
                print("Reading " + param + " from " + flnm)

            try:
                return eval(parsed[ind])
            except:
                return parsed[ind]
    print()
    print("Couldn't find ", param)
    print("Returning default ", default)
    print()

    return default


def clean_lines(orig_lines, stringlist=[""]):
    lines = copy.deepcopy(orig_lines)
    # Start by getting rid of spaces.

    lines = [item.strip() for item in lines]

    # Check for strings to exclude.
    lines = [item for item in lines if stringlist.count(item) == 0]

    # Get rid of comments
    lines = [item for item in lines if item[0] != "#"]
    return lines


def load_sdss_template(filename):
    """Load the continuum subtracted spectrum from an SDSS fits file."""
    hdulist = fits.open(filename)

    # The sky subtracted continuum is in the 2nd row of the first extension's
    # data.
    ext = hdulist[0]

    # Get the wavelength info.
    # The 1 offsets seem wrong, but match both the header and where lines
    # should actually be. Good enough. SDSS is weird.
    f_wcs = wcs.WCS(ext.header, fobj=hdulist)
    num_elements = ext.data.shape[1]
    log_wave = f_wcs.all_pix2world(
        np.array([np.arange(num_elements), np.ones(num_elements)]).T, 1  # Note: no +1!
    )[:, 0]

    vac_wave = 10**log_wave

    # SDSS wavelengths are in vacuum wavelength! Convert to STP units.
    # This conversion is from:
    # http://classic.sdss.org/dr7/products/spectra/vacwavelength.html
    wave = vac_wave / (
        1.0 + 2.735182e-4 + 131.4182 / vac_wave**2 + 2.76249e8 / vac_wave**4
    )

    flux = ext.data[0, :]

    # Keep track of the steps in logarithmic space so that we can do the cross
    # correlation.
    log_step = log_wave[1] - log_wave[0]
    ind = np.where(flux != 0)

    return wave[ind], flux[ind], log_step


def sigma_a(c, z, dz=0.001):
    if np.isclose(dz, 0.001):
        size = 149
    elif np.isclose(dz, 0.01):
        size = 15
    elif np.isclose(dz, 0.1):
        size = 3
    elif np.isclose(dz, 0.0001):
        size = 1499
    elif np.isclose(dz, 0.00001):
        size = 14999
    elif np.isclose(dz, 0.000001):
        size = 149999
    elif np.isclose(dz, 0.0000001):
        size = 1499999
    else:
        raise (Exception)
    # print 'size: %s' %size
    maxind = np.argmax(c)
    lowind = maxind - size
    hiind = maxind + size + 1
    comparison = c[lowind:hiind]
    xarr = np.arange(0, size, 1)
    tmpsig = []
    for i in xarr:
        if i >= len(comparison):
            continue
        left = comparison[i]
        right = comparison[(-i + 1)]
        tmpsig.append((left - right) ** 2)

    sig2 = (1.0 / (2 * len(comparison))) * np.sum(tmpsig)
    return sig2


def weightedCC(
    inspec,
    intemp,
    zmin,
    zmax,
    dz,
    out_name="weightedCC",
    pull_cut=200,
    binAngstroms=0,
    plotz=None,
    plot=True,
    figname="default",
    show=True,
    verbose=False,
):

    z = np.arange(zmin, zmax, dz)

    specs = {}
    smspecs = {}
    for s, sp in enumerate(inspec):
        d = Table.read(sp, format="ascii")
        w = d["wave"]
        g = d["flux"]
        try:
            ge = d["err"]
        except:
            ge = np.ones(len(d["flux"]))
            ind = ~np.isnan(g)
            ge *= 0.25 * np.median(g[ind])
        weights = 1.0 / ge**2
        weights[ge == 0] = 1e-9
        ge[ge == 0] = 1e9

        if binAngstroms != 0:
            wave_range = np.max(w) - np.min(w)
            nbins = int(wave_range / binAngstroms)
            bins = np.linspace(np.min(w), np.max(w), nbins)
            digitized = np.digitize(w, bins)

            newg = []
            newge = []
            neww = []

            for i in range(1, len(bins)):
                ind = np.where(digitized == i)
                tmpw1 = 1 / ge[ind] ** 2
                try:
                    tmpg = np.average(g[ind], weights=tmpw1)
                except Exception as e:
                    print(e)
                    # from IPython import embed; embed()
                    sys.exit()

                binned_weight = np.sum(tmpw1)
                tmpge = 1 / np.sqrt(binned_weight)

                newg.append(tmpg)
                newge.append(tmpge)
                neww.append(np.average(w[ind]))

            g = np.array(newg)
            ge = np.array(newge)
            w = np.array(neww)

        galspec = Spectrum(w, g, err=ge)
        galdx = galspec.getDx()
        galsmf, galsme = galspec.smooth(width=galsmooth / galdx, replace=False)

        ind = np.where(galspec.flux == 0)[0]
        for i in ind:
            galsmf[i] = 0

        if verbose:
            print(f"{sp} S/N: {np.mean(g / np.std(g - galsmf))}")

        specs[sp] = galspec
        smspecs[sp] = galsmf

    c = []
    tempspecs = {}
    for t, tp in enumerate(intemp):
        if "sdss" in tp:
            tw, tf, dummy = load_sdss_template(tp)
            tempspecs[tp] = (tw, tf)
            continue
        elif tp.endswith("fits") or tp.endswith("fit"):
            d = Table.read(tp, format="fits")
        else:
            d = Table.read(tp, format="ascii.no_header")
        tw = np.array(d.values()[0])
        tf = np.array(d.values()[1])
        tempspecs[tp] = (tw, tf)

    overlp = []
    for ind, i in tqdm(enumerate(z), total=len(z)):
        tmpc = []
        overtmp = 0
        ctmp = 0
        for t, tp in enumerate(intemp):
            tw, tf = tempspecs[tp]
            for s, sp in enumerate(inspec):

                galspec = specs[sp]
                galsmf = smspecs[sp]

                wz = tw * (1 + i)

                tempspec = Spectrum(wz, tf)
                resx, resf, rese = tempspec.resample(galspec.x, replace=False)
                tempspec = Spectrum(resx, resf)
                tempdx = tempspec.getDx()
                tempsmf, tempsme = tempspec.smooth(
                    width=tempsmooth / tempdx, replace=False
                )

                M = np.ones(len(galspec.x))

                ind = galsmf == 0
                M[np.where(~ind)[0][0:5]] = 0
                M[np.where(~ind)[0][-5:]] = 0

                M[(galsmf == 0) | (tempsmf == 0.0)] = 0
                if pull_cut is not None:
                    ind = np.where(np.abs(galspec.flux / galspec.err) > pull_cut)
                    M[ind] = 0
                Cw = (
                    (M / np.array(galspec.err) ** 2)
                    * (np.array(galsmf) ** 2)
                    * (galspec.flux / galsmf - 1)
                    * (tempspec.flux / tempsmf - 1)
                )

                overlapmask = np.where((galspec.x >= wz[0]) & (galspec.x <= wz[-1]))
                overtmp += len(galspec.x[overlapmask])

                Cw[np.isnan(Cw)] = 0
                ctmp += Cw[5:-5].sum()

            tmpc.append(ctmp)
        overlp.append(overtmp)
        c.append(np.mean(tmpc))

    # cross correlation plot
    c = np.array(c)
    overlp = np.array(overlp)

    cmax = np.argmax(c)
    zmax = z[cmax]

    i = zmax

    sig = sigma_a(c, z, dz=dz)
    if verbose:
        print(f"z: {zmax}")
        print(str(z[2] - z[1]))
        print(f"sigma_a {sig}")
    r = c / (np.sqrt(2) * np.sqrt(sig))

    if plot:
        if verbose:
            print("Plotting cross-correlation")
        plt.clf()
        plt.figure(figsize=(12, 2))
        plt.subplot(111)
        plt.plot(z, c, alpha=0)
        plt.xlabel("z")
        plt.ylabel("weighted cross-correlation")

        ax2 = plt.gca().twinx()
        ax2.plot(z, r)
        ax2.set_ylabel("tonry r")
        plt.savefig(f"{figname}_wcc.png")

        if show:
            plt.show()

    # if plotz was set in the config file, plot at that value, else, use the best fit (highest r value)
    if plotz is None:
        plotz = i

    if plot:
        plt.clf()
        plt.figure(figsize=(12, 8))
        if verbose:
            print("Plotting spectra comparison")
    # spectrum and pulls plots
    for t, tp in enumerate(intemp):
        tw, tf = tempspecs[tp]
        for s, sp in enumerate(inspec):
            d = Table.read(sp, format="ascii")
            w = d["wave"]
            g = d["flux"]
            g[np.isnan(g)] = 0
            try:
                ge = d["err"]
            except:
                ge = np.ones(len(g))
            weights = 1.0 / ge**2
            weights[ge == 0] = 1e-9
            ge[ge == 0] = 1e9

            if binAngstroms != 0:
                wave_range = np.max(w) - np.min(w)
                nbins = int(wave_range / binAngstroms)
                bins = np.linspace(np.min(w), np.max(w), nbins)
                digitized = np.digitize(w, bins)
                newg = []
                newge = []
                neww = []

                for index in range(1, len(bins)):
                    ind = np.where(digitized == index)
                    tmpw1 = 1 / ge[ind] ** 2

                    tmpg = np.average(g[ind], weights=tmpw1)
                    binned_weight = np.sum(tmpw1)
                    tmpge = 1 / np.sqrt(binned_weight)

                    newg.append(tmpg)
                    newge.append(tmpge)
                    neww.append(np.average(w[ind]))

                g = np.array(newg)
                ge = np.array(newge)
                w = np.array(neww)

            galspec = Spectrum(w, g, err=ge)
            galdx = galspec.getDx()
            galsmf, galsme = galspec.smooth(width=galsmooth / galdx, replace=False)
            ind = ~np.isnan(g) & ~np.isnan(galsmf)
            ind = np.where(galspec.flux == 0)[0]
            for i in ind:
                galsmf[i] = 0

            wz = tw * (1 + plotz)

            tempspec = Spectrum(wz, tf)
            resx, resf, rese = tempspec.resample(galspec.x, replace=False)
            tempspec = Spectrum(resx, resf)
            tempdx = tempspec.getDx()
            tempsmf, tempsme = tempspec.smooth(width=tempsmooth / tempdx, replace=False)

            mask = ((galsmf != 0) | (tempsmf != 0.0)) & ~np.isnan(galspec.flux)
            M = np.ones(len(galspec.x))

            M[(galsmf == 0) | (tempsmf == 0.0)] = 0

            ind = galsmf == 0
            M[np.where(~ind)[0][0:5]] = 0
            M[np.where(~ind)[0][-5:]] = 0

            Cw = (
                (M / np.array(galspec.err) ** 2)
                * (np.array(galsmf) ** 2)
                * (galspec.flux / galsmf - 1)
                * (tempspec.flux / tempsmf - 1)
            )

            sm2, sm2e = galspec.smooth(width=7.0 / galdx, replace=False)
            if plot:
                plt.subplot(311)
                plt.plot(
                    np.array(galspec.x)[mask],
                    np.array(galspec.flux)[mask]
                    / np.sum(np.abs(np.array(galspec.flux)[mask])),
                    label="gal",
                    color="red",
                )
                plt.plot(
                    np.array(galspec.x)[mask],
                    np.array(galsmf)[mask]
                    / np.sum(np.abs(np.array(galspec.flux)[mask])),
                    "r--",
                    label="gal spline",
                )
                plt.plot(
                    np.array(galspec.x)[mask],
                    np.array(tempspec.flux)[mask]
                    / np.sum(np.abs(np.array(tempspec.flux)[mask])),
                    label="tmp",
                    color="black",
                )
                plt.plot(
                    np.array(galspec.x)[mask],
                    np.array(tempsmf)[mask]
                    / np.sum(np.abs(np.array(tempspec.flux)[mask])),
                    "k--",
                    label="tmp spline",
                )
                plt.legend(fontsize=8, loc="upper left")
                ax = plt.gca()
                plt.subplot(312, sharex=ax)
                plt.plot(
                    np.array(galspec.x)[mask],
                    np.array(galspec.flux)[mask] / np.array(galspec.err)[mask],
                )
                plt.ylabel("pull")
                plt.subplot(313, sharex=ax)
                plt.plot(np.array(galspec.x)[mask], np.array(Cw)[mask])
                plt.ylabel("contribution to correlation function")

    if plot:
        plt.savefig(f"{figname}_comparison.png")
        if show:
            plt.show()

    # error estimation
    spline = UnivariateSpline(z, c - (c.max() - 0.5), s=0)
    roots = spline.roots()
    try:
        lowz = np.max(roots[roots < i])
        minus = i - lowz
    except:
        minus = -1
    try:
        hiz = np.min(roots[roots > i])
        plus = hiz - i
    except:
        plus = -1

    if verbose:
        print(out_name)
        print(f"{i} +{plus} -{minus}")
        print(f"r: {r[cmax]}")

    # peak-finding
    ext = argrelextrema(r, np.greater)
    peaks = np.where(r[ext] >= 3.0)
    zpeaks = z[ext][peaks]

    peak_dict = {}
    peakz = []
    peak_plus = []
    peak_minus = []
    peak_r = []
    for zp in zpeaks:
        ind = np.where(z == zp)
        spline = UnivariateSpline(z, c - (c[ind] - 0.5), s=0)
        roots = spline.roots()
        try:
            lowz = np.max(roots[roots < zp])
            minus = zp - lowz
        except:
            minus = -1
        try:
            hiz = np.min(roots[roots > zp])
            plus = hiz - zp
        except:
            plus = -1

        peakz.append(zp)
        peak_plus.append(plus)
        peak_minus.append(minus)
        peak_r.append(r[ind][0])

        if verbose:
            print(f"{zp} +{plus} -{minus} r {r[ind][0]}")

    peak_dict["z"] = np.array(peakz)
    peak_dict["plus"] = np.array(peak_plus)
    peak_dict["minus"] = np.array(peak_minus)
    peak_dict["r"] = np.array(peak_r)
    indsort = np.argsort(peak_dict["r"])[::-1]
    peak_dict["z"] = peak_dict["z"][indsort]
    peak_dict["plus"] = peak_dict["plus"][indsort]
    peak_dict["minus"] = peak_dict["minus"][indsort]
    peak_dict["r"] = peak_dict["r"][indsort]
    if verbose:
        print(str(peak_dict))
    if plot:
        plt.close()
    return (zmax, plus, minus, r[cmax], peak_dict)


def weightedCC_mesh(
    inspec,
    intemp,
    zmin,
    zmax,
    out_name="weightedCC",
    pull_cut=200,
    binAngstroms=0,
    plotz=None,
    plot=True,
    figname="default",
    show=True,
    verbose=False,
    precision=3,
):
    while precision < 7:
        dz = 10 ** (-precision)
        z, plus, minus, r, d = weightedCC(
            inspec=inspec,
            intemp=intemp,
            zmin=zmin,
            zmax=zmax,
            dz=dz,
            out_name=out_name,
            pull_cut=pull_cut,
            binAngstroms=binAngstroms,
            plotz=plotz,
            plot=plot,
            figname=figname,
            show=show,
            verbose=verbose,
        )
        z = np.round(z, precision)
        print(f"Found z={z} using [{zmin}:{zmax}:{dz}]")
        zmin = z - 10 ** (1 - precision)
        zmax = z + 10 ** (1 - precision)
        precision += 1
    return z, plus, minus, r, d, precision


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(dest="param", type=str)
    args = parser.parse_args()

    param = args.param
    weightedCC_with_config(param)
