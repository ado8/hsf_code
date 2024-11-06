import base64
import copy
import datetime
import glob
import io
import json
import math
import os
import re
import shutil
import time
import urllib.parse
from collections import OrderedDict
from itertools import chain, combinations
from os import environ
from sys import stdout

import astropy.units as u
import corner
import dotenv
import emcee
import gspread
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
import pytz
import pyvo as vo
import requests
from apiclient.http import MediaFileUpload, MediaIoBaseDownload
from astropy.coordinates import EarthLocation, SkyCoord, SphericalRepresentation
from astropy.io import fits
from astropy.table import Table
from astropy.table.column import MaskedColumn
from astropy.time import Time
from astropy.wcs import WCS
from bs4 import BeautifulSoup
from googleapiclient.discovery import build
from IPython import get_ipython
from loguru import logger as custom_logger
from oauth2client.service_account import ServiceAccountCredentials
from paramiko import AutoAddPolicy, RSAKey, SSHClient
from paramiko.auth_handler import AuthenticationException, SSHException
from scipy.ndimage import rotate, shift
from scp import SCPClient, SCPException
from django.db import models

import constants
from hsf.settings import BASE_DIR

dotenv_file = os.path.join(BASE_DIR, ".env")
if os.path.isfile(dotenv_file):
    dotenv.load_dotenv(dotenv_file)
ssh_key_filepath = environ.get("SSH_KEY")
remote_path = environ.get("REMOTE_PATH")


def create_logger():
    """Create custom logger."""
    custom_logger.remove()
    custom_logger.add(
        stdout,
        colorize=True,
        level="INFO",
        format="<light-cyan>{time:MM-DD-YYYY HH:mm:ss}</light-cyan> | \
                    <light-green>{level}</light-green>: \
                    <light-white>{message}</light-white>",
    )
    custom_logger.add(
        "logs/errors.log",
        colorize=True,
        level="ERROR",
        rotation="200 MB",
        catch=True,
        format="<light-cyan>{time:MM-DD-YYYY HH:mm:ss}</light-cyan> | \
                     <light-red>{level}</light-red>: \
                     <light-white>{message}</light-white>",
    )
    return custom_logger


logger = create_logger()


def print_verb(v, *args):
    if v:
        print(*args)


def get_graph():
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=300)
    buf.seek(0)
    image_png = buf.getvalue()
    graph = base64.b64encode(image_png).decode("utf-8").replace("\n", "")
    buf.close()
    return graph


def light_plot():
    import matplotlib as mpl

    for key, val in constants.DARK_DICT.items():
        if val == "white":
            val = "black"
        elif val == "black":
            val = "white"
        mpl.rcParams[key] = val


def dark_plot():
    import matplotlib as mpl

    for key, val in constants.DARK_DICT.items():
        mpl.rcParams[key] = val


def MJD(
    today=False,
    year=0,
    month=0,
    day=0,
    hour=0,
    minute=0,
    second=0,
    ymdhms=None,
    d=None,
    dt=None,
):
    if today == True:
        year, month, day, hour, minute, second = np.array(
            re.split("-|:| ", str(datetime.datetime.now())), dtype=float
        )
    elif ymdhms:
        year, month, day, hour, minute, second = re.split("-|:| ", str(ymdhms))
    elif d:
        year, month, day = d.year, d.month, d.day
    elif dt:
        year, month, day, hour, minute, second = (
            dt.year,
            dt.month,
            dt.day,
            dt.hour,
            dt.minute,
            dt.second,
        )
    year, month, day, hour, minute, second = (
        int(year),
        int(month),
        int(day),
        int(hour),
        int(minute),
        float(second),
    )
    if month == 1 or month == 2:
        yearp = year - 1
        monthp = month + 12
    else:
        yearp = year
        monthp = month
    A = np.trunc(yearp / 100.0)
    B = 2 - A + np.trunc(A / 4.0)

    if yearp < 0:
        C = np.trunc((365.25 * yearp) - 0.75)
    else:
        C = np.trunc(365.25 * yearp)

    D = np.trunc(30.6001 * (monthp + 1))
    E = day
    F = hour / 24.0 + minute / (24.0 * 60.0) + second / (24.0 * 3600.0)
    return B + C + D + E + F + 1720994.5 - 2400000.5


def MJD_to_ut(MJD: float) -> str:
    jd = MJD + 2400000.5 + 0.5
    F, I = math.modf(jd)
    I = int(I)
    A = math.trunc((I - 1867216.25) / 36524.25)
    if I > 2299160:
        B = I + 1 + A - math.trunc(A / 4.0)
    else:
        B = I
    C = B + 1524
    D = math.trunc((C - 122.1) / 365.25)
    E = math.trunc(365.25 * D)
    G = math.trunc((C - E) / 30.6001)
    day = C - E + F - math.trunc(30.6001 * G)
    if G < 13.5:
        month = G - 1
    else:
        month = G - 13
    if month > 2.5:
        year = D - 4716
    else:
        year = D - 4715
    return int(year), int(month), day


def ab_to_ujy(mag, dmag, vega_fix=0):
    mag += vega_fix
    ujy = 10 ** (0.4 * (23.9 - mag))
    dujy = ujy * (10 ** (dmag / 2.5) - 1)
    return ujy, dujy


def ujy_to_ab(ujy, dujy):
    if ujy <= 0:
        return 99, 99
    mag = 23.9 - 2.5 * np.log10(ujy)
    dmag = 2.5 * np.log10(1 + dujy / ujy)
    return mag, dmag


def empty_dict(test_dict):
    res = True
    for key in test_dict.keys():
        if test_dict[key] != []:
            res = False
            break
    return res


def deg2HMS(ra="", dec="", rounding=False):
    RA, DEC, rs, ds = "", "", "", ""
    if dec:
        if str(dec)[0] == "-":
            ds, dec = "-", abs(dec)
        deg = int(dec)
        decM = abs(int((dec - deg) * 60))
        if rounding == False:
            decS = (abs((dec - deg) * 60) - decM) * 60
        else:
            decS = np.round((abs((dec - deg) * 60) - decM) * 60, rounding)
        DEC = "{0}{1} {2} {3}".format(ds, deg, decM, decS)

    if ra:
        if str(ra)[0] == "-":
            rs, ra = "-", abs(ra)
        raH = int(ra / 15)
        raM = int(((ra / 15) - raH) * 60)
        if rounding == False:
            raS = ((((ra / 15) - raH) * 60) - raM) * 60
        else:
            raS = np.round(((((ra / 15) - raH) * 60) - raM) * 60, rounding)
        RA = "{0}{1} {2} {3}".format(rs, raH, raM, raS)

    if ra and dec:
        return (RA, DEC)
    else:
        return RA or DEC


def HMS2deg(ra="", dec=""):
    RA, DEC, rs, ds = "", "", 1, 1
    if dec:
        dec = dec.replace(":", " ")
        if len(dec.split()) == 1:
            D = float(dec)
            M = 0
            S = 0
        elif len(dec.split()) == 2:
            D, M = [float(i) for i in dec.split()]
            S = 0
        else:
            D, M, S = [float(i) for i in dec.split()]
        if str(D)[0] == "-":
            ds, D = -1, abs(D)
        deg = D + (M / 60) + (S / 3600)
        DEC = deg * ds

    if ra:
        ra = ra.replace(":", " ")
        if len(ra.split()) == 1:
            H = float(ra)
            M = 0
            S = 0
        if len(ra.split()) == 2:
            H, M = [float(i) for i in ra.split()]
            S = 0
        else:
            H, M, S = [float(i) for i in ra.split()]
        if str(H)[0] == "-":
            rs, H = -1, abs(H)
        deg = (H * 15) + (M / 4) + (S / 240)
        RA = deg * rs

    if ra and dec:
        return (RA, DEC)
    else:
        return RA or DEC


def remove_empty_dict(obj, kind="lc"):
    """
    Goes through an object's lc_dict and removes empty lightcurves.
    If lc_dict keys are supposed to indicate the presence of lightcurves,
    I don't want any keys that correspond to empty lightcurves
    """
    del_list = []
    if kind == "lc":
        d = obj.lc_dict
    elif kind == "snpy" and "model_dict" in obj.snpy_dict:
        d = obj.snpy_dict["model_dict"]
    elif kind == "sncosmo" and "model_dict" in obj.sncosmo_dict:
        d = obj.sncosmo_dict["model_dict"]
    else:
        return obj
    for filt in d.keys():
        if d[filt] == None:
            del_list.append(filt)
        elif empty_dict(d[filt]):
            del_list.append(filt)
    for item in del_list:
        del d[item]
    if kind == "lc":
        obj.lc_dict = d
    elif kind == "snpy" and "model_dict" in obj.snpy_dict:
        obj.snpy_dict["model_dict"] = d
    elif kind == "sncosmo" and "model_dict" in obj.sncosmo_dict:
        obj.sncosmo_dict["model_dict"] = d
    return obj


def round_up_to_odd(f):
    return int(np.ceil(f) // 2 * 2)


def rebin_image(array, binning):
    """
    Program to rebin array
    array= 2D array
    new_shape= (X,Y) where X,Y are the reshape values. Binning of 2, X,Y=2
    operation=mean or sum
    """
    array = np.array(array)
    # Select the maximum of row, column but odd number
    a = array[
        0 : round_up_to_odd(np.shape(array)[0]), 0 : round_up_to_odd(np.shape(array)[1])
    ]

    new_shape = (a.shape[0] // binning[0], a.shape[1] // binning[1])

    compression_pairs = [(d, c // d) for d, c in zip(new_shape, a.shape)]
    flattened = [l for p in compression_pairs for l in p]
    a = a.reshape(flattened)
    for i in range(len(new_shape)):
        op = getattr(a, "mean")
        a = op(-1 * (i + 1))
    return a


def valid_file(parser, arg):
    if not os.path.isfile(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return arg


def rotate_header(num_of_rot, header):
    tmp_header = copy.deepcopy(header)
    for i in range(num_of_rot):
        x0 = tmp_header["CRPIX1"]
        y0 = tmp_header["CRPIX2"]
        tmp_header["CRPIX1"] = y0
        tmp_header["CRPIX2"] = (
            -x0 + tmp_header["NAXIS1"]
        )  # pixel conventions put lower left corner of lower left pixel at -0.5,-0.5
        tmp_header["NAXIS1"], tmp_header["NAXIS2"] = (
            tmp_header["NAXIS2"],
            tmp_header["NAXIS1"],
        )
        CD = [
            tmp_header["CD1_1"],
            tmp_header["CD1_2"],
            tmp_header["CD2_1"],
            tmp_header["CD2_2"],
        ]
        tmp_header["CD1_1"] = CD[2]
        tmp_header["CD1_2"] = CD[3]
        tmp_header["CD2_1"] = -CD[0]
        tmp_header["CD2_2"] = -CD[1]
    return tmp_header


def rotate_w_header(array, header, num_of_rot=1):
    num_of_rot = num_of_rot % 4
    new_header = rotate_header(num_of_rot, header)
    new_array = np.rot90(array, k=4 - num_of_rot, axes=(1, 0))
    return new_array, new_header


def align_wcs(array, header):
    """
    Will lead to ~2-5 pixel scale errors for ZPN projections
    """
    tmp_header = copy.deepcopy(header)
    if np.abs(header["CD1_1"]) < np.abs(header["CD1_2"]):
        array, tmp_header = rotate_w_header(array, tmp_header)
    if header["CD1_1"] > 0:
        array, tmp_header = flip(array, tmp_header, axis="vertical")
    if header["CD2_2"] < 0:
        array, tmp_header = flip(array, tmp_header, axis="horizontal")
    return array, tmp_header


def align_image(image_path, header):
    with PIL.Image.open(image_path) as align_im:
        lr_flip = header["CD1_1"] > 0
        tb_flip = header["CD2_2"] < 0
        if np.abs(header["CD1_1"]) < np.abs(header["CD1_2"]):
            align_im = align_im.rotate(90)
            lr_flip = header["CD2_1"] > 0
            tb_flip = header["CD1_2"] > 0
        if lr_flip:
            align_im = align_im.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        if tb_flip:
            align_im = align_im.transpose(PIL.Image.FLIP_TOP_BOTTOM)
        align_im.save(image_path)


def crop_fits(x0, x1, y0, y1, data, header, out_path=None, overwrite=True):
    data = data[y0:y1, x0:x1]
    tmp_header = copy.deepcopy(header)
    tmp_header["CRPIX1"] -= x0
    tmp_header["CRPIX2"] -= y0
    hdu = fits.PrimaryHDU(data, header=tmp_header)
    hdul = fits.HDUList([hdu])
    if out_path is not None:
        hdul.writeto(out_path, overwrite=overwrite)
    return hdul


def flip(array, header, axis="vertical"):
    array = np.array(array)
    tmp_header = copy.deepcopy(header)
    if axis == "horizontal":
        tmp_header["CRPIX2"] *= -1
        tmp_header["CRPIX2"] += tmp_header["NAXIS2"]
        tmp_header["CD1_2"] *= -1
        tmp_header["CD2_2"] *= -1
        array = array[::-1]
    elif axis == "vertical":
        tmp_header["CRPIX1"] *= -1
        tmp_header["CRPIX1"] += tmp_header["NAXIS1"]
        tmp_header["CD1_1"] *= -1
        tmp_header["CD2_1"] *= -1
        array = array[:, ::-1]
    return array, tmp_header


def build_service(folder_id=environ.get("UKIRT_DRIVE_ID_MSBS")):
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive.file",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_name(
        f"{constants.HSF_DIR}/hawaii-supernova-flows-1332a295a90b.json", scope
    )
    return build("drive", "v3", credentials=creds)


def download_file(
    file_name, dl_path, folder_id=environ.get("UKIRT_DRIVE_ID_MSBS"), force=False
):
    if os.path.exists(file_name) and not force:
        return
    service = build_service()
    file_id = (
        service.files()
        .list(
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
            q=f"parents in '{folder_id}' and trashed = false and name='{file_name}'",
            fields="nextPageToken, files(id, name)",
        )
        .execute()["files"][0]["id"]
    )
    fh = io.FileIO(dl_path, mode="wb")
    request = service.files().get_media(fileId=file_id)
    downloader = MediaIoBaseDownload(fh, request)
    try:
        downloader = MediaIoBaseDownload(fh, request, chunksize=1024 * 1024)
        done = False
        while done is False:
            status, done = downloader.next_chunk(num_retries=2)
            if status:
                print("Download %d%%." % int(status.progress() * 100))
        print("Download Complete!")
    finally:
        fh.close()


def upload_file(
    local_path,
    new_name=None,
    folder_id=environ.get("UKIRT_DRIVE_ID_MSBS"),
    overwrite=True,
):
    service = build_service()
    name = local_path.split("/")[-1]
    if new_name:
        name = new_name
    file_metadata = {"name": name}
    media = MediaFileUpload(local_path)
    file_id = file_exists(new_name, folder_id)
    if overwrite and file_id:
        service.files().update(
            fileId=file_id,
            body=file_metadata,
            media_body=media,
            fields="id",
            supportsAllDrives=True,
        ).execute()
    else:
        file_metadata["parents"] = [folder_id]
        service.files().create(
            body=file_metadata, media_body=media, fields="id", supportsAllDrives=True
        ).execute()


def file_exists(file_name, folder_id=environ.get("UKIRT_DRIVE_ID_MSBS")):
    service = build_service()
    file_list = (
        service.files()
        .list(
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
            q=f"parents in '{folder_id}' and trashed = false and name='{file_name}'",
            fields="nextPageToken, files(id, name)",
        )
        .execute()["files"]
    )
    if len(file_list) == 0:
        return False
    else:
        return file_list[0]["id"]


def list_files_in(folder_id):
    service = build_service()
    list_of_files = []
    page_token = None
    while True:
        response = (
            service.files()
            .list(
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
                q=f"parents in '{folder_id}' and trashed = false",
                fields="nextPageToken, files(id, name)",
                pageToken=page_token,
            )
            .execute()
        )
        list_of_files += response.get("files")
        page_token = response.get("nextPageToken", None)
        if page_token is None:
            break
    return list_of_files


def get_TNS_name(AT_name, mode="sheets"):
    AT_name = AT_name.strip("-").split("_")[0]
    bad_names = pd.read_csv(
        f"{constants.STATIC_DIR}/bad_names.txt", delim_whitespace=True
    )
    if AT_name in bad_names.bad.tolist():
        AT_name = bad_names.good[bad_names.bad == AT_name].tolist()[0]

    # Before, we used exclusively atlas names. When shifting to TNS, used an AT string to mark TNS
    if AT_name[:2] != "AT":
        # check from downloaded google sheets
        if mode == "sheets":
            t = pd.read_csv(f"{constants.MEDIA_DIR}/sheets_targets.csv", skiprows=4)
            try:
                TNS_name = t["TNS_name"][np.where(t["ATLAS_name"] == AT_name)[0][0]]
            except IndexError:
                print(f"ATLAS name {AT_name} does not seem to exist")
                TNS_matches = np.where(t["TNS_name"] == AT_name)[0]
                if len(TNS_matches) == 0:
                    print(
                        f"The sheet does not have a TNS_name that matches either. Zoinks!"
                    )
                    return None
                else:
                    TNS_name = t["TNS_name"][TNS_matches[0]]
                    print(
                        f"The sheet has one TNS name for that. Assuming it was supposed to be AT{TNS_name}"
                    )
        # check from database. Slow lookups.
        # elif mode == 'db':
        #    try:
        #        TNS_name = Target.objects.get(ATLAS_name=f'ATLAS{AT_name}').TNS_name
        #    except ObjectDoesNotExist:
        #        print(f'Do not see a TNS entry for ATLAS{AT_name}')
        #        print(f'Not sure what TNS_name it corresponds to. Check ATLAS site')
        #        return None
    else:
        TNS_name = AT_name[2:]
    return TNS_name


def get_observable_ra(date="today"):
    if date == "today":
        d = datetime.date.today()
    elif len(date) == 1:
        d = MJD_to_ut(MJD(date))
    elif type(date) == type(datetime.date.today()):
        d = date
    ra_of_sun = (
        360 / 365.24 * (d - datetime.date(datetime.date.today().year, 3, 21)).days
    )
    start = (ra_of_sun + 40) % 360
    end = (ra_of_sun - 40) % 360
    return start, end


def weighted_quantile(
    values, quantiles, sample_weight=None, values_sorted=False, old_style=False
):
    """From https://stackoverflow.com/questions/21844024/weighted-percentile-using-numpy/29677616#29677616
    Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
    initial array
    :param old_style: if True, will correct output to be consistent
    with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(
        quantiles <= 1
    ), "quantiles should be in [0, 1]"

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)


def sigmaclip(
    data,
    errors=None,
    niters=1,
    sigmalow=5,
    sigmahigh=5,
    center="median",
    use_pull=False,
):
    i = 0
    if errors is None:
        errors = np.ones(len(data))
    data = np.array(data)
    errors = np.array(errors)
    if center == "median":
        cen = np.median(data)
    elif center in ["mean", "average"]:
        cen = np.average(data, weights=1 / errors)
        errors = np.array(errors)
    std = np.std(data)
    if std == 0 or np.isnan(std):
        return data, errors, data == data[0], i
    while niters != 0:
        if use_pull:
            mlow = (data - cen) / errors > -sigmalow
            mhigh = (data - cen) / errors < sigmahigh
        else:
            mlow = data > cen - std * sigmalow
            mhigh = data < cen + std * sigmahigh
        if len(data[mlow & mhigh]) == len(data):
            break
        niters -= 1
        i += 1
        if center == "median":
            cen = np.median(data[mlow & mhigh])
        elif center in ["mean", "average"]:
            cen = np.average(data[mlow & mhigh], weights=1 / errors[mlow & mhigh])
        std = np.std(data[mlow & mhigh])
    if errors is None:
        return data[mlow & mhigh], None, mlow & mhigh, i
    return data[mlow & mhigh], errors[mlow & mhigh], mlow & mhigh, i


def line_fit(
    x,
    y,
    dy,
    xcentre,
    nwalkers=50,
    nburn=500,
    nsamp=500,
    scatter_outliers=50.0,
    plots=1,
    return_samples=False,
):
    """
    modified from https://github.com/mpags-python/examples/blob/master/emcee/fitting_a_line_with_emcee.ipynb
    """
    x = np.array(x)
    y = np.array(y)
    e = np.array(dy)
    # scatter of outlier points, just needs to be much larger than normal scatter
    scatter_outliers = 50.0
    # better-looking plots
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["figure.figsize"] = (10.0, 8.0)
    plt.rcParams["font.size"] = 12

    # define some helper functions
    def flatten_without_burn(sampler, nburn):
        c = sampler.chain
        if c.ndim == 4:
            c = c[0]
        c = c[:, nburn:]
        return c.reshape((np.product(c.shape[:-1]), c.shape[-1]))

    def weight_without_burn(sampler, nburn):
        c = sampler.lnprobability
        if c.ndim == 3:
            c = c[0]
        c = c[:, nburn:]
        w = np.exp(c.reshape(np.product(c.shape)))
        return w / w.sum()

    def get_samples(sampler, nburn, minweight=None):
        sample = flatten_without_burn(sampler, nburn)
        if minweight is not None:
            weight = weight_without_burn(sampler, nburn)
            sample = sample[weight > minweight]
        return sample

    def minmaxpad(x, p=0.05):
        xmin = x.min()
        xmax = x.max()
        xrange = xmax - xmin
        xmin = xmin - p * xrange
        xmax = xmax + p * xrange
        return xmin, xmax

    def plot_MCMC_model(ax, xdata, ydata, trace, show=False):
        """Plot the linear model and 2sigma contours"""
        ax.plot(xdata, ydata, "ow", mec="black")

        xmin, xmax = minmaxpad(x)
        dx = (xmax - xmin) / 100.0
        xfit = np.linspace(xmin, xmax, 100)
        yfit = model(xfit[..., np.newaxis], trace.T)
        mu = yfit.mean(-1)
        sig = yfit.std(-1)

        ax.fill_between(xfit, mu - 2 * sig, mu + 2 * sig, color="lightgray")
        ax.fill_between(xfit, mu - sig, mu + sig, color="darkgray")
        ax.plot(xfit, mu, "-k")

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim([xmin, xmax])
        if show:
            plt.show()

    def round_sig(x, sig=1):
        d = sig - int(math.floor(np.log10(x))) - 1
        d = max(0, d)
        return round(x, d), d

    def summary(samples, truths=None):
        mean = samples.mean(0)
        sigma = samples.std(0)
        for i, p in enumerate(par):
            err, dp = round_sig(sigma[i], 1)
            val = round(mean[i], dp)
            dp = str(dp)
            dp += "e}" if abs(np.log10(val)) > 3 else "f}"
            outstr = ("{:16s} = {:8." + dp + " ± {:<8." + dp).format(p, val, err)
            if truths is not None:
                outstr += ("   (" + dp + ")").format(truths[i])
            print(outstr)

    def model(x, theta):
        intercept, slope, scatter, prob_outlier = theta
        return intercept + slope * (x - xcentre)

    def log_prior(theta):
        intercept, slope, scatter, prob_outlier = theta
        # scatter must be greater than zero
        if scatter <= 0:
            return -np.inf
        # prob_outlier must be between zero and one
        if prob_outlier < 0 or prob_outlier > 1:
            return -np.inf
        # prior on intercept, slope and scatter; as explained at
        # http://jakevdp.github.io/blog/2014/06/14/frequentism-and-bayesianism-4-bayesian-in-python/#The-Prior
        # P(intercept) = 1
        # P(slope) = (1 + slope ** 2)^(-1.5)  # uniform in sin(angle)
        # P(scatter) = 1/scatter  # Jeffreys prior, invariant to rescaling
        return -1.5 * np.log(1 + slope**2) - np.log(scatter)

    def log_likelihood(theta, x, y, e, scatter_outliers=scatter_outliers):
        intercept, slope, scatter, prob_outlier = theta
        # square residuals
        y_model = model(x, theta)
        dy2 = (y - y_model) ** 2
        # avoid NaNs in logarithm
        prob_outlier = np.clip(prob_outlier, 1e-99, 1 - 1e-99)
        # compute effective variances by combining errors and intrinsic scatter
        eff_var = scatter**2 + e**2
        eff_var_outliers = scatter_outliers**2 + e**2
        # logL for good (normal) and bad (outlier) distributions
        logL_good = (
            np.log(1 - prob_outlier)
            - 0.5 * np.log(2 * np.pi * eff_var)
            - 0.5 * dy2 / eff_var
        )
        logL_bad = (
            np.log(prob_outlier)
            - 0.5 * np.log(2 * np.pi * eff_var_outliers)
            - 0.5 * dy2 / eff_var_outliers
        )
        # using np.logaddexp helps maintain numerical precision
        return np.sum(np.logaddexp(logL_good, logL_bad))

    def logl(theta):
        # PT sampler needs us to use global variables
        return log_likelihood(theta, x, y, e, scatter_outliers)

    def log_posterior(theta, x, y, e, scatter_outliers=scatter_outliers):
        return log_prior(theta) + log_likelihood(theta, x, y, e, scatter_outliers)

    # set up emcee
    par = ("intercept", "slope", "scatter", "prob_outlier")
    ndim = len(par)
    initial_theta = np.zeros((nwalkers, ndim))
    initial_theta[:, :2] = np.random.normal((np.mean(y), 0.0), 0.05, (nwalkers, 2))
    initial_theta[:, 2:] = np.random.uniform((0.1, 0.1), (1.0, 0.5), (nwalkers, 2))
    # perform sampling
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_posterior, args=[x, y, e], threads=1
    )
    r = sampler.run_mcmc(initial_theta, nburn + nsamp)
    # plot the chains to visually assess convergence
    if plots == 2:
        plt.figure(figsize=[20, 10])
        for i, p in enumerate(par):
            plt.subplot(2, 2, i + 1)
            for w in range(nwalkers):
                plt.plot(
                    np.arange(sampler.chain.shape[1]),
                    sampler.chain[w, :, i],
                    "b-",
                    alpha=0.1,
                )
            plt.ylabel(p)
            aymin, aymax = plt.ylim()
            plt.vlines(nburn, aymin, aymax, linestyle=":")
            plt.ylim(aymin, aymax)
    # plot the chains to visually assess auto correlation time at equilibrium
    if plots == 2:
        plt.figure(figsize=[20, 10])
        for i, p in enumerate(par):
            plt.subplot(2, 2, i + 1)
            for w in range(0, nwalkers, 10):
                plt.plot(np.arange(100), sampler.chain[w, nburn : nburn + 100, i], "b-")
            plt.ylabel(p)
    # convert chains into samples for each parameter
    # clip chains with very low weights for normal method
    samples = get_samples(sampler, nburn)
    # examine parameter histograms and compare normal and parallel methods
    if plots == 2:
        plt.figure(figsize=[20, 10])
        for i, p in enumerate(par):
            plt.subplot(2, (ndim + 1) // 2, i + 1)
            n, b, patches = plt.hist(
                samples[:, i], bins=100, color="b", histtype="stepfilled", log=True
            )
            plt.xlabel(p)
    # create mega plot for normal method
    if plots > 0:
        xmin, xmax = minmaxpad(x)
        corner.corner(samples, labels=par)
        ax = plt.subplot(2, 2, 2)
        plot_MCMC_model(ax, x, y, samples)
        plt.plot([xmin, xmax], [xmin, xmax], lw=2)
        plt.subplots_adjust(wspace=0.15, hspace=0.15)
    summary(samples)
    res = {}
    mean = samples.mean(0)
    sigma = samples.std(0)
    for i, p in enumerate(par):
        err, dp = round_sig(sigma[i], 1)
        val = round(mean[i], dp)
        dp = str(dp)
        res[p] = (val, err)
    if return_samples:
        return res, samples
    return res


def pix2sky(header, x=None, y=None, coords=None, convention="TSK", coord_axis=0):
    """
    If the pixel coordinates of the center of the lower left pixel is:
        0,0 - convention = C, Astropy
        0.5,0.5 - convention = TSK
        1,1 - convetion = FITS, WCS
    """
    w = WCS(header)
    if x is None and y is None and coords is None:
        return
    if x is not None and y is not None:
        try:
            x = float(x)
            y = float(y)
        except TypeError:
            if len(x) != len(y):
                raise SyntaxError(
                    "Number of RA coords not equal to number of Dec coords"
                )
            x = np.array(x)
            y = np.array(y)
    elif coords is not None:
        try:
            coords = np.array(coords)
        except:
            raise TypeError("Unable to convert coords into np.ndarray")
        if coord_axis == 0:
            x = coords[:, 0]
            y = coords[:, 1]
        elif coord_axis == 1:
            x = coords[0]
            y = coords[1]
    # Weird issues with -= and numpy integers vs floats
    if convention.lower() == "tsk":
        x = x - 0.5
        y = y - 0.5
    elif convention.lower() in ["fits", "wcs"]:
        x = x - 1
        y = y - 1
    return w.pixel_to_world(x, y)


def sky2pix(header, ra=None, dec=None, coords=None, convention="TSK", coord_axis=0):
    """
    If the pixel coordinates of the center of the lower left pixel is:
        0,0 - convention = C, Astropy
        0.5,0.5 - convention = TSK
        1,1 - convetion = FITS, WCS
    """
    w = WCS(header)
    if ra is None and dec is None and coords is None:
        return
    if coords is not None:
        try:
            coords = np.array(coords)
        except:
            raise TypeError("Unable to convert coords to np.ndarray")
        if coord_axis == 0:
            world = SkyCoord(coords[:, 0], coords[:, 1], unit="deg")
        elif coord_axis == 1:
            world = SkyCoord(coords[0], coords[1], unit="deg")
    elif ra is not None and dec is not None:
        try:
            ra = float(ra)
            dec = float(dec)
        except TypeError:
            if len(ra) != len(dec):
                raise SyntaxError(
                    "Number of RA coords not equal to number of Dec coords"
                )
        world = SkyCoord(ra, dec, unit="deg")
    pix = np.array(w.world_to_pixel(world))
    if convention.lower() == "tsk":
        pix += 0.5
    elif convention.lower() in ["fits", "wcs"]:
        pix += 1
    return pix


def add_ellipses(ax, reg, color):
    for i, _ in enumerate(reg["x"]):
        ax.add_patch(
            patches.Ellipse(
                (float(reg["x"][i]), float(reg["y"][i])),
                float(reg["major"][i]),
                float(reg["minor"][i]),
                float(reg["phi"][i]),
                edgecolor=color,
                lw=2,
                facecolor="none",
            )
        )
    return ax


def get_ukirt_systematics(obs_header, x=None, y=None, ra=None, dec=None):
    if ra is not None:
        if type(ra) == list:
            ra = np.array(ra)
        if type(dec) == list:
            dec = np.array(dec)
        dim_1 = (ra - obs_header["CRVAL1"]) * np.cos(dec * np.pi / 180)
        dim_2 = dec - obs_header["CRVAL2"]
    elif x is not None:
        dim_1 = (x - obs_header["CRPIX1"]) * obs_header["PIXLSIZE"] / 3600
        dim_2 = (y - obs_header["CRPIX2"]) * obs_header["PIXLSIZE"] / 3600
    r = np.sqrt(dim_1**2 + dim_2**2) * np.pi / 180
    k = obs_header["PV2_3"]
    radial_distortion = -2.5 * np.log10(
        (1 + k * r**2) / (1 + 3 * k * r**2)
    )  # eq 1 Hodgkin et al. 2009
    return radial_distortion


def tphot(
    path,
    x=None,
    y=None,
    grid=False,
    force=False,
    tph_force=False,
    instr_mag=True,
    arg_dict={},
):
    """
    Python wrapper for tphot with instr_mags and refactored x y inputs.
    Returns pd.DataFrame and writes to out

    General options:
    out         Output file name (stdout)
    resid       Write a residual FITS image (NULL)
    moment      Write moments for each output object (NULL)
    subtract    Subtract fits successively from image (default no)
    residsky    Subtract sky model from residual image (default no)
    trail       Fit a trailed PSF (default no)
    verb        Diagnostic verbosity
    VERB        Diagnostic verbosity
    hdr         Write details about all fields
    binning     Bin input image by a factor NxN (1)
    invert      Invert the sign of the image? (no)

    Trigger parameters:
    border      Image border to avoid on all sides (2)
    xlft        Left x image border to avoid (2)
    xrgt        Right x image border to avoid (2)
    ybot        Bottom y image border to avoid (2)
    ytop        Top y image border to avoid (2)
    badata      Ignore data values of B (0.0)
    rad         Local max required over radius r to trigger (5.0)
    sig         Minimum value = sky+sig*rms to trigger (5.0)
    min         Minimum value above sky to trigger (100.0)
    max         Maximum peak to trigger (100000.0)

    Fit acceptance parameters:
    fwmax       Maximum FW to keep (100.00)
    fwmin       Minimum FW to keep (0.50)
    net         Minimum peak-sky to keep (10.0)
    move        Max offset between peak and fit (3.0)
    chin        Maximum chi/N to keep (1000.00)
    snr         Minimum flux SNR to keep (-1.0)
    okfit       Insist (0/1) on successful profile fit (1)
    force   Output even if some tests failed (default not)

    Fit arguments:
    major       Set major axis M for 4 param fit [pix]
    minor       Set minor axis M for 4 param fit [pix]
    phi         major axis angle to P for 4 param fit [deg]
    npar        How many parameter fit (2/4/7)? (7)
    ninit       Does trigger file have x,y or x,y,maj,min,phi (2/5)? (0)

    Noise, signal, and aperture parameters:
    eadu        Electrons per ADU to estimate noise.
                if set to None, infers gain as a UKIRT file
                if E is 'auto' eadu is set to mean/variance
    bias        Image background exceeds sky by B (0.0)
    wgt         Weight (inv variance) file name (NULL)
    sat         Saturation level is S (100000.0)
    aprad       Aperture radius for photometry (15)
    skyrad      Sky radius for photometry (40)
    nxsky       Subdivide image by M in x for sky (0)
    nysky       Subdivide image by N in y for sky (0)
    mosiacsky   Use an MxN chunk discontinuous sky model
    flat        Read flat file and divide by it
    flatnorm    Divide flat file by N before application

    cfits       Read with CFITSIO instead of default FITS reader
    nitf        Input is NITF, not FITS
    """
    os.makedirs("/tmp/hsf/tphot", exist_ok=True)
    # sorting out cases, the obj flag
    tmp_name = path.split("/")[-1]
    if "out" in arg_dict:
        out = arg_dict["out"]
    else:
        out = f"/tmp/hsf/tphot/{tmp_name}.tph"
    # Run on entire image
    if x is None and y is None:
        cmd_str = f"tphot {path}"
    # Run on single coordinate
    elif type(x) in [int, float, np.float32, np.float64] and type(y) in [
        int,
        float,
        np.float32,
        np.float64,
    ]:
        cmd_str = f"echo {x} {y} | tphot {path} -obj -"
    # run on array-like
    elif type(x) in [list, set, np.array, np.ndarray, pd.Series] and type(y) in [
        list,
        set,
        np.array,
        np.ndarray,
        pd.Series,
    ]:
        cmd_str = f"tphot {path} -obj /tmp/hsf/tphot/{tmp_name}.xy"
        # Make list of xy pairs for tphot
        with open(f"/tmp/hsf/tphot/{tmp_name}.xy", "w") as f:
            if grid:
                for a in x:
                    for b in y:
                        f.write(f"{a} {b}\n")
            elif len(x) == len(y):
                for a, b in zip(x, y):
                    f.write(f"{a} {b}\n")
    else:
        return TypeError(
            """x and y not recognized. Their types must match.
                          They can both be None for a full image search,
                          float-likes for a search at those pixel coordinates,
                          or array-likes of matching lengths for a search at those coordinates."""
        )
    if "eadu" not in arg_dict:
        cmd_str += f" -eadu {fits.open(path)['PRIMARY'].header['GAIN']}"
    if tph_force:
        cmd_str += " -force"
    for name, value in arg_dict.items():
        cmd_str += f" -{name}"
        if value is not None:
            cmd_str += f" {value}"
    # Handles BITPIX = -64
    cmd_str += " -cfits"
    if os.path.exists(out) and not force:
        tph = pd.read_csv(out)
    else:
        os.system(cmd_str)
        tph = pd.read_csv(
            out, delim_whitespace=True, skiprows=1, names=constants.TPHOT_COLUMNS
        )
        tph["radial_distortion"] = get_ukirt_systematics(
            fits.open(path)["PRIMARY"].header, x=tph["x"], y=tph["y"]
        )
        if instr_mag:
            tph = tph[tph.peakfit > 0]
            tph["instr_mag"] = -2.5 * np.log10(
                tph["peakfit"] * tph["major"] * tph["minor"]
            )
            tph["instr_mag_err"] = 2.5 / np.log(10) * tph["dpeak"] / tph["peakfit"]
        tph.to_csv(out, index=False)
    return tph


def rolling_fn(
    x, y, fn, dy=None, xcenter=None, window=5, fill_value=np.nan, units="idx"
):
    if len(x) != len(y):
        raise ValueError("x and y are not the same size")
    if (units == "idx" and len(x) < window) or (
        units != "idx" and max(x) - min(x) < window
    ):
        raise ValueError(f"x is too small to fit a window of {window} {units}")
    if units != "idx" and xcenter is None:
        raise ValueError(
            "If not using indices, provide an array of xcenter values to evaluate"
        )
    if dy is None:
        dy = np.ones(len(y))
    # sort x, y, and dy by x
    x, y, dy = np.array(list(zip(*sorted(list(zip(x, y, dy))))))
    if units == "idx":
        res, xcenter = [np.zeros(len(x) - window) for i in range(2)]
        for i in range(len(x) - window):
            try:
                res[i] = fn(y[i : i + window], dy[i : i + window])
            except:
                res[i] = fn(y[i : i + window])
            xcenter[i] = (x[i] + x[i + window]) / 2
        return res, xcenter
    res = np.zeros(len(xcenter))
    for i, xc in enumerate(xcenter):
        idx = np.where((x > xc - window / 2) & (x < xc + window / 2))
        if not len(idx[0]):
            res[i] = fill_value
        else:
            try:
                res[i] = fn(y[idx], dy[idx])
            except:
                res[i] = fn(y[idx])
    return res, xcenter


def linear_model_uncertainty(
    x,
    e_slope=None,
    e_intercept=None,
    e_slope_int=None,
    cov=None,
):
    if e_slope is None and e_intercept is None and cov is None:
        return TypeError("Provide either e_slope and e_intercept or cov")
    if cov is not None:
        if np.array(cov).shape != (2, 2):
            return TypeError("Covariance matrix should be a 2x2 matrix")
        e_slope = np.sqrt(cov[0][0])
        e_intercept = np.sqrt(cov[1][1])
        e_slope_int = cov[0][1]
    return np.sqrt(x**2 * e_slope**2 + e_intercept**2 + 2 * x * e_slope_int)


@logger.catch
def atlas_command(command, destring=True):
    host = os.environ["ATLAS_DOMAIN"]
    user = os.environ["ATLAS_USER"]
    port = os.environ["ATLAS_PORT"]
    ssh_key_filepath = environ.get("SSH_KEY")
    remote_path = environ.get("REMOTE_PATH")
    if isinstance(command, str):
        command_list = [
            command,
        ]
    else:
        command_list = command
    remote = RemoteClient(host, port, user, ssh_key_filepath, remote_path)
    all_results = []
    for c in command_list:
        result = remote.execute_commands([c])
        if destring:
            for i, _ in enumerate(result):
                result[i] = result[i].strip("\n").split()
        all_results.append(result)
    remote.disconnect()
    if isinstance(command, str):
        return all_results[0]
    else:
        return all_results


def dl_atlas_file(file_path, local_path):
    """dl_atlas_file.

    Parameters
    ----------
    file_path :
        file_path
    local_path :
        local_path
    """
    host = os.environ["ATLAS_DOMAIN"]
    user = os.environ["ATLAS_USER"]
    port = os.environ["ATLAS_PORT"]
    ssh_key_filepath = environ.get("SSH_KEY")
    remote_path = environ.get("REMOTE_PATH")
    if isinstance(file_path, str) and isinstance(local_path, str):
        file_path = [
            file_path,
        ]
        local_path = [
            local_path,
        ]
    remote = RemoteClient(host, port, user, ssh_key_filepath, remote_path)
    for f, l in zip(file_path, local_path):
        remote.download_file(f, l)
        for i in range(5):
            if not os.path.exists(f.split("/")[-1]):
                time.sleep(0.2)
            else:
                shutil.move(f.split("/")[-1], l)
                break
    remote.disconnect()


def refcat(
    ra,
    dec,
    rad=None,
    dr=None,
    dd=None,
    all_cols=True,
    mlim=None,
    force=False,
    file_path=None,
):
    os.makedirs("/tmp/hsf/refcat", exist_ok=True)
    default_file_path = f"/tmp/hsf/refcat/{ra}-{dec}-"
    command = f"refcat {ra} {dec}"
    if rad is None and dr is not None and dd is not None:
        default_file_path += f"rect-{dr}-{dd}"
        command += f" -rect {dr},{dd}"
    elif rad is not None and dr is None and dd is None:
        default_file_path += f"rad-{rad}"
        command += f" -rad {rad}"
    else:
        return TypeError("Provide either a value for rad or values for dr and dd")
    if mlim:
        default_file_path += f"-mlim-{mlim}"
        command += f" -mlim {mlim}"
    col_choice = constants.REFCAT_SHORT_COLUMNS
    if all_cols:
        default_file_path += "-all"
        command += " -all"
        col_choice = constants.REFCAT_ALL_COLUMNS
    default_file_path += ".dat"
    if file_path is None:
        file_path = default_file_path
    if not force and os.path.exists(file_path):
        return pd.read_csv(file_path)
    results = atlas_command(command)
    rc = pd.DataFrame(results[1:], columns=col_choice, dtype=float)
    # Need to convert contrib hexadecimal to integers
    if all_cols:
        for i, row in rc.iterrows():
            for filt in ["g", "r", "i", "z"]:
                # contribs in hexadecimal, dtype float in DataFrame messes up integers
                rc.loc[i, f"{filt}contrib"] = int(
                    str(row[f"{filt}contrib"]).split(".")[0], 16
                )

    rc.to_csv(file_path, index=False)
    return rc


def atlas_force(ra, dec, parallel=constants.ATLAS_CORES, force=False, **kwargs):
    for key in kwargs:
        if ";" in key or ";" in str(kwargs[key]):
            raise Exception("Please do not try anything funny on the ATLAS servers.")
    os.makedirs("/tmp/hsf/atlas_force", exist_ok=True)
    file_path = f"/tmp/hsf/atlas_force/{ra}-{dec}.dat"
    if not force and os.path.exists(file_path):
        return pd.read_csv(file_path)
    command = f"force.sh {ra} {dec} parallel={parallel}"
    for key in kwargs:
        command += f" {key}={kwargs[key]}"
    if "m0" in kwargs or "m1" in kwargs:
        command += " dodb=1"
    results = atlas_command(f"{command} | sort -n")
    if results is not None:
        lc = pd.DataFrame(results[1:], columns=constants.FORCE_COLUMNS)
        for key in constants.FORCE_COLUMNS:
            if key not in ("bandpass", "obs"):
                lc[key] = lc[key].astype(float)
        lc.to_csv(file_path, index=False)
        return lc


def epoch_coverage(target_epoch, existing_epochs, force=False, stop_at_today=True):
    target_epoch = np.array(target_epoch)
    if stop_at_today:
        target_epoch[1] = min(target_epoch[1], MJD(today=True))
    existing_epochs = np.array(existing_epochs)
    if len(existing_epochs.shape) == 1:
        existing_epochs = np.array([existing_epochs])
    if None in existing_epochs or force:
        return np.array([target_epoch])
    early = min(existing_epochs[:, 0])
    late = max(existing_epochs[:, 1])
    # Dummy array to allow for appending first and second intervals, with neither being required
    return_epoch = np.array([[None, None]])
    if target_epoch[0] < early:
        return_epoch = np.append(return_epoch, [[target_epoch[0], early]], axis=0)
    if target_epoch[1] > late:
        return_epoch = np.append(return_epoch, [[late, target_epoch[1]]], axis=0)
    return return_epoch[1:]


def fit_images(a_path, b_path, a_mef=0, b_mef=0, rebin=None):
    """
    Use WCS info to crop science (image a) and reference (image b) to their intersection.
    Rotate and adjust WCS to make images match.
    """
    # get appropriate headers and read WCS info
    a_hdu = fits.open(a_path)
    a_header = a_hdu[a_mef].header
    a_wcs = WCS(a_header)
    b_hdu = fits.open(b_path)
    b_header = b_hdu[b_mef].header
    b_wcs = WCS(b_header)

    # create x and y pairs to span four corners of obs and ref images
    a_edge1 = [0, 0, a_header["NAXIS1"], a_header["NAXIS1"]]
    a_edge2 = [0, a_header["NAXIS2"], 0, a_header["NAXIS2"]]
    b_edge1 = [0, 0, b_header["NAXIS1"], b_header["NAXIS1"]]
    b_edge2 = [0, b_header["NAXIS2"], 0, b_header["NAXIS2"]]

    # WCS convention, pixel centers are integers
    # the 1 argument says there's no 0th pixel (from -0.5 to 0.5)
    a_coord_edges = a_wcs.all_pix2world(a_edge1, a_edge2, 1)
    b_coord_edges = b_wcs.all_pix2world(b_edge1, b_edge2, 1)

    # find overlap by comparing RA and Dec at edges
    a_mins = (min(a_coord_edges[0]), min(a_coord_edges[1]))
    a_maxs = (max(a_coord_edges[0]), max(a_coord_edges[1]))
    b_mins = (min(b_coord_edges[0]), min(b_coord_edges[1]))
    b_maxs = (max(b_coord_edges[0]), max(b_coord_edges[1]))
    overlap_mins = (max(a_mins[0], b_mins[0]), max(a_mins[1], b_mins[1]))
    overlap_maxs = (min(a_maxs[0], b_maxs[0]), min(a_maxs[1], b_maxs[1]))

    # overlap mins should be < overlap maxs unless images don't overlap
    if overlap_mins[0] > overlap_maxs[0] or overlap_mins[1] > overlap_maxs[1]:
        raise ValueError("The two images don't seem to overlap.")

    # convert back into pixel space
    a_overlap_coord_edges = a_wcs.all_world2pix(
        [overlap_mins[0], overlap_maxs[0]], [overlap_mins[1], overlap_maxs[1]], 1
    )
    b_overlap_coord_edges = b_wcs.all_world2pix(
        [overlap_mins[0], overlap_maxs[0]], [overlap_mins[1], overlap_maxs[1]], 1
    )

    # ra, dec not necessarily x, y. also, parity means an interval in RA or Dec might result in a pixel interval [max, min]. Straightening things out.
    a_lim = [
        [int(min(a_overlap_coord_edges[0])), int(max(a_overlap_coord_edges[0]))],
        [int(min(a_overlap_coord_edges[1])), int(max(a_overlap_coord_edges[1]))],
    ]
    b_lim = [
        [int(min(b_overlap_coord_edges[0])), int(max(b_overlap_coord_edges[0]))],
        [int(min(b_overlap_coord_edges[1])), int(max(b_overlap_coord_edges[1]))],
    ]

    # make sure pixels stay in image
    if a_lim[0][0] < 0:
        a_lim[0][0] = 0
    if a_lim[0][1] > a_header["NAXIS1"]:
        a_lim[0][1] = a_header["NAXIS1"]
    if a_lim[1][0] < 0:
        a_lim[1][0] = 0
    if a_lim[1][1] > a_header["NAXIS2"]:
        a_lim[1][1] = a_header["NAXIS2"]
    if b_lim[0][0] < 0:
        b_lim[0][0] = 0
    if b_lim[0][1] > b_header["NAXIS1"]:
        b_lim[0][1] = b_header["NAXIS1"]
    if b_lim[1][0] < 0:
        b_lim[1][0] = 0
    if b_lim[1][1] > b_header["NAXIS2"]:
        b_lim[1][1] = b_header["NAXIS2"]

    # crop images based on overlap. Might have issues if the images have intersecting edges. c-like array y,x indexing.
    a_im = a_hdu[a_mef].data[a_lim[1][0] : a_lim[1][1], a_lim[0][0] : a_lim[0][1]]
    b_im = b_hdu[b_mef].data[b_lim[1][0] : b_lim[1][1], b_lim[0][0] : b_lim[0][1]]

    # change CRPIX in b_header to maintain wcs information
    b_header["CRPIX1"] -= b_lim[0][0]
    b_header["CRPIX2"] -= b_lim[1][0]
    b_header["NAXIS1"] = b_lim[0][1] - b_lim[0][0]
    b_header["NAXIS2"] = b_lim[1][1] - b_lim[0][0]

    # photometry based off reference. Need to copy over headers.
    for keyword in ["EXP_TIME"]:
        b_header[keyword] = b_hdu["PRIMARY"].header[keyword]
    for keyword in ["MAGZPT", "MAGZRR", "EXTINCT"]:
        b_header[keyword] = b_hdu[b_mef].header[keyword]
    for keyword in ["AMSTART", "AMEND", "MJD-OBS"]:
        b_header[keyword] = a_hdu["PRIMARY"].header[keyword]

    a_im, a_header = align_wcs(a_im, a_header)
    b_im, b_header = align_wcs(b_im, b_header)

    # Rebin image if using UHS reference images
    if rebin:
        a_im = rebin_image(a_im, rebin)

    a_shape = np.shape(a_im)
    b_shape = np.shape(b_im)
    x_min = min(a_shape[0], b_shape[0])
    y_min = min(a_shape[1], b_shape[1])
    a_im = a_im[0:x_min, 0:y_min]
    b_im = b_im[0:x_min, 0:y_min]

    # write the cropped images to prepare for ISIS. a_wcs no longer valid after rebinning.
    fits.writeto(f"{constants.ISIS_DIR}/new.fits", np.array(a_im), overwrite=True)
    fits.writeto(f"{constants.ISIS_DIR}/ref.fits", b_im, b_header, overwrite=True)

    return b_header


def crop_bright(a_path, b_path, sn_ra, sn_dec, overwrite=True, new_path_end="cropped"):
    """
    Works on the new.fits and ref.fits in ISIS DIR to make sure it's
    looking at the intersection of both images when deciding what to crop
    """
    a_hdu = fits.open(a_path)["PRIMARY"]
    b_hdu = fits.open(b_path)["PRIMARY"]

    b_header = b_hdu.header
    b_data = b_hdu.data
    a_data = a_hdu.data
    w = WCS(b_header)
    mid_pix = np.array([[b_header["NAXIS1"] / 2, b_header["NAXIS2"] / 2]])
    mid_world = w.wcs_pix2world(mid_pix, 1)
    dra = (
        b_header["NAXIS1"]
        * b_header["PIXLSIZE"]
        / 1.6
        / 3600
        / np.cos(mid_world[0][1] * np.pi / 180)
    )
    ddec = b_header["NAXIS2"] * b_header["PIXLSIZE"] / 1.6 / 3600

    # do refcat cone search to get catalog of nearby stars brighter than 10th mag in any filter
    results = refcat(mid_world[0][0], mid_world[0][1], dr=dra, dd=ddec, mlim=10)
    if len(results):
        bad_ra, bad_dec = np.zeros(len(results)), np.zeros(len(results))
        for i, row in enumerate(results):
            bad_ra[i] = row[0]
            bad_dec[i] = row[1]
    else:
        return

    bad_world = np.reshape(np.append(bad_ra, bad_dec), (2, len(results))).T
    bad_pix = w.wcs_world2pix(bad_world, 1)
    bad_x = bad_pix.T[0]
    bad_y = bad_pix.T[1]
    world = np.array([[sn_ra, sn_dec]])
    pix = w.wcs_world2pix(world, 1)
    x = np.append(bad_x, [pix[0][0], 0, b_header["NAXIS1"]])
    y = np.append(bad_y, [pix[0][1], 0, b_header["NAXIS2"]])
    x.sort()
    y.sort()
    x_idx = np.where(x == pix[0][0])[0][0]
    y_idx = np.where(y == pix[0][1])[0][0]
    min_x = x[x_idx - 1]
    max_x = x[x_idx + 1]
    min_y = y[y_idx - 1]
    max_y = y[y_idx + 1]

    if min_x != 0:
        min_x += constants.BRIGHT_PAD
    if max_x != b_header["NAXIS1"]:
        max_x -= constants.BRIGHT_PAD
    if min_y != 0:
        min_y += constants.BRIGHT_PAD
    if max_y != b_header["NAXIS2"]:
        max_y -= constants.BRIGHT_PAD

    if (max_x - min_x) * b_header["NAXIS2"] > (max_y - min_y) * b_header["NAXIS1"]:
        a_crop = a_data[:, int(min_x) : int(max_x)]
        b_crop = b_data[:, int(min_x) : int(max_x)]
        b_header["CRPIX1"] -= min_x
        b_header["NAXIS1"] = max_x - min_x
    else:
        a_crop = a_data[int(min_y) : int(max_y)]
        b_crop = b_data[int(min_y) : int(max_y)]
        b_header["CRPIX2"] -= min_y
        b_header["NAXIS2"] = max_y - min_y
    if overwrite:
        # weird overwrite issue where overwrite=True throws error if there's nothing to overwrite
        fits.writeto(a_path, a_crop, b_header, overwrite=overwrite)
        fits.writeto(b_path, b_crop, b_header, overwrite=overwrite)
    else:
        fits.writeto(a_path + f"_{new_path_end}", a_crop, b_header, overwrite=overwrite)
        fits.writeto(b_path + f"_{new_path_end}", b_crop, b_header, overwrite=overwrite)


def ISIS(
    a_path,
    b_path,
    a_mef="PRIMARY",
    b_mef="PRIMARY",
    nstamps_x=11,
    nstamps_y=5,
    sub_x=1,
    sub_y=1,
    half_mesh_size=11,
    half_stamp_size=15,
    deg_bg=2,
    saturation=3500.0,
    pix_min=50.0,
    min_stamp_center=1000.0,
    ngauss=3,
    deg_gauss1=6,
    deg_gauss2=4,
    deg_gauss3=3,
    sigma_gauss1=0.7,
    sigma_gauss2=2.0,
    sigma_gauss3=4.0,
    deg_spatial=2,
):
    """
    Run ISIS. ISIS is coded to use files from current directory, so need to change.
    """
    os.chdir(constants.ISIS_DIR)
    a_hdu = fits.open(a_path)
    fits.writeto(
        f"{constants.ISIS_DIR}/new.fits",
        a_hdu[a_mef].data,
        a_hdu[a_mef].header,
        overwrite=True,
    )
    b_hdu = fits.open(b_path)
    fits.writeto(
        f"{constants.ISIS_DIR}/ref.fits",
        b_hdu[b_mef].data,
        b_hdu[b_mef].header,
        overwrite=True,
    )
    data_hdu = fits.open(f"{constants.ISIS_DIR}/new.fits")
    avg = np.average(a_hdu[a_mef].data)
    std = np.std(a_hdu[a_mef].data)
    with open(f"{constants.ISIS_DIR}/default_config", "w") as f:
        f.write(
            f"nstamps_x        {nstamps_x}     /*** Number of stamps along X axis ***/\n"
        )
        f.write(
            f"nstamps_y        {nstamps_y}      /*** Number of stamps along Y axis ***/\n"
        )
        f.write(
            f"sub_x             {sub_x}       /*** Number of sub_division of the image along X axis ***/\n"
        )
        f.write(
            f"sub_y             {sub_y}       /*** Number of sub_division of the image along Y axis ***/\n"
        )
        f.write(f"half_mesh_size    {half_mesh_size}     /*** Half kernel size ***/\n")
        f.write(f"half_stamp_size   {half_stamp_size}      /*** Half stamp size ***/\n")
        f.write(
            f"deg_bg            {deg_bg}       /** degree to fit differential background variations **/\n"
        )
        f.write(
            f"saturation       {saturation} /** degree to fit background variations **/\n"
        )
        f.write(
            f"pix_min           {pix_min}    /*** Minimum vaue of the pixels to fit *****/\n"
        )
        f.write(
            f"min_stamp_center  {min_stamp_center}     /*** Minimum value for object to enter kernel fit *****/\n"
        )
        f.write(f"ngauss            {ngauss}       /*** Number of Gaussians ****/\n")
        f.write(
            f"deg_gauss1        {deg_gauss1}       /*** Degree associated with 1st Gaussian ****/\n"
        )
        f.write(
            f"deg_gauss2        {deg_gauss2}       /*** Degree associated with 2nd Gaussian ****/\n"
        )
        f.write(
            f"deg_gauss3        {deg_gauss3}       /*** Degree associated with 3rd Gaussian ****/\n"
        )
        f.write(
            f"sigma_gauss1      {sigma_gauss1}     /*** Sigma of 1st Gaussian ****/\n"
        )
        f.write(
            f"sigma_gauss2      {sigma_gauss2}     /*** Sigma of 2nd Gaussian ****/\n"
        )
        f.write(
            f"sigma_gauss3      {sigma_gauss3}     /*** Sigma of 3rd Gaussian ****/\n"
        )
        f.write(
            f"deg_spatial       {deg_spatial}   /*** Degree of the fit of the spatial variations of the Kernel ****/\n"
        )
    if os.path.exists(f"{constants.ISIS_DIR}/interp_new.fits"):
        os.remove(f"{constants.ISIS_DIR}/interp_new.fits")
    os.system(f"{constants.ISIS_DIR}/sexterp.sh new.fits -r ref.fits")
    os.system(f"{constants.ISIS_DIR}/sub.csh")
    for path in glob.glob(f"{constants.ISIS_DIR}/log_*"):
        os.remove(path)
    for path in glob.glob(f"{constants.ISIS_DIR}/dates-sexterp-*"):
        os.remove(path)


def rotational_subtract(
    path, mef, gal_ra, gal_dec, mask_ra, mask_dec, rot_ra, rot_dec, out_path
):
    # Turn RA and Dec into pixel coordinates
    hdulist = fits.open(path)
    obs = hdulist[mef]
    obs_im = obs.data
    obs_header = obs.header
    # obs_im, obs_header = align_wcs(obs_im, obs_header)

    w = WCS(obs_header)
    gal_world = np.array([[gal_ra, gal_dec]])
    d_world = np.array(
        [
            [
                gal_ra - rot_ra * np.cos(gal_dec * np.pi / 180) / 3600,
                gal_dec - rot_dec / 3600,
            ]
        ]
    )
    gal_pix = w.wcs_world2pix(gal_world, 1)
    d_pix = w.wcs_world2pix(d_world, 1)
    gal_x = gal_pix[0][0]
    gal_y = gal_pix[0][1]
    rot_dx = d_pix[0][0] - gal_x
    rot_dy = d_pix[0][1] - gal_y
    dx, dy = get_rotsub_center(mask_ra, mask_dec, gal_ra, gal_dec, obs_im, obs_header)

    rot = rotate(obs_im, 180)
    shifted = shift(
        rot,
        [
            2 * gal_y - obs_im.shape[0] + dy + rot_dy,
            2 * gal_x - obs_im.shape[1] + dx + rot_dx,
        ],
    )
    for key in ["MJD-OBS", "AMSTART", "AMEND", "EXP_TIME"]:
        obs_header[key] = hdulist["PRIMARY"].header[key]
    rot_sub = obs_im - shifted
    fits.writeto(out_path, rot_sub, obs_header, overwrite=True)
    return dx, dy


def get_rotsub_center(
    sn_ra,
    sn_dec,
    gal_ra,
    gal_dec,
    obs_im,
    obs_header,
    stamp_size=10,
    grid_size=5,
    grid_res=0.1,
    sn_size=5,
):
    obs_im = copy.deepcopy(obs_im)

    sn_x, sn_y = sky2pix(obs_header, ra=sn_ra, dec=sn_dec)
    gal_x, gal_y = sky2pix(obs_header, ra=gal_ra, dec=gal_dec)
    large_size = stamp_size + grid_size
    # mask out sn
    obs_im[
        int(sn_y - sn_size) : int(sn_y + sn_size),
        int(sn_x - sn_size) : int(sn_x + sn_size),
    ] = np.ones((2 * sn_size, 2 * sn_size)) * np.median(obs_im)
    # crop to galaxy field
    obs_im = obs_im[
        int(gal_y - large_size) : int(gal_y + large_size),
        int(gal_x - large_size) : int(gal_x + large_size),
    ]
    rot = rotate(obs_im, 180)
    current_least_sum = np.inf
    for dx, dy in [
        (dx, dy)
        for dx in np.arange(-grid_size, grid_size + grid_res, grid_res)
        for dy in np.arange(-grid_size, grid_size + grid_res, grid_res)
    ]:
        shifted = shift(rot, [dy, dx])
        rot_sub = obs_im - shifted
        rot_sub = rot_sub[grid_size : -grid_size - 1, grid_size : -grid_size - 1]
        least_sum = np.sum(np.abs(rot_sub))
        if least_sum < current_least_sum:
            current_least_sum = least_sum
            min_dx = dx
            min_dy = dy
    shifted = shift(rot, [min_dy, min_dx])
    rot_sub = obs_im - shifted
    return min_dx, min_dy


def get_error_around_sn(
    sn_ra, sn_dec, fname, major, minor, phi, grid_res=2, grid_size=7
):
    header = fits.open(fname)["PRIMARY"].header
    w = WCS(header)
    sn_world = np.array([[sn_ra, sn_dec]])
    sn_pix = w.wcs_world2pix(sn_world, 1)
    sn_x = sn_pix[0][0]
    sn_y = sn_pix[0][1]
    eadu = header["GAIN"]
    with open(f"{fname}_tmp_grid", "w") as f:
        for x in np.arange(
            sn_x - grid_res * np.floor(grid_size / 2),
            sn_x + grid_res * (np.floor(grid_size / 2) + grid_res),
            grid_res,
        ):
            for y in np.arange(
                sn_y - grid_res * np.floor(grid_size / 2),
                sn_y + grid_res * (np.floor(grid_size / 2) + grid_res),
                grid_res,
            ):
                f.write(f"{x} {y}\n")
    os.system(
        f"tphot {fname} -obj {fname}_tmp_grid -major {major} -minor {minor} -phi {phi} -force -eadu {eadu} -out {fname}_tmp_results"
    )
    os.remove(f"{fname}_tmp_grid")
    res = pd.read_csv(
        f"{fname}_tmp_results",
        delim_whitespace=True,
        skiprows=1,
        names=constants.TPHOT_COLUMNS,
    )
    os.remove(f"{fname}_tmp_results")
    return res


def convert_z(z, ra, dec, z0="hel", z1="cmb", mjd=None):
    sc = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
    if z0 == "obs":
        obstime = Time(mjd, format="mjd")
        loc = EarthLocation.from_geodetic(
            lat=constants.MK_LAT * u.deg,
            lon=constants.MK_LON * u.deg,
            height=constants.MK_HEIGHT * u.m,
        )
        heliocorr = sc.radial_velocity_correction(
            "heliocentric", obstime=obstime, location=loc
        )
        vcorr = heliocorr.to(u.km / u.s)
        v_obs = z * constants.C
        v_hel = v_obs + vcorr.value + v_obs * vcorr.value / constants.C
        z_hel = v_hel / constants.C
    if z1 == "hel":
        return z_hel
    if z0 == "hel":
        v_hel = z * constants.C
    cmb_sc = SkyCoord(
        l=constants.CMB_L, b=constants.CMB_B, frame="galactic", unit="deg"
    )
    ang_sep = cmb_sc.separation(sc).value * np.pi / 180
    v_corr = constants.CMB_V * np.cos(ang_sep)
    v_cmb = v_hel + v_corr
    z_cmb = v_cmb / constants.C
    if z1 == "cmb":
        return z_cmb


def mu_lcdm(z_hel, z_cmb, H0, q0=-0.53, j0=None):
    # standard expansion, e.g. eqn 9 in
    # https://iopscience.iop.org/article/10.3847/1538-4357/aae51c
    # https://arxiv.org/pdf/gr-qc/0309109.pdf
    arg = 1 + (1 - q0) / 2 * z_cmb
    if j0 is not None:  # 3rd order
        arg -= (1 - q0 - 3 * q0**2 + j0) / 6 * z_cmb**2
    return 5 * np.log10((1 + z_hel) / (1 + z_cmb) * constants.C * z_cmb / H0 * arg) + 25


def z_lcdm(H0, mu=None, dl=None, q0=-0.53):
    if mu is None and dl is None:
        raise TypeError("mu and dl cannot both be None")
    if mu is not None:
        dl = 10 ** (mu / 5 - 5)
    if dl is not None:
        hdc = H0 * dl / constants.C
        return 1 / (1 - q0) * (-1 + np.sqrt(1 + 2 * hdc * (1 - q0)))


def google_sheets_scat(title="SCAT Targets", sheet_index=2):
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_name(
        f"{constants.HSF_DIR}/hawaii-supernova-flows-1332a295a90b.json", scope
    )
    client = gspread.authorize(creds)
    sheet = client.open(title)

    # Read through list of observations to see if host targets have been observed
    if sheet_index == "all":
        return sheet
    else:
        return sheet.get_worksheet(sheet_index)


def inverse_variance(array, std_err):
    ivar = 1 / np.array(std_err) ** 2
    ave = np.average(array, weights=ivar)
    std = np.sqrt(1 / sum(ivar))
    return ave, std


def sub_type_suffix(sub_type="nosub"):
    sub_type_suffix = ""
    if sub_type in ["refsub", "rotsub"]:
        sub_type_suffix = f"_{sub_type[:3]}"
    return sub_type_suffix


def edit_component_coords(component, ra, dec, name, attrib):
    """
    Edit MSB leaf with new coords/names
    """
    component.attrib["TYPE"] = attrib
    component.find("target").find("targetName").text = name
    coords = component.find("target").find("spherSystem")
    coords.find("c1").text = ra
    coords.find("c2").text = dec
    # return component


def read_ukirt_log(year, month, day, semester="22A", program="H06"):
    """
    Scrape UKIRT project site for observations completed on year, month, day
    """
    if type(month) != str:
        month = str(month).zfill(2)
    if type(day) != str:
        day = str(day).zfill(2)
    url = f"https://ukirt.ifa.hawaii.edu/web/cgi/utprojlog.pl?project=U/{semester}/{program}&utdate={year}-{month}-{day}&noretrv=1"
    payload = {
        "username": os.environ["UKIRT_UNAME"],
        "password": os.environ["UKIRT_PWD"],
        "submit_log_in": "Submit",
        "provider": "staff",
        "show_content": "1",
    }  # pain in the neck. found by logging in and looking at dev tools, network, POST, Request payload
    observed_list = []
    with requests.Session() as session:
        post = session.post(url, data=payload)
        resp = session.get(url)
    soup = BeautifulSoup(resp.text, "html.parser")
    for i, tr in enumerate(soup.find(class_="infobox").select("tr")):
        split = tr.text.split()
        if not len(split):
            continue
        elif split[0][-1] == ".":
            obj = get_TNS_name(split[1])
            bandpasses = split[3]
        elif "done" in split:
            observed_list.append((obj, bandpasses))
    print(f"Found {len(observed_list)} completed observations")
    return observed_list


def read_ukirt_page(semester="22A", program="H06"):
    """
    Scrape UKIRT project site for MSBs
    """
    url = f"https://ukirt.ifa.hawaii.edu/web/cgi/projecthome.pl?project=U%2F{semester}%2F{program}"
    payload = {
        "username": os.environ["UKIRT_UNAME"],
        "password": os.environ["UKIRT_PWD"],
        "submit_log_in": "Submit",
        "provider": "staff",
        "show_content": "1",
    }  # pain in the neck. found by logging in and looking at dev tools, network, POST, Request payload
    with requests.Session() as session:
        post = session.post(url, data=payload)
        resp = session.get(url)
    soup = BeautifulSoup(resp.text, "html.parser")
    return soup


def skycoord_to_hms_dgs(
    sc=None, sc_ra=None, sc_dec=None, add_plus=True, ra_sig=3, dec_sig=2
):
    if sc is not None:
        ra_h, ra_m, ra_s = sc.ra.hms
        dec_d, dec_m, dec_s = sc.dec.dms
        d = sc.dec.value
    elif sc_ra is not None and sc_dec is not None:
        ra_h, ra_m, ra_s = sc_ra.hms
        dec_d, dec_m, dec_s = sc_dec.dms
        d = sc_dec.value

    ra_str = f"{int(ra_h)}:{int(ra_m):02d}:{np.round(ra_s, ra_sig):06.3f}"
    dec_str = f"{int(np.abs(dec_d))}:{int(np.abs(dec_m)):02d}:{np.abs(np.round(dec_s, dec_sig)):05.2f}"
    if d < 0:
        dec_str = f"-{dec_str}"
    if add_plus and d > 0:
        dec_str = f"+{dec_str}"
    return f"{ra_str} {dec_str}"


def ukirt_str_to_datetime(ukirt_str):
    d, t = ukirt_str.split("T")
    dy, dm, dd = d.split("-")
    th, tm, ts = t.split(":")
    return datetime.datetime(
        year=int(dy),
        month=int(dm),
        day=int(dd),
        hour=int(th),
        minute=int(tm),
        second=int(float(ts)),
        tzinfo=pytz.timezone("UTC"),
    )


def generic_format(value, blanks_to_null=True):
    if isinstance(value, bytes):
        temp_val = value.decode("utf-8").strip()
        if temp_val == "" and blanks_to_null:
            return
        else:
            return temp_val
    if isinstance(value, np.ma.core.MaskedConstant):
        return None
    if isinstance(value, MaskedColumn):
        return generic_format(value[0])
    if type(value) in (int, np.int16, np.int32, np.int64):
        return int(value)
    if type(value) in (float, np.float16, np.float32, np.float64):
        if np.isnan(value):
            return None
        return float(value)
    if value == "" and blanks_to_null:
        return None
    return value


def return_kwargs(new_kwargs, old_kwargs):
    for key, val in old_kwargs.items():
        if key not in new_kwargs:
            new_kwargs[key] = val
    return new_kwargs


def pset(array_like, remove_empty=True):
    p = list(
        chain.from_iterable(
            combinations(array_like, r) for r in range(len(array_like) + 1)
        )
    )
    if remove_empty:
        p = [q for q in p if q != []]
    return p


def combo(input_value, d="bandpasses"):
    """combo.

    Parameters
    ----------
    input_value :
        input_value
    d :
        d
    """
    if d == "bandpasses":
        d = constants.BPS_COMBO_DICT
    elif d == "variants":
        d = constants.VARIANT_COMBO_DICT
    if type(input_value) in (int, float):
        retval = []
        for key, val in d.items():
            if input_value >= val:
                retval.append(key)
                input_value -= val
    else:
        retval = 0
        for key in input_value:
            retval += d[key]
    return retval


def mollweide(long, lat, meridian=180, lat_thresh=0.0001 * np.pi / 180):
    rad = np.pi / 180
    try:
        arr_len = len(lat)
    except TypeError:
        arr_len = 1
        long = np.array([long])
        lat = np.array([lat])
    theta = np.ones(arr_len) * np.inf
    new_theta = lat * rad
    while all(np.abs(list(new_theta - theta)) > lat_thresh):
        theta = new_theta
        new_theta = theta - (
            2 * theta + np.sin(2 * theta) - np.pi * np.sin(lat * rad)
        ) / (4 * np.cos(theta) ** 2)
    x = 2**1.5 / np.pi * (long * rad - meridian * rad) * np.cos(theta)
    y = 2**0.5 * np.sin(theta)
    return x, y


def plot_mollweide(meridian=180, lat_thresh=0.0001 * np.pi / 180, **kwargs):
    fig, ax = plt.subplots(figsize=(12, 6))
    for grid_x, grid_y in zip(np.linspace(0, 360, 19), np.linspace(-90, 90, 19)):

        lat_x, lat_y = mollweide(
            np.ones(181) * grid_x, np.linspace(-90, 90, 181), meridian, lat_thresh
        )
        long_x, long_y = mollweide(
            np.linspace(0, 360, 361), np.ones(361) * grid_y, meridian, lat_thresh
        )
        if grid_x == 0 or grid_x == 360:
            alpha = 1
        else:
            alpha = 0.3
        ax.plot(lat_x, lat_y, color="black", alpha=alpha)
        ax.plot(long_x, long_y, color="black", alpha=alpha)
    plt.xticks([])
    plt.yticks([])
    return fig, ax


def get_extrema_idx(array, which="both"):
    if len(array) <= 2:
        return np.array([])
    extr = np.where((array[1:-1] - array[:-2]) * (array[1:-1] - array[2:]) >= 0)[0] + 1
    if which == "both":
        return extr
    slope_left = array[extr] - array[extr - 1]
    slope_right = array[extr + 1] - array[extr]
    if which == "max":
        return extr[np.where(slope_left - slope_right > 0)[0]]
    elif which == "min":
        return extr[np.where(slope_left - slope_right < 0)[0]]


def twod_block_stack(m1, m2):
    """twod_block_stack.
    combine two square matrices
    [[m1, 0], [0, m2]]

    Parameters
    ----------
    m1 :
        m1 twod square
    m2 :
        m2 twod square
    """
    assert m1.shape[0] == m1.shape[1]
    assert m2.shape[0] == m2.shape[1]

    m3 = np.zeros([len(m1) + len(m2)] * 2, dtype=np.float64)
    m3[: len(m1), : len(m1)] = m1
    m3[len(m1) :, len(m1) :] = m2

    return m3


def synthetic_flux(wl, flux, filter_wl, filter_trans):
    trans = np.interp(wl, filter_wl, filter_trans)
    return flux * wl * trans


def odr_polyfit(x, y, e_x, e_y, degree=2):
    import scipy.odr as odr

    def poly_func(p, x):
        return sum(p[i] * x ** (degree - i) for i in range(degree + 1))

    init_guess = np.polyfit(x, y, degree)

    poly_model = odr.Model(poly_func)
    data = odr.RealData(x, y, sx=e_x, sy=e_y)
    return odr.ODR(data, poly_model, beta0=init_guess).run()


def chauvenet(arr):
    from scipy.special import erfc

    N = len(arr)
    criterion = 1 / (2 * N)
    d = np.abs(arr - np.average(arr)) / np.std(arr)
    d /= 2.0**0.5
    prob = erfc(d)
    return prob >= criterion


def get_unity_outl_mask(unity_path):
    unity = pd.read_csv(unity_path, delimiter="\s+", index_col="param_names")
    outl = unity["mean"][unity.index.str.startswith("outl_loglike_by_SN")]
    inl = unity["mean"][unity.index.str.startswith("inl_loglike_by_SN")]
    likely_outl_idx = np.where(inl.values - outl.values < 0)
    return inl.values - outl.values > 0


def mahalanobis_distances(arr, origin=False, cov=None):
    # arr should be an array of coordinates
    # e.g. xy coords np.array([[x0, y0], [x1, y1], ...[xN, yN]])
    x = np.array(arr)
    if origin:
        mu = np.zeros(len(x[0]))
    else:
        mu = np.average(x, axis=0)
    if cov is None:
        cov = np.cov(x.T)
    elif cov == "bootstrap":
        cov = bootstrap_covariance(x.T)
    dists = np.zeros(len(x))
    for i in range(len(x)):
        dists[i] = np.sqrt(
            np.matmul(np.matmul(x[i] - mu, np.linalg.inv(cov)), (x[i] - mu).T)
        )
    return dists


def bootstrap_fn(arrs, fns=[], iter_num=5000):
    # feed in (m, N) array to be passed to fn as fn(m[0], m[1], ...)
    ret_dict = {}
    N = len(arrs[0])
    for fn in fns:
        ret_dict[fn.__name__] = np.zeros(iter_num)
    for i in range(iter_num):
        idx = np.random.randint(N, size=(N))
        for fn in fns:
            ret_dict[fn.__name__][i] = fn(*[arr[idx] for arr in arrs])
    return ret_dict


def collated_bootstrap_fn(arrs_dict, fns=[], iter_num=5000):
    # feed in (k (m, N)) array to be passed to fn and return
    # ret_dict{k_0: fn(m[0], m[1], ... m[N]), k_1: ...}
    ret_dict = dict()
    for key in arrs_dict:
        ret_dict[key] = {}
        for fn in fns:
            ret_dict[key][fn.__name__] = np.zeros(iter_num)
    N_arr = [len(arrs[0]) for arrs in arrs_dict.values()]
    if np.std(N_arr) != 0:
        raise ValueError("Arrays should be the same size")
    N = N_arr[0]
    for i in range(iter_num):
        idx = np.random.randint(N, size=(N))
        for key in arrs_dict:
            for fn in fns:
                ret_dict[key][fn.__name__][i] = fn(
                    *[arr[idx] for arr in arrs_dict[key]]
                )
    return ret_dict


def bootstrap_covariance(arr, iter_num=5000):
    cov_bs = np.zeros((iter_num, len(arr), len(arr)))
    for i in range(iter_num):
        idx = np.random.randint(len(arr[0]), size=len(arr[0]))
        cov_bs[i] = np.cov(np.array([param[idx] for param in arr]))
    return np.average(cov_bs, axis=0)


def RMS(arr, *args):
    return np.sqrt(np.average(arr**2))


def WRMS(arr, err, *args):
    return np.sqrt(np.average(arr**2, weights=1 / err**2))


def NMAD(arr, *args):
    return 1.486 * np.median(np.abs(arr - np.median(arr)))


def STD(arr, *args):
    return np.std(arr)


def sigma_int(arr, err, precision=5, *args):
    def neg_loglikelihood(int_err, arr, err):
        return 0.5 * sum(np.log(err**2 + int_err**2) + arr**2 / (err**2 + int_err**2))

    guess = 0.5
    for i in range(1, precision + 1):
        int_arr = [guess + (j - 5) * 10 ** (-i) for j in range(10)]
        guess = np.round(
            int_arr[np.argmin([neg_loglikelihood(i, arr, err) for i in int_arr])], i + 1
        )
    return np.abs(guess)


def stan(
    input_dict,
    num_samples=5000,
    jobs=7,
    data="",
    transformed_data="",
    params="",
    transformed_params="",
    model="",
    generated_quantities="",
    pickle_path="",
):
    import pickle

    import stan

    code = ""
    for name, block in zip(
        (
            "data",
            "transformed data",
            "parameters",
            "transformed parameters",
            "model",
            "generated quantities",
        ),
        (
            data,
            transformed_data,
            params,
            transformed_params,
            model,
            generated_quantities,
        ),
    ):
        if block != "":
            code += f"{name} \u007b {block} \u007d\n"
    posterior = stan.build(
        code, data=input_dict
    )  # extra_compile_args=["-pthread", "-DSTAN_THREADS"]
    pickle.dump((posterior, code), open(pickle_path, "wb"))
    fit = posterior.sample(
        num_samples=num_samples,
        num_chains=jobs,
    )
    return fit


def stan_line_fit(
    x,
    y,
    dx,
    dy,
    iters=5000,
    jobs=7,
    outl_frac_prior_lnmean=None,
    outl_frac_prior_lnwidth=None,
    priors={},
):
    # Something is still funky, it doesn't do as well as ODR in tests.
    if len(x) != len(y):
        raise ValueError("x and y are not the same length")
    N = len(x)
    if dx is None:
        dx = np.ones(len(x)) / 1e3
    if dy is None:
        dy = np.ones(len(y)) / 1e3
    input_dict = dict(
        N=N,
        x=x,
        y=y,
        x_err=dx,
        y_err=dy,
    )
    pickle_path = f"{constants.STATIC_DIR}/stan_line.pickle"
    data = """
    int<lower=1> N;
    vector[N] x;
    vector[N] y;
    vector<lower=0>[N] x_err;
    vector<lower=0>[N] y_err;
    """
    params = """
    // For uniform distribution in angle rather than slope
    real slope;
    real intercept;
    vector[N] x_t; // true x values
    real<lower=0> sigma_x;
    real<lower=0> sigma_y;
    """
    transformed_params = ""
    model = """
    target += -1.5*log(1+square(slope));
    target += -log(sigma_x);
    target += -log(sigma_y);
    x ~ normal(x_t, sqrt(square(x_err)+square(sigma_x)));
    y ~ normal(intercept+slope*x_t, sqrt(square(y_err)+square(sigma_y)));
    """
    if outl_frac_prior_lnmean is not None and outl_frac_prior_lnwidth is not None:
        pickle_path = f"{constants.STATIC_DIR}/stan_line_mix_model.pickle"
        input_dict["outl_frac_prior_lnmean"] = outl_frac_prior_lnmean
        input_dict["outl_frac_prior_lnwidth"] = outl_frac_prior_lnwidth
        data += """
        real outl_frac_prior_lnmean;
        real outl_frac_prior_lnwidth;
        """
        params += """
        real <lower=0.001, upper=0.1> outl_frac;
        real <lower=1, upper=10> outl_x_err;
        real <lower=1, upper=10> outl_y_err;
        """
        transformed_params = """
        vector [N] outl_loglike;
        vector [N] inl_loglike;
        for (i in 1:N) {
            outl_loglike[i] = log(outl_frac)
                + normal_lpdf(x[i] | x_t[i], outl_x_err)
                + normal_lpdf(y[i] | intercept + slope*x_t[i], outl_y_err);
            inl_loglike[i] = log(1-outl_frac)
                + normal_lpdf(x[i] |
                    x_t[i], sqrt(square(x_err[i])+square(sigma_x)))
                + normal_lpdf(y[i] |
                    intercept + slope*x_t[i], sqrt(square(y_err[i])+square(sigma_y)));
        }
        """
        model = """
        target += -1.5*log(1+square(slope));
        for (i in 1:N) {
            target += log_sum_exp(outl_loglike[i], inl_loglike[i]);
        }
        outl_frac ~ lognormal(outl_frac_prior_lnmean, outl_frac_prior_lnwidth);
        outl_x_err ~ normal(1, 1);
        outl_y_err ~ normal(2, 2);
        """
    for kw in priors:
        if kw not in ("slope", "intercept"):
            continue
        if len(priors[kw]) == 3:
            if priors[kw][0] == "N":
                distr = "normal"
            elif priors[kw][0] == "U":
                distr = "uniform"
            model += f"""
            {kw} ~ {distr}({priors[kw][1]}, {priors[kw][2]});
            """
        elif len(priors[kw]) == 2:
            model += f"""
            {kw} ~ normal({priors[kw][0]}, {priors[kw][1]});
            """
        pickle_path = pickle_path.replace(".pickle", ".custom_priors.pickle")
    return stan(
        input_dict,
        iters=iters,
        jobs=jobs,
        data=data,
        params=params,
        transformed_params=transformed_params,
        model=model,
        pickle_path=pickle_path,
    )


def get_log_dist(mu, z, H0=72):
    return np.log10(10 ** ((mu_lcdm(z, z, H0) - mu) / 5))


def get_log_dist_err(mu, e_mu, z, e_z, H0=72, q0=-0.53):
    return np.sqrt(
        (e_z / (z * np.log(10))) ** 2
        + (e_z * 5 * (1 - q0) / (2 * np.log(10))) ** 2
        # + 2 * (5 * (1 - q0) / (2 * z * np.log(10) ** 2)) * e_z ** 2
        + (e_mu / 5) ** 2
    )


def get_log_dist_err_2(mu, e_mu, z, e_z, H0=72, n_sample=5000):
    dz = np.random.normal(0, e_z, size=n_sample)
    dmu = np.random.normal(0, e_mu, size=n_sample)
    return np.std(np.log10(10 ** ((mu_lcdm(z + dz, z + dz, H0) - (mu + dmu)) / 5)))


# TNS functions from help page
def TNS_set_bot_tns_marker():
    tns_marker = (
        'tns_marker{"tns_id": "'
        + str(constants.TNS_BOT_ID)
        + '", "type": "bot", "name": "'
        + constants.TNS_BOT_NAME
        + '"}'
    )
    return tns_marker


def TNS_format_to_json(source):
    parsed = json.loads(source, object_pairs_hook=OrderedDict)
    result = json.dumps(parsed, indent=4)
    return result


def TNS_is_string_json(string):
    try:
        json_object = json.loads(string)
    except Exception:
        return False
    return json_object


def TNS_print_status_code(response):
    json_string = TNS_is_string_json(response.text)
    if json_string != False:
        print(json_string)
        print(
            "status code ---> [ "
            + str(json_string["id_code"])
            + " - '"
            + json_string["id_message"]
            + "' ]\n"
        )
    else:
        status_code = response.status_code
        if status_code == 200:
            status_msg = "OK"
        elif status_code in constants.TNS_EXT_HTTP_ERRORS:
            status_msg = constants.ERR_MSG[ext_http_errors.index(status_code)]
        else:
            status_msg = "Undocumented error"
        print("status code ---> [ " + str(status_code) + " - '" + status_msg + "' ]\n")


def TNS_search(search_obj):
    search_url = constants.TNS_URL_API + "/search"
    tns_marker = TNS_set_bot_tns_marker()
    headers = {"User-Agent": tns_marker}
    json_file = OrderedDict(search_obj)
    search_data = {
        "api_key": str(os.environ["TNS_API_KEY"]),
        "data": json.dumps(json_file),
    }
    return requests.post(search_url, headers=headers, data=search_data)


def TNS_get(get_obj):
    get_url = constants.TNS_URL_API + "/object"
    tns_marker = TNS_set_bot_tns_marker()
    headers = {"User-Agent": tns_marker}
    json_file = OrderedDict(get_obj)
    get_data = {"api_key": os.environ["TNS_API_KEY"], "data": json.dumps(json_file)}
    return requests.post(get_url, headers=headers, data=get_data)


def TNS_get_file(file_url):
    filename = os.path.basename(file_url)
    tns_marker = TNS_set_bot_tns_marker()
    headers = {"User-Agent": tns_marker}
    api_data = {"api_key": str(os.environ["TNS_API_KEY"])}
    print("Downloading file '" + filename + "' from the TNS...\n")
    response = requests.post(file_url, headers=headers, data=api_data, stream=True)
    TNS_print_status_code(response)
    path = os.path.join(constants.MEDIA_DIR, filename)
    if response.status_code == 200:
        with open(path, "wb") as f:
            for chunk in response:
                f.write(chunk)
        print("File was successfully downloaded.\n")
    else:
        print("File was not downloaded.\n")


def TNS_print_response(response, json_file, counter):
    response_code = (
        str(response.status_code) if json_file == False else str(json_file["id_code"])
    )
    stats = (
        "Attempt #"
        + str(counter)
        + "| return code: "
        + response_code
        + " | Total Rate-Limit: "
        + str(response.headers.get("x-rate-limit-limit"))
        + " | Remaining: "
        + str(response.headers.get("x-rate-limit-remaining"))
        + " | Reset: "
        + str(response.headers.get("x-rate-limit-reset"))
    )
    if response.headers.get("x-cone-rate-limit-limit") != None:
        stats += (
            " || Cone Rate-Limit: "
            + str(response.headers.get("x-cone-rate-limit-limit"))
            + " | Cone Remaining: "
            + str(response.headers.get("x-cone-rate-limit-remaining"))
            + " | Cone Reset: "
            + str(response.headers.get("x-cone-rate-limit-reset"))
        )
    print(stats)


def TNS_get_reset_time(response):
    # If any of the '...-remaining' values is zero, return the reset time
    for name in response.headers:
        value = response.headers.get(name)
        if name.endswith("-remaining") and (value == "0" or value == "Exceeded"):
            return int(response.headers.get(name.replace("remaining", "reset")))
    return None


def TNS_query(query, obj):
    counter = 0
    while True:
        counter = counter + 1
        if query == "search":
            response = TNS_search(obj)
        elif query == "get":
            response = TNS_get(obj)
        elif query == "classification":
            response = get_TNS_classification(obj)
        json_file = TNS_is_string_json(response.text)
        TNS_print_response(response, json_file, counter)
        # Checking if rate-limit reached (...-remaining = 0)
        reset = TNS_get_reset_time(response)
        # A general verification if not some error
        if response.status_code == 200:
            if reset is not None:
                # Sleeping for reset + 1 sec
                print("Sleep for " + str(reset + 1) + " sec")
                time.sleep(reset + 1)
                # Can continue to submit requests...
                print("Continue to submit requests...")
                counter = counter + 1
                if query == "search":
                    response = TNS_search(obj)
                elif query == "get":
                    response = TNS_get(obj)
                json_file = TNS_is_string_json(response.text)
                TNS_print_response(response, json_file, counter)
                break
            break
        elif response.status_code == 429:
            reset = TNS_get_reset_time(response)
            print("Sleep for " + str(reset + 1) + " sec")
            time.sleep(reset + 1)
        else:
            TNS_print_status_code(response)
            break
    if query == "classification":
        soup = BeautifulSoup(response.text, "html.parser")
        tab = soup.find_all("table", class_="atreps-results-table")[-1]
        dates = tab.find_all("td", class_="cell-time_received")[::-1]
        types = tab.find_all("td", class_="cell-type")[::-1]
        groups = tab.find_all("td", class_="cell-source_group_name")[::-1]
        classifiers = tab.find_all("td", class_="cell-classifier_name")[::-1]
        return dates, types, groups, classifiers
    else:
        return json.loads(TNS_format_to_json(response.text))["data"]["reply"]


def get_TNS_classification(TNS_name):
    day = datetime.datetime.utcnow()
    old = int(TNS_name[:2]) + 2000 > day.year
    url = f"https://www.wis-tns.org/object/20{TNS_name}/"
    if old:
        url = f"https://www.wis-tns.org/object/19{TNS_name}/"
    tns_marker = TNS_set_bot_tns_marker()
    headers = {"User-Agent": tns_marker}
    api_data = {"api_key": str(os.environ["TNS_API_KEY"])}
    return requests.get(url, headers=headers, data=api_data)


class RemoteClient:
    """
    Client to interact with a remote host via SSH & SCP
    """

    def __init__(self, host, port, user, ssh_key_filepath, remote_path):
        self.host = host
        self.port = port
        self.user = user
        self.ssh_key_filepath = ssh_key_filepath
        self.remote_path = remote_path
        self.client = None
        self.scp = None
        self.conn = None
        # self._upload_ssh_key()

    @logger.catch
    def _get_ssh_key(self):
        """
        Fetch locally stored SSH key.
        """
        try:
            self.ssh_key = RSAKey.from_private_key_file(self.ssh_key_filepath)
            logger.info(f"Found SSH key at self {self.ssh_key_filepath}")
        except SSHException as error:
            logger.error(error)
        return self.ssh_key

    @logger.catch
    def _upload_ssh_key(self):
        try:
            system(
                f"ssh-copy-id -i {self.ssh_key_filepath} {self.user}@{self.host}>/dev/null 2>&1"
            )
            system(
                f"ssh-copy-id -i {self.ssh_key_filepath}.pub {self.user}@{self.host}>/dev/null 2>&1"
            )
            logger.info(f"{self.ssh_key_filepath} uploaded to {self.host}")
        except FileNotFoundError as error:
            logger.error(error)

    @logger.catch
    def _connect(self):
        """Open connection to remote host."""
        if self.conn is None:
            try:
                self.client = SSHClient()
                self.client.load_system_host_keys()
                self.client.set_missing_host_key_policy(AutoAddPolicy())
                self.client.connect(
                    self.host,
                    self.port,
                    username=self.user,
                    key_filename=self.ssh_key_filepath,
                    look_for_keys=True,
                    timeout=100,
                )
                self.scp = SCPClient(self.client.get_transport())
            except AuthenticationException as error:
                logger.error(
                    f"Authentication failed: \
                    did you remember to create an SSH key? {error}"
                )
                raise error
        return self.client

    def disconnect(self):
        """
        Close ssh connection.
        """
        if self.client:
            self.client.close()
        if self.scp:
            self.scp.close()

    @logger.catch
    def execute_commands(self, commands):
        """
        Execute list of command (str).

        :param commands: List of UNIX command as strings.
        :type commands: str
        """
        if self.conn is None:
            self.conn = self._connect()
        for command in commands:
            stdin, stdout, stderr = self.client.exec_command(command)
            stdout.channel.recv_exit_status()
            response = stdout.readlines()
            for line in response:
                logger.info(f"INPUT: {command} | OUTPUT: {line}")
        return response

    def bulk_upload(self, files):
        """
        Upload multiple files to a remote directory.

        :param files: List of local files to be uploaded.
        :type files: List[str]
        """
        try:
            if self.conn is None:
                self.conn = self._connect()
            self.scp.put(files, remote_path=self.remote_path)
            logger.info(
                f"Finished uploading {len(files)} files to {self.remote_path} on {self.host}"
            )
        except SCPException as e:
            raise e

    def download_file(self, file_name, local_path):
        """
        Download file from remote host.
        """
        if self.conn is None:
            self.conn = self._connect()
        self.scp.get(file_name)


"""
NED lookups
"""


def ned_aliases(object_name):
    encoded_name = urllib.parse.quote_plus(
        object_name
    )  # need to encode special characters in object name

    NED_object_lookup = "http://ned.ipac.caltech.edu/srs/ObjectLookup?"
    param_dict = {
        "name": {"v": encoded_name},
        "aliases": {"v": True},
    }
    object_name_packet = "json=" + json.dumps(param_dict, separators=(",", ":"))
    NED_object_lookup_response = requests.post(
        NED_object_lookup, data=object_name_packet
    )
    if NED_object_lookup_response.status_code == 200:
        ned_object_basic_info = json.loads(NED_object_lookup_response.content)
        return ned_object_basic_info


def ned_cone_search(coord_ra, coord_dec, sr_deg=0.2 / 60, columns="most"):
    """
    for valid colnames see
    https://ned.ipac.caltech.edu/tap/sync?QUERY=SELECT+*+FROM+TAP_SCHEMA.columns+WHERE+table_name=%27NEDTAP.objdir%27&REQUEST=doQuery&LANG=ADQL&FORMAT=text
    """
    if isinstance(columns, set):
        columns = list(columns)
    if isinstance(columns, list):
        columns = ",".join(columns)
    elif columns == "all":
        columns = "*"
    elif columns == "most":
        columns = "objid,prefname,pretype,ra,dec,zflag,z,zunc,n_crosref,n_notes,n_gphot,n_posd,n_zdf,n_assoc,n_ddf,n_images,n_spectra,n_dist,n_class"
    ned_tab = "https://ned.ipac.caltech.edu/tap"
    table_name = "objdir"
    coord_sys = "J2000"
    cone = (
        "CONTAINS(POINT('"
        + str(coord_sys)
        + "', ra, dec),CIRCLE('"
        + str(coord_sys)
        + "',"
        + str(coord_ra)
        + ","
        + str(coord_dec)
        + ","
        + str(sr_deg)
        + " ))=1"
    )
    query = "SELECT " + columns + " FROM " + table_name + " WHERE " + cone
    ned_TAP = vo.dal.TAPService(ned_tab)
    return ned_TAP.search(query).table.to_pandas()


def sed_plot():
    """
    example from https://ned.ipac.caltech.edu/docs/Notebooks/Workshop_20180603.ipynb
    """
    object_names = ("arp220", "3c273", "BL Lac")
    color = ("r", "b", "g")

    NED_sed = "http://vo.ned.ipac.caltech.edu/services/accessSED?"

    plt.figure(figsize=(15, 12))
    plt.rcParams.update({"font.size": 22})

    i = 0
    for name in object_names:
        paramters = {"REQUEST": "getData", "TARGETNAME": name}
        NED_sed_response = requests.get(NED_sed, params=paramters)
        if NED_sed_response.status_code == 200:
            ned_data_table = Table.read(io.BytesIO(NED_sed_response.content))
            x = ned_data_table["DataSpectralValue"]
            y = ned_data_table["DataFluxValue"]
            plt.plot(x, y, color[i])
            j = -i * 15
            plt.annotate(
                name,
                xy=(x[j], y[j]),
                xytext=(x[j] * 100, y[j] * 100),
                arrowprops=dict(facecolor=color[i], shrink=0.05),
                color=color[i],
                fontsize=28,
            )
            i += 1
    plt.xlim(1e6, 1e20)
    plt.ylim(1e-9, 1e3)
    plt.yscale("log")
    plt.xscale("log")
    plt.show()


"""
Simbad lookups
"""


def simbad_set_output_options(session):
    url = "https://simbad.cds.unistra.fr/simbad/sim-fout"
    payload = {
        "httpRadio": "Get",
        "output.format": "HTML",
        "output.max": "1000",
        "list.idsel": "on",
        "list.idopt": "FIRST",
        "list.idcat": "",
        "list.otypesel": "on",
        "otypedisp": "3",
        "otypedispall": "on",
        "list.coo1": "on",
        "frame1": "ICRS",
        "epoch1": "J2000",
        "coodisp1": "d2",
        "frame2": "FK5",
        "epoch2": "J2000",
        "equi2": "2000",
        "coodisp2": "s2",
        "frame3": "FK4",
        "epoch3": "B1950",
        "equi3": "1950",
        "coodisp3": "s2",
        "frame4": "Gal",
        "epoch4": "J2000",
        "equi4": "2000",
        "coodisp4": "d2",
        "obj.pmsel": "on",
        "obj.plxsel": "on",
        "obj.rvsel": "on",
        # "list.rvsel": "on",
        "rvRedshift": "on",
        "obj.fluxsel": "on",
        "obj.spsel": "on",
        "obj.mtsel": "on",
        # "list.mtsel": "on",
        "obj.sizesel": "on",
        # "list.sizesel": "on",
        "obj.hierarchysel": "on",
        "obj.bibsel": "on",
        "bibyear1": "1850",
        "bibyear2": "$currentYear",
        "bibjnls": "",
        "bibdisplay": "bibnum",
        "bibcom": "on",
        "bibtabular": "on",
        "obj.notesel": "on",
        "notedisplay": "A",
        "obj.messel": "on",
        "list.messel": "on",
        "list.mescat": "Diameter Distance Velocities",
        "mesdisplay": "N",
        "obj.extsel": "on",
        "save": "SAVE",
    }
    session.post(url, data=payload)


def simbad_cone_search(ra, dec, sr_arcmin=2):
    """simbad_cone_search.

    2 total calls to simbad.
    One to set output options, one to cone search.
    Can investigate keeping session alive later to improve
    call efficiency

    Parameters
    ----------
    ra :
        ra
    dec :
        dec
    sr_arcmin :
        sr_arcmin
    """
    url = (
        "https://simbad.cds.unistra.fr/simbad/sim-coo?"
        + f"Coord={ra}+{dec}+"
        + "&CooFrame=FK5&CooEpoch=2000&CooEqui=2000&CooDefinedFrames=none"
        + f"&Radius={sr_arcmin}"
        + "&Radius.unit=arcmin&submit=submit+query&CoordList="
    )
    with requests.Session() as session:
        simbad_set_output_options(session)
        resp = session.get(url)
    soup = BeautifulSoup(resp.text, "html.parser")
    if soup.find("div", class_="simbaderr"):
        return
    name_parent = soup.find("td", id="basic_data")
    if name_parent:
        cols = ["main_id", "objid"]
        data = [name_parent.find("b").text.strip("\n ")]  # main_id
        data.append(soup.find_all("input", id="Ident")[-1].get("value")[1:])  # objid
        form = soup.find("form", action="sim-id#lab_meas")
        if form:
            tables = form.find_all("b")
            for i in range(len(tables) // 2):
                table_type = tables[2 * i].text.strip("\n ")
                if table_type in ("velocities", "distance", "diameter"):
                    cols.append(tables[2 * i].text.strip("\n "))
                    data.append(int(tables[2 * i + 1].text.strip("\n ")))
        for t in ("velocities", "distance", "diameter"):
            if t not in cols:
                cols.append(t)
                data.append(0)
        return pd.DataFrame(data=[data], columns=cols)
    d = {
        "objid": [],  # two values from identifiers field
        "main_id": [],
        # "galdim_majaxis": [],  # angular size has three values
        # "galdim_minaxis": [],
        # "galdim_angle": [],
    }
    t = soup.find_all("table")[-1]
    for row in t.find("tbody").find_all("tr"):
        for td, th in zip(row.find_all("td"), t.find_all("th")):
            # Parse th to get key
            if th.a:
                key = th.a.text
            else:
                key = "_".join(
                    [i.strip("\n ") for i in th.contents if isinstance(i, str)]
                )
            key = key.lower().replace(" ", "_").replace(".", "").strip("#_")

            if key in ("dist(asec)", "n", "otype"):
                continue
            if key.startswith("icrs_(j2000)"):
                key = key.split("_")[2]

            # parse td to get value
            if td.a:
                val = td.a.text
            else:
                val = td.text.strip("\n ")
            if val == "~":
                val = None

            if key == "identifier":
                d["objid"].append(
                    int(td.a.get("href").split("Ident=%40")[1].split("&Name")[0])
                )
                d["main_id"].append(val)
                continue
            elif key == "all_types":
                val = val.split(",")
            elif key in ("ra", "dec", "redshift") and val:
                val = float(val)
            elif key in ("diameter", "distance", "velocities"):
                val = int(val)

            # splitting majaxis, minaxis, phi into 3
            if key == "angular_size":
                majaxis, minaxis, phi = [
                    None if i == "~" else float(i) for i in val.split()
                ]
                d["galdim_majaxis"].append(majaxis)
                d["galdim_minaxis"].append(minaxis)
                d["galdim_angle"].append(phi)
            # write to d
            else:
                if key not in d:
                    d[key] = []
                d[key].append(val)

    res = pd.DataFrame(data=d)
    return res


def simbad_object_search(objid, name):
    encoded_name = urllib.parse.quote_plus(name)
    url = (
        "https://simbad.cds.unistra.fr/simbad/sim-id?"
        f"Ident=%40{objid}&Name={encoded_name}"
        "&submit=display+all+measurements#lab_meas"
    )
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, "html.parser")
    return soup


"""
decorators
"""


# def coord_input(func):
#    def wrapper_decorator(*args, **kwargs):
#        ra, dec, coord, size, rd_coord = [None for i in range(5)]
#
#        # arg case
#        if len(args) == 1:  # Either coord or error
#            if isinstance(args[0], SkyCoord):
#                coord = args[0]
#            else:
#                raise SyntaxError(f"Cannot interpret {args[0]} as a coordinate")
#        elif len(args) == 2:  # Either coord/size, ra/dec, or error
#            for i, j in ((0, 1), (1, 0)):
#                if isinstance(args[i], SkyCoord):
#                    coord = args[i]
#                    size = args[j]
#            if not coord:
#                ra, dec = args
#        elif len(args) == 3:  # Either ra/dec/size or error
#            ra, dec, size = args
#
#        # kwarg case
#        if isinstance(kwargs.get("coord"), SkyCoord):
#            coord = kwargs["coord"]
#        if kwargs.get("ra") is not None and kwargs.get("dec") is not None:
#            ra, dec = kwargs["ra"], kwargs["dec"]
#
#        if isinstance(ra, str):
#            ra = HMS2deg(ra=ra)
#        if isinstance(dec, str):
#            dec = HMS2deg(dec=dec)
#        if ra and dec:
#            rd_coord = SkyCoord(ra, dec, unit="deg")
#
#        if coord and rd_coord and coord != rd_coord:
#            raise ValueError(
#                f"Given RA/Dec ({ra}, {dec}) do not match given coordinate {coord}"
#            )
#        elif rd_coord:
#            coord = rd_coord
#        elif not coord and not rd_coord:
#            raise SyntaxError(
#                f"Received RA/Dec ({ra}, {dec}) and coord {coord}. Need more info"
#            )
#        if size:
#            return func(coord=coord, size=size)
#        else:
#            return func(coord=coord)
#
#    return wrapper_decorator


def compare_iterables(arr1, arr2, key_str=""):
    diff = False
    if isinstance(arr1, str) and isinstance(arr2, str):
        if arr1 != arr2:
            print(key_str, arr1, arr2)
            diff = True
    elif isinstance(arr1, dict) and isinstance(arr1, dict):
        for key in arr1:
            compare_iterables(arr1[key], arr2[key], key_str=key_str + " " + key)
    elif (
        hasattr(arr1, "__iter__")
        and hasattr(arr2, "__iter__")
        and len(arr1) == len(arr2)
    ):
        for idx, (element_1, element_2) in enumerate(zip(arr1, arr2)):
            compare_iterables(element_1, element_2, key_str=key_str + " " + str(idx))
    elif (
        hasattr(arr1, "__iter__")
        and hasattr(arr2, "__iter__")
        and len(arr1) != len(arr2)
    ):
        print(key_str, "different lengths:", arr1, arr2)
        diff = True
    elif arr1 != arr2:
        print(key_str, arr1, arr2)
        diff = True
    return diff


def dump_as_str(
    obj,
    delimiter="\t",
    excluded_relations=[
        "detected_stars",
        "reference_stars",
        "target_details",
        "images",
        "msb_list",
        "ot_details",
    ],
    recursion=0,
    recursion_limit=3,
):
    if not hasattr(obj, "_meta"):
        raise TypeError("obj should be a django model.")
    outlist = [
        obj._meta.model_name,
    ]
    for field in obj._meta.fields:
        if isinstance(field, models.fields.json.JSONField):
            outlist.append(json.dumps(field.value_from_object(obj)))
        else:
            outlist.append(field.value_to_string(obj))
        if delimiter in outlist[-1]:
            raise ValueError(
                f"The current delimiter {delimiter} is used in the {obj} {attr} field. {getattr(self, attr)}"
            )
    relations = []
    for relation in obj._meta.related_objects:
        if relation.related_name not in excluded_relations:
            relations.append(relation.related_name)
    if not len(relations) or recursion == recursion_limit:
        return delimiter.join(outlist)
    else:
        outlist = [
            delimiter.join(outlist),
        ]
        for relation in relations:
            try:
                manager = getattr(obj, relation)
            except TypeError:
                continue
            excluded_relations.append(relation)
            for related_obj in manager.all():
                outlist.append(
                    dump_as_str(
                        related_obj,
                        delimiter=delimiter,
                        excluded_relations=excluded_relations,
                        recursion=recursion + 1,
                        recursion_limit=recursion_limit,
                    )
                )
        return "\n".join(outlist)


def rank_norm_chains(arr):
    """arr should be 2d with shape (num_chains, num_samples)"""
    from scipy.stats import rankdata

    M = len(arr)
    N = len(arr[0])
    rank = rankdata(arr)


def rhat(arr, split=True, rank_norm=True):
    """arr should be 2d with shape (num_chains, num_samples)"""
    if split:
        arr = arr.reshape(int(arr.shape[0] * 2), int(arr.shape[1] / 2))
    M = len(arr)
    N = len(arr[0])
    if rank_norm:
        from scipy.stats import rankdata
        from scipy.special import ndtri

        rank = rankdata(arr).reshape(M, N)
        arr = ndtri((rank - 3 / 8) / (M * N - 1 / 4))
    between_chains_var = (
        N
        / (M - 1)
        * sum([(np.average(arr[m]) - np.average(arr)) ** 2 for m in range(M)])
    )
    within_chains_var = (
        1
        / M
        * sum([1 / (N - 1) * sum((arr[m] - np.average(arr[m])) ** 2) for m in range(M)])
    )
    posterior_var = (N - 1) / N * within_chains_var + 1 / N * between_chains_var
    return np.sqrt(posterior_var / within_chains_var)
