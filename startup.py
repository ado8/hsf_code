import os
import sys
import shutil
import gzip
import pickle
from glob import glob

import arviz
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
from astropy.coordinates import SkyCoord
from astropy.io import fits
from tqdm import tqdm

import constants
import targets.tasks as tasks
import utils
from data.models import (
    CalspecSpectrum,
    DetectedStar,
    FocasSpectrum,
    Image,
    Observation,
    ReferenceStar,
    ReferenceStarDetails,
    SnifsSpectrum,
    TemplateSpectrum,
)
from data.weightedCC import weightedCC
from fitting.models import FitResults, salt_fit, snpy_fit, spline_fit
from galaxies.models import (
    FocasEntry,
    Galaxy,
    GladeEntry,
    NedDiameter,
    NedEntry,
    NedPhotometry,
    NedRedshifts,
    PanStarrsEntry,
    SimbadEntry,
    SnifsEntry,
)
from lightcurves.models import (
    AsassnDetection,
    AtlasDetection,
    Lightcurve,
    UkirtDetection,
    ZtfDetection,
)
from ot.models import MSBList, TargetOTDetails
from ot.tasks import checkout, new_ot, sync_from_drive, sync_w_omp, task_get_coords
from targets.models import Target, TransientType
