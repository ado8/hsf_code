from django.db import models
from django.core.exceptions import ObjectDoesNotExist

import os

from alerce.core import Alerce
import pandas as pd
import numpy as np

import constants
import utils


# Create your models here.

class NewLightcurve(models.Model):
    BANDPASS_CHOICES = [
            ('c', 'cyan'),
            ('o', 'orange'),
            ('ztfg', 'ZTF green'),
            ('ztfr', 'ZTF red'),
            ('asg', 'ASAS-SN g'),
            ('Y', 'WFCAM Y'),
            ('J', 'WFCAM J'),
            ('H', 'WFCAM H'),
            ]
    SOURCE_CHOICES = [
            ('survey', 'Survey'),
            ('observed', 'Observed'),
            ('model', 'Model'),
            ]
    SURVEY_CHOICES = [
            ('ATLAS', 'ATLAS'),
            ('ZTF', 'ZTF'),
            ('ASAS-SN', 'ASAS-SN'),
            ]
    FITTER_CHOICES = [
            ('snpy', 'SNooPy'),
            ('salt3', 'salt3'),
            ('salt2-extended', 'salt2-extended'),
            ('salt3-experimental', 'salt3-experimental'),
            ]
    SUB_CHOICES = [
            ('none', 'Survey Source'),
            ('nosub', 'No subtraction'),
            ('refsub', 'Reference Subtraction'),
            ('rotsub', 'Rotational Subtraction'),
            ]
    target = models.ForeignKey("targets.Target", on_delete=models.CASCADE, related_name='lightcurves')
    bandpass = models.CharField(max_length=10, choices=BANDPASS_CHOICES)
    source = models.CharField(max_length=8, choices=SOURCE_CHOICES)
    # for survey lcs
    survey = models.CharField(max_length=8, null=True, choices=SURVEY_CHOICES)
    # for observed lcs, none for others
    sub_type = models.CharField(max_length=8, default='none', choices=SUB_CHOICES)
    # for model lcs
    fitter = models.CharField(max_length=20, null=True, choices=FITTER_CHOICES)
    bandpasses_str = models.CharField(max_length=50, null=True)
    mjd = models.JSONField(default=list)
    mag = models.JSONField(default=list)
    dmag = models.JSONField(default=list)
    ujy = models.JSONField(default=list)
    dujy = models.JSONField(default=list)

    def __str__(self):
        return self.target.TNS_name + ' lightcurve'

    def save(self, *args, **kwargs):
        if self.detections.count():
            qset = self.detections.all()
            self.mjd = [det.mjd for det in qset]
            self.mjd = [det.mag for det in qset]
            self.mjd = [det.dmag for det in qset]
            self.mjd = [det.ujy for det in qset]
            self.mjd = [det.dujy for det in qset]
        super(Bandpass, self).save(*args, **kwargs)

    def bin(self, bin_period='mjd', statistic='median', weight='inverse variance', clip=3):
        # replace intranight observations with a single photometric point
        bin_dict = {}
        for i in zip(self.mjd, self.ujy, self.dujy, self.mag, self.dmag):
            if bin_period in ['mjd', 'day', 'daily']:
                bin_str = str(int(i[0]))
            elif bin_period in ['none', 'None']:
                bin_str = str(i[0])
            else:
                print(f'bin_period={bin_period} not valid, see help if it exists')
                print('proceeding with mjd binning')
                bin_str = str(int(i[0]))
            if bin_str not in bin_dict:
                bin_dict[bin_str] = []
            bin_dict[bin_str].append(i)
        holder = {}
        for i,param in enumerate(['mjd', 'ujy', 'dujy', 'mag', 'dmag']):
            holder[param] = []
            for b in bin_dict.keys():
                nparr = np.array(bin_dict[b])
                for j in range(len(nparr[:,2])):
                    if float(nparr[:,2][j]) == 0.:
                        nparr[:,2][j] = 1e5
                if clip:
                    clip_results = utils.sigmaclip(data=nparr[:,1], errors=nparr[:,2], sigmalow=clip, sigmahigh=clip)
                    nparr = nparr[clip_results[2]==1]
                if str(weight).lower() in ['inverse variance', 'ivar', 'iv', 'least squares', 'least square', 'ls']:
                    w = 1/nparr[:,2]**2
                elif str(weight).lower() in ['none', 'flat', 'even']:
                    w = np.ones(len(nparr[:,2]))
                else:
                    print(f'weight={weight} not valid, see help if it exists')
                    print('proceeding with no weights')
                    w = np.ones(len(nparr[:,2]))
                if statistic in ['ave', 'average', 'mean']:
                    value = np.average(nparr[:,i], weights=w)
                elif statistic in ['median', 'med']:
                    value = utils.weighted_quantile(nparr[:,i], [0.5], sample_weight=w)[0]
                holder[param].append(value)
        if self.bandpass not in ['Y', 'J', 'H', 'K']:
            for i in range(len(holder['mjd'])):
                mag, dmag = utils.ujy_to_ab(holder['ujy'][i], holder['dujy'][i])
                holder['mag'][i] = mag
                holder['dmag'][i] = dmag
        return holder
    def move_to_bad(self, mjd):
        if mjd in self.detections:
            self.bad_data[mjd] = self.detections.pop(mjd)

class AtlasObservation(models.Model):
    CAMERA_CHOICES = [
            ('01a', 'Mauna Loa'),
            ('02a', 'Haleakala'),
            ('03a', 'Southern Telescope 1'),
            ('04a', 'Southern Telescope 2'),
            ]
    lc = models.ForeignKey('Lightcurve', related_name='detections', on_delete=models.CASCADE)
    mjd = models.FloatField()
    mag = models.FloatField()
    dmag = models.FloatField()
    ujy = models.IntegerField()
    dujy = models.PositiveIntegerField()
    err = models.BooleanField(default=False)
    chin = models.FloatField()
    ra = models.FloatField()
    dec = models.FloatField()
    x = models.FloatField()
    y = models.FloatField()
    major = models.FloatField()
    minor = models.FloatField()
    phi = models.FloatField()
    apfit = models.FloatField()
    mag5sig = models.FloatField()
    sky = models.FloatField()
    camera = models.CharField(max_length=3, choices=CAMERA_CHOICES)
    exp_num = models.PositiveIntegerField()

    def atlas_obs_name(self):
        return f'{self.camera}{int(self.mjd)}o{self.exp_num}{self.bandpass}'
class ZTFNonDetection(models.Model):
    g = 1
    r = 2
    i = 3
    FID_CHOICES = [
            (g, 'ztfg'),
            (r, 'ztfr'),
            (i, 'ztfi'),
            ]
    lc = models.ForeignKey('Lightcurve', related_name='nondetections', on_delete=models.CASCADE)
    mjd = models.FloatField()
    fid = models.PositiveIntegerField(choices=FID_CHOICES)
    diffmaglim = models.FloatField()
class ZTFDetection(ZTFNonDetection):
    # https://alerce.readthedocs.io/_/downloads/en/develop/pdf/
    SCI_MINUS_REF = 1
    REF_MINUS_SCI = -1
    ISDIFFPOS_CHOICES = [
            (SCI_MINUS_REF, 'Candidate is from positive (sci minus ref) subtraction')
            (REF_MINUS_SCI, 'Candidate is from negative (ref minus sci) subtraction')
    lc = models.ForeignKey('Lightcurve', related_name='detections', on_delete=models.CASCADE)
    candid = models.CharField(max_length=20) 
    pid = models.PositiveBigIntegerField() 
    isdiffpos = models.FloatField(choices=ISDIFFPOS_CHOICES)
    nid = models.SmallIntegerField() 
    distnr = models.FloatField()
    magpsf = models.FloatField()
    magpsf_corr = models.FloatField(null=True)
    magpsf_corr_ext = models.FloatField(null=True)
    magap = models.FloatField()
    magap_corr = models.FloatField()
    sigmapsf = models.FloatField()
    sigmapsf_corr = models.FloatField(null=True)
    sigmapsf_corr_ext = models.FloatField(null=True)
    sigmagap = models.FloatField()
    sigmagap_corr = models.FloatField(null=True)
    ra = models.FloatField()
    dec = models.FloatField()
    rb = models.FloatField()
    rbversion = models.CharField(max_length=9)
    drb = models.FloatField()
    magapbig = models.FloatField()
    sigmagapbig = models.FloatField()
    rfid = models.PositiveIntegerField()
    has_stamp = models.BooleanField()
    corrected = models.BooleanField()
    dubious = models.BooleanField()
    candid_alert 
    step_id_corr = models.CharField(max_length=10)
    phase = models.FloatField()
    parent_candid = models.IntegerField(null=True)
class ASASSNNonDetection(models.Model):
    lc = models.ForeignKey('Lightcurve', on_delete=models.CASCADE, related_name='nondetections')
    mjd = models.FloatField()
    hjd = models.FloatField()
    ut_date = models.DateTimeField()
    image = models.CharField(max_length=15)
    fwhm = models.FloatField()
    diff = models.FloatField()
    limit = models.FloatField()
    mag = models.FloatField()
    dmag = models.FloatField()
    counts = models.FloatField()
    count_err = models.FloatField()
    ujy = models.SmallIntegerField()
    dujy = models.SmallIntegerField()
class ASASSNDetection(ASASSNNonDetection):
    lc = models.ForeignKey('Lightcurve', on_delete=models.CASCADE, related_name='detections')
