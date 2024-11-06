import glob
import os
import shutil
import time
from datetime import date

import constants
import numpy as np
import pexpect
import sncosmo
import utils
from astropy.io import fits
from celery import Task, chain, chord, group, shared_task
from celery_progress.backend import ProgressRecorder
from data.models import Image, Observation
from dotenv import load_dotenv
from galaxies.models import Galaxy, NedEntry, SimbadEntry
from utils import MJD, HMS2deg, get_TNS_name

from targets.email import send_email
from targets.log import logger
from targets.models import Target, TransientType

"""
Restart celery to push to production
"""


class BaseTaskWithRetry(Task):
    autoretry_for = (Exception,)
    retry_kwargs = {"max_retries": 4}
    retry_backoff = 3
    retry_jitter = True


@shared_task(name="targets.test")
def test():
    pass


@shared_task(name="targets.sync_w_tns")
def sync_w_TNS(m=31):
    res = Target.sync_w_tns(m=m)
    if hasattr(res, "__iter__"):
        new, updated = res
        return group(
            group(initialize_target.s(name) for name in new),
            group(update_tns_info.s(name) for name in updated),
        ).apply_async()


@shared_task(name="targets.update_tns_info")
def update_tns_info(TNS_name):
    Target.quick_get(TNS_name).update_TNS()
    return TNS_name


@shared_task(name="targets.update_values")
def update_values(updated_lcs=None, TNS_name=None):
    if updated_lcs is None:
        Target.quick_get(TNS_name).update_current_values()
        Target.quick_get(TNS_name).update_peak_values()
    elif isinstance(updated_lcs, list) and any(updated_lcs):
        Target.quick_get(TNS_name).update_current_values()
        Target.quick_get(TNS_name).update_peak_values()


@shared_task(name="targets.update_target")
def update_target(TNS_name, force=False):
    return chain(
        group(
            update_atlas.s(TNS_name), update_ztf.s(TNS_name), update_asassn.s(TNS_name)
        ),
        group(
            task_fit.s(TNS_name=TNS_name, bandpasses=combo, model_name=m, force=force)
            for m in constants.MODEL_NAMES
            for combo in utils.pset(Target.quick_get(TNS_name).detected_bandpasses)
            if len(combo) >= Target.quick_get(TNS_name).detected_bandpasses.count() - 1
        ),
        update_values.s(TNS_name=TNS_name),
    ).apply_async()


@shared_task(name="targets.update_candidates")
def update_candidates():
    return group(
        update_target.s(TNS_name)
        for TNS_name in Target.objects.filter(
            queue_status__in=("q", "c"), detection_date__gt=MJD(today=True) - 14
        ).values_list("TNS_name", flat=True)
    ).apply_async()


@shared_task(name="targets.initialize_target")
def initialize_target(TNS_name):
    """initialize_target.

    Parameters
    ----------
    TNS_name :
        TNS_name
    """
    Target.objects.get_or_create(
        TNS_name=TNS_name, defaults={"sn_type": TransientType.objects.get(name="?")}
    )
    return chain(
        update_tns_info.s(TNS_name),
        chord(
            (
                group(update_atlas.s(), update_ztf.s(), update_asassn.s()),
                chain(
                    group(query_ned.s(), query_simbad.s(), query_ps1.s()),
                    get_closest_galaxy.si(TNS_name),
                    group(
                        query_ned.s(job="table redshifts"),
                        query_simbad.s(job="measurements"),
                    ),
                ),
            ),
            group(
                task_fit.si(TNS_name=TNS_name, bandpasses=combo, model_name=m)
                for m in constants.MODEL_NAMES
                for combo in utils.pset(Target.quick_get(TNS_name).detected_bandpasses)
            ),
        ),
        update_values.si(TNS_name=TNS_name),
    ).apply_async()


@shared_task(name="galaxies.query_ned")
def query_ned(
    TNS_name=None, ra=None, dec=None, radius=2.5, ned_pk=None, job="cone search"
):
    """query_ned.

    Parameters
    ----------
    TNS_name:
        TNS_name
    ra :
        ra
    dec :
        dec
    radius :
        radius
    ned_pk :
        ned_pk
    job :
        job
    """
    if job == "cone search":
        if TNS_name:
            obj = Target.quick_get(TNS_name)
            ra, dec = obj.ra, obj.dec
        Galaxy.query.ned(ra=ra, dec=dec, radius=radius)
    elif job.startswith("table"):
        if TNS_name:
            obj = Target.quick_get(TNS_name)
            if obj.galaxy and obj.galaxy.ned_entries.exists():
                for ned in obj.galaxy.ned_entries.all():
                    query_ned(ned_pk=ned.pk, job=job)
            return
        table_type = job.split()[1]
        if table_type == "all":
            for param in (
                "diameters",
                "redshifts",
                "photometry",
                "classification",
                "distance",
            ):
                query_ned(ned_pk=ned_pk, job=f"table {param}")
            return
        NedEntry.objects.get(pk=ned_pk).query_table(table_type=table_type)


@shared_task(name="galaxies.query_simbad")
def query_simbad(
    TNS_name=None, ra=None, dec=None, radius=2.5, sb_pk=None, job="cone search"
):
    """query_simbad.

    Parameters
    ----------
    TNS_name:
        TNS_name
    ra :
        ra
    dec :
        dec
    radius :
        radius
    sb_pk :
        sb_pk
    job :
        job
    """
    if job == "cone search":
        if TNS_name:
            obj = Target.quick_get(TNS_name)
            ra, dec = obj.ra, obj.dec
        Galaxy.query.simbad(ra=ra, dec=dec, radius=radius)
    if job == "measurements":
        if TNS_name:
            obj = Target.quick_get(TNS_name)
            if obj.galaxy and obj.galaxy.simbad_entries.exists():
                for sb in obj.galaxy.simbad_entries.all():
                    query_simbad(sb_pk=sb.pk, job=job)
            return
        SimbadEntry.objects.get(pk=sb_pk).query_measurements()


@shared_task(name="galaxies.query_ps1")
def query_ps1(TNS_name=None, ra=None, dec=None, radius=2.5):
    """query_ps1.

    Parameters
    ----------
    ra :
        ra
    dec :
        dec
    radius :
        radius
    """
    if TNS_name:
        obj = Target.quick_get(TNS_name)
        ra, dec = obj.ra, obj.dec
    Galaxy.query.ps1(ra=ra, dec=dec, radius=radius)


@shared_task(name="targets.get_closest_galaxy")
def get_closest_galaxy(TNS_name):
    Target.quick_get(TNS_name).get_closest_galaxy()
    return TNS_name


@shared_task(name="lightcurves.update_atlas")
def update_atlas(TNS_name, get_stamps=False):
    try:
        return Target.quick_get(TNS_name).update_atlas(get_stamps=get_stamps)
    except TypeError:
        pass


@shared_task(name="lightcurves.update_ztf")
def update_ztf(TNS_name):
    return Target.quick_get(TNS_name).update_ztf()


@shared_task(name="lightcurves.update_asassn")
def update_asassn(TNS_name):
    return Target.quick_get(TNS_name).update_asassn()


@shared_task(name="fitting.fit")
def task_fit(
    updated_lcs=[True, True],
    TNS_name=None,
    fit_result_pk=None,
    bandpasses=[],
    variants=[],
    calibration=6,
    redlaw="F19",
    bandpasses_str="",
    variants_str="",
    model_name=None,
    priors={},
    force=False,
):
    if TNS_name is None and fit_result_pk is None:
        return False
    if fit_result_pk is not None:
        Target.quick_get(TNS_name).fit_results.get(pk=fit_result_pk).refit(force=force)
        return True
    if model_name is None:
        return False
    if not len(bandpasses) and bandpasses_str != "":
        bandpasses = bandpasses_str.split("-")
    if not len(variants) and variants_str != "":
        variants = variants_str.split("-")
    if not len(bandpasses):
        return False
    if updated_lcs and any(i is True for i in updated_lcs):
        Target.quick_get(TNS_name).fit(
            bandpasses=bandpasses,
            variants=variants,
            model_name=model_name,
            calibration=calibration,
            redlaw=redlaw,
            priors=priors,
            force=force,
        )
        return True
    return False


@shared_task(name="incorporate_SCAT")
def incorporate_SCAT():
    Target.add_SCAT_to_db()


@shared_task(name="download_UKIRT_data")
def dl_ukirt(semester="u22bh01"):
    load_dotenv(f"{constants.HSF_DIR}/.env")
    dl_uname = os.environ.get("UKIRT_FTP_UNAME")
    dl_pwd = os.environ.get("UKIRT_FTP_PWD")

    if not os.path.exists(f"{constants.UKIRT_HSF_DIR}/{semester}/"):
        os.mkdir(f"{constants.UKIRT_HSF_DIR}/{semester}/")
    elif os.path.exists(f"{constants.UKIRT_HSF_DIR}/{semester}/{semester}"):
        os.remove(f"{constants.UKIRT_HSF_DIR}/{semester}/{semester}")
    os.system(
        f"wget --cut-dirs=3 -P {constants.UKIRT_HSF_DIR}/{semester}/ -np -nH -N -r -l1 http://apm3.ast.cam.ac.uk/~mike/wfcam/{semester} --http-user={dl_uname} --http-password={dl_pwd}"
    )
    for path in glob.glob(f"{constants.UKIRT_HSF_DIR}/{semester}/*_sf_st.fit"):
        if Observation.objects_all.filter(path=path).exists():
            continue
        else:
            hdu = fits.open(path)
            name = hdu[0].header["MSBTITLE"].split()[0].strip("-")
            print(name)
            if name in ["CALSPEC", "NGC4242", "G191B2B", "2M055914"]:
                continue
            if name.endswith("_temp"):
                name = name.strip("_temp")
            if name[:2] != "AT":
                program = hdu[0].header["PROJECT"]
                if program in ["U/20A/DJH01", "U/20A/DJH02"]:
                    name = f"AT{name}"
            TNS_name = get_TNS_name(name)
            t = Target.quick_get_or_create(TNS_name)
            y, m, d = hdu[0].header["DATE"].split("T")[0].split("-")
            bp = hdu[0].header["FILTER"]
            defaults = {"status": science}
            obs = Observation.objects.update_or_create(
                path=path,
                name=path.split("/")[-1].split("_s")[0],
                date=date(int(y), int(m), int(d)),
                bandpass=bp,
                target=t,
                defaults=defaults,
            )[0]
            try:
                obs.make_image(mef=(2, 3))
                for im in obs.images.all():
                    im.process()
            except:
                continue


@shared_task(name="redownload one UKIRT file")
def redownload_ukirt(obs_name):
    obs = Observation.objects_all.get(name=obs_name)
    load_dotenv(f"{constants.HSF_DIR}/.env")
    dl_uname = os.environ.get("UKIRT_FTP_UNAME")
    dl_pwd = os.environ.get("UKIRT_FTP_PWD")
    semester = obs.program.lower().replace("/", "")

    for path in glob.glob(f"{constants.UKIRT_HSF_DIR}/{semester}/{obs.name}*"):
        os.remove(path)
        os.system(
            f"wget --cut-dirs=3 -P {constants.UKIRT_HSF_DIR}/{semester} -np -nH -r -l1 http://apm3.ast.cam.ac.uk/~mike/wfcam/{semester}/{path.split('/')[-1]} --http-user={dl_uname} --http-password={dl_pwd}"
        )


@shared_task(name="download Subaru data")
def dl_subaru():
    if os.path.exists("/data/projects/Subaru/holding/S2Query.tar"):
        os.chdir("/data/projects/Subaru/holding")
        os.system("tar -xvf S2Query.tar")
        child = pexpect.spawn("./zadmin/unpack.py")
        child.sendline("N")
        child.sendline("1")
        child.sendline("2")
        child.sendline("y")
        child.sendline("n")
        child.sendline("n")
        child.sendline("7")
        done = False
        print("waiting for download to complete")
        while not done:
            num_in_dir = len(glob.glob("*.fits"))
            time.sleep(3)
            if num_in_dir == len(glob.glob("*.fits")):
                done = True
        for path in glob.glob("./*.fits"):
            header = fits.open(path)["PRIMARY"].header
            date_obs = header["DATE-OBS"]
            short_date = date_obs.replace("-", "")[2:]
            dir_path = f"{constants.SUBARU_HSF_DIR}/{short_date}"
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)
            name = header["OBJECT"]
            shutil.move(path, dir_path)
            try:
                obj = Target.quick_get(name)
                if obj.galaxy.z_flag not in ["su1", "su2"]:
                    print(f"updating galaxy status for {name}")
                    obj.galaxy.z_flag = "su1"
                    obj.galaxy.save()
                else:
                    print(f"galaxy status for {name} already {obj.galaxy.z_flag}")
            except:
                continue
        os.chdir("/data/poohbah/1/assassin/Subaru")
        shutil.rmtree("holding")
        os.mkdir("holding")


@shared_task(name="rotsub", bind=True)
def update_obs(
    self,
    obs_name,
    redo_sn=False,
    redo_rot=False,
    redo_sub=False,
    sn_dx=None,
    sn_dy=None,
    rot_dx=None,
    rot_dy=None,
    sub_type=None,
):
    obs = Observation.objects.get(name=obs_name)
    if redo_sn:
        # pixel scale 0.2
        sn_ra = (sn_dx * 0.2 / 3600) * np.cos(obs.target.dec * np.pi / 180.0)
        sn_dec = sn_dy * 0.2 / 3600
        if sn_ra != obs.sn_ra or sn_dec != obs.sn_dec:
            obs.sn_ra = sn_ra
            obs.sn_dec = sn_dec
            for im in obs.images.all():
                im.photometry(force=True)
    if redo_rot:
        rot_ra = (rot_dx * 0.2 / 3600) * np.cos(obs.target.dec * np.pi / 180.0)
        rot_dec = rot_dy * 0.2 / 3600
        if rot_ra != obs.rot_ra or rot_dec != obs.rot_dec:
            obs.rot_ra = rot_ra
            obs.rot_dec = rot_dec
            obs.get_rotsub(force=True)
            obs.images.get(sub_type="rotsub").photometry(force=True)
    if redo_sub:
        obs.preferred_sub_type = sub_type
    obs.save()


@shared_task(name="redo_rot", bind=True)
def redo_rot(self, im_pk_list):
    progress_recorder = ProgressRecorder(self)
    for i, im_pk in enumerate(im_pk_list):
        im = Image.objects.get(pk=im_pk)
        progress_recorder.set_progress(
            i,
            len(im_pk_list),
            description=f"Doing rotational subtraction for {im.observation.name}",
        )
        im.get_rotsub(force=True)


@shared_task(name="email_mike_irwin", bind=True)
def email_mike(self):
    message = """
Hi Mike,
This is Aaron Do with an automatically sent email.
If you'd like to go back to 'handwritten' emails instead of automatic ones, or if something seems amiss, please let me know at ado@hawaii.edu.

Could you update the ftp sites again?
We could use data from the following programs: u23bh01, u22adu01

Thank you for your attention,
Aaron, on behalf of the rest of the HSF team.
"""
    send_email("Automated update request", message, "mike@ast.cam.ac.uk")
