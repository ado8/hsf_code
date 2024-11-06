from datetime import date, datetime

import pytz
import utils
from celery import shared_task
from celery_progress.backend import ProgressRecorder
from django.db import transaction
from django.db.models import Q

import constants
from ot.models import MSBList, TargetOTDetails


@shared_task(name="ot.new_ot")
def new_ot():
    """
    Make a new, empty list with today's date and the latest msb as previous
    copy all previous TargetOTDetails with null rejection_date and at least
    one remaining observation.
    """
    new = MSBList.create_todays_list()
    new.import_from_previous()


@shared_task(name="ot.sync_w_omp")
def sync_w_omp():
    if MSBList.objects.last().date != date.today():
        new = MSBList.create_todays_list()
        new.import_from_previous()
    else:
        new = MSBList.objects.last()
    # Download html of UKIRT omp site observing log, parse for observations,
    # create pending observations
    new.process_ukirt_log()
    # Check TNS for new classifications of active targets (not references),
    # drop if no longer Ia or ?
    new.update_tns_info()
    # check obj.tmax for unobserved targets (remaining==3)
    new.update_phase()
    # check lcs for estimate of current mag. adjust exposure times accordingly
    new.update_exp_times()
    new.save()


@shared_task(name="sync from drive")
def sync_from_drive():
    today = datetime.now(pytz.timezone("US/Hawaii"))
    today = date(today.year, today.month, today.day)
    name = "%02d%02d%02d.xml" % (today.year - 2000, today.month, today.day)
    path = f"{constants.MEDIA_DIR}/{name}"
    try:
        utils.download_file(name, path)
    except IndexError:
        print("doesn't exist yet")
    msb = MSBList.create_todays_list()
    msb.import_from_xml(path)


@shared_task(name="task get coords")
def task_get_coords(tod_pk):
    """
    Quick task for finding guide stars in the background
    """
    TargetOTDetails.objects.get(pk=tod_pk).get_coords()


@shared_task(name="task_get_exp_times")
def task_get_exp_times(tod_pk):
    """
    Quick task for finding guide stars in the background
    """
    TargetOTDetails.objects.get(pk=tod_pk).get_exp_times()


@shared_task(name="checkout")
def checkout(msblist_date):
    # arg is converted to str, need to put back into date
    from targets.models import Target

    d, t = msblist_date.split("T")
    dy, dm, dd = d.split("-")
    msb = MSBList.objects.get(date=date(int(dy), int(dm), int(dd)))
    msb.write_out()
    msb.update_current_targets_sheet()
    msb.get_add_set("targets").update(
        queue_status="q",
        status_reason=f"Lightcurve manually accepted {utils.MJD(today=True)}",
    )
    msb.get_in_set("targets").update(queue_status="q")
    for tod in msb.target_details.filter(Q(drop=True) | Q(remaining=0)):
        """update_current_targets_sheet.
        """
        obj = tod.target
        if obj.observations.exists() and (
            obj.sn_type.subtype_of("?") or obj.sn_type.subtype_of("Ia")
        ):
            obj.queue_status = "d"
        else:
            obj.queue_status = "j"
            obj.status_reason = f"Lightcurve manually rejected {utils.MJD(today=True)}"
        obj.save()
    for tod in msb.target_details.filter(rejection_date__isnull=False):
        obj = tod.target
        obj.queue_status = "j"
        obj.save()
