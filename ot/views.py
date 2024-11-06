from datetime import date, datetime
from io import BytesIO

import constants
import pytz
import targets.tasks as tasks
from django.contrib.auth.decorators import login_required
from django.core.exceptions import ObjectDoesNotExist
from django.http import FileResponse, HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils.decorators import method_decorator
from django.views.generic.list import ListView, MultipleObjectMixin
from galaxies.models import Galaxy
from targets.models import Target, TransientType
from targets.views import SearchView, TargetDetailView, TaskView
from utils import MJD, deg2HMS, file_exists

from ot.models import MSBList, TargetOTDetails
from ot.tasks import (checkout, new_ot, sync_from_drive, task_get_coords,
                      task_get_exp_times)


# Create your views here.
@method_decorator(login_required, name="dispatch")
class UKIRTListView(SearchView):
    """UKIRTListView.

    Adding context to SearchView
    """

    template_name = "ot/UKIRT_list.html"
    model = Target

    def get_queryset(self):
        return Target.objects.ukirt_candidates().exclude(
            TNS_name__in=self.object.targets.values_list("TNS_name")
        )

    def get(self, request, *args, **kwargs):
        self.object = MSBList.objects.latest("date")
        self.object_list = self.get_queryset()
        context = self.get_context_data()
        return self.render_to_response(context)

    def post(self, request, *args, **kwargs):
        self.object = MSBList.objects.latest("date")
        self.object_list = self.get_queryset()
        task = None
        context = self.get_context_data()
        if "checkout" in request.POST:
            task = checkout.delay(self.object.pk)
        if "sync_from_drive" in request.POST:
            task = sync_from_drive.delay()
        for tod in context["drop_set"]:
            if f"drop_{tod.target.TNS_name}" in request.POST:
                self.object.target_details.filter(
                    target__TNS_name=tod.target.TNS_name
                ).update(drop=False)
        if task:
            context.update({"task_id": task.id, "task_status": task.status})
        return render(request, self.template_name, context)

    def get_context_data(self, **kwargs):
        context = super(UKIRTListView, self).get_context_data(**kwargs)
        now = datetime.now(pytz.timezone("US/Hawaii"))
        today = date(now.year, now.month, now.day)
        msblist = MSBList.objects.latest("date")

        context["today_str"] = "%02d%02d%02d" % (
            today.year - 2000,
            today.month,
            today.year,
        )
        context["latest"] = msblist.date
        context["today"] = today
        context["add_set"] = msblist.get_add_set()
        context["observed_set"] = msblist.target_details.filter(observed=True)
        context["finished_set"] = msblist.target_details.filter(remaining=0)
        context["in_set"] = msblist.get_in_set()
        context["drop_set"] = msblist.target_details.filter(drop=True)
        change_list = []
        for tod in msblist.target_details.all():
            try:
                old = msblist.previous.target_details.get(target=tod.target)
                if old.exp_times != tod.exp_times:
                    change_list.append(tod.target.TNS_name)
            except:
                continue
        context["change_set"] = msblist.target_details.filter(
            target__TNS_name__in=change_list
        )
        context["reject_set"] = msblist.target_details.filter(
            rejection_date__isnull=False
        )
        context["filters"] = ("c", "o", "ztfg", "ztfr", "asg", "Y", "J", "H")
        context["filters_exp_times"] = {}
        for tod in msblist.target_details.filter(
            remaining__gt=0, drop=False, rejection_date__isnull=True
        ):
            context["filters_exp_times"][tod.target.TNS_name] = []
            for i, filt in enumerate(tod.filters):
                context["filters_exp_times"][tod.target.TNS_name].append(
                    f"{filt}={int(float(tod.exp_times.split()[i]))}"
                )
        context["module"] = "UKIRT_helper"
        return context


@method_decorator(login_required, name="dispatch")
class UKIRTDetailView(TargetDetailView):
    template_name = "ot/UKIRT_detail.html"
    model = Target

    def get_context_data(self, **kwargs):
        context = super(UKIRTDetailView, self).get_context_data(**kwargs)
        now = datetime.now(pytz.timezone("US/Hawaii"))
        today = date(now.year, now.month, now.day)
        context["MSBList"], new = MSBList.objects.get_or_create(date=today)
        if new:
            context["MSBList"].previous = MSBList.objects.filter(date__lt=today).latest(
                "date"
            )
            context["MSBList"].import_from_previous()
        if self.object.TNS_name in context["MSBList"].get_in_set().values_list(
            "target__TNS_name", flat=True
        ):
            context["page_type"] = "keep/drop"
            self.object_list = Target.objects.filter(
                TNS_name__in=context["MSBList"]
                .get_in_set()
                .values_list("target__TNS_name", flat=True)
            )
        else:
            context["page_type"] = "add/junk"
            self.object_list = Target.objects.ukirt_candidates().exclude(
                TNS_name__in=context["MSBList"].targets.values_list("TNS_name")
            )
        context["idx"] = (
            self.object_list.filter(TNS_name__lt=self.object.TNS_name).count() + 1
        )
        context["count"] = self.object_list.count()
        if self.object == self.object_list.last():
            context["next"] = self.object_list.last()
        else:
            context["next"] = self.object_list.filter(
                TNS_name__gt=self.object.TNS_name
            ).first()
        if context["next"]:
            context["next_path"] = f"/UKIRT_helper/{context['next'].TNS_name}"
        return context

    def post(self, request, *args, **kwargs):
        today = datetime.now(pytz.timezone("US/Hawaii"))
        self.object = Target.quick_get(self.kwargs["TNS_name"])
        context = self.get_context_data()
        if self.object == self.object_list.last():
            next_path = "/UKIRT_helper/"
        else:
            next_path = context["next_path"]
        if (
            "c_next" in request.POST
            or "j_next" in request.POST
            or "q_next" in request.POST
        ):
            if "j_next" in request.POST:
                TargetOTDetails.objects.update_or_create(
                    target=self.object,
                    msblist=context["MSBList"],
                    defaults={"rejection_date": today},
                )
                self.object.status_reason += (
                    f"Lightcurve manually rejected on MJD={MJD(today=True)}"
                )
            if "q_next" in request.POST:
                self.object.save()
                tod, new = TargetOTDetails.objects.update_or_create(
                    target=self.object,
                    msblist=context["MSBList"],
                    defaults={"filters": "YJ"},
                )
                self.object.status_reason += (
                    f"Lightcurve manually accepted on MJD={MJD(today=True)}"
                )
                task_get_coords.delay(tod.pk)
                task_get_exp_times.delay(tod.pk)
            context["MSBList"].save()
            return redirect(next_path, context)
        if "k_next" in request.POST or "d_next" in request.POST:
            if "d_next" in request.POST:
                TargetOTDetails.objects.filter(
                    target=self.object, msblist=context["MSBList"],
                ).update(drop=True)
                self.object.status_reason += (
                    f"Lightcurve manually rejected on MJD={MJD(today=True)}"
                )
                self.object.save()
            return redirect(next_path, context)
        if "set_host_galaxy_btn" in request.POST:
            msg = request.POST.get("aladin_coords")
            host_msg = request.POST.get("set_host_galaxy_btn")
            if host_msg == "Create galaxy at coordinates":
                g = Galaxy.objects.create(
                    ra=float(msg.split(": ")[1].split("Dec")[0]),
                    dec=float(msg.split(": ")[2]),
                )
                self.object.galaxy = g
            elif host_msg == "Set as host":
                hsf_id = int(msg.split()[2])
                self.object.galaxy = Galaxy.objects.get(pk=hsf_id)
                self.object.galaxy_status = "c"
                self.object.galaxy.manually_inspected.add(request.user)
            self.object.save()
            self.object.galaxy.save()
        return render(request, self.template_name, context)


@login_required
def get_xml(request):
    """get_xml.


    Parameters
    ----------
    request :
        request
    """
    mem_file = BytesIO(MSBList.objects.latest("date").get_xml_as_bytes())
    return FileResponse(mem_file, filename=f"ot.xml")
