from io import BytesIO

import utils
from astropy.coordinates import SkyCoord
from bokeh.embed import components
from bokeh.io.export import get_screenshot_as_png
from bokeh.resources import CDN
from celery import current_app
from data.models import Image
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.http import FileResponse, Http404, HttpResponse, JsonResponse
from django.shortcuts import redirect, render
from django.template.defaulttags import register
from django.utils.decorators import method_decorator
from django.views.decorators.cache import never_cache
from django.views.generic import CreateView, DetailView, ListView, View
from django_filters.views import FilterView
from galaxies.forms import GalaxyForm
from galaxies.models import Galaxy
from selenium import webdriver
from selenium.webdriver.firefox.options import Options

import targets.tasks as tasks

from .filters import TargetFilter
from .forms import CustomListForm, EmailPreferencesForm, RedoForm, TargetForm
from .models import CustomList, Target, TransientType


@register.filter
def get_item(dictionary, key):
    return dictionary.get(key)


@register.filter
def get_value_in_qs(queryset, key):
    # adapted from https://stackoverflow.com/questions/53386786/access-a-querysets-values-list-in-django-template
    if queryset is None:
        return
    if " " in key:
        keys = key.split()
        return queryset.values_list(*keys)
    else:
        return queryset.values_list(key, flat=True)


# Create your views here.


@login_required
def custom_list_view(request):
    CL = CustomList.objects.get_or_create(user=request.user)[0]
    sn_list = Target.objects.filter(TNS_name__in=CL.sn_list)
    if len(sn_list) == 0:
        new = True
    else:
        new = False
    context = {
        "email_preferences": CL.email_preferences,
        "email_frequency": CL.EMAIL_CHOICES[
            [ord(char) - 96 for char in CL.email_frequency.lower()][0] - 1
        ][1],
        "sn_list": sn_list,
        "new": new,
        "cl_form": CustomListForm(),
        "ep_form": EmailPreferencesForm(),
    }
    if request.method == "POST" and "cl" in request.POST:
        cl_form = CustomListForm(request.POST)
        if cl_form.is_valid():
            context["cl_form"] = cl_form
            sn_set = set(CL.sn_list)
            sn_set.update(cl_form.cleaned_data["add_list"].replace(",", " ").split())
            sn_set.difference_update(
                cl_form.cleaned_data["remove_list"].replace(",", " ").split()
            )
            CL.sn_list = list(sn_set)
            CL.save()
            context["sn_list"] = Target.objects.filter(TNS_name__in=CL.sn_list)
            return render(request, "custom_list.html", context)
    elif request.method == "POST" and "ep" in request.POST:
        ep_form = EmailPreferencesForm(request.POST)
        if ep_form.is_valid():
            context["ep_form"] = ep_form
            CL.email_preferences = ep_form.cleaned_data["email_preferences"]
            CL.email_frequency = ep_form.cleaned_data["email_frequency"]
            context["email_preferences"] = CL.email_preferences
            context["email_frequency"] = CL.EMAIL_CHOICES[
                [ord(char) - 96 for char in CL.email_frequency.lower()][0] - 1
            ][1]

            CL.save()
            return render(request, "custom_list.html", context)
    return render(request, "custom_list.html", context)


@method_decorator([login_required], name="dispatch")
class TargetDetailView(DetailView):
    template_name = "targets/detail_w_obs.html"
    model = Target
    pk_url_kwarg = "TNS_name"

    def get(self, request, *args, **kwargs):
        try:
            self.object = Target.quick_get(self.kwargs["TNS_name"])
            return render(request, self.template_name, self.get_context_data())
        except Target.DoesNotExist:
            return redirect(f"/targets/{self.kwargs['TNS_name']}/create")

    def post(self, request, *args, **kwargs):
        self.object = Target.quick_get(self.kwargs["TNS_name"])
        task = None
        context = self.get_context_data()
        if "prev" in request.POST:
            return redirect(context["prev_path"])
        if "next" in request.POST:
            return redirect(context["next_path"])
        for status in ("queue_status", "fit_status", "sub_status", "galaxy_status"):
            if status in request.POST:
                setattr(self.object, status, request.POST.get(status))
        if "sn_type" in request.POST:
            self.object.sn_type = TransientType.objects.get(
                name=request.POST.get("sn_type")
            )
        for z_param in ("z", "z_err", "z_flag"):
            if not self.object.galaxy:
                continue
            if request.POST.get(z_param):
                setattr(self.object.galaxy, z_param, request.POST.get(z_param))
                if z_param == "z_flag":
                    continue
                setattr(self.object, f"manual_{z_param}", request.POST.get(z_param))
            if request.POST.get("pgc_no"):
                self.object.galaxy.pgc_no = request.POST.get("pgc_no")
            self.object.galaxy.manually_inspected.add(request.user)
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
        if "get_candidates_btn" in request.POST:
            msg = request.POST["aladin_coords"]
            if msg.startswith("HSF ID"):
                g = Galaxy.objects.get(pk=int(msg.split()[2]))
                ra = g.ra
                dec = g.dec
            else:
                ra = (float(msg.split(": ")[1].split("\n")[0]),)
                dec = (float(msg.split(": ")[2]),)
            Galaxy.query.all_catalogs(SkyCoord(ra, dec, unit="deg"))
        self.object.galaxy.save()
        self.object.save()
        if request.POST.get("submission_action") == "next_target":
            return redirect(context["next_path"])

        if task:
            context.update({"task_id": task.id, "task_status": task.status})
        return render(request, self.template_name, context)

    def get_queryset(self):
        queryset = super(TargetDetailView, self).get_queryset()
        filt = TargetFilter(self.request.GET, queryset)
        return filt.qs

    def get_context_data(self, **kwargs):
        context = super(TargetDetailView, self).get_context_data(**kwargs)
        p = self.object.plot()
        if p:
            script, div = components(p)  # , CDN)
            context["script"] = script
            context["div"] = div
        else:
            context["script"] = ""
            context["div"] = "No photometry available"
        init = {
            "queue_status": self.object.queue_status,
            "sn_type": self.object.sn_type.name,
            "galaxy_status": self.object.galaxy_status,
            "fit_status": self.object.fit_status,
            "sub_status": self.object.sub_status,
        }
        context["target_form"] = TargetForm(init, instance=self.object)
        context["redo_form"] = RedoForm(self.object.TNS_name)
        context["today"] = utils.MJD(today=True)
        if "J" in self.object.tmax:
            context["time_to_peak"] = self.object.tmax["J"] - utils.MJD(today=True)
        context["epochs"] = int(self.object.observations.count() / 2)
        context["observations"] = self.object.observations.all()
        context["photometry"] = {}
        context["fitresults"] = self.object.fit_results.all()
        context["nearby_galaxies"] = Galaxy.objects.box_search(
            self.object.ra, self.object.dec, 90.0 / 3600
        )
        if self.object.galaxy:
            gal_z = self.object.galaxy.z
            gal_z_err = self.object.galaxy.z_err
            if self.object.manual_z:
                gal_z = self.object.manual_z
            if self.object.manual_z_err:
                gal_z_err = self.object.manual_z_err
            gal_init = {
                "z": gal_z,
                "z_err": gal_z_err,
                "z_flag": self.object.galaxy.z_flag,
                "pgc_no": self.object.galaxy.pgc_no,
            }
            context["galaxy_form"] = GalaxyForm(gal_init, instance=self.object.galaxy)

        queryset = self.get_queryset()
        context["sub_set"] = True
        if queryset.count == Target.objects.count():
            context["sub_set"] = False
        try:
            if int(self.object.TNS_name[0]) < 4:
                context["millenium"] = "20"
            else:
                context["millenium"] = "19"
        except ValueError:
            context["millenium"] = ""
        if not context["sub_set"]:
            return context
        context["query_string"] = f"?{self.request.GET.urlencode()}"
        context["count"] = queryset.count()
        context["prev_path"] = f"/targets/search/{context['query_string']}"
        context["next_path"] = f"/targets/search/{context['query_string']}"
        prev_set = queryset.filter(TNS_name__lt=self.object.TNS_name)
        next_set = queryset.filter(TNS_name__gt=self.object.TNS_name)
        if prev_set.exists():
            context["prev_path"] = (
                f"/targets/{prev_set.last().TNS_name}/{context['query_string']}"
            )
        if next_set.exists():
            context["next_path"] = (
                f"/targets/{next_set.first().TNS_name}/{context['query_string']}"
            )
        context["idx"] = prev_set.count()
        if self.object not in queryset:
            context["idx"] = -1
        return context


@method_decorator(login_required, name="dispatch")
class SearchView(FilterView):
    template_name = "targets/target_list.html"
    model = Target
    paginate_by = 100
    ordering = ["TNS_name"]
    filterset_class = TargetFilter

    def get_context_data(self, **kwargs):
        context = super(SearchView, self).get_context_data(**kwargs)
        context["filters"] = ("c", "o", "ztfg", "ztfr", "asg", "Y", "J", "H")
        queryset = self.get_queryset()
        context["count"] = queryset.count()
        context["query_string"] = f"?{self.request.GET.urlencode()}"
        context["module"] = "targets"
        return context


@method_decorator([never_cache, login_required], name="dispatch")
class RotsubView(DetailView):
    template_name = "targets/rotsub.html"
    model = Target
    pk_url_kwarg = "TNS_name"

    def get(self, request, *args, **kwargs):
        self.object = Target.quick_get(self.kwargs["TNS_name"])
        return render(request, self.template_name, self.get_context_data())

    def post(self, request, *args, **kwargs):
        self.object = Target.quick_get(self.kwargs["TNS_name"])
        task = None
        redo_list = []
        for obs in self.object.observations.all():
            im = obs.images.get(mef=obs.sn_mef)
            if im.status not in ["g", "b"]:
                if request.POST[f"rot_ra_{obs.name}"] != str(im.rot_ra) or request.POST[
                    f"rot_dec_{obs.name}"
                ] != str(im.rot_dec):
                    im.rot_ra = float(request.POST[f"rot_ra_{obs.name}"])
                    im.rot_dec = float(request.POST[f"rot_dec_{obs.name}"])
                    im.save()
                    redo_list.append(im.pk)
                if request.POST[f"status_{obs.name}"] != im.status:
                    im.status = request.POST[f"status_{obs.name}"]
                    im.save()
        if len(redo_list) < 3:
            for im_pk in redo_list:
                Image.objects.get(pk=im_pk).get_rotsub(force=True)
        else:
            task = tasks.redo_rot.delay(redo_list)
        context = self.get_context_data()
        if task:
            context.update({"task_id": task.id, "task_status": task.status})
        return render(request, self.template_name, context)

    def get_context_data(self, **kwargs):
        context = super(RotsubView, self).get_context_data(**kwargs)
        context["obs_im"] = []
        for obs in self.object.observations.all():
            if obs.images.get(mef=obs.sn_mef).status not in ["g", "b"]:
                image = obs.images.get(mef=obs.sn_mef)
                context["obs_im"].append((obs, image))
        return context


@method_decorator([login_required], name="dispatch")
class AtlasImageView(TargetDetailView):
    template_name = "targets/atlas_images.html"

    def get_context_data(self, **kwargs):
        context = super(AtlasImageView, self).get_context_data(**kwargs)
        ATLAS_epochs = {}
        for lc in self.object.lightcurves.filter(
            source="survey", bandpass__in=["c", "o"]
        ):
            for det in lc.detections():
                if int(det.mjd) not in ATLAS_epochs:
                    ATLAS_epochs[int(det.mjd)] = []
                ATLAS_epochs[int(det.mjd)].append(
                    [f"{det.mjd:6f}", lc.bandpass, det.ujy, det.dujy]
                )
        sorted_ATLAS = sorted(ATLAS_epochs.items())
        context["ATLAS_epochs"] = {}
        for i in sorted_ATLAS:
            context["ATLAS_epochs"][i[0]] = i[1]
        return context


@method_decorator(login_required, name="dispatch")
class TaskView(View):
    """TaskView."""

    def get(self, request, task_id):
        task = current_app.AsyncResult(task_id)
        response_data = {"task_status": task.status, "task_id": task.id}
        if task.status == "SUCCESS":
            response_data["results"] = task.get()
        return JsonResponse(response_data)


@method_decorator(login_required, name="dispatch")
class NeedHostRedshiftListView(SearchView):
    def get_queryset(self):
        return (
            Target.objects.get_by_type("Ia")
            .needs_host_z()
            .number_of_observations(1, logic="gte")
        )


@method_decorator(login_required, name="dispatch")
class NeedSpectraListView(SearchView):
    def get_queryset(self):
        recent_mjd = utils.MJD(today=True) - 5
        return Target.objects.get_by_type("?").filter(
            status__in=["q", "c"], dec__range=(-40, 60), detection_date__gt=recent_mjd
        )


@method_decorator(login_required, name="dispatch")
class NeedBrentView(SearchView):
    def get_queryset(self):
        return (
            Target.objects.get_by_type("Ia")
            .number_of_observations(num=1, logic="gte")
            .exclude(galaxy__in=User.objects.get(username="tully").hostgalaxy_set.all())
        )


@method_decorator(login_required, name="dispatch")
class HubbleView(SearchView):
    filterset_class = TargetFilter

    def get_context_data(self, **kwargs):
        context = super(HubbleView, self).get_context_data(**kwargs)
        if context["filter"].qs.count() == Target.objects.count():
            script, div, stats = None, None, None
        else:
            hubble_diagram, stats = context["filter"].qs.get_hubble_diagram(
                model_name="snpy_ebv_model2",
                bandpasses=["ztfg", "ztfr", "J"],
                dm_bandpass="J",
                sigma=3,
            )
            script, div = components(hubble_diagram, CDN)
            context["script"] = script
            context["div"] = div
            context["stats"] = stats
        return context


@method_decorator(login_required, name="dispatch")
class NewTargetView(CreateView):
    template_name = "targets/new_target.html"

    def get(self, request, *args, **kwargs):
        context = {"check_404": False}
        return render(request, self.template_name, context)

    def post(self, request, *args, **kwargs):
        self.object = Target.objects.get_or_create(
            TNS_name=self.kwargs["TNS_name"],
            defaults={
                "queue_status": "c",
                "sn_type": TransientType.objects.get(name="?"),
                "discovering_group": "?",
            },
        )[0]
        # check_404 = self.object.update_TNS()
        # context = {"check_404": check_404}
        # if check_404 == "404":
        #     return render(request, self.template_name)
        self.object.update_all()
        return redirect(self.object)

    def get_context_data(self, **kwargs):
        context = super(NewTargetView, self).get_context_data(**kwargs)
        context["TNS_name"] = self.kwargs["TNS_name"]
        return context


@method_decorator(login_required, name="dispatch")
class BrightView(SearchView):
    def get_queryset(self):
        qset = Target.objects.filter(detection_date__gt=utils.MJD(today=True) - 5)
        bright_names = []
        for obj in qset.get_by_type("Ia").union(qset.get_by_type("?")):
            for lc in obj.lightcurves.filter(source__in=["survey", "model"]):
                if len(lc.ujy) and max(lc.ujy) > 229:
                    bright_names.append(obj.TNS_name)
                    break
        return qset.filter(TNS_name__in=bright_names)


@login_required
def get_lc(request):
    try:
        TNS_name = request.GET["TNS_name"]
        sub_type = request.GET["sub_type"]
    except KeyError:
        raise Http404
    mem_file = BytesIO(Target.quick_get(TNS_name).get_lc_as_bytes(sub_type=sub_type))
    return FileResponse(mem_file, filename=f"lc_{TNS_name}.dat")


@login_required
def get_plot(request):
    try:
        TNS_name = request.GET["TNS_name"]
    except KeyError:
        raise Http404
    options = Options()
    options.headless = True
    executable_path = "/home/ado/.conda/envs/flows/bin/geckodriver"
    image = get_screenshot_as_png(
        Target.quick_get(TNS_name).plot(),
        driver=webdriver.Firefox(executable_path=executable_path, options=options),
    )
    buf = BytesIO()
    image.save(buf, "PNG")
    return HttpResponse(buf.getvalue(), content_type="image/png")
    # return FileResponse(buf.getvalue(), filename=f'lc_{TNS_name}.png')


@login_required
def get_qset(request):
    try:
        qset_name = request.GET["q"]
        cols = request.GET["cols"].split(",")
    except KeyError:
        raise Http404
    if qset_name == "hubble" or qset_name == "good":
        qset = Target.objects.good()
    if qset_name == "galaxies":
        qset = Target.objects.filter(galaxy__identity="c")
    if qset_name == "custom":
        filt = TargetFilter(request.GET, Target.objects.all())
        qset = filt.qs
    mem_file = BytesIO(qset.get_qset_as_bytes(cols=cols))
    return FileResponse(mem_file, filename=f"{qset_name}_queryset.dat")
