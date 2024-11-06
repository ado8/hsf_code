import sys

sys.path.append(
    ".."
)  # Adds a higher directory to python module path for importing goals

from datetime import date

from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import Group, User
from django.db.models import Q
from django.http import HttpResponse, HttpResponseForbidden
from django.shortcuts import get_object_or_404, redirect, render
from django.views.generic import FormView, View
from django.views.static import serve
from lightcurves.models import UkirtDetection
from targets.forms import SearchForm
from targets.models import Target
from timeline import deadlines
from utils import MJD

# Create your views here.


def thanks_view(request):
    if request.user.is_authenticated():
        return redirect(request.META["HTTP_REFERER"])
    return render(request, "thanks.html", context)


class HomeView(View):
    template_name = "home.html"

    def get(self, request, *args, **kwargs):
        return render(request, self.template_name, self.get_context_data(**kwargs))

    def get_context_data(self, **kwargs):
        context = {}
        context["unknown"] = (
            int(
                Target.objects.get_by_type("?")
                .number_of_observations(logic="gte")
                .count()
            ),
        )
        context["ias"] = (
            int(
                Target.objects.get_by_type("Ia")
                .number_of_observations(logic="gte")
                .count()
            ),
        )
        context["unknown_w_obs_and_z"] = (
            int(
                Target.objects.get_by_type("?")
                .number_of_observations(logic="gte")
                .with_good_host_z()
                .count()
            ),
        )
        context["unknown_w_obs_needs_z"] = (
            int(
                Target.objects.get_by_type("?")
                .number_of_observations(logic="gte")
                .needs_host_z()
                .count()
            ),
        )
        context["ias_w_obs_and_z"] = (
            int(
                Target.objects.get_by_type("Ia")
                .number_of_observations(logic="gte")
                .with_good_host_z()
                .count()
            ),
        )
        context["ias_w_obs_needs_z"] = (
            int(
                Target.objects.get_by_type("Ia")
                .number_of_observations(logic="gte")
                .needs_host_z()
                .count()
            ),
        )
        context["good_z"] = (
            context["unknown_w_obs_and_z"][0] + context["ias_w_obs_and_z"][0]
        )
        context["needs_z"] = (
            context["unknown_w_obs_needs_z"][0] + context["ias_w_obs_needs_z"][0]
        )
        context["all"] = context["ias"][0] + context["unknown"][0]
        context["3unknown"] = (
            int(
                Target.objects.get_by_type("?")
                .number_of_observations(num=6, logic="gte")
                .count()
            ),
        )
        context["3ias"] = (
            int(
                Target.objects.get_by_type("Ia")
                .number_of_observations(num=6, logic="gte")
                .count()
            ),
        )
        context["3unknown_w_obs_and_z"] = (
            int(
                Target.objects.get_by_type("?")
                .number_of_observations(num=6, logic="gte")
                .with_good_host_z()
                .count()
            ),
        )
        context["3unknown_w_obs_needs_z"] = (
            int(
                Target.objects.get_by_type("?")
                .number_of_observations(num=6, logic="gte")
                .needs_host_z()
                .count()
            ),
        )
        context["3ias_w_obs_and_z"] = (
            int(
                Target.objects.get_by_type("Ia")
                .number_of_observations(num=6, logic="gte")
                .count()
            ),
        )
        context["3ias_w_obs_needs_z"] = (
            int(
                Target.objects.get_by_type("Ia")
                .number_of_observations(num=6, logic="gte")
                .needs_host_z()
                .count()
            ),
        )
        context["3good_z"] = (
            context["3unknown_w_obs_and_z"][0] + context["3ias_w_obs_and_z"][0]
        )
        context["3needs_z"] = (
            context["3unknown_w_obs_needs_z"][0] + context["3ias_w_obs_needs_z"][0]
        )
        context["3all"] = context["3ias"][0] + context["3unknown"][0]
        context["base_str"] = "/targets/search?"
        context["ia_str"] = "sn_type=Ia&"
        context["unknown_str"] = "sn_type=%3F&"
        context["gal_str"] = ""
        for flag in ("sp", "spu", "sn1", "sn2", "su1", "su2"):
            context["gal_str"] += f"galaxy_z_flag={flag}&"
        context["nogal_str"] = ""
        for flag in ("n", "l", "lu", "p", "pu", "sn3", "su3"):
            context["nogal_str"] += f"galaxy_z_flag={flag}&"
        return context


class AboutView(View):
    template_name = "about.html"

    def get(self, request, *args, **kwargs):
        return render(request, self.template_name, self.get_context_data())

    def get_context_data(self, *args, **kwargs):
        context = {}
        return context


class TodoView(View):
    template_name = "todo.html"

    def get_context_data(self, *args, **kwargs):
        context = {
            "today": date.today(),
            "mjd_today": round(MJD(today=True)),
            "deadlines": [
                (entry[0], (entry[0] - date.today()).days, entry[1])
                for entry in deadlines
            ],
            "ud_inspected": UkirtDetection.objects.filter(status="?"),
            "gal_ident": Target.objects.filter(galaxy_status="?")
            .get_by_type("Ia")
            .number_of_observations(num=1, logic="gte")
            .count(),
            "gal_inspected": Target.objects.get_by_type("Ia")
            .number_of_observations(num=1, logic="gte")
            .filter(galaxy_status="?")
            .count(),
            "tully_uninspected": Target.objects.get_by_type("Ia")
            .number_of_observations(num=1, logic="gte")
            .exclude(galaxy__in=User.objects.get(username="tully").galaxy_set.all())
            .count(),
        }
        return context

    def get(self, request, *args, **kwargs):
        return render(request, self.template_name, self.get_context_data())


def login_view(request):
    username = request.POST["username"]
    password = request.POST["password"]
    user = authenticate(request, username=username, password=password)
    if user is not None:
        login(request, user)
        # Redirect to a success page.
    else:
        # Return an 'invalid login' error message
        context = {"username": username, "password": password, "status": fail}
        return render(request, login.html, context)


@login_required
def media_access(request, path):
    """
    When trying to access :
    myproject.com/media/uploads/passport.png

    If access is authorized, the request will be redirected to
    myproject.com/protected/media/uploads/passport.png

    This special URL will be handle by nginx we the help of X-Accel
    """
    response = HttpResponse()
    # Content-type will be detected by nginx
    del response["Content-Type"]
    response["X-Accel-Redirect"] = "/protected/media/" + path
    return response
