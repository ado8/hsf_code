from django.shortcuts import render, get_object_or_404, redirect
from django.http import HttpResponse, JsonResponse, Http404, FileResponse
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib import messages
from django.utils.decorators import method_decorator
from django.views.generic import DetailView, ListView, FormView, CreateView, View
from django.views.decorators.cache import never_cache, patch_cache_control
from django.core.exceptions import ObjectDoesNotExist
from django.template.defaulttags import register
from django.forms import formset_factory
from django.db.models import Count

from .models import Galaxy
from targets.models import Target

@register.filter
def get_item(dictionary, key):
    return dictionary.get(key)

# Create your views here.
@method_decorator([login_required], name='dispatch')
class GalaxyDetailView(DetailView):
    template_name = 'galaxies/detail.html'
    model = Galaxy
    pk_url_kwarg = 'TNS_name'

    def get(self, request, *args, **kwargs):
        try:
            self.object = Galaxy.objects.get(TNS_name=self.kwargs['TNS_name'])
            return render(request, self.template_name, self.get_context_data())
        except ObjectDoesNotExist:
            return redirect(f"/targets/{self.kwargs['TNS_name']}/create")

    def get_context_data(self, **kwargs):
        context = super(GalaxyDetailView, self).get_context_data(**kwargs)
        context['z_cols'] = ['No.', 'Frequency Targeted', 'Published Velocity', 'Published Velocity Uncertainty', 'Published Redshift', 'Published Redshift Uncertainty', 'Refcode']
        return context

