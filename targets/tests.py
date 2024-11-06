from django.test import TestCase

# Create your tests here.
from .models import Target
from bokeh.plotting import show

def get_cand():
    a = Target.quick_get('18iyn')
    a.get_candidate_galaxies()
    a.save()

def plot():
    a = Target.quick_get('20lon')
    p = a.plot()
    show(p)

plot()
