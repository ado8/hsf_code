from django.contrib import admin
from .models import Observation, Image, SnifsSpectrum, FocasSpectrum

# Register your models here.

admin.site.register(Observation)
admin.site.register(Image)
admin.site.register(SnifsSpectrum)
admin.site.register(FocasSpectrum)
