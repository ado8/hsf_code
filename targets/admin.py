from django.contrib import admin

from .models import Target, CustomList, TransientType
# Register your models here.

admin.site.register(Target)
admin.site.register(CustomList)
admin.site.register(TransientType)
