"""hsf URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.urls import path

from targets import views

app_name = "targets"
urlpatterns = [
    path("search/", views.SearchView.as_view(), name="search"),
    path("hubble/", views.HubbleView.as_view(), name="hubble"),
    path(
        "galaxies_needing_z/",
        views.NeedHostRedshiftListView.as_view(),
        name="galaxies_needing_z/",
    ),
    path("needs_brent/", views.NeedBrentView.as_view(), name="needs_brent/"),
    path("bright/", views.BrightView.as_view(), name="bright/"),
    path(
        "classification_needed/",
        views.NeedSpectraListView.as_view(),
        name="classification_needed/",
    ),
    path("download_lc/", views.get_lc, name="get_lc"),
    path("download_plot/", views.get_plot, name="get_plot"),
    path("download_qset/", views.get_qset, name="get_qset"),
    path("task/<str:task_id>", views.TaskView.as_view(), name="task"),
    path("<str:TNS_name>/", views.TargetDetailView.as_view(), name="detail"),
    path("<str:TNS_name>/atlas_images/", views.AtlasImageView.as_view(), name="atlas"),
    path("<str:TNS_name>/create/", views.NewTargetView.as_view(), name="new-target"),
    path("<str:TNS_name>/rotsub_fix", views.RotsubView.as_view(), name="rotsub"),
]
