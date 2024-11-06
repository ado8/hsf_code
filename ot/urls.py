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
from ot import views

app_name = 'ot'
urlpatterns = [
    path('', views.UKIRTListView.as_view(), name='ukirt_list'),
    path('task/<str:TNS_name>/', views.TaskView.as_view(), name='task'),
    path('download_xml/', views.get_xml, name='get_xml'),
    path('<str:TNS_name>/', views.UKIRTDetailView.as_view(), name='ukirt_detail'),
]

