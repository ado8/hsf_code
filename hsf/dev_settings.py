"""
Django settings for hsf project.

Generated by 'django-admin startproject' using Django 2.0.7.

For more information on this file, see
https://docs.djangoproject.com/en/2.0/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/2.0/ref/settings/
"""

DEBUG = True
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': 'db_dev.sqlite3',
        }
}

INSTALLED_APPS = [
    "django_extensions",
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    #extra
    'django_cron',
    'django_celery_beat',
    # 'django_celery_results',
    'celery_progress',
    "django_filters",
    "debug_toolbar",
    #own
    "pages",
    "targets",
    "ot",
    "data",
    "lightcurves",
    "fitting",
    "galaxies",
]
