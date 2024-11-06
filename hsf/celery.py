import os

import dotenv

from celery import Celery
from celery.schedules import crontab

# just for getting redis pwd
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dotenv_file = os.path.join(BASE_DIR, ".env")
if os.path.isfile(dotenv_file):
    dotenv.load_dotenv(dotenv_file)

# Set the default Django settings module for the 'celery' program.
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "hsf.settings")

# Get the base REDIS URL, default to redis' default
BASE_REDIS_URL = os.environ.get(
    "REDIS_URL", f"redis://:{os.environ.get('REDIS_PWD')}@localhost:6379"
)

app = Celery("hsf", backend=BASE_REDIS_URL)

# Using a string here means the worker doesn't have to serialize
# the configuration object to child processes.
# - namespace='CELERY' means all celery-related configuration keys
#   should have a `CELERY_` prefix.
app.config_from_object("django.conf:settings")

# Load task modules from all registered Django apps.
app.autodiscover_tasks()

app.conf.broker_url = BASE_REDIS_URL
app.conf.beat_scheduler = "django_celery_beat.schedulers.DatabaseScheduler"

# restart celerybeat to update schedule
app.conf.beat_schedule = {
    "new_candidates_half_hour": {
        "task": "targets.sync_w_tns",
        "schedule": crontab(minute="*/30"),
    },
    "update_candidates": {
        "task": "targets.update_candidates",
        "schedule": crontab(minute=15),
    },
    # "dl_ukirt_each_week": {
    #     "task": "download_UKIRT_data",
    #     "schedule": crontab(minute=0, hour=8, day_of_week="sun"),
    # },
    # "email_mike_once_a_month": {
    #     "task": "email_mike_irwin",
    #     "schedule": crontab(minute=0, hour=0, day_of_month=5),
    # },
    # "new_ot": {"task": "ot.new_ot", "schedule": crontab(hour=0, minute=1)},
    # "sync_w_omp": {"task": "ot.sync_w_omp", "schedule": crontab(hour=8, minute=5)},
    "incorporate_SCAT": {
        "task": "incorporate_SCAT",
        "schedule": crontab(hour=7, minute=45),
    },
}

# outside queries
app.conf.task_annotations = {
    "targets.update_tns_info": {"rate_limit": "30/m"},
    "galaxies.query_ned": {"rate_limit": "2/s"},
    "galaxies.query_simbad": {"rate_limit": "2/s"},
    "galaxies.query_ps1": {"rate_limit": "2/s"},
    "lightcurves.update_atlas": {"rate_limit": "1/s"},
    "lightcurves.update_ztf": {"rate_limit": "1/s"},
    "lightcurves.update_ps1": {"rate_limit": "1/s"},
}
