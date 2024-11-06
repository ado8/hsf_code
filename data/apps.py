from django.apps import AppConfig


class DataConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'data'

class ObservationConfig(AppConfig):
    name = 'observation'

class ReferenceConfig(AppConfig):
    name = 'reference'
