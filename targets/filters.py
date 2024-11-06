import django_filters
from django.db.models import Count, Q
from django_filters import (BooleanFilter, CharFilter, ChoiceFilter,
                            ModelChoiceFilter, MultipleChoiceFilter,
                            NumberFilter, RangeFilter)
from galaxies.models import Galaxy

from .models import Target, TransientType


class TargetFilter(django_filters.FilterSet):
    """TargetFilter.

    Parse fields and return a queryset
    """

    ra = RangeFilter(field_name="ra")
    dec = RangeFilter(field_name="dec")
    detection_date = RangeFilter(field_name="detection_date")
    name = CharFilter(method="name_filt")
    galaxy_z_flag = MultipleChoiceFilter(
        choices=[
            ("n", "No redshift available"),
            ("l", "Literature redshift available"),
            ("lu", "Literature redshift and uncertainty available"),
            ("p", "Photometric redshift from literature"),
            ("pu", "Photometric redshift and uncertainty from literature"),
            ("sp", "Spectroscopic redshift from literature"),
            ("spu", "Spectroscopic redshift and uncertainty from literature"),
            ("sn1", "SNIFS spectrum available, not yet reduced"),
            ("sn2", "SNIFS spectrum available, reduced"),
            ("sn3", "SNIFS spectrum available, reduced, SNR too low"),
            ("su1", "Subaru spectrum available, not yet reduced"),
            ("su2", "Subaru spectrum available, reduced"),
            ("su3", "Subaru spectrum available, reduced, SNR too low"),
            ("spx1", "Literature spectrum available not yet reduced"),
            ("spx2", "Literature spectrum available, reduced"),
        ],
        field_name="galaxy__z_flag",
    )
    galaxy_name = CharFilter(method="gal_name_filt")
    galaxy_manually_inspected = BooleanFilter(
        field_name="galaxy__manually_inspected", method="galaxy_manually_inspected_filt"
    )
    min_obs_num = NumberFilter(field_name="min_obs_num", method="min_obs_num_filt")
    spectrum = ChoiceFilter(
        field_name="spectrum",
        method="spectrum_filt",
        choices=[
            ("good", "SNIFS or FOCAS spectrum has strong features"),
            ("need", "Does not have a spectrum or spectrum is weak"),
            ("uninspected", "Has an uninspected SNIFS or FOCAS spectrum"),
        ],
    )
    sn_type = ModelChoiceFilter(
        queryset=TransientType.objects.all(),
        field_name="sn_type",
        method="sn_type_filt",
    )

    class Meta:
        model = Target
        fields = [
            "TNS_name",
            "ra",
            "dec",
            "sn_type",
            "detection_date",
            "discovering_group",
            "ATLAS_name",
            "ZTF_name",
            "galaxy__z_flag",
            "queue_status",
            "fit_status",
            "sub_status",
            "galaxy_status",
        ]

    def name_filt(self, qs, name, value):
        if not value or value == "null":
            return qs
        return qs.filter(
            Q(TNS_name__contains=value)
            | Q(ATLAS_name__contains=value)
            | Q(ZTF_name__contains=value)
        )

    def gal_name_filt(self, qs, name, value):
        if not value or value == "null":
            return qs
        return qs.filter(
            Q(galaxy__ned_entries__aliases__contains=value)
            | Q(galaxy__simbad_entries__ids__contains=value)
        )

    def galaxy_manually_inspected_filt(self, qs, name, value):
        if not value or value == "null":
            qs = qs.filter(Q(galaxy__manually_inspected=None))
        return qs

    def min_obs_num_filt(self, qs, name, value):
        if not value or value == "null":
            return qs
        return qs.number_of_observations(num=value, logic='gte')

    def spectrum_filt(self, qs, name, value):
        if not value or value == "null":
            return qs
        if value == "need":
            return qs.needs_host_z()
        elif value == "good":
            return qs.filter(
                Q(galaxy__snifs_entries__z_flag="s")
                | Q(galaxy__focas_entries__z_flag="s")
            )
        elif value == "uninspected":
            return qs.filter(
                Q(galaxy__snifs_entries__z_flag="?")
                | Q(galaxy__focas_entries__z_flag="?")
            )

    def sn_type_filt(self, qs, name, value):
        if not value or value == "null":
            return qs
        subtypes = TransientType.objects.get(name=value.name).get_children()
        return qs.filter(sn_type__name__in=subtypes)
