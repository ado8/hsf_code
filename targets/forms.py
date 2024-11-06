from data.models import Observation
from django import forms

from .models import Target, TransientType


class TargetForm(forms.ModelForm):
    class Meta:
        model = Target
        fields = [
            "queue_status",
            "sn_type",
            "fit_status",
            "sub_status",
            "galaxy_status",
        ]


class RedoForm(forms.ModelForm):
    def __init__(self, TNS_name, *args, **kwargs):
        super(RedoForm, self).__init__(*args, **kwargs)
        self.fields["observations"].queryset = Target.quick_get(
            TNS_name
        ).observations.all()

    ZP_CHOICES = [
        ("trust", "Trust the headers"),
        ("tphot", "Tonry tphot"),
    ]
    SUBTRACTION_CHOICES = [
        ("all", "All available methods (3)"),
        ("nosub", "no subtraction"),
        ("refsub", "UHS reference subtraction"),
        ("rotsub", "rotational subtraction"),
    ]
    FITTING_CHOICES = [
        ("all", "All available methods (2)"),
        ("snpy", "SNooPy"),
        ("sncosmo", "SNCosmo"),
    ]
    observations = forms.ModelChoiceField(
        queryset=Observation.objects.all(),
        empty_label="All observations",
        required=False,
    )
    sub_type = forms.ChoiceField(choices=SUBTRACTION_CHOICES)
    fit_type = forms.ChoiceField(choices=FITTING_CHOICES)
    sub = forms.BooleanField(required=False)
    fit = forms.BooleanField(required=False)
    rad_dist = forms.BooleanField(required=False, initial=True)
    extinct = forms.BooleanField(required=False, initial=True)
    thresh = forms.FloatField(required=False, initial=10)
    zp_method = forms.ChoiceField(choices=ZP_CHOICES)

    class Meta:
        model = Target
        fields = "__all__"


class SearchForm(forms.Form):
    STATUS_CHOICES = [("d", "Done"), ("c", "Candidate"), ("j", "Junk"), ("q", "Queue")]
    GALAXY_STATUS_CHOICES = [
        ("?", "Uninspected"),
        ("c", "Clear host"),
        ("m", "Multiple possible hosts"),
        ("no", "No apparent galaxy"),
    ]
    Z_CHOICES = [
        ("n", "No z"),
        ("p", "Photometric redshift from literature"),
        ("sp", "Spectroscopic redshift from literature"),
        ("sn1", "SNIFS spectrum available, not yet reduced"),
        ("sn2", "SNIFS spectrum available, reduced"),
    ]
    CONE_CHOICE = [
        ("arcsec", "arcsec"),
        ("arcmin", "arcmin"),
        ("deg", "deg"),
    ]
    queue_status = forms.MultipleChoiceField(
        choices=STATUS_CHOICES, widget=forms.CheckboxSelectMultiple, required=False
    )
    galaxy_status = forms.MultipleChoiceField(
        choices=GALAXY_STATUS_CHOICES,
        widget=forms.CheckboxSelectMultiple,
        required=False,
    )
    galaxy_z_flag = forms.MultipleChoiceField(
        choices=Z_CHOICES, widget=forms.CheckboxSelectMultiple, required=False
    )
    sn_type = forms.ModelChoiceField(queryset=TransientType.objects.all())
    name = forms.CharField(max_length=20, required=False)
    ra = forms.FloatField(required=False)
    dec = forms.FloatField(required=False)
    search_radius = forms.FloatField(required=False)
    cone_unit = forms.ChoiceField(choices=CONE_CHOICE, required=False)
    discover_earliest = forms.DateField(required=False)
    discover_latest = forms.DateField(required=False)
    with_observations = forms.IntegerField(required=False)


class CustomListForm(forms.Form):
    add_list = forms.CharField(label="Add List", max_length=500, required=False)
    remove_list = forms.CharField(label="Remove List", max_length=500, required=False)


class EmailPreferencesForm(forms.Form):
    email_preferences = forms.CharField(
        label="Email Preferences", max_length=500, required=False
    )
    EMAIL_CHOICES = (
        ("a", "ASAP"),
        ("b", "1/hr"),
        ("c", "Morning and Noon"),
        ("d", "Once a day"),
    )
    email_frequency = forms.ChoiceField(choices=EMAIL_CHOICES, required=False)


class UpdateForceForm(forms.Form):
    force = forms.BooleanField(required=False)


class RawTargetForm(forms.Form):
    TNS_name = forms.CharField(label="TNS Name")
    ra = forms.FloatField(label="RA (degrees)")
    dec = forms.FloatField(label="Dec (degrees)")
