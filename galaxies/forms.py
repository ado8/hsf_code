from django import forms
from .models import Galaxy

class GalaxyForm(forms.ModelForm):
    class Meta:
        model = Galaxy
        fields = '__all__'

    def __init__(self, *args, **kwargs):
        super(GalaxyForm, self).__init__(*args, **kwargs)
        self.fields['z'].required = False
        self.fields['z_err'].required = False
        self.fields['pgc_no'].required = False
