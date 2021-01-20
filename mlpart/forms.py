from django import forms
from django.utils.translation import ugettext_lazy as _

from .models import FileOK


class UploadForm(forms.ModelForm):
    class Meta:
        model = FileOK
        fields = ('file',)

        labels = {
            'file': _(''),
        }

        # widgets = {
        #     'file': TextInput(attrs={
        #
        #         'class': 'form-control',
        #         'data-validation': 'custom',
        #
        #     }),}