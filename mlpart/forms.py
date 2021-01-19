from django import forms
from .models import FileOK


class UploadForm(forms.ModelForm):
    class Meta:
        model = FileOK
        fields = ('file',)