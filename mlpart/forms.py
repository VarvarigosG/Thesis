from django import forms
from django.utils.translation import ugettext_lazy as _

from .models import FileOK, approvals


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


from django import forms


# gia to BankLoan

class ApprovalForm(forms.Form):
    Firstname = forms.CharField(max_length=15)
    Lastname = forms.CharField(max_length=15)
    Dependants = forms.IntegerField()
    ApplicantIncome = forms.IntegerField()
    CoapplicatIncome = forms.IntegerField()
    LoanAmount = forms.IntegerField()
    Loan_Amount_Term = forms.IntegerField()
    Credit_History = forms.IntegerField()
    Gender = forms.ChoiceField(choices=[('Male', 'Male'), ('Female', 'Female')])
    Married = forms.ChoiceField(choices=[('Yes', 'Yes'), ('No', 'No')])
    Education = forms.ChoiceField(choices=[('Graduate', 'Graduated'), ('Not_Graduate', 'Not_Graduate')])
    Self_Employed = forms.ChoiceField(choices=[('Yes', 'Yes'), ('No', 'No')])
    Property_Area = forms.ChoiceField(choices=[('Rural', 'Rural'), ('Semiurban', 'Semiurban'), ('Urban', 'Urban')])
