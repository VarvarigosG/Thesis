from django import forms

from .models import Choice


# class DetailForm(forms.ModelForm):
#     text = forms.CharField()
#
#
#     class Meta:
#         model = Choice
#         # me poio model antistoixizetai auto to form
#         fields = ('answer_text',)
#         # poio field apo to model pairnei

class PostForm(forms.ModelForm):

    class Meta:
        model = Choice
        fields = ('answer_text',)
