from django import forms

class CommentForm(forms.Form):
    name= forms.CharField(max_length=20,
    widget=forms.TextInput(attrs={'class': 'formcontrol', 'placeholder' : 'Name'}))
    comment = forms.CharField
    #CharField(widget=forms.Textarea(attrs={'class': 'formcontrol', 'placeholder' : 'Comment'}))
