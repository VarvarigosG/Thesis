from django.shortcuts import render, HttpResponseRedirect, reverse
from rest_framework import mixins
from .forms import UploadForm


def FileUploadView(request):
    if request.method == 'POST':
        form = UploadForm(request.POST,request.FILES)
        if form.is_valid():
            form.save()
            # messages.success(request, 'Your file had been uploaded successfully.')
            return HttpResponseRedirect(reverse('upload'))
    else:
        form = UploadForm()
        context = {
            'form': form,
        }
    return render(request, 'mlpart/upload.html', context)




