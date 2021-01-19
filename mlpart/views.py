from django.shortcuts import render, HttpResponseRedirect, reverse
from django.contrib import messages
from .forms import UploadForm
#
# class Home(TemplateView):
#     template_name = 'mlpart/home.html'

# def upload(request):
#     context={}
#     if request.method == 'POST':
#         uploaded_file = request.FILES['document']
#         fs = FileSystemStorage()
#         name=fs.save(uploaded_file.name, uploaded_file )
#         context['url'] = fs.url(name)
#         print(uploaded_file.name)
#         print(uploaded_file.size)
#     return render(request, 'mlpart/upload.html', context)
#
# def upload_file (request):
#     return render(request,'mlpart/upload.html')

def FileUploadView(request):
    if request.method == 'POST':
        form = UploadForm(request.POST,request.FILES)
        if form.is_valid():
            form.save()
            messages.success(request, 'Your file had been uploaded successfully.')
            return HttpResponseRedirect(reverse('upload'))
    else:
        form = UploadForm()
        context = {
            'form':form,
        }
    return render(request, 'mlpart/upload.html', context)

