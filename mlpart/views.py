from django.http import HttpResponse
from django.shortcuts import render
from django.views.generic import TemplateView
from django.core.files.storage import FileSystemStorage

# class Home(TemplateView):
#     template_name = 'mlpart/home.html'

def upload(request):
    if request.method == 'POST':
        uploaded_file = request.FILES['document']
        fs = FileSystemStorage()
        fs.save(uploaded_file.name, uploaded_file )
        print(uploaded_file.name)
        print(uploaded_file.size)
    return render(request, 'mlpart/upload.html')



from django.shortcuts import render

# Create your views here.
