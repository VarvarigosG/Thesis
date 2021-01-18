from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
#
# class Home(TemplateView):
#     template_name = 'mlpart/home.html'

def upload(request):
    context={}
    if request.method == 'POST':
        uploaded_file = request.FILES['document']
        fs = FileSystemStorage()
        name=fs.save(uploaded_file.name, uploaded_file )
        context['url'] = fs.url(name)
        print(uploaded_file.name)
        print(uploaded_file.size)
    return render(request, 'mlpart/upload.html', context)
#
# def upload_file (request):
#     return render(request,'mlpart/upload.html')
