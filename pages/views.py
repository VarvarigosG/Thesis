from django.shortcuts import render

def index(request):
    return render(request, 'pages/index2.html')