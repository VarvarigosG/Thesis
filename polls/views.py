from django.http import HttpResponse


def polls(request):
    return HttpResponse("Hello, world. You're at the polls index.")
