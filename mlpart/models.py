from django.db import models
from django.shortcuts import render


class File(models.Model):
    file = models.FileField(upload_to='')
    updated = models.DateTimeField(auto_now=True)
    created = models.DateTimeField(auto_now_add=True)


    def __str__(self):
        return str(self.id)

    # def upload_file(request):
    #     return render(request, 'upload.html')