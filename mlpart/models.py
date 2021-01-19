from django.db import models
from django.shortcuts import render


class FileOK(models.Model):
    file = models.FileField()
    result = models.CharField(max_length=2, blank=True)
    updated = models.DateTimeField(auto_now=True)
    created = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['id']


    def __str__(self):
        return str(self.id)

    # def upload_file(request):
    #     return render(request, 'upload.html')