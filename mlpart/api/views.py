from rest_framework import viewsets
from ..models import FileOK
from .serializers import FileSerializer

class FileOKViewSet(viewsets.ModelViewSet):
    serializer_class = FileSerializer
    queryset = FileOK.objects.all()