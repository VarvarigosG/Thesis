from django.db import transaction
from rest_framework import mixins
from rest_framework import viewsets
from rest_framework.exceptions import APIException


from .serializers import FileSerializer

from ..models import Endpoint
from ..models import FileOK
from ..models import MLAlgorithm
from ..models import MLAlgorithmStatus
from ..models import MLRequest


class FileOKViewSet(viewsets.ModelViewSet):
    serializer_class = FileSerializer
    queryset = FileOK.objects.all()


