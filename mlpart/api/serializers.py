from rest_framework import serializers
from ..models import FileOK

class FileSerializer(serializers.ModelSerializer):
    class Meta:
        model = FileOK
        fields = ('id', 'file', 'result')