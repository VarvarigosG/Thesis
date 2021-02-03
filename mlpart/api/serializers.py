from rest_framework import serializers

from ..models import FileOK, approvals


class FileSerializer(serializers.ModelSerializer):
    class Meta:
        model = FileOK
        fields = ('id', 'file', 'result')


# aytos tha einai gia to BankLoanNN
class approvalsSerializers(serializers.ModelSerializer):
    class Meta:
        model = approvals
        fields = '__all__'
