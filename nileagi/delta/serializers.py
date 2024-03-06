from rest_framework import serializers
from .models import SearchData, FileData

class SearchDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = SearchData
        fields = '__all__'

class FileDataSerializer(serializers.ModelSerializer):

    class Meta:

        model = FileData

        fields = ('__all__')