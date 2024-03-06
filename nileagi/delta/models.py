from django.db import models
from django.utils import timezone
from django.contrib.auth.models import User
import os
from django.conf import settings
# Create your models here.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.relpath(__file__)))

class FileData(models.Model):
    filepath = models.FileField(max_length=200,upload_to=os.path.join(BASE_DIR,"files"))
    filename = models.CharField(max_length=50)
    # user_id = models.ForeignKey(auto_users, on_delete=models.CASCADE)
    upload_date = models.DateTimeField(default=timezone.now)

class SearchData(models.Model):
    source_type = models.CharField(max_length=50)
    data_content = models.TextField()

    def __str__(self):
        return f"{self.source_type} - {self.id}"
    

class SearchHistory(models.Model):
    ip_address = models.GenericIPAddressField()
    query = models.CharField(max_length=255)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.ip_address} - {self.query} - {self.timestamp}"
