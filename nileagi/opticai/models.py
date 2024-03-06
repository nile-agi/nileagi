from django.db import models
# Create your models here.
import os

from django.utils import timezone

BASE_DIR = os.path.dirname(os.path.dirname(os.path.relpath(__file__)))

class DemoRequests(models.Model):
    email_id = models.CharField(max_length=100)
    email = models.EmailField(max_length=100)
    upload_date = models.DateTimeField(default=timezone.now)

class OpticaiData(models.Model):
    image_id = models.CharField(max_length=100)
    filepaths = models.FileField(max_length=200,upload_to=os.path.join(BASE_DIR,'images'))
    filename = models.CharField(max_length=100)
    upload_date = models.DateTimeField(default=timezone.now)
 