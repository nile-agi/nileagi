from django.db import models
import os
from django.utils import timezone

# Create your models here.
class Messages(models.Model):
    message_id = models.CharField(max_length=200)
    name = models.CharField(max_length=200)
    email = models.EmailField(max_length=200)
    subject = models.CharField(max_length=200)
    message = models.CharField(max_length=255)
    upload_date = models.DateTimeField(default=timezone.now)  


class Subscribers(models.Model):
    email_id = models.CharField(max_length=200)
    email = models.EmailField(max_length=200)
    upload_date = models.DateTimeField(default=timezone.now) 