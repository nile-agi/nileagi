from django.urls import path, include
from . import views

app_name = "home"  

urlpatterns = [
    path('', view=views.home, name='home'),
    path('about/', view=views.about, name='about'),
    path('services/',view=views.services_products, name='services_products'),
    path('blog/', view = views.blog, name='blog'),
    path('contact/', view=views.contact, name='contact'),
    path('subscribe',views.subscribe, name='subscribe'),
]