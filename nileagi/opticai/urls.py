from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

app_name = "opticai"  

urlpatterns = [
    path('about-opticai',views.about_opticai, name='about-opticai'),
    path('request-demo',views.request_demonstration, name='request-demo'),
    path('opticai/', views.opticai, name='opticai'),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

if not settings.DEBUG:
   urlpatterns += static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)