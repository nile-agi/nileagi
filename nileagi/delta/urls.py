from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

app_name = "delta"  

urlpatterns = [
path('search-api/', views.SearchAPI.as_view(), name='search-api'),
path('search/', view=views.SearchAPI.search_engine, name='search')
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

if not settings.DEBUG:
   urlpatterns += static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)