from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static



from django.conf.urls import url, include
from rest_framework.routers import DefaultRouter



urlpatterns = [
    path('', views.FileUploadView, name='upload'),
    path('status/', views.approvereject),
    path('form/', views.cxcontact, name='cxform'),

    path('multipleupload/', views.multipleupload, name='multipleupload'),
    path('multipleupload_save/', views.multipleupload_save, name='multipleupload_save'),

    path('mpg/', views.mpg, name='mpg'),
    path('mpg/predictMPG/', views.predictMPG, name='predictMPG'),

    path('diabetes/', views.diabetesinsideDjango, name='diabetesinsideDjango'),
    path('diabetes/predictDiabetes/', views.DiabetesModel, name='DiabetesModel'),

    path('iris/', views.Iris, name='Iris'),
    path('iris/predictIris/', views.IrisModel, name='IrisModel'),


]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
