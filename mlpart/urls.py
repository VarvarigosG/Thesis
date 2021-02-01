from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static



from django.conf.urls import url, include
from rest_framework.routers import DefaultRouter

from mlpart.api.views import EndpointViewSet
from mlpart.api.views import MLAlgorithmViewSet
from mlpart.api.views import MLRequestViewSet
from mlpart.views import PredictView




urlpatterns = [
    path('', views.FileUploadView, name='upload'),
    path('status/', views.approvereject),
    path('form/', views.cxcontact, name='cxform'),
    path('mpg/', views.mpg, name='mpg'),
    path('mpg/predictMPG/', views.predictMPG, name='predictMPG'),

    path('diabetes/', views.diabetesinsideDjango, name='diabetesinsideDjango'),
    path('diabetes/predictDiabetes/', views.DiabetesModel, name='DiabetesModel'),

    path('iris/', views.Iris, name='Iris'),
    path('iris/predictIris/', views.IrisModel, name='IrisModel'),

    #path('diabetesDjango/', views.diabetesinsideDjango, name='diabetesinsideDjango'),
    #path('diabetesDjango/predict', views.DiabetesModel, name='DiabetesModel'),
    #path(r'^site_media/(?P<path>.*)$', 'django.views.static.serve', {'document_root': '/path/to/media'}),

]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
