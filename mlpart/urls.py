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

]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)