from .views import FileOKViewSet
from rest_framework.routers import DefaultRouter
from django.urls import path,include
from . import views
from django.conf import settings
from django.conf.urls.static import static


router = DefaultRouter(trailing_slash=False)
router.register(r'mlpart', FileOKViewSet)

# urlpatterns = router.urls

urlpatterns = [
    path('', include(router.urls)),
]