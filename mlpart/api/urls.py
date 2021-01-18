from .views import FileViewSet
from rest_framework.routers import DefaultRouter

router = DefaultRouter()
router.register(r'mlpart', FileViewSet)
urlpatterns = router.urls