from .views import FileOKViewSet
from rest_framework.routers import DefaultRouter

router = DefaultRouter()
router.register(r'mlpart', FileOKViewSet)
urlpatterns = router.urls