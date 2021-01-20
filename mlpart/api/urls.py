from .views import FileOKViewSet
from rest_framework.routers import DefaultRouter


from .views import EndpointViewSet
from .views import MLAlgorithmViewSet
from .views import MLAlgorithmStatusViewSet
from .views import MLRequestViewSet

router = DefaultRouter(trailing_slash=False)
router.register(r'mlpart', FileOKViewSet)
router.register(r"endpoints", EndpointViewSet, basename="endpoints")
router.register(r"mlalgorithms", MLAlgorithmViewSet, basename="mlalgorithms")
router.register(r"mlalgorithmstatuses", MLAlgorithmStatusViewSet, basename="mlalgorithmstatuses")
router.register(r"mlrequests", MLRequestViewSet, basename="mlrequests")
urlpatterns = router.urls