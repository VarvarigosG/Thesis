from .views import FileOKViewSet
from rest_framework.routers import DefaultRouter
from django.urls import path,include
from . import views
from django.conf import settings
from django.conf.urls.static import static


from .views import EndpointViewSet
from .views import MLAlgorithmViewSet
from .views import MLAlgorithmStatusViewSet
from .views import MLRequestViewSet
from ..views import PredictView

router = DefaultRouter(trailing_slash=False)
router.register(r'mlpart', FileOKViewSet)
router.register(r"endpoints", EndpointViewSet, basename="endpoints")
router.register(r"mlalgorithms", MLAlgorithmViewSet, basename="mlalgorithms")
router.register(r"mlalgorithmstatuses", MLAlgorithmStatusViewSet, basename="mlalgorithmstatuses")
router.register(r"mlrequests", MLRequestViewSet, basename="mlrequests")
# urlpatterns = router.urls

urlpatterns = [
    path('', include(router.urls)),
    path('<endpoint_name>/predict', PredictView.as_view(), name="predict"),
]