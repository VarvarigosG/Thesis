from django.urls import path
from . import views

app_name = 'polls'
# dinoume to onoma app_name san namespace gia na mhn mperdeutoune ta URLS me ta upoloipa apps

urlpatterns = [
    # ex: /polls/
    path('', views.IndexView.as_view(), name='index'),
    # ex: /polls/5/
    path('<int:pk>/', views.DetailView.as_view(), name='detail'),
    # ex: /polls/5/results/
    path('<int:pk>/results/', views.ResultsView.as_view(), name='results'),
    # ex: /polls/5/vote/
    path('<int:question_id>/vote/', views.vote, name='vote'),
    #has changed from <question_id> to <pk>.
    path('<int:question_id>/choice-chart/', views.choice_chart, name='choice-chart'),
    path('<int:question_id>/get-percentage/', views.get_percentage, name='get-percentage'),
    path('<int:question_id>/leap/', views.leapquestion, name='leapquestion'),
    path('leap/totalpercentage/', views.totalpercentage, name='totalpercentage'),

]
