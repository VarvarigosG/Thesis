from django.http import HttpResponseRedirect
from django.shortcuts import get_object_or_404, render
from django.urls import reverse
from django.views import generic
from .models import Choice, Question
from django.utils import timezone
from django.contrib import messages
from django.http import HttpResponse
from django.db.models import Sum
from django.http import JsonResponse


# # kanei display  ena question text xwris apotelesmata kai ena form gia na kaneis vote
# def detail(request, question_id):
#     question = get_object_or_404(Question, pk=question_id)
#     return render(request, 'polls/detail.html', {'question': question})

# # kanei dispaly ta results gia ena particular question
# def results(request, question_id):
#     question = get_object_or_404(Question, pk=question_id)
#     # einai shortcut , yparxei kai to  get_list_or_404() function, which works just as get_object_or_404()
#     return render(request, 'polls/results.html', {'question': question})

# kanei handling to voting gia ena particular choice se ena particular question


# def index(request):
#     latest_question_list = Question.objects.order_by('-pub_date')[:5]
#     context = {'latest_question_list': latest_question_list}
#     return render(request, 'polls/index.html', context)
#     # it displays the latest 5 poll questions separated by commas according to publication date

# These views represent a common case of basic Web development: getting data from the database according to a parameter
# passed in the URL, loading a template and returning the rendered template. Because this is so common,
# Django provides a shortcut, called the “generic views” system.


# Each generic view needs to know what model it will be acting upon. This is provided using the model attribute.

class IndexView(generic.ListView):
    template_name = 'polls/index.html'
    context_object_name = 'latest_question_list'

    # For DetailView the question variable is provided automatically – since we’re using a Django model (Question),
    # Django is able to determine an appropriate name for the context variable. However, for ListView, the automatically
    # generated context variable is question_list. To override this we provide the context_object_name attribute, specifying
    # that we want to use latest_question_list instead.

    def get_queryset(self):
        """Return the last five published questions."""
        return Question.objects.order_by('-pub_date')[:5]
        # return Question.objects.filter(pub_date__lte=timezone.now())


class DetailView(generic.DetailView):
    model = Question
    template_name = 'polls/detail.html'

    def get_queryset(self):
        """
        Excludes any questions that aren't published yet.
        """
        return Question.objects.filter(pub_date__lte=timezone.now())


def vote(request, question_id):
    question = get_object_or_404(Question, pk=question_id)
    try:
        selected_choice = question.choice_set.get(pk=request.POST['choice'])
    # to request.POST einai ena object poy se afhnei na kaneis access ta dedomena me ena kapoio KEYNAME
    # Sthn prokeimenh periptwsh gyrnaei to ID ths epiloghs('choice') san string(request.POST values are always strings)
    except (KeyError, Choice.DoesNotExist):
        messages.info(request, "You didn't select a choice!")
        return HttpResponseRedirect(reverse('polls:detail', args=(question.id,)))

        # return HttpResponse("Welcome to poll's index!")
        # # # Redisplay the question voting form.
        # # return render(request, 'polls/index.html', {
        # #     'question': question,
        # #     'error_message': "You didn't select a choice.",
        #
        # })
    else:
        selected_choice.votes += 1
        selected_choice.save()
        # Always return an HttpResponseRedirect after successfully dealing
        # with POST data. This prevents data from being posted twice if a
        # user hits the Back button.
        return HttpResponseRedirect(reverse('polls:results', args=(question.id,)))
    # to HttpResponseRedirect pairnei mono ena argument : to URL sto opoio tha ginei redirected o xrhsths
    # otan exoume teleiwsei me POST data kalo tha htan na epistrfoyme me HttpResponseRedirect
    # to reverse einai function , bohthaei ston na mhn exoume hardcoded url , twra tha epistrefei to /polls/question.id/results/'


class ResultsView(generic.DetailView):
    model = Question
    template_name = 'polls/results.html'


def get_queryset(self):
    """
    Return the last five published questions (not including those set to be
    published in the future).
    """
    return Question.objects.filter(
        pub_date__lte=timezone.now()
    ).order_by('-pub_date')[:5]


def choice_chart(request, question_id):
    labels = []
    data = []
    queryset = Choice.objects.filter(question_id=question_id).values().annotate(votes=Sum('votes')).order_by(
        '-question_id')
    for entry in queryset:
        labels.append(entry['choice_text'])
        data.append(entry['votes'])

    return JsonResponse(data={
        'labels': labels,
        'data': data,
    })
