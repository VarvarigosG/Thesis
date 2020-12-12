from django.contrib import messages
from django.db.models import Sum
from django.http import HttpResponseRedirect
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, render, redirect
from django.urls import reverse
from django.utils import timezone
from django.views import generic

from .forms import PostForm
# from .forms import DetailForm
from .models import Choice, Question


class IndexView(generic.ListView):
    template_name = 'polls/index.html'
    context_object_name = 'latest_question_list'

    # For DetailView the question variable is provided automatically – since we’re using a Django model (Question),
    # Django is able to determine an appropriate name for the context variable. However, for ListView, the automatically
    # generated context variable is question_list. To override this we provide the context_object_name attribute, specifying
    # that we want to use latest_question_list instead.

    def get_queryset(self):
        """Return the last five published questions."""
        return Question.objects.filter(id__gte=101).filter(id__lt=200)
        # return Question.objects.order_by('-pub_date')[:5]

    def get_context_data(self, **kwargs):
        context = super(IndexView, self).get_context_data(**kwargs)
        context.update({
            'latest_question_list2': Question.objects.filter(id__gte=201).filter(id__lt=300).values(),
            'latest_question_list3': Question.objects.filter(id__gte=301).filter(id__lte=400).values(),
            'latest_question_list4': Question.objects.filter(id__gte=401).filter(id__lte=500).values(),
            'latest_question_list5': Question.objects.filter(id__gte=501).filter(id__lte=600).values(),
            'latest_question_list6': Question.objects.filter(id__gte=601).filter(id__lte=700).values(),
            'latest_question_list7': Question.objects.filter(id__gte=701).filter(id__lte=800).values(),
            'latest_question_list8': Question.objects.filter(id__gte=801).filter(id__lte=900).values(),
            'latest_question_list9': Question.objects.filter(id__gte=901).filter(id__lte=1000).values(),
            'latest_question_list10': Question.objects.filter(id__gte=1001).filter(id__lte=1100).values(),
            'latest_question_list11': Question.objects.filter(id__gte=2001).filter(id__lte=2100).values(),
            'latest_question_list12': Question.objects.filter(id__gte=3001).filter(id__lte=3100).values(),

        })
        return context


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
    else:
        selected_choice.votes += 1
        selected_choice.save()
        # Always return an HttpResponseRedirect after successfully dealing
        # with POST data. This prevents data from being posted twice if a
        # user hits the Back button.

        if question_id == 108 or question_id == 205 or question_id == 304 or question_id == 407 or question_id == 506 or question_id == 604 or question_id == 704 or question_id == 807 or question_id == 906 or question_id == 1007 or question_id == 2004 or question_id == 3006:

            # Choice.objects.filter(question_id=102).update(votes=0)
            # douleuei

            return HttpResponseRedirect(reverse('polls:results', args=(question.id,)))
        else:
            return HttpResponseRedirect(reverse('polls:detail', args=(question.id + 1,)))

        if question_id == 108:
            # Choice.objects.filter(question_id=102).update(votes=0)
            Choice.objects.filter(id__range=[41, 76]).update(votes=0)
            return HttpResponseRedirect(reverse('polls:results', args=(question.id,)))
        elif question_id == 205:
            Choice.objects.filter(id__range=[82, 103]).update(votes=0)
            return HttpResponseRedirect(reverse('polls:results', args=(question.id,)))
        elif question_id == 304:
            Choice.objects.filter(id__range=[104, 117]).update(votes=0)
            return HttpResponseRedirect(reverse('polls:results', args=(question.id,)))
        elif question_id == 407:
            Choice.objects.filter(id__range=[118, 141]).update(votes=0)
            return HttpResponseRedirect(reverse('polls:results', args=(question.id,)))
        elif question_id == 506:
            Choice.objects.filter(id__range=[142, 162]).update(votes=0)
            return HttpResponseRedirect(reverse('polls:results', args=(question.id,)))
        elif question_id == 604:
            Choice.objects.filter(id__range=[163, 179]).update(votes=0)
            return HttpResponseRedirect(reverse('polls:results', args=(question.id,)))
        elif question_id == 704:
            Choice.objects.filter(id__range=[180, 199]).update(votes=0)
            return HttpResponseRedirect(reverse('polls:results', args=(question.id,)))
        elif question_id == 807:
            Choice.objects.filter(id__range=[200, 231]).update(votes=0)
            return HttpResponseRedirect(reverse('polls:results', args=(question.id,)))
        elif question_id == 906:
            Choice.objects.filter(id__range=[232, 261]).update(votes=0)
            return HttpResponseRedirect(reverse('polls:results', args=(question.id,)))
        elif question_id == 1007:
            Choice.objects.filter(id__range=[262, 294]).update(votes=0)
            return HttpResponseRedirect(reverse('polls:results', args=(question.id,)))
        elif question_id == 2004:
            Choice.objects.filter(id__range=[295, 311]).update(votes=0)
            return HttpResponseRedirect(reverse('polls:results', args=(question.id,)))
        elif question_id == 3006:
            Choice.objects.filter(id__range=[312, 338]).update(votes=0)
            return HttpResponseRedirect(reverse('polls:results', args=(question.id,)))
        else:
            return HttpResponseRedirect(reverse('polls:detail', args=(question.id + 1,)))

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


# def get_form(request, ):
#     if request.method == "POST":
#         form = PostForm(request.POST)
#         if form.is_valid():
#             post = form.save(commit=False)
#             # post.author = request.user
#             # post.published_date = timezone.now()
#             post.save()
#             return redirect('post_detail', pk=post.pk)
#     else:
#         form = PostForm()
#     return render(request, 'polls/percentage.html', {'form': form})


def get_percenatage(request):
    Choice.objects.filter(id__range=[41, 76]).exclude(id__range=[42, 45]).exclude(id__range=[51, 54]).exclude(id__range=[70, 71]).filter(votes=1)
    x = 0
    y = []
    if 'choice_text' == 'Absolutely':
            x = x + 1
    elif 'choice_text' == 'Much':
            x = x + 2
    elif 'choice_text' == 'Moderate':
            x = x + 3
    elif 'choice_text' == 'A little bit':
            x = x + 4
    elif 'choice_text' == 'Not at all':
            x = x + 5
    y = (100 / (5 * 5)) * x;
    return render(request, 'polls/percentage.html', {y})