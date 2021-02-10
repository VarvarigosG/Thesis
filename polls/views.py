from django.contrib import messages
from django.db.models import Sum
from django.http import HttpResponseRedirect
from django.http import JsonResponse, FileResponse
from django.shortcuts import get_object_or_404
from django.shortcuts import render
from django.urls import reverse
from django.utils import timezone
from django.views import generic

from .models import Choice, Question


class IndexView(generic.ListView):
    template_name = 'polls/index2.html'
    context_object_name = 'latest_question_list'

    # For DetailView the question variable is provided automatically – since we’re using a Django model (Question),
    # Django is able to determine an appropriate name for the context variable. However, for ListView, the automatically
    # generated context variable is question_list. To override this we provide the context_object_name attribute, specifying
    # that we want to use latest_question_list instead.

    def get_queryset(self):
        """Return the last five published questions."""
        return Question.objects.filter(id__gte=101).filter(id__lt=102)
        # return Question.objects.order_by('-pub_date')[:5]

    def get_context_data(self, **kwargs):
        context = super(IndexView, self).get_context_data(**kwargs)
        context.update({
            'latest_question_list2': Question.objects.filter(id__gte=201).filter(id__lt=202).values(),
            'latest_question_list3': Question.objects.filter(id__gte=301).filter(id__lt=302).values(),
            'latest_question_list4': Question.objects.filter(id__gte=401).filter(id__lt=402).values(),
            'latest_question_list5': Question.objects.filter(id__gte=501).filter(id__lt=502).values(),
            'latest_question_list6': Question.objects.filter(id__gte=601).filter(id__lt=602).values(),
            'latest_question_list7': Question.objects.filter(id__gte=701).filter(id__lt=702).values(),
            'latest_question_list8': Question.objects.filter(id__gte=801).filter(id__lt=802).values(),
            'latest_question_list9': Question.objects.filter(id__gte=901).filter(id__lt=902).values(),
            'latest_question_list10': Question.objects.filter(id__gte=1001).filter(id__lt=1002).values(),
            'latest_question_list11': Question.objects.filter(id__gte=2001).filter(id__lt=2002).values(),
            'latest_question_list12': Question.objects.filter(id__gte=3001).filter(id__lt=3002).values(),

        })
        return context


class DetailView(generic.DetailView):
    model = Question
    template_name = 'polls/detail2.html'

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
            return HttpResponseRedirect(reverse('polls:results', args=(question.id,)))
        else:
            return HttpResponseRedirect(reverse('polls:detail', args=(question.id + 1,)))

    # to HttpResponseRedirect pairnei mono ena argument : to URL sto opoio tha ginei redirected o xrhsths
    # otan exoume teleiwsei me POST data kalo tha htan na epistrfoyme me HttpResponseRedirect
    # to reverse einai function , bohthaei ston na mhn exoume hardcoded url , twra tha epistrefei to /polls/question.id/results/'


class ResultsView(generic.DetailView):
    model = Question
    template_name = 'polls/results2.html'


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


##deixnei ta percentages se kathe result graph
def get_percentage(request, question_id):
    if question_id == 108:
        queryset1 = Choice.objects.filter(id__range=[41, 76]).exclude(id__range=[42, 45]).exclude(
            id__range=[65, 71]).exclude(id__range=[51, 54]).filter(votes=1).values()
        queryset2 = Choice.objects.filter(id__range=[65, 69]).filter(votes=1).values()
        x = 0
        y = 0
        labels2 = []
        labels = []
        data = []
        category = ['Responsibilty']
        for entry in queryset1:
            labels.append(entry['choice_text'])
        for entry in labels:
            if entry == 'Absolutely' in labels:
                x = x + 5
            elif entry == 'Much' in labels:
                x = x + 4
            elif entry == 'Moderate' in labels:
                x = x + 3
            elif entry == 'A little bit' in labels:
                x = x + 2
            elif entry == 'Not at all' in labels:
                x = x + 1

        for entry in queryset2:
            labels2.append(entry['choice_text'])
        for entry in labels2:
            if entry == 'Absolutely' in labels2:
                x = x + 1
            elif entry == 'Much' in labels2:
                x = x + 2
            elif entry == 'Moderate' in labels2:
                x = x + 3
            elif entry == 'A little bit' in labels2:
                x = x + 4
            elif entry == 'Not at all' in labels2:
                x = x + 5
        y = (100 / (5 * 5)) * (x)
        data.append(y)
        # Choice.objects.filter(id__range=[41, 76]).update(votes=0)


    elif question_id == 205:
        queryset1 = Choice.objects.filter(id__range=[82, 103]).exclude(id__range=[92, 93]).filter(votes=1).values()
        x = 0
        y = 0
        labels = []
        data = []
        category = ['Explainability']
        for entry in queryset1:
            labels.append(entry['choice_text'])
        for entry in labels:
            if entry == 'Absolutely' in labels:
                x = x + 5
            elif entry == 'Much' in labels:
                x = x + 4
            elif entry == 'Moderate' in labels:
                x = x + 3
            elif entry == 'A little bit' in labels:
                x = x + 2
            elif entry == 'Not at all' in labels:
                x = x + 1
        y = (100 / (4 * 5)) * (x)
        data.append(y)
        # Choice.objects.filter(id__range=[82, 103]).update(votes=0)

    elif question_id == 304:
        queryset1 = Choice.objects.filter(id__range=[104, 117]).exclude(id__range=[104, 105]).exclude(
            id__range=[111, 112]).filter(votes=1).values()
        x = 0
        y = 0
        labels = []
        data = []
        category = ['Auditability']
        for entry in queryset1:
            labels.append(entry['choice_text'])
        for entry in labels:
            if entry == 'Absolutely' in labels:
                x = x + 5
            elif entry == 'Much' in labels:
                x = x + 4
            elif entry == 'Moderate' in labels:
                x = x + 3
            elif entry == 'A little bit' in labels:
                x = x + 2
            elif entry == 'Not at all' in labels:
                x = x + 1
        y = (100 / (2 * 5)) * (x)
        data.append(y)
        # Choice.objects.filter(id__range=[104, 117]).update(votes=0)

    elif question_id == 407:
        queryset1 = Choice.objects.filter(id__range=[122, 361]).exclude(id__range=[136, 356]).filter(votes=1).values()
        queryset2 = Choice.objects.filter(id__range=[137, 141]).filter(votes=1).values()
        x = 0
        y = 0
        labels2 = []
        labels = []
        data = []
        category = ['Accuracy']
        for entry in queryset1:
            labels.append(entry['choice_text'])
        for entry in labels:
            if entry == 'Absolutely' in labels:
                x = x + 5
            elif entry == 'Much' in labels:
                x = x + 4
            elif entry == 'Moderate' in labels:
                x = x + 3
            elif entry == 'A little bit' in labels:
                x = x + 2
            elif entry == 'Not at all' in labels:
                x = x + 1

        for entry in queryset2:
            labels2.append(entry['choice_text'])
        for entry in labels2:
            if entry == 'Absolutely' in labels2:
                x = x + 1
            elif entry == 'Much' in labels2:
                x = x + 2
            elif entry == 'Moderate' in labels2:
                x = x + 3
            elif entry == 'A little bit' in labels2:
                x = x + 4
            elif entry == 'Not at all' in labels2:
                x = x + 5
        y = (100 / (5 * 5)) * (x)
        data.append(y)
        # Choice.objects.filter(id__range=[118, 361]).exclude(id__range=[142, 356]).update(votes=0)

    elif question_id == 506:
        queryset1 = Choice.objects.filter(id__range=[144, 160]).exclude(id__range=[154, 155]).filter(votes=1).values()
        x = 0
        y = 0
        labels = []
        data = []
        category = ['Fairness']
        for entry in queryset1:
            labels.append(entry['choice_text'])
        for entry in labels:
            if entry == 'Absolutely' in labels:
                x = x + 5
            elif entry == 'Much' in labels:
                x = x + 4
            elif entry == 'Moderate' in labels:
                x = x + 3
            elif entry == 'A little bit' in labels:
                x = x + 2
            elif entry == 'Not at all' in labels:
                x = x + 1
        y = (100 / (3 * 5)) * (x)
        data.append(y)
        # Choice.objects.filter(id__range=[142, 162]).update(votes=0)

    elif question_id == 604:
        queryset1 = Choice.objects.filter(id__range=[165, 179]).filter(votes=1).values()
        x = 0
        y = 0
        labels = []
        data = []
        category = ['Communication']
        for entry in queryset1:
            labels.append(entry['choice_text'])
        for entry in labels:
            if entry == 'Absolutely' in labels:
                x = x + 5
            elif entry == 'Much' in labels:
                x = x + 4
            elif entry == 'Moderate' in labels:
                x = x + 3
            elif entry == 'A little bit' in labels:
                x = x + 2
            elif entry == 'Not at all' in labels:
                x = x + 1
        y = (100 / (3 * 5)) * (x)
        data.append(y)
        # Choice.objects.filter(id__range=[162, 179]).update(votes=0)

    elif question_id == 704:
        queryset1 = Choice.objects.filter(id__range=[180, 199]).filter(votes=1).values()
        x = 0
        y = 0
        labels = []
        data = []
        category = ['Accesibility']
        for entry in queryset1:
            labels.append(entry['choice_text'])
        for entry in labels:
            if entry == 'Absolutely' in labels:
                x = x + 5
            elif entry == 'Much' in labels:
                x = x + 4
            elif entry == 'Moderate' in labels:
                x = x + 3
            elif entry == 'A little bit' in labels:
                x = x + 2
            elif entry == 'Not at all' in labels:
                x = x + 1
        y = (100 / (4 * 5)) * (x)
        data.append(y)
        # Choice.objects.filter(id__range=[180, 199]).update(votes=0)

    elif question_id == 807:
        queryset1 = Choice.objects.filter(id__range=[200, 231]).exclude(id__range=[215, 216]).exclude(
            id__range=[222, 226]).filter(votes=1).values()
        queryset2 = Choice.objects.filter(id__range=[222, 226]).filter(votes=1).values()
        x = 0
        y = 0
        labels2 = []
        labels = []
        data = []
        category = ['Data Evaluation']
        for entry in queryset1:
            labels.append(entry['choice_text'])
        for entry in labels:
            if entry == 'Absolutely' in labels:
                x = x + 5
            elif entry == 'Much' in labels:
                x = x + 4
            elif entry == 'Moderate' in labels:
                x = x + 3
            elif entry == 'A little bit' in labels:
                x = x + 2
            elif entry == 'Not at all' in labels:
                x = x + 1

        for entry in queryset2:
            labels2.append(entry['choice_text'])
        for entry in labels2:
            if entry == 'Absolutely' in labels2:
                x = x + 1
            elif entry == 'Much' in labels2:
                x = x + 2
            elif entry == 'Moderate' in labels2:
                x = x + 3
            elif entry == 'A little bit' in labels2:
                x = x + 4
            elif entry == 'Not at all' in labels2:
                x = x + 5
        y = (100 / (6 * 5)) * (x)
        data.append(y)
        # Choice.objects.filter(id__range=[200, 231]).update(votes=0)

    elif question_id == 906:
        queryset1 = Choice.objects.filter(id__range=[232, 261]).exclude(id__range=[252, 256]).filter(votes=1).values()
        queryset2 = Choice.objects.filter(id__range=[252, 256]).filter(votes=1).values()
        x = 0
        y = 0
        labels2 = []
        labels = []
        data = []
        category = ['Model Evaluation']
        for entry in queryset1:
            labels.append(entry['choice_text'])
        for entry in labels:
            if entry == 'Absolutely' in labels:
                x = x + 5
            elif entry == 'Much' in labels:
                x = x + 4
            elif entry == 'Moderate' in labels:
                x = x + 3
            elif entry == 'A little bit' in labels:
                x = x + 2
            elif entry == 'Not at all' in labels:
                x = x + 1

        for entry in queryset2:
            labels2.append(entry['choice_text'])
        for entry in labels2:
            if entry == 'Absolutely' in labels2:
                x = x + 1
            elif entry == 'Much' in labels2:
                x = x + 2
            elif entry == 'Moderate' in labels2:
                x = x + 3
            elif entry == 'A little bit' in labels2:
                x = x + 4
            elif entry == 'Not at all' in labels2:
                x = x + 5
        y = (100 / (6 * 5)) * (x)
        data.append(y)
        # Choice.objects.filter(id__range=[232, 261]).update(votes=0)

    elif question_id == 1007:
        queryset1 = Choice.objects.filter(id__range=[262, 271]).filter(votes=1).values()
        queryset2 = Choice.objects.filter(id__range=[275, 294]).filter(votes=1).values()
        x = 0
        y = 0
        labels2 = []
        labels = []
        data = []
        category = ['Inferencing']
        for entry in queryset1:
            labels.append(entry['choice_text'])
        for entry in labels:
            if entry == 'Absolutely' in labels:
                x = x + 5
            elif entry == 'Much' in labels:
                x = x + 4
            elif entry == 'Moderate' in labels:
                x = x + 3
            elif entry == 'A little bit' in labels:
                x = x + 2
            elif entry == 'Not at all' in labels:
                x = x + 1

        for entry in queryset2:
            labels2.append(entry['choice_text'])
        for entry in labels2:
            if entry == 'Absolutely' in labels2:
                x = x + 1
            elif entry == 'Much' in labels2:
                x = x + 2
            elif entry == 'Moderate' in labels2:
                x = x + 3
            elif entry == 'A little bit' in labels2:
                x = x + 4
            elif entry == 'Not at all' in labels2:
                x = x + 5
        y = (100 / (6 * 5)) * (x)
        data.append(y)
        # Choice.objects.filter(id__range=[262, 294]).update(votes=0)

    elif question_id == 2004:
        queryset1 = Choice.objects.filter(id__range=[295, 311]).exclude(id__range=[300, 301]).filter(votes=1).values()
        x = 0
        y = 0
        labels = []
        data = []
        category = ['Algorithmic Presence']
        for entry in queryset1:
            labels.append(entry['choice_text'])
        for entry in labels:
            if entry == 'Absolutely' in labels:
                x = x + 5
            elif entry == 'Much' in labels:
                x = x + 4
            elif entry == 'Moderate' in labels:
                x = x + 3
            elif entry == 'A little bit' in labels:
                x = x + 2
            elif entry == 'Not at all' in labels:
                x = x + 1
        y = (100 / (3 * 5)) * (x)
        data.append(y)
        # Choice.objects.filter(id__range=[295, 311]).update(votes=0)

    elif question_id == 3006:
        queryset1 = Choice.objects.filter(id__range=[312, 338]).exclude(id__range=[332, 333]).filter(votes=1).values()
        x = 0
        y = 0
        labels = []
        data = []
        category = ['Performance Evaluation']
        for entry in queryset1:
            labels.append(entry['choice_text'])
        for entry in labels:
            if entry == 'Absolutely' in labels:
                x = x + 5
            elif entry == 'Much' in labels:
                x = x + 4
            elif entry == 'Moderate' in labels:
                x = x + 3
            elif entry == 'A little bit' in labels:
                x = x + 2
            elif entry == 'Not at all' in labels:
                x = x + 1
        y = (100 / (5 * 5)) * (x)
        data.append(y)
        # Choice.objects.filter(id__range=[312, 338]).update(votes=0)

    return JsonResponse(data={
        'labels': labels,
        'responsibilty': category,
        "data": data,
    })


def leapquestion(request, question_id):
    if question_id == 108:
        return HttpResponseRedirect(reverse('polls:detail', args=(question_id + 93,)))
    elif question_id == 205:
        return HttpResponseRedirect(reverse('polls:detail', args=(question_id + 96,)))
    elif question_id == 304:
        return HttpResponseRedirect(reverse('polls:detail', args=(question_id + 97,)))
    elif question_id == 407:
        return HttpResponseRedirect(reverse('polls:detail', args=(question_id + 94,)))
    elif question_id == 506:
        return HttpResponseRedirect(reverse('polls:detail', args=(question_id + 95,)))
    elif question_id == 604:
        return HttpResponseRedirect(reverse('polls:detail', args=(question_id + 97,)))
    elif question_id == 704:
        return HttpResponseRedirect(reverse('polls:detail', args=(question_id + 97,)))
    elif question_id == 807:
        return HttpResponseRedirect(reverse('polls:detail', args=(question_id + 94,)))
    elif question_id == 906:
        return HttpResponseRedirect(reverse('polls:detail', args=(question_id + 95,)))
    elif question_id == 1007:
        return HttpResponseRedirect(reverse('polls:detail', args=(question_id + 994,)))
    elif question_id == 2004:
        return HttpResponseRedirect(reverse('polls:detail', args=(question_id + 997,)))
    elif question_id == 3006:

        #Responsibility
        queryset1 = Choice.objects.filter(id__range=[41, 76]).exclude(id__range=[42, 45]).exclude(
            id__range=[65, 71]).exclude(id__range=[51, 54]).filter(votes=1).values()
        queryset2 = Choice.objects.filter(id__range=[65, 69]).filter(votes=1).values()
        x = 0
        y = 0
        labels2 = []
        labels = []
        for entry in queryset1:
            labels.append(entry['choice_text'])
        for entry in labels:
            if entry == 'Absolutely' in labels:
                x = x + 5
            elif entry == 'Much' in labels:
                x = x + 4
            elif entry == 'Moderate' in labels:
                x = x + 3
            elif entry == 'A little bit' in labels:
                x = x + 2
            elif entry == 'Not at all' in labels:
                x = x + 1

        for entry in queryset2:
            labels2.append(entry['choice_text'])
        for entry in labels2:
            if entry == 'Absolutely' in labels2:
                x = x + 1
            elif entry == 'Much' in labels2:
                x = x + 2
            elif entry == 'Moderate' in labels2:
                x = x + 3
            elif entry == 'A little bit' in labels2:
                x = x + 4
            elif entry == 'Not at all' in labels2:
                x = x + 5
        y = (100 / (5 * 5)) * (x)
        # Choice.objects.filter(id__range=[41, 76]).update(votes=0)

        #Explainability

        queryset3 = Choice.objects.filter(id__range=[82, 103]).exclude(id__range=[92, 93]).filter(votes=1).values()
        x3 = 0
        y3 = 0
        labels3 = []

        for entry in queryset3:
            labels3.append(entry['choice_text'])
        for entry in labels3:
            if entry == 'Absolutely' in labels3:
                x3 = x3 + 5
            elif entry == 'Much' in labels3:
                x3 = x3 + 4
            elif entry == 'Moderate' in labels3:
                x3 = x3 + 3
            elif entry == 'A little bit' in labels3:
                x3 = x3 + 2
            elif entry == 'Not at all' in labels3:
                x3 = x3 + 1
        y3 = (100 / (4 * 5)) * (x3)

        #Auditability

        queryset4 = Choice.objects.filter(id__range=[104, 117]).exclude(id__range=[104, 105]).exclude(
            id__range=[111, 112]).filter(votes=1).values()
        x4 = 0
        y4 = 0
        labels4 = []
        for entry in queryset4:
            labels4.append(entry['choice_text'])
        for entry in labels4:
            if entry == 'Absolutely' in labels4:
                x4 = x4 + 5
            elif entry == 'Much' in labels4:
                x4 = x4 + 4
            elif entry == 'Moderate' in labels4:
                x4 = x4 + 3
            elif entry == 'A little bit' in labels4:
                x4 = x4 + 2
            elif entry == 'Not at all' in labels4:
                x = x4 + 1
        y4 = (100 / (2 * 5)) * (x4)

        #Accuracy

        queryset5 = Choice.objects.filter(id__range=[122, 361]).exclude(id__range=[136, 356]).filter(votes=1).values()
        queryset6 = Choice.objects.filter(id__range=[137, 141]).filter(votes=1).values()
        x5 = 0
        y5 = 0
        labels6 = []
        labels5 = []

        for entry in queryset5:
            labels5.append(entry['choice_text'])
        for entry in labels5:
            if entry == 'Absolutely' in labels5:
                x5 = x5 + 5
            elif entry == 'Much' in labels5:
                x5 = x5 + 4
            elif entry == 'Moderate' in labels5:
                x5 = x5 + 3
            elif entry == 'A little bit' in labels5:
                x5 = x5 + 2
            elif entry == 'Not at all' in labels5:
                x5= x5 + 1

        for entry in queryset6:
            labels6.append(entry['choice_text'])
        for entry in labels6:
            if entry == 'Absolutely' in labels6:
                x5 = x5 + 1
            elif entry == 'Much' in labels6:
                x5 = x5 + 2
            elif entry == 'Moderate' in labels6:
                x5= x5 + 3
            elif entry == 'A little bit' in labels6:
                x5 = x5 + 4
            elif entry == 'Not at all' in labels6:
                x5 = x5 + 5
        y5 = (100 / (5 * 5)) * (x5)

        #Fairness

        queryset7 = Choice.objects.filter(id__range=[144, 160]).exclude(id__range=[154, 155]).filter(votes=1).values()
        x6 = 0
        y6 = 0
        labels7 = []

        for entry in queryset7:
            labels7.append(entry['choice_text'])
        for entry in labels7:
            if entry == 'Absolutely' in labels7:
                x6 = x6 + 5
            elif entry == 'Much' in labels7:
                x6 = x6 + 4
            elif entry == 'Moderate' in labels7:
                x6 = x6 + 3
            elif entry == 'A little bit' in labels7:
                x6 = x6 + 2
            elif entry == 'Not at all' in labels7:
                x6 = x6 + 1
        y6 = (100 / (3 * 5)) * (x6)

        #Communication

        queryset8 = Choice.objects.filter(id__range=[165, 179]).filter(votes=1).values()
        x7 = 0
        y7 = 0
        labels8 = []
        for entry in queryset8:
            labels8.append(entry['choice_text'])
        for entry in labels8:
            if entry == 'Absolutely' in labels8:
                x7 = x7 + 5
            elif entry == 'Much' in labels8:
                x7 = x7 + 4
            elif entry == 'Moderate' in labels8:
                x7 = x7 + 3
            elif entry == 'A little bit' in labels8:
                x7 = x7 + 2
            elif entry == 'Not at all' in labels8:
                x7 = x7 + 1
        y7 = (100 / (3 * 5)) * (x7)

        #Accesibility
        queryset9 = Choice.objects.filter(id__range=[180, 199]).filter(votes=1).values()
        x8 = 0
        y8 = 0
        labels9 = []
        for entry in queryset9:
            labels9.append(entry['choice_text'])
        for entry in labels9:
            if entry == 'Absolutely' in labels9:
                x8 = x8 + 5
            elif entry == 'Much' in labels9:
                x8 = x8 + 4
            elif entry == 'Moderate' in labels9:
                x8 = x8 + 3
            elif entry == 'A little bit' in labels9:
                x8 = x8 + 2
            elif entry == 'Not at all' in labels9:
                x8 = x8 + 1
        y8 = (100 / (4 * 5)) * (x8)

        #Data

        queryset10 = Choice.objects.filter(id__range=[200, 231]).exclude(id__range=[215, 216]).exclude(
            id__range=[222, 226]).filter(votes=1).values()
        queryset11 = Choice.objects.filter(id__range=[222, 226]).filter(votes=1).values()
        x9 = 0
        y9 = 0
        labels11 = []
        labels10 = []
        for entry in queryset10:
            labels10.append(entry['choice_text'])
        for entry in labels10:
            if entry == 'Absolutely' in labels10:
                x9 = x9 + 5
            elif entry == 'Much' in labels10:
                x9 = x9 + 4
            elif entry == 'Moderate' in labels10:
                x9 = x9 + 3
            elif entry == 'A little bit' in labels10:
                x9 = x9 + 2
            elif entry == 'Not at all' in labels10:
                x9 = x9 + 1

        for entry in queryset11:
            labels11.append(entry['choice_text'])
        for entry in labels11:
            if entry == 'Absolutely' in labels11:
                x9 = x9 + 1
            elif entry == 'Much' in labels11:
                x9 = x9 + 2
            elif entry == 'Moderate' in labels11:
                x9 = x9 + 3
            elif entry == 'A little bit' in labels11:
                x9 = x9 + 4
            elif entry == 'Not at all' in labels11:
                x9 = x9 + 5
        y9 = (100 / (6 * 5)) * (x9)

        #The Model

        queryset12 = Choice.objects.filter(id__range=[232, 261]).exclude(id__range=[252, 256]).filter(votes=1).values()
        queryset13 = Choice.objects.filter(id__range=[252, 256]).filter(votes=1).values()
        x10 = 0
        y10 = 0
        labels13 = []
        labels12 = []
        for entry in queryset12:
            labels12.append(entry['choice_text'])
        for entry in labels12:
            if entry == 'Absolutely' in labels12:
                x10 = x10 + 5
            elif entry == 'Much' in labels12:
                x10 = x10 + 4
            elif entry == 'Moderate' in labels12:
                x10 = x10 + 3
            elif entry == 'A little bit' in labels12:
                x10 = x10 + 2
            elif entry == 'Not at all' in labels12:
                x10 = x10 + 1

        for entry in queryset13:
            labels13.append(entry['choice_text'])
        for entry in labels13:
            if entry == 'Absolutely' in labels13:
                x10 = x10 + 1
            elif entry == 'Much' in labels13:
                x10 = x10 + 2
            elif entry == 'Moderate' in labels13:
                x10 = x10 + 3
            elif entry == 'A little bit' in labels13:
                x10 = x10 + 4
            elif entry == 'Not at all' in labels13:
                x10 = x10 + 5
        y10 = (100 / (6 * 5)) * (x10)

        #Inferencing

        queryset14 = Choice.objects.filter(id__range=[262, 271]).filter(votes=1).values()
        queryset15 = Choice.objects.filter(id__range=[275, 294]).filter(votes=1).values()
        x11 = 0
        y11 = 0
        labels15 = []
        labels14 = []

        for entry in queryset14:
            labels14.append(entry['choice_text'])
        for entry in labels14:
            if entry == 'Absolutely' in labels14:
                x11 = x11 + 5
            elif entry == 'Much' in labels14:
                x11 = x11 + 4
            elif entry == 'Moderate' in labels14:
                x11 = x11 + 3
            elif entry == 'A little bit' in labels14:
                x11 = x11 + 2
            elif entry == 'Not at all' in labels14:
                x11 = x11 + 1

        for entry in queryset15:
            labels15.append(entry['choice_text'])
        for entry in labels15:
            if entry == 'Absolutely' in labels15:
                x11 = x11 + 1
            elif entry == 'Much' in labels15:
                x11 = x11 + 2
            elif entry == 'Moderate' in labels15:
                x11 = x11 + 3
            elif entry == 'A little bit' in labels15:
                x11 = x11 + 4
            elif entry == 'Not at all' in labels15:
                x11 = x11 + 5
        y11 = (100 / (6 * 5)) * (x11)

        #Algorithmic Presence

        queryset16 = Choice.objects.filter(id__range=[295, 311]).exclude(id__range=[300, 301]).filter(votes=1).values()
        x12 = 0
        y12 = 0
        labels16 = []
        for entry in queryset16:
            labels16.append(entry['choice_text'])
        for entry in labels16:
            if entry == 'Absolutely' in labels16:
                x12 = x12 + 5
            elif entry == 'Much' in labels16:
                x12 = x12 + 4
            elif entry == 'Moderate' in labels16:
                x12 = x12 + 3
            elif entry == 'A little bit' in labels16:
                x12 = x12 + 2
            elif entry == 'Not at all' in labels16:
                x12 = x12 + 1
        y12 = (100 / (3 * 5)) * (x12)

        #Performance Evaluation

        queryset17 = Choice.objects.filter(id__range=[312, 338]).exclude(id__range=[332, 333]).filter(votes=1).values()
        x13 = 0
        y13 = 0
        labels17 = []
        for entry in queryset17:
            labels17.append(entry['choice_text'])
        for entry in labels17:
            if entry == 'Absolutely' in labels17:
                x13 = x13 + 5
            elif entry == 'Much' in labels17:
                x13 = x13 + 4
            elif entry == 'Moderate' in labels17:
                x13 = x13 + 3
            elif entry == 'A little bit' in labels17:
                x13 = x13 + 2
            elif entry == 'Not at all' in labels17:
                x13 = x13 + 1
        y13 = (100 / (5 * 5)) * (x13)


        Org_Total=(y+y3+y4+y5+y6+y7+y8)/7
        Alg_Total=(y9+y10+y11+y12+y13)/5
        Total_Total=(Org_Total+Alg_Total)/2
        Choice.objects.filter(id__range=[41, 361]).update(votes=0)

        context = {'scoreval': y, 'scoreval2': y3, 'scoreval3': y4,'scoreval4': y5,'scoreval5': y6,'scoreval6': y7,'scoreval7': y8,'scoreval8': y9,'scoreval9': y10,'scoreval10': y11,'scoreval11': y12,'scoreval12': y13, 'scoreval13': Org_Total,'scoreval14': Alg_Total,'scoreval15':Total_Total}
        return render(request, 'polls/dashboard.html', context)

