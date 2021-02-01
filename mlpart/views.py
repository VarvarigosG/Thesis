import json
import pickle

import joblib
import matplotlib
import numpy as np
import pandas as pd
import shap
from django.contrib import messages
from django.http import JsonResponse
from django.shortcuts import HttpResponseRedirect, reverse
from django.shortcuts import render
from numpy.random import rand
from rest_framework import status
from rest_framework import views
from rest_framework import viewsets
from rest_framework.decorators import api_view
from rest_framework.response import Response
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mlpart.api.serializers import approvalsSerializers
from thesis.wsgi import registry
from .forms import UploadForm, ApprovalForm
from .models import MLAlgorithm, MLRequest
from .models import approvals


def FileUploadView(request):
    # gia to upload
    if request.method == 'POST':
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            # messages.success(request, 'Your file had been uploaded successfully.')
            return HttpResponseRedirect(reverse('upload'))
    else:
        form = UploadForm()
        context = {
            'form': form,
        }
    return render(request, 'mlpart/upload.html', context)


class PredictView(views.APIView):
    def post(self, request, endpoint_name, format=None):

        algorithm_status = self.request.query_params.get("status", "production")
        algorithm_version = self.request.query_params.get("version")

        algs = MLAlgorithm.objects.filter(parent_endpoint__name=endpoint_name, status__status=algorithm_status,
                                          status__active=True)

        if algorithm_version is not None:
            algs = algs.filter(version=algorithm_version)

        if len(algs) == 0:
            return Response(
                {"status": "Error", "message": "ML algorithm is not available"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        if len(algs) != 1 and algorithm_status != "ab_testing":
            return Response(
                {"status": "Error",
                 "message": "ML algorithm selection is ambiguous. Please specify algorithm version."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        alg_index = 0
        if algorithm_status == "ab_testing":
            alg_index = 0 if rand() < 0.5 else 1

        algorithm_object = registry.endpoints[algs[alg_index].id]
        prediction = algorithm_object.compute_prediction(request.data)

        label = prediction["label"] if "label" in prediction else "error"
        ml_request = MLRequest(
            input_data=json.dumps(request.data),
            full_response=prediction,
            response=label,
            feedback="",
            parent_mlalgorithm=algs[alg_index],
        )
        ml_request.save()

        prediction["request_id"] = ml_request.id

        return Response(prediction)


# gia to bankLoanNN
class ApprovalsView(viewsets.ModelViewSet):
    queryset = approvals.objects.all()
    serializer_class = approvalsSerializers
    # kathe fora poy kanoume ena request toy leme pare ola ta pragmata apo to approvals Class


# einai h methodos pou kaloume to hdh trained model
@api_view(["POST"])
def approvereject(unit):
    try:
        mdl = pickle.load(open("/Users/gvarv/anaconda3/envs/thesis/Bank Loan/loan_model.pkl", "rb"))
        scalers = pickle.load(open("/Users/gvarv/anaconda3/envs/thesis/Bank Loan/scalers.pkl", "r"))
        X = scalers.transform(unit)
        y_pred = mdl.predict(X)
        y_pred = (y_pred > 0.58)
        newdf = pd.DataFrame(y_pred, columns=['Status'])
        newdf = newdf.replace({True: 'Approved', False: 'Rejected'})
        return JsonResponse('Your Status is {}'.format(newdf), encoding="utf-8", safe=False)
    except ValueError as e:
        return Response(e.args[0], status.HTTP_400_BAD_REQUEST)


def ohevalue(df):
    ohe_col = pickle.load(open("/Users/gvarv/anaconda3/envs/thesis/Bank Loan/allcol.pkl", errors="ignore"))
    cat_columns = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']

    df_processed = pd.get_dummies(df, columns=cat_columns)
    newdict = {}
    for i in ohe_col:
        if i in df_processed.columns:
            newdict[i] = df_processed[i].values
        else:
            newdict[i] = 0
    newdf = pd.DataFrame(newdict)
    return newdf


# einai h methodos gia to form ayth pou kanei to submit
def cxcontact(request):
    if request.method == "POST":
        form = ApprovalForm(request.POST)
        if form.is_valid():
            Firstname = form.cleaned_data['Firstname']
            Lastname = form.cleaned_data['Lastname']
            Dependants = form.cleaned_data['Dependants']
            ApplicantIncome = form.cleaned_data['ApplicantIncome']
            CoapplicatIncome = form.cleaned_data['CoapplicatIncome']
            LoanAmount = form.cleaned_data['LoanAmount']
            Loan_Amount_Term = form.cleaned_data['Loan_Amount_Term']
            Credit_History = form.cleaned_data['Credit_History']
            Gender = form.cleaned_data['Gender']
            Married = form.cleaned_data['Married']
            Education = form.cleaned_data['Education']
            Self_Employed = form.cleaned_data['Self_Employed']
            Property_Area = form.cleaned_data['Property_Area']
            # kanei submit ena dictionary me kanonikes lexeis oi opoies den einai one hot encoded pou eixame
            myDict = (request.POST).dict()
            df = pd.DataFrame(myDict, index=[0])
            answer = (approvereject(ohevalue(df)))
            messages.success(request, "Application Status: {0}".format(answer))
            # return JsonResponse('Your Status is {}'.format(answer), safe=False)
            # print(ohevalue(df))
    form = ApprovalForm()

    return render(request, 'mlpart/form.html', {'form': form})


# kanei render thn selida
def mpg(request):
    return render(request, 'mlpart/indexMPG.html')


reloadModel = joblib.load(r"C:\Users\gvarv\anaconda3\envs\thesis\MPG\RFModelforMPG.pkl")


# einai h methodos poy tha mas kanei predict to apotelesma
# epishs gia na paroyme ta submited pragmata apo xrhsth prepei na exoume dictionary
def predictMPG(request):
    # print (request)
    if request.method == 'POST':
        # print (request.POST.dict()) #ektypwnei cmd ta submited form
        # print (request.POST.get('cylinderVal')) #afou einai dictionary mporw na kanw access tis times
        temp = {}
        temp['cylinders'] = request.POST.get('cylinderVal')
        temp['displacement'] = request.POST.get('dispVal')
        temp['horsepower'] = request.POST.get('hrsPwrVal')
        temp['weight'] = request.POST.get('weightVal')
        temp['acceleration'] = request.POST.get('accVal')
        temp['model_year'] = request.POST.get('modelVal')
        temp['origin'] = request.POST.get('originVal')

        temp2 = temp.copy()
        temp2['model year'] = temp['model_year']
        print(temp.keys(), temp2.keys())
        # del temp2['model_year']
        # ayto giati sthn arxh me to model_year barage error afou sto dataset einai model year

        # kai twra pou exw ta dedomena prepei na fortwsw to modelo

        # to model perimenei DATAFRAME enw emeis exoume dictionary opote prepei na kanoume thn allagh
        testDtaa = pd.DataFrame({'x': temp2}).transpose()  # kai twra poy egine h allagh prepei na kanoume to predict
        scoreval = reloadModel.predict(testDtaa)[0]  # kai to pername mesa sto context
        context = {'scoreval': scoreval}
    return render(request, 'mlpart/resultsMPG.html', context)


def diabetesinsideDjango(request):
    return render(request, 'mlpart/indexDiabetes.html')


def DiabetesModel(request):
    if request.method == 'POST':
        temp3 = {}
        temp3['Pregnancies'] = request.POST.get('Val1')
        temp3['Glucose'] = request.POST.get('Val2')
        temp3['BloodPressure'] = request.POST.get('Val3')
        temp3['SkinThickness'] = request.POST.get('Val4')
        temp3['Insulin'] = request.POST.get('Val5')
        temp3['BMI'] = request.POST.get('Val6')
        temp3['DiabetesPedigreeFunction'] = request.POST.get('Val7')
        temp3['Age'] = request.POST.get('Val8')

        diabetesDF = pd.read_csv(r"C:\Users\gvarv\anaconda3\envs\thesis\Diabetes\diabetes.csv")
        dfTrain = diabetesDF[:650]
        dfTest = diabetesDF[650:750]
        dfCheck = diabetesDF[750:]
        trainLabel = np.asarray(dfTrain['Outcome'])
        trainData = np.asarray(dfTrain.drop('Outcome', 1))
        testLabel = np.asarray(dfTest['Outcome'])
        testData = np.asarray(dfTest.drop('Outcome', 1))
        means = np.mean(trainData, axis=0)
        stds = np.std(trainData, axis=0)
        trainData = (trainData - means) / stds
        testData = (testData - means) / stds
        diabetesCheck = LogisticRegression()
        diabetesCheck.fit(trainData, trainLabel)
        joblib.dump(diabetesCheck, 'diabeteseModelInsideDjango.pkl')
        # sampleDataFeatures = np.asarray(temp3)
        # sampleDataFeatures =np.array(temp3.items()))
        ##y = temp3.astype(np.float)

        sampleDataFeatures = np.array(
            [list(item.values()) for item in ({'x': temp3}).values()])  # to temp3 to kanoyme numpyarray
        sampleDataFeatures = np.asarray(sampleDataFeatures, dtype=np.float64,
                                        order='C')  # ta kanoyme float giati einai string
        # print(sampleDataFeatures)
        # print(means)
        sampleDataFeatures = np.subtract(sampleDataFeatures, means)
        sampleDataFeatures = sampleDataFeatures / stds
        # print(sampleDataFeatures)

        # # predict
        diabeteseModelInsideDjango = joblib.load(r"C:\Users\gvarv\anaconda3\envs\thesis\diabeteseModelInsideDjango.pkl")
        predictionProbability = diabeteseModelInsideDjango.predict_proba(sampleDataFeatures)
        prediction = diabeteseModelInsideDjango.predict(sampleDataFeatures)
        print(predictionProbability)
        print(prediction)

        if prediction[0] == 0:
            print("Diabetes will NOT occur")
            messages.success(request, "Diabetes will NOT occur")
        elif prediction[0] == 1:
            print("Diabetes will  occur")
            messages.success(request, "Diabetes will occur")

        print(sampleDataFeatures)
        print(trainData[0, :])

        # Summaryplot kai dependence plot me ta test Data
        # ex = shap.KernelExplainer(diabeteseModelInsideDjango.predict, testData)
        # shap_values = ex.shap_values(testData)
        #
        # fig = shap.summary_plot(shap_values, testData, plot_type="bar",
        #                         feature_names=['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness', 'Insulin',
        #                                        'BMI', 'Diabetes Pedigree', 'Age'], show=False, sort=False)
        # plt.savefig("mlpart/static/mlpart/SPbartestdata.jpeg", format='jpeg', dpi=130, bbox_inches='tight')
        #
        #
        #
        # fig = shap.summary_plot(shap_values, testData, show=False, plot_type='dot',
        #                         feature_names=['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness', 'Insulin',
        #                                        'BMI', 'Diabetes Pedigree', 'Age'])
        # plt.savefig("mlpart/static/mlpart/SPtestdata.jpeg", format='jpeg', dpi=130, bbox_inches='tight')
        #
        #
        #
        # fig = shap.dependence_plot("Age", shap_values, testData, show=False, feature_names=['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness', 'Insulin',
        #                                        'BMI', 'Diabetes Pedigree', 'Age'])
        #
        # plt.savefig("mlpart/static/mlpart/DPAgetestdata.jpeg", format='jpeg', dpi=130, bbox_inches='tight')
        #
        #
        #
        # #Force Plots gia thn 14 timh twn test Data
        ex = shap.KernelExplainer(diabeteseModelInsideDjango.predict, testData)
        shap_values = ex.shap_values(testData[61, :])
        shap.initjs()
        fig = shap.force_plot(ex.expected_value, shap_values, testData[61, :], matplotlib=True, show=False,
                              feature_names=['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness', 'Insulin',
                                             'BMI', 'Diabetes Pedigree', 'Age'])

        fig.tight_layout()
        # fig = plt.gcf()
        # plt.tight_layout()
        # plt.tight_layout(pad=1.9, w_pad=3.5, h_pad=1.5)
        # fig.set_size_inches(5, 15)
        plt.savefig("mlpart/static/mlpart/FPtestdata.jpeg", format='jpeg', dpi=850)

        # # # Summaryplot kai dependence mlpart/static/mlpartplot me ta input Data tou xrhsth
        # #
        # # # ex = shap.KernelExplainer(diabeteseModelInsideDjango.predict, sampleDataFeatures)
        # # # shap_values = ex.shap_values(sampleDataFeatures)
        # # #
        # # # fig = shap.summary_plot(shap_values, sampleDataFeatures, plot_type="bar", show=False)
        # # # shap.initjs()
        # # # plt.savefig("mlpart/static/mlpart/SPinputdata.svg", format='svg', dpi=150, bbox_inches='tight')
        # # #
        # # # fig = shap.dependence_plot("Feature 0", shap_values, sampleDataFeatures, show=False,)
        # # # shap.initjs()
        # # # plt.savefig("mlpart/static/mlpart/DPinputdata.svg", format='svg', dpi=150, bbox_inches='tight')
        #
        # #Force Plots gia thn prwth timh twn test Data kai toy Input toy xrhsth

    return render(request, 'mlpart/resultsDiabetes.html')


def Iris(request):
    return render(request, 'mlpart/indexIris.html')


def IrisModel(request):
    if request.method == 'POST':
        temp3 = {}
        temp3['SepalLength'] = request.POST.get('Val1')
        temp3['SepalWidth'] = request.POST.get('Val2')
        temp3['PetalLength'] = request.POST.get('Val3')
        temp3['PetalWidth'] = request.POST.get('Val4')

        df = pd.read_csv(r"C:\Users\gvarv\anaconda3\envs\thesis\Iris\iris.csv")
        df = df.drop(columns=['Id'])
        le = LabelEncoder()
        df['Species'] = le.fit_transform(df['Species'])
        X = df.drop(columns=['Species'])
        Y = df['Species']
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)
        model = KNeighborsClassifier()
        model.fit(x_train, y_train)

        print("Accuracy", model.score(x_test, y_test) * 100)
        joblib.dump(model, 'irismodel.pkl')

        LoadedModel = joblib.load('irismodel.pkl')
        accuracyModel = LoadedModel.score(x_test, y_test)
        print("accuracy = ", accuracyModel * 100, "%")

        sampleDataFeatures = np.array(
            [list(item.values()) for item in ({'x': temp3}).values()])  # to temp3 to kanoyme numpyarray
        sampleDataFeatures = np.asarray(sampleDataFeatures, dtype=np.float64,
                                        order='C')  # ta kanoyme float giati einai string

        print(sampleDataFeatures)

        prediction = LoadedModel.predict(sampleDataFeatures)
        print(prediction)

        if prediction[0] == 0:
            print("The species is Iris Setosa ")
            messages.success(request, "The species is Iris Setosa ")
        elif prediction[0] == 1:
            print("The species is Iris Versicolor ")
            messages.success(request, "The species is Iris Versicolor")
        elif prediction[0] == 2:
            print("The species is Iris Virginica ")
            messages.success(request, "The species is Iris Virginica")

        ex = shap.KernelExplainer(LoadedModel.predict, x_train)
        shap_values = ex.shap_values(x_test)
        shap.summary_plot(shap_values, x_test, plot_type="bar", show=False)
        plt.savefig("mlpart/static/mlpart/IRISSPxtest.jpeg", format='jpeg', dpi=150, bbox_inches='tight')

    return render(request, 'mlpart/resultsIris.html')
