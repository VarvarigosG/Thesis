import pickle

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from django.contrib import messages
from django.http import JsonResponse
from django.shortcuts import HttpResponseRedirect, reverse
from django.shortcuts import render
from rest_framework import status
from rest_framework import viewsets
from rest_framework.decorators import api_view
from rest_framework.response import Response
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

matplotlib.use('Agg')
from mlpart.api.serializers import approvalsSerializers

from .forms import UploadForm, ApprovalForm, UploadDataForm

from .models import approvals, FileOK, MLmodeldata


# def loadmlpart(request):
#     return render(request, 'mlpart/upload.html')


def FileUploadView(request):
    # gia to upload
    if request.method == 'POST':
        model_form = UploadForm(request.POST, request.FILES)
        data_form = UploadDataForm(request.POST, request.FILES)
        if model_form.is_valid() and data_form.is_valid():
            model_form.save()
            data_form.save()

            return HttpResponseRedirect(reverse('agnosticExplanation'))
        else:
            messages.warning(request, 'Unsupported file type!')
    else:
        form = UploadForm()
    context = {
        'model_form': UploadForm,
        'data_form': UploadDataForm,
    }
    return render(request, 'mlpart/upload.html', context)


def agnosticExplanation(request):
    # kane recover tis teleytaies egrafes
    model = FileOK.objects.latest('created').file.path
    data = MLmodeldata.objects.latest('created').data.path

    print(model)
    print(data)
    # fortwse tis
    modelReloaded = joblib.load(model)
    df = pd.read_csv(data)

    column_name_list = df.columns.tolist()
    print(column_name_list)
    print(len(column_name_list))
    print(column_name_list[0])
    list_length = len(column_name_list)
    print (list_length)
    context = {'length': list_length}
    # shap gia to modelo toy xrhsth
    ex = shap.KernelExplainer(modelReloaded.predict, df)
    shap_values = ex.shap_values(df)
    # shap.summary_plot(shap_values, df, show=False, sort=True)
    shap.summary_plot(shap_values, df, show=False, plot_type='bar', sort=True)
    plt.savefig("mlpart/static/mlpart/agnostic/RandombarGlobaldata.jpeg", format='jpeg', dpi=130, bbox_inches='tight')



    # ex = shap.KernelExplainer(modelReloaded.predict, df)
    # shap_values = ex.shap_values(df)
    # shap.summary_plot(shap_values, df, show=False, sort=False)
    # plt.savefig("mlpart/static/mlpart/agnostic/RandomGlobaldata.jpeg", format='jpeg', dpi=130, bbox_inches='tight')


    shap.dependence_plot(column_name_list[0], shap_values, df, show=False)
    plt.savefig("mlpart/static/mlpart/agnostic/RandomSpecificData1.jpeg", format='jpeg', dpi=130, bbox_inches='tight')

    shap.dependence_plot(column_name_list[1], shap_values, df, show=False)
    plt.savefig("mlpart/static/mlpart/agnostic/RandomSpecificData2.jpeg", format='jpeg', dpi=130, bbox_inches='tight')

    shap.dependence_plot(column_name_list[2], shap_values, df, show=False)
    plt.savefig("mlpart/static/mlpart/agnostic/RandomSpecificData3.jpeg", format='jpeg', dpi=130, bbox_inches='tight')

    shap.dependence_plot(column_name_list[3], shap_values, df, show=False)
    plt.savefig("mlpart/static/mlpart/agnostic/RandomSpecificData4.jpeg", format='jpeg', dpi=130, bbox_inches='tight')

    if list_length == 5:
            shap.dependence_plot(column_name_list[4], shap_values, df, show=False)
            plt.savefig("mlpart/static/mlpart/agnostic/RandomSpecificData5.jpeg", format='jpeg', dpi=130,
                        bbox_inches='tight')
    elif list_length == 6:
            shap.dependence_plot(column_name_list[4], shap_values, df, show=False)
            plt.savefig("mlpart/static/mlpart/agnostic/RandomSpecificData5.jpeg", format='jpeg', dpi=130,
                        bbox_inches='tight')
            shap.dependence_plot(column_name_list[5], shap_values, df, show=False)
            plt.savefig("mlpart/static/mlpart/agnostic/RandomSpecificData6.jpeg", format='jpeg', dpi=130,
                        bbox_inches='tight')
    elif list_length == 7:
            shap.dependence_plot(column_name_list[4], shap_values, df, show=False)
            plt.savefig("mlpart/static/mlpart/agnostic/RandomSpecificData5.jpeg", format='jpeg', dpi=130,
                        bbox_inches='tight')
            shap.dependence_plot(column_name_list[5], shap_values, df, show=False)
            plt.savefig("mlpart/static/mlpart/agnostic/RandomSpecificData6.jpeg", format='jpeg', dpi=130,
                        bbox_inches='tight')
            shap.dependence_plot(column_name_list[6], shap_values, df, show=False)
            plt.savefig("mlpart/static/mlpart/agnostic/RandomSpecificData7.jpeg", format='jpeg', dpi=130,
                        bbox_inches='tight')
    elif list_length == 8:
            shap.dependence_plot(column_name_list[4], shap_values, df, show=False)
            plt.savefig("mlpart/static/mlpart/agnostic/RandomSpecificData5.jpeg", format='jpeg', dpi=130,
                        bbox_inches='tight')
            shap.dependence_plot(column_name_list[5], shap_values, df, show=False)
            plt.savefig("mlpart/static/mlpart/agnostic/RandomSpecificData6.jpeg", format='jpeg', dpi=130,
                        bbox_inches='tight')
            shap.dependence_plot(column_name_list[6], shap_values, df, show=False)
            plt.savefig("mlpart/static/mlpart/agnostic/RandomSpecificData7.jpeg", format='jpeg', dpi=130,
                        bbox_inches='tight')
            shap.dependence_plot(column_name_list[7], shap_values, df, show=False)
            plt.savefig("mlpart/static/mlpart/agnostic/RandomSpecificData8.jpeg", format='jpeg', dpi=130,
                        bbox_inches='tight')
    elif list_length == 9:
            shap.dependence_plot(column_name_list[4], shap_values, df, show=False)
            plt.savefig("mlpart/static/mlpart/agnostic/RandomSpecificData5.jpeg", format='jpeg', dpi=130,
                        bbox_inches='tight')
            shap.dependence_plot(column_name_list[5], shap_values, df, show=False)
            plt.savefig("mlpart/static/mlpart/agnostic/RandomSpecificData6.jpeg", format='jpeg', dpi=130,
                        bbox_inches='tight')
            shap.dependence_plot(column_name_list[6], shap_values, df, show=False)
            plt.savefig("mlpart/static/mlpart/agnostic/RandomSpecificData7.jpeg", format='jpeg', dpi=130,
                        bbox_inches='tight')
            shap.dependence_plot(column_name_list[7], shap_values, df, show=False)
            plt.savefig("mlpart/static/mlpart/agnostic/RandomSpecificData8.jpeg", format='jpeg', dpi=130,
                        bbox_inches='tight')
            shap.dependence_plot(column_name_list[8], shap_values, df, show=False)
            plt.savefig("mlpart/static/mlpart/agnostic/RandomSpecificData9.jpeg", format='jpeg', dpi=130,
                        bbox_inches='tight')
    elif list_length == 10:
        shap.dependence_plot(column_name_list[4], shap_values, df, show=False)
        plt.savefig("mlpart/static/mlpart/agnostic/RandomSpecificData5.jpeg", format='jpeg', dpi=130,
                    bbox_inches='tight')
        shap.dependence_plot(column_name_list[5], shap_values, df, show=False)
        plt.savefig("mlpart/static/mlpart/agnostic/RandomSpecificData6.jpeg", format='jpeg', dpi=130,
                    bbox_inches='tight')
        shap.dependence_plot(column_name_list[6], shap_values, df, show=False)
        plt.savefig("mlpart/static/mlpart/agnostic/RandomSpecificData7.jpeg", format='jpeg', dpi=130,
                    bbox_inches='tight')
        shap.dependence_plot(column_name_list[7], shap_values, df, show=False)
        plt.savefig("mlpart/static/mlpart/agnostic/RandomSpecificData8.jpeg", format='jpeg', dpi=130,
                    bbox_inches='tight')
        shap.dependence_plot(column_name_list[8], shap_values, df, show=False)
        plt.savefig("mlpart/static/mlpart/agnostic/RandomSpecificData9.jpeg", format='jpeg', dpi=130,
                    bbox_inches='tight')
        shap.dependence_plot(column_name_list[9], shap_values, df, show=False)
        plt.savefig("mlpart/static/mlpart/agnostic/RandomSpecificData10.jpeg", format='jpeg', dpi=130,
                    bbox_inches='tight')
    elif list_length == 11:
        shap.dependence_plot(column_name_list[4], shap_values, df, show=False)
        plt.savefig("mlpart/static/mlpart/agnostic/RandomSpecificData5.jpeg", format='jpeg', dpi=130,
                    bbox_inches='tight')
        shap.dependence_plot(column_name_list[5], shap_values, df, show=False)
        plt.savefig("mlpart/static/mlpart/agnostic/RandomSpecificData6.jpeg", format='jpeg', dpi=130,
                    bbox_inches='tight')
        shap.dependence_plot(column_name_list[6], shap_values, df, show=False)
        plt.savefig("mlpart/static/mlpart/agnostic/RandomSpecificData7.jpeg", format='jpeg', dpi=130,
                    bbox_inches='tight')
        shap.dependence_plot(column_name_list[7], shap_values, df, show=False)
        plt.savefig("mlpart/static/mlpart/agnostic/RandomSpecificData8.jpeg", format='jpeg', dpi=130,
                    bbox_inches='tight')
        shap.dependence_plot(column_name_list[8], shap_values, df, show=False)
        plt.savefig("mlpart/static/mlpart/agnostic/RandomSpecificData9.jpeg", format='jpeg', dpi=130,
                    bbox_inches='tight')
        shap.dependence_plot(column_name_list[9], shap_values, df, show=False)
        plt.savefig("mlpart/static/mlpart/agnostic/RandomSpecificData10.jpeg", format='jpeg', dpi=130,
                    bbox_inches='tight')
        shap.dependence_plot(column_name_list[10], shap_values, df, show=False)
        plt.savefig("mlpart/static/mlpart/agnostic/RandomSpecificData11.jpeg", format='jpeg', dpi=130,
                    bbox_inches='tight')
    elif list_length == 12:
        shap.dependence_plot(column_name_list[4], shap_values, df, show=False)
        plt.savefig("mlpart/static/mlpart/agnostic/RandomSpecificData5.jpeg", format='jpeg', dpi=130,
                    bbox_inches='tight')
        shap.dependence_plot(column_name_list[5], shap_values, df, show=False)
        plt.savefig("mlpart/static/mlpart/agnostic/RandomSpecificData6.jpeg", format='jpeg', dpi=130,
                    bbox_inches='tight')
        shap.dependence_plot(column_name_list[6], shap_values, df, show=False)
        plt.savefig("mlpart/static/mlpart/agnostic/RandomSpecificData7.jpeg", format='jpeg', dpi=130,
                    bbox_inches='tight')
        shap.dependence_plot(column_name_list[7], shap_values, df, show=False)
        plt.savefig("mlpart/static/mlpart/agnostic/RandomSpecificData8.jpeg", format='jpeg', dpi=130,
                    bbox_inches='tight')
        shap.dependence_plot(column_name_list[8], shap_values, df, show=False)
        plt.savefig("mlpart/static/mlpart/agnostic/RandomSpecificData9.jpeg", format='jpeg', dpi=130,
                    bbox_inches='tight')
        shap.dependence_plot(column_name_list[9], shap_values, df, show=False)
        plt.savefig("mlpart/static/mlpart/agnostic/RandomSpecificData10.jpeg", format='jpeg', dpi=130,
                    bbox_inches='tight')
        shap.dependence_plot(column_name_list[10], shap_values, df, show=False)
        plt.savefig("mlpart/static/mlpart/agnostic/RandomSpecificData11.jpeg", format='jpeg', dpi=130,
                    bbox_inches='tight')
        shap.dependence_plot(column_name_list[11], shap_values, df, show=False)
        plt.savefig("mlpart/static/mlpart/agnostic/RandomSpecificData12.jpeg", format='jpeg', dpi=130,
                    bbox_inches='tight')
    return render(request, 'mlpart/randomExplanation.html', context)


# def FileUploadView(request):
#     # gia to upload
#     if request.method == 'POST':
#         form = UploadForm(request.POST, request.FILES)
#         if form.is_valid():
#             form.save()
#             # messages.success(request, 'Your file had been uploaded successfully.')
#             return HttpResponseRedirect(reverse('FileUploadView'))
#     else:
#         form = UploadForm()
#     context = {
#             'form': form,
#         }
#     return render(request, 'mlpart/uploadtest.html', context)

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


# einai h methodos poy tha mas kanei predict to apotelesma
# epishs gia na paroyme ta submited pragmata apo xrhsth prepei na exoume dictionary
def predictMPG(request):
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

        temp2 = temp.copy()
        temp2['model year'] = temp['model_year']
        # del temp2['model_year']
        # ayto giati sthn arxh me to model_year barage error afou sto dataset einai model year

        # kai twra pou exw ta dedomena prepei na fortwsw to modelo

        # to model perimenei DATAFRAME enw emeis exoume dictionary opote prepei na kanoume thn allagh
        df = pd.read_csv(r"C:\Users\gvarv\anaconda3\envs\thesis\MPG\mpg_data_example.csv")
        df['horsepower'] = df['horsepower'].fillna(df['horsepower'].mean())
        df[['cylinders', 'weight', 'model year', 'origin']] = df[
            ['cylinders', 'weight', 'model year', 'origin']].astype(float)
        x = df.drop(columns=['mpg', 'car name', 'origin'], axis=1)
        y = df['mpg']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)
        model = RandomForestRegressor(n_estimators=200, max_depth=None, min_samples_split=2, random_state=0)
        model.fit(x_train, y_train)
        joblib.dump(model, 'MPGmodel.pkl')
        sampleDataFeatures = np.asarray(
            [list(item.values()) for item in ({'x': temp}).values()])
        print(sampleDataFeatures)
        modelReload = joblib.load('MPGmodel.pkl')
        scoreval = modelReload.predict(sampleDataFeatures)
        scoreval = np.around(scoreval, 2)
        scoreval1 = scoreval[0]

        print(scoreval1)
        context = {'scoreval': scoreval1}

        # Summary Plot bar gia global
        # explainer = shap.TreeExplainer(modelReload)
        # shap_values = explainer.shap_values(x_train)
        # shap.summary_plot(shap_values, x_train)
        # plt.savefig("mlpart/static/mlpart/MpgSPGlobaldata.jpeg", format='jpeg', dpi=130, bbox_inches='tight')

        # Summary Plot gia global
        # explainer = shap.TreeExplainer(modelReload)
        # shap_values = explainer.shap_values(x_train)
        # shap.summary_plot(shap_values, x_train, show=False, plot_type='bar', sort=True)
        # plt.savefig("mlpart/static/mlpart/MpgSPbarGlobaldata.jpeg", format='jpeg', dpi=130, bbox_inches='tight')

        # Dependence Plots (idio plot allazw to feature kathe fora)
        # explainer = shap.TreeExplainer(modelReload)
        # shap_values = explainer.shap_values(x_train)
        # fig=shap.dependence_plot("model year", shap_values, x_train)
        # plt.savefig("mlpart/static/mlpart/MpgDPModelyeardata.jpeg", format='jpeg', dpi=130, bbox_inches='tight')

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
        print(sampleDataFeatures)
        # print(means)
        sampleDataFeatures = np.subtract(sampleDataFeatures, means)
        sampleDataFeatures = sampleDataFeatures / stds
        # print(sampleDataFeatures)

        # # predict
        diabeteseModelInsideDjango = joblib.load(r"C:\Users\gvarv\anaconda3\envs\thesis\diabeteseModelInsideDjango.pkl")
        predictionProbability = diabeteseModelInsideDjango.predict_proba(sampleDataFeatures)
        prediction = diabeteseModelInsideDjango.predict(sampleDataFeatures)
        print(sampleDataFeatures)
        print(predictionProbability)
        print(prediction)


        if prediction[0] == 0:
            print("Diabetes will NOT occur")
            messages.success(request, "Diabetes will NOT occur")
        elif prediction[0] == 1:
            print("Diabetes will  occur")
            messages.success(request, "Diabetes will occur")


        # print(trainData[0, :])

        #Summary Plot gia global

        # ex = shap.KernelExplainer(diabeteseModelInsideDjango.predict, trainData)
        # shap_values = ex.shap_values(trainData)
        # fig = shap.summary_plot(shap_values, trainData,
        #                         feature_names=['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness', 'Insulin',
        #                                        'BMI', 'Diabetes Pedigree', 'Age'], show=False, sort=True)
        # plt.savefig("mlpart/static/mlpart/DiabetesSPGlobaldataNEW.jpeg", format='jpeg', dpi=130, bbox_inches='tight')

        # Summary Plot bar gia global

        # ex = shap.KernelExplainer(diabeteseModelInsideDjango.predict, trainData)
        # shap_values = ex.shap_values(trainData)
        # fig = shap.summary_plot(shap_values, trainData, show=False, plot_type='bar', sort=True,
        #                         feature_names=['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness', 'Insulin',
        #                                        'BMI', 'Diabetes Pedigree', 'Age'])
        # plt.savefig("mlpart/static/mlpart/DiabetesSPbarGlobaltestdata.jpeg", format='jpeg', dpi=130,
        #             bbox_inches='tight')

        # Dependence Plots (idio plot allazw to feature kathe fora)

        # ex = shap.KernelExplainer(diabeteseModelInsideDjango.predict, trainData)
        # shap_values = ex.shap_values(trainData)
        # fig = shap.dependence_plot("Age", shap_values, trainData, show=False,
        #                            feature_names=['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness',
        #                                           'Insulin',
        #                                           'BMI', 'Diabetes Pedigree', 'Age'])
        # plt.savefig("mlpart/static/mlpart/DiabetesDPAgedata.jpeg", format='jpeg', dpi=130, bbox_inches='tight')

        # Force Plots gia tyxaia observations

        # ex = shap.KernelExplainer(diabeteseModelInsideDjango.predict, trainData)
        # shap_values = ex.shap_values(trainData[593, :])
        # shap.force_plot(ex.expected_value, shap_values, trainData[593, :],
        #                 feature_names=['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness', 'Insulin',
        #                                'BMI', 'Diabetes Pedigree', 'Age'])

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
            print("The species is Iris-Setosa ")
            messages.success(request, "The species is Iris-Setosa ")
        elif prediction[0] == 1:
            print("The species is Iris-Versicolor ")
            messages.success(request, "The species is Iris-Versicolor")
        elif prediction[0] == 2:
            print("The species is Iris-Virginica ")
            messages.success(request, "The species is Iris-Virginica")

        # Summary Plot bar gia Global

        # ex = shap.KernelExplainer(LoadedModel.predict, x_train)
        # shap_values = ex.shap_values(x_train)
        # shap.summary_plot(shap_values, x_train, plot_type="bar", show=False, sort=True,)
        # plt.savefig("mlpart/static/mlpart/IrisSPbarGlobal.jpeg", format='jpeg', dpi=130, bbox_inches='tight')

        # Summary Plot gia Global

        # ex = shap.KernelExplainer(LoadedModel.predict, x_train)
        # shap_values = ex.shap_values(x_train)
        # shap.summary_plot(shap_values, x_train, show=False)
        # plt.savefig("mlpart/static/mlpart/IrisSPGlobal.jpeg", format='jpeg', dpi=130, bbox_inches='tight')

        # Dependence Plot gia Global(idio plot allazw to feature kathe fora)

        # ex = shap.KernelExplainer(LoadedModel.predict, x_train)
        # shap_values = ex.shap_values(x_train)
        # fig = shap.dependence_plot("PetalWidthCm", shap_values, x_train, show=False)
        # plt.savefig("mlpart/static/mlpart/IrisDPPetalWdata.jpeg", format='jpeg', dpi=130, bbox_inches='tight')

        # Force plots gia tyxaia observations
        # explainer = shap.KernelExplainer(LoadedModel.predict_proba, x_train)
        # shap_values = explainer.shap_values(x_train.iloc[75, :])
        # shap.force_plot(explainer.expected_value[0], shap_values[0], x_train.iloc[75, :])

    return render(request, 'mlpart/resultsIris.html')
