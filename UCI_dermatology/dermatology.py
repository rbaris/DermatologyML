# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 14:23:20 2022

@author: baris
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.inspection import partial_dependence,PartialDependenceDisplay,permutation_importance
from sklearn.metrics import accuracy_score


# veri setini dataframe çevirip sonrasında optimizasyon yapılmalı.

#df=pd.read_csv("DFdermatology.csv")
features ="erythema,scaling,definite borders,itching,koebner phenomenon,polygonal papules,follicular papules,oral mucosal involvement,knee and elbow involvement,scalp involvement,family history,melanin incontinence,eosinophils in the infiltrate,PNL infiltrate,fibrosis of the papillary dermis,exocytosis,acanthosis,hyperkeratosis,parakeratosis,clubbing of the rete ridges,elongation of the rete ridges,thinning of the suprapapillary epidermis,spongiform pustule,munro microabcess,focal hypergranulosis,disappearance of the granular layer,vacuolisation and damage of basal layer,spongiosis,saw-tooth appearance of retes,follicular horn plug,perifollicular parakeratosis,inflammatory monoluclear inflitrate,band-like infiltrate,Age".split(",")


filename = "dermatology.csv"
data = pd.read_csv(filename,sep=',')
data = data.astype({'Age':'int64','disease':'int64'})
#print(data.dtypes)
dataDescription = data.describe()

#Clean Data
# df.at[34, 'Age'] = 36
# df.at[35, 'Age'] = 36
# df.at[36, 'Age'] = 36
# df.at[37, 'Age'] = 36
# df.at[264, 'Age'] = 36
# df.at[265, 'Age'] = 36
# df.at[266, 'Age'] = 36
# df.at[267, 'Age'] = 36

#disease = {1= pityriasis rubra pilaris,2=cronic dermatitis,3=pityriasis rosea,
#...4=lichen planus,5=seboreic dermatitis ,6= psoriasis}


Y = data['disease']
data = data.drop('disease',axis=1)
X = data

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3,random_state=True)

modelKNN = KNeighborsClassifier(n_neighbors=3,metric="euclidean")
modelKNN.fit(x_train, y_train)
modelNB = GaussianNB()
modelNB.fit(x_train,y_train)
modelDT = tree.DecisionTreeClassifier()
modelDT.fit(x_train,y_train) 

testsayisi = len(x_test)
print("Test veri sayısı: %d" %testsayisi)

#tahmin yap
tahminKNN = modelKNN.predict(x_test)
tahminNB = modelNB.predict(x_test)
tahminDT = modelDT.predict(x_test)

print("KNN algoritması")
print(classification_report(y_test,tahminKNN))
print("*"*50)
print("Naive Bayes algoritması")
print(classification_report(y_test,tahminNB))
print("*"*50)
print("Karar Ağacı algoritması")
print(classification_report(y_test,tahminDT))
#3 a=tree.plot_tree(modelDT,impurity=(False),filled=(True),feature_names=features,class_names=(1))
#plt.show(a)
print(tree.export_text(modelDT,feature_names=features))
print("*"*50)

cmKNN = confusion_matrix(tahminKNN, y_test)
cmNB = confusion_matrix(tahminNB, y_test)
cmDT = confusion_matrix(tahminDT, y_test)


def objective_functionKNN(solution):
    if(sum(solution) == 0):
        return 0
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(x_train.loc[:,solution],y_train)
    score = model.score(x_test.loc[:,solution], y_test)
    #print(solution)
    #print(score)
    return score

best_objectiveKNN = 0;

for i in range(100):
    solution = np.random.random(34)>0.5
    sol_objectiveKNN = objective_functionKNN(solution)    
    if(sol_objectiveKNN>best_objectiveKNN):
        best_solution = solution.copy()
        best_objectiveKNN = sol_objectiveKNN
print("KNN Optimizasyon")
best_solution = best_solution.astype(int).astype(str)
print("ÖZelliklerin Hedef Sınıfa Etkisi:\n",best_solution)
print("Optimizasyon Sonrası İsabet Oranı:\n",best_objectiveKNN)

def objective_functionNB(solution):
    if(sum(solution) == 0):
        return 0
    model = GaussianNB()
    model.fit(x_train.loc[:,solution],y_train)
    score = model.score(x_test.loc[:,solution], y_test)
    #print(solution)
    #print(score)
    return score

best_objectiveNB = 0;

for i in range(100):
    solution = np.random.random(34)>0.5
    sol_objectiveNB= objective_functionNB(solution)    
    if(sol_objectiveNB>best_objectiveNB):
        best_solution = solution.copy()
        best_objectiveNB = sol_objectiveNB

print("Naïve Bayes Optimizasyon")
best_solution = best_solution.astype(int).astype(str)
print("ÖZelliklerin Hedef Sınıfa Etkisi:\n",best_solution)
print("Optimizasyon Sonrası İsabet Oranı:\n",best_objectiveNB)

def objective_functionDT(solution):
    if(sum(solution) == 0):
        return 0
    model = tree.DecisionTreeClassifier()
    model.fit(x_train.loc[:,solution],y_train)
    score = model.score(x_test.loc[:,solution], y_test)
    #print(solution)
    #print(score)
    return score

best_objectiveDT = 0;

for i in range(100):
    solution = np.random.random(34)>0.5
    sol_objectiveDT= objective_functionDT(solution)    
    if(sol_objectiveDT>best_objectiveDT):
        best_solution = solution.copy()
        best_objectiveDT = sol_objectiveDT

print("Karar Ağacı Optimizasyon")
best_solution = best_solution.astype(int).astype(str)
print("ÖZelliklerin Hedef Sınıfa Etkisi:\n",best_solution)
print("Optimizasyon Sonrası İsabet Oranı:\n",best_objectiveDT)


