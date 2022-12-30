# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 17:04:07 2022

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
from sklearn.model_selection import cross_val_score
from sklearn.inspection import partial_dependence,PartialDependenceDisplay,permutation_importance
import math
from sklearn.metrics import accuracy_score


features ="erythema,scaling,definite borders,itching,koebner phenomenon,polygonal papules,follicular papules,oral mucosal involvement,knee and elbow involvement,scalp involvement,family history,melanin incontinence,eosinophils in the infiltrate,PNL infiltrate,fibrosis of the papillary dermis,exocytosis,acanthosis,hyperkeratosis,parakeratosis,clubbing of the rete ridges,elongation of the rete ridges,thinning of the suprapapillary epidermis,spongiform pustule,munro microabcess,focal hypergranulosis,disappearance of the granular layer,vacuolisation and damage of basal layer,spongiosis,saw-tooth appearance of retes,follicular horn plug,perifollicular parakeratosis,inflammatory monoluclear inflitrate,band-like infiltrate,Age".split(",")


filename = "dermatology.csv"
data = pd.read_csv(filename,sep=',')
data = data.astype({'Age':'int64','disease':'int64'})
#print(data.dtypes)

Y = data['disease']
data = data.drop('disease',axis=1)
a = data

# Changing column names of x dataframe with 0 and 1
c = np.random.random(34) > 0.5
c = c.astype(int).astype(str)

a.columns = c


# Taking only 1's of feature for knn
X = a.loc[:,["1"]]

#train test split
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size = 0.3,random_state=True)
modelDT = tree.DecisionTreeClassifier()
modelDT.fit(x_train,y_train) 
y_pred = modelDT.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print("Karar Ağacı Sınıflandırması için isabet oranı: ",acc)


def get_neighbors(a):
    c = np.random.random(34) > 0.5
    c = c.astype(int).astype(str)
    a.columns = c
    a = a.xs('1', axis=1)
    return a
    
# neigbor parameter is created by get_neighbor so it has randomly selected 1's
# y is target to predicited
def get_cost(x,y):
    
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3)
    modelDT = tree.DecisionTreeClassifier()
    modelDT.fit(x_train,y_train) 
    y_pred = modelDT.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    return acc


# SA
# param = dataframe a with 0's and 1's, initial_state is knn score with df a (only 1's selected)
def simulated_annealing(param, initial_state, Y):
    """Peforms simulated annealing to find a solution"""
    
    scores = list()
    bestscores =list()
    initial_temp = 100
    final_temp = 0
    alpha = 1
    
    current_temp = initial_temp

    # Start by initializing the current state with the initial state
    current_state = initial_state
    solution = current_state
    bestSolution = 0;
    i = 0        
    while current_temp >=  final_temp:
        
        neighbor = get_neighbors(param)
        # Check if neighbor is best so far
        cost_diff = get_cost(neighbor,Y)
        print(f"{i} . iterasyonda isabet oranı :",cost_diff)
        scores.append(cost_diff)
      
        if cost_diff < math.exp(cost_diff / current_temp):
            solution = cost_diff
        # decrement the temperature
        current_temp = current_temp - alpha
        i = i+1
        
        if solution > bestSolution:
            bestSolution = solution
            bestscores.append(bestSolution)
            
    return bestSolution,scores,bestscores

print("Karar Ağacı Sınıflandırması")
bestSolution, scores,bestScores = simulated_annealing(a, acc , Y)
print("Simulated Annealing(Benzetilmiş Tavlama) Optimizasyonu Sonrası İsabet Oranı :", bestSolution)
print("Optimizasyon öncesi isabet oranı:\n",classification_report(y_test,y_pred))

a = plt.stem(scores,bottom=0.75,markerfmt=("D"))
plt.plot(bestScores)

#a = plt.plot(scores,".-",bestScores,".-")
plt.title("Karar Ağacı")
plt.xlabel('İterasyon')
plt.ylabel('İsabet Oranı')
plt.legend(["bestScores","scores"],loc='lower right')
plt.show()