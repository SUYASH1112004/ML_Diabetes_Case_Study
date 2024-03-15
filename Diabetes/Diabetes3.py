#Diabetes Predictor application using K Nearest Neighbour Algorithm 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

print("---------Diabetes Predictor Using K Nearest Neighbour--------------")
diabetes=pd.read_csv('diabetes.csv')
print("The Columns Of Dataset :",diabetes.columns)
print("Top 5 Records Of Dataset :",diabetes.head())

print("The Dimension of dataset :",diabetes.shape)


x_train,x_test,y_train,y_test=train_test_split(diabetes.loc[:,diabetes.columns != 'Outcome'],diabetes['Outcome'],stratify=diabetes['Outcome'],
                                               random_state=66)

training_accuracy=[]
test_accuracy=[]
neighbours_setting = range(1,11)

# knn=KNeighborsClassifier(n_neighbors=1)
# knn.fit(x_train,y_train)
# train_accuracy=knn.score(x_train,y_train)
# test_Accuracy=knn.score(x_test,y_test)
for n in neighbours_setting:
    knn=KNeighborsClassifier(n_neighbors=n)
    knn.fit(x_train,y_train)
    training_accuracy.append(knn.score(x_train,y_train))
    test_accuracy.append(knn.score(x_test,y_test))





plt.plot(neighbours_setting,training_accuracy,label="Training Accuracy")
plt.plot(neighbours_setting,test_accuracy,label="Testing Accuracy")
plt.ylabel=("Accuracy")
plt.xlabel=("n_neighbors")
plt.legend()
plt.savefig('knn_compare_model')
plt.show()


knn2=KNeighborsClassifier(n_neighbors=10)
knn2.fit(x_train,y_train)

print("Accuracy Of Knn Classifier On Training Set(Hyper Parameter =9) :",knn2.score(x_train,y_train))
print("Accuracy of knn classifier on test set (hyper parameter =9 ):",knn2.score(x_test,y_test))


