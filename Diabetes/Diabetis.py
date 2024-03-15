import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

print("----------Diabetes predictor using Decision Tree --------------")

diabetes=pd.read_csv("Diabetes.csv")

print("columns of Dataset",diabetes.columns)
print("First five record of data set :\n",diabetes.head())
print("Dimension of data set :{}".format(diabetes.shape))

x_train,x_test,y_train,y_test=train_test_split(diabetes.loc[:,diabetes.columns != 'Outcome'],diabetes['Outcome'],stratify=diabetes['Outcome'],
                                               random_state=66)

tree=DecisionTreeClassifier(random_state=0)
tree.fit(x_train,y_train)

print("Accuracy on training set :{:.3f}".format(tree.score(x_train,y_train)))
print("Accuracy on test set : {:.3f}".format(tree.score(x_test,y_test)))
tree=DecisionTreeClassifier(max_depth=3,random_state=0)
tree.fit(x_train,y_train)

print("Accuracy on training set :{:.3f}".format(tree.score(x_train,y_train)))
print("Accuracy on testing data set :{:.3f}".format(tree.score(x_test,y_test)))
print("Feature Importance : \n{}".format(tree.feature_importances_))

def Plot_feature_importances_diabetes(model):
    plt.figure(figsize=(8,6))
    n_features=8
    plt.barh(range(n_features),model.feature_importances_,align='center')
    diabetes_features=[X for i ,X in enumerate(diabetes.columns)if i!=8 ]
    plt.yticks(np.arange(n_features),diabetes_features)
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.ylim(-1,n_features)
    plt.show()

Plot_feature_importances_diabetes(tree)
