#svm = support vector machine
#used to find a decision boundry

import pandas as pd
import matplotlib.pyplot as mp
import seaborn as sns
from sklearn import datasets
from  sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics

dataa=datasets.load_breast_cancer()
print(dataa.keys())
cancer_data=pd.DataFrame(dataa.data)
cancer_data.columns=dataa.feature_names
cancer_data["isCancer"]=dataa.target

print(cancer_data.info())
print(cancer_data.head(10))

y=cancer_data["isCancer"]
cancer_data.drop("isCancer",axis=1)
X=cancer_data

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=50)
print(X_train)
print(X_test)
print(y_train)
print(y_test)

cls=svm.SVC(kernel="linear")
cls.fit(X_train,y_train)
y_pred=cls.predict(X_test)

accu=metrics.accuracy_score(y_test,y_pred)
print("accuracy = " , accu)

matrix=metrics.confusion_matrix(y_test,y_pred)
sns.heatmap(matrix,annot = True,fmt="d")
mp.title("CONFUSIONN")
mp.xlabel("actual")
mp.ylabel("predicted")
mp.show()