import pandas as pd
import matplotlib.pyplot as mp
import seaborn as sns
from sklearn import svm
from sklearn import metrics

data= pd.read_csv("heart.csv")
print(data.info())

X=data[['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope',"ca","thal"]]
y=data["target"]

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=5)

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