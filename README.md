# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packages and print the present data. 
2.Print the placement data and salary data. 
3.Find the null and duplicate values. 
4.Using logistic regression find the predicted values of accuracy. 
5.Display the results.
## Program:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: VIKAASH K S
RegisterNumber: 212223240179

import pandas as pd
data=pd.read_csv("C:/Users/admin/OneDrive/Documents/INTRO TO ML/Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data.head()

data1.isnull()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```
## Output:
![11](https://github.com/23003250/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/139331462/99c5e42c-e37b-4729-b177-1eb035096776)

![12](https://github.com/23003250/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/139331462/f3422c1f-8aed-47f4-af06-4af0b4cdfe92)

![13](https://github.com/23003250/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/139331462/4b012cec-3cb0-488c-8ece-95953ce94c52)

![14](https://github.com/23003250/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/139331462/2418b531-794b-4710-9547-974328e86725)

![15](https://github.com/23003250/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/139331462/5f5ee347-465a-429d-9f95-1811be46f288)

![16](https://github.com/23003250/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/139331462/c0487ce5-46a0-47fa-ac9f-d540b780a93a)

![17](https://github.com/23003250/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/139331462/594200e1-7b2f-4380-9575-d00827d75896)

![18](https://github.com/23003250/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/139331462/7d28a49d-1b5a-4f67-8adf-6a7bb5c71c72)

![19](https://github.com/23003250/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/139331462/fee40242-4269-4b2b-8d05-c6690b05ccdc)

![20](https://github.com/23003250/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/139331462/f863daec-deef-4480-8730-b7d9220fceea)

![21](https://github.com/23003250/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/139331462/6edfe502-2ed9-406d-b1e0-806a84fd5122)

![22](https://github.com/23003250/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/139331462/f39c2775-e274-491e-b2ec-dc112a28474b)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
