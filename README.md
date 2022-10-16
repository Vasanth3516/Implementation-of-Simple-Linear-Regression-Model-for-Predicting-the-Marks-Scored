# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: VASANTHAN 
RegisterNumber:  212220220052
*/
```
import pandas as pd

import numpy as np

df=pd.read_csv('student_scores.csv')

print(df)

X=df.iloc[:,:-1].values

Y=df.iloc[:,1].values

print(X,Y)

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression

reg=LinearRegression()

reg.fit(X_train,Y_train)

Y_pred=reg.predict(X_test)

import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error , mean_squared_error

plt.scatter(X_train,Y_train,color='green')

plt.plot(X_train,reg.predict(X_train),color='purple')

plt.title(' Training set (Hours Vs Scores)')

plt.xlabel('Hours')

plt.ylabel('Scores')

plt.scatter(X_test,Y_test,color='green')

plt.plot(X_test,reg.predict(X_test),color='purple')

plt.title(' Training set (Hours Vs Scores)')

plt.xlabel('Hours')

plt.ylabel('Scores')

mse=mean_squared_error(Y_test,Y_pred)

print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)

print('MAE = ',mae)

rmse=np.sqrt(mse)

print('RMSE = ',rmse)

## Output:
![ss1](https://user-images.githubusercontent.com/115924983/196040012-31b1301c-b0f7-410b-8590-c2931dfc16ec.png)

![ss2](https://user-images.githubusercontent.com/115924983/196040025-efd7c896-2cf7-4e1c-9286-a8d8723b4f93.png)

![ss3](https://user-images.githubusercontent.com/115924983/196040039-0de0d835-d5ef-4dba-bc67-e26a277a3cf8.png)

![ss4](https://user-images.githubusercontent.com/115924983/196040052-3388d3ce-1c88-401c-8a40-84b90b1c1a50.png)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
