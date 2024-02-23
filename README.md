# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5. Predict the regression for the marks by using the representation of the graph.
6. Hence we obtained the linear regression for the given dataset.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Lokesh N
RegisterNumber:212222100023  
*/
```
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('/content/ex1_os.csv')
df.head(10)
plt.scatter(df['x'],df['y'])
plt.xlabel('x')
plt.ylabel('y')
x=df.iloc[:,0:-1]
y=df.iloc[:,-1]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
x_train
y_train
lr.predict(x_test.iloc[0].values.reshape(1,1))
plt.scatter(df['x'],df['y'])
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x_train,lr.predict(x_train),color='orange')
lr.coef_
lr.intercept_
```

## Output:
## 1.) Dataset:
![image](https://github.com/lokeshnarayanan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393019/e361ad83-5966-4fbd-b5ec-6a02c31f75ef)

## 2.) Graph of plotted data:
![image](https://github.com/lokeshnarayanan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393019/e1189e3e-1e84-4fdd-b35f-0a8eae5e16a7)


## 3.) Performing Linear Regression:
![image](https://github.com/lokeshnarayanan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393019/5f95b265-8bc5-45b8-bc17-4004981f3141)


## 4.) Trained data:
![image](https://github.com/lokeshnarayanan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393019/6fe2147a-28b3-44ae-aea9-a8c33dcf2779)


## 5.) Predicting the line of Regression:
![image](https://github.com/lokeshnarayanan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393019/3ca90e9a-17bc-4223-90fa-9ff325bd47af)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
