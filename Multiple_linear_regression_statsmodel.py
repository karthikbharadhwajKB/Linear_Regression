# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 18:36:46 2019

@author: Karthik Bharadhwaj
"""
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sb 
import statsmodels.api as sm 
#sb.set()

data = pd.read_csv('SAT.csv')

#converting categorical varible  to numerical 
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
data['Attendance']=labelencoder.fit_transform(data['Attendance'])
print(data)

  
#Regression 

#dependent variable
y = data['GPA']
#independent variables 
x1=data[['SAT','Attendance']]
 
#intercept

x = sm.add_constant(x1)

results = sm.OLS(y,x).fit()

print(results.summary())

#intercept and constant 

print(results.params)

#scatterplot 

y_hat_yes =0.8665+0.00140*x1

y_hat_no = 0.6438+0.00140*x1

plt.scatter(data['SAT'],y)

fig = plt.plot(data['SAT'],y_hat_yes,c='orange',lw=2)

fig = plt.plot(data['SAT'],y_hat_no,c='red',lw=2)
plt.ylim(2)
plt.xlabel('SAT',fontsize=20)
plt.ylabel('GPA',fontsize=20)
plt.show()

#prediction 

new_data = pd.DataFrame({'const':1,'SAT':[1800,1800],'Attendance':[0,1]})

new_data = new_data.rename(index={0:'Karthik',1:'Bharadhwaj'})

print(new_data)

predictions = results.predict(new_data)

print(predictions)