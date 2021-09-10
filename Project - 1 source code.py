# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 13:15:28 2021

@author: BHARGAV BHAT
"""
#IMPORTING THE LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt



#READING THE DATA FROM FILES
data=pd.read_csv('advertising.csv')
data.head()

#VISUALIZE THE DATA

f , axs=plt.subplots(1, 3,sharey=True)
data.plot(kind='scatter',x='TV',y='Sales',ax=axs[0],figsize=(14,7))
data.plot(kind='scatter',x='Radio',y='Sales',ax=axs[1])
data.plot(kind='scatter',x='Newspaper',y='Sales',ax=axs[2])

#CREATING X&Y FOR LINEAR REGRESSION
f_cols=['TV']
X=data[f_cols]
y=data.Sales


#IMPORTING LINEAR REGRESSION ALGO FOR LINEAR REGRESSION 
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X, y)

print(lr.intercept_)
print(lr.coef_)

r=6.9748214882298925 + 0.05546477*50
print(r)


#CREATE A DATAFRAME WITH MIN AND MAX VALUE OF THE TABLE
X_new=pd.DataFrame({'TV':[data.TV.min(),data.TV.max()]})
X_new.head()


pr=lr.predict(X_new)
print(pr)

data.plot(kind='scatter',x='TV',y='Sales')

plt.plot(X_new,pr,c='red',linewidth=3)

#SUMMERIZE
import statsmodels.formula.api as smf
lm=smf.ols(formula='Sales ~ TV',data=data).fit()
lm.conf_int()

#FINDING THE PROBABILITY VALUES
print(lm.pvalues)


#FINDING THE R-SQUARE VALUES
print(lm.rsquared)



#MULTI LINEAR REGRESSION
f_cols=['TV','Radio','Newspaper']
X=data[f_cols]
y=data.Sales

lr=LinearRegression()
lr.fit(X, y)


print(lr.intercept_)
print(lr.coef_)

lm=smf.ols(formula='Sales ~ TV+Radio+Newspaper',data=data).fit()
lm.conf_int()
print(lm.summary())


