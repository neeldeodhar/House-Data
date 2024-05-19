#!/usr/bin/env python
# coding: utf-8

# In[1]:


#downloading dataset, importing libraries

import pandas as pd

import numpy as np

import os

import matplotlib.pyplot as plt

import seaborn as sns

import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import PolynomialFeatures

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


# In[2]:


#reading the dataset, deleting missing value entries
df = pd.read_csv("C:/Users/ideod/OneDrive/Documents/new folder zip data/house_data.csv")



df.head()

df.dropna()


# In[3]:


df.columns


# In[4]:


#encoding and selecting features
enc = OrdinalEncoder()


df[['zipcode']] = enc.fit_transform(df[['zipcode']])


# In[5]:


#selecting features: X-axis
selected_features = df[['bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated','zipcode']]

display(selected_features)


# In[6]:


#selecting and scaling X variable (selected features)
scaler = StandardScaler()

X = scaler.fit_transform(selected_features)


# In[7]:


# defining y variable

y = df['price'].values


# In[8]:


#creating a split (train/test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)


# In[9]:


#training decision tree regressor
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
DTRscore = []
for i in ['friedman_mse', 'squared_error', 'poisson']:

    regressor = DecisionTreeRegressor(random_state =0, criterion = i)
    regressor.fit(X_train,y_train)
    y_pred = regressor.predict(X_test)
    DTRscore.append(regressor.score(X_test,y_test))
    print (DTRscore)


# In[10]:


max(DTRscore)


# In[11]:


#SVR regression linear, non-linear 

from sklearn.svm import SVR
score1 = []
for k in ['linear', 'poly', 'rbf', 'sigmoid']:

    svr_model = SVR(kernel = k, gamma ='auto', C =100, epsilon = 0.1)
    svr_model.fit(X_train,y_train)
    y_pred1 = svr_model.predict(X_test)
    score1.append (svr_model.score(X_test,y_test))
                   
  
    print (score1)


# In[12]:


max(score1)


# In[20]:


#training RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
score2 = []

for k in range (290,300):
    optRFR = RandomForestRegressor(n_estimators = k, random_state = 0)
    optRFR = optRFR.fit(X_train, y_train)
    y_predOPT = optRFR.predict(X_test)
    score2.append(optRFR.score(X_test, y_test))
  


# In[21]:


print (score2)


# In[22]:


max(score2)


# In[23]:


#training KNN regressor model
from sklearn.neighbors import KNeighborsRegressor
score3 = []
for n in range (1,10):
    knn = KNeighborsRegressor(n_neighbors =n)
    knn.fit(X_train,y_train)
    y_predict= knn.predict(X_test)
    score3.append(knn.score(X_test,y_test))
    print(score3)
   


# In[24]:


max(score3)


# In[27]:


#training Polynomial regression model
scoresPOLY = []
for i in range(1,4):
    poly=PolynomialFeatures(degree=i)
    x_poly = poly.fit_transform(X)
    
    x_trainp, x_testp, y_trainp, y_testp = train_test_split(x_poly, y, test_size=0.2, random_state=0)
    
    model = LinearRegression()
    model.fit(x_trainp, y_trainp)
    
    score = model.score(x_testp, y_testp)
    scoresPOLY.append(score)
   


# In[28]:


max(scoresPOLY)


# In[31]:


# plotting scores for all models  tested above for comparison
models = ['DTR', 'POLY' , 'SVR','RFR', 'KNN']

maxscores = [max(DTRscore), max(scoresPOLY), max(score1), max(score2), max(score3)]



# In[32]:


#comparison of max scores by models (PLOTTING)
fig = plt.figure(figsize= (10,5))
plt.bar(models, maxscores, color = 'magenta')
plt.xlabel("MODELS")
plt.ylabel("scores")
plt.title("comparison of scores by models")
plt.ylim(0.59,0.82)

plt.show


# In[33]:


#conclusion:

print ("as per the figure above Random Forest Regression provides highest accuracy score")


# In[34]:


fd = pd.read_csv("C:/Users/ideod/OneDrive/Documents/new folder zip data/house_data.csv")


fd.head()

fd.dropna()


# In[36]:


selected_features1 = fd[['bedrooms', 'bathrooms', 'floors', 'zipcode']]

display(selected_features1)


# In[37]:


#selecting and scaling X variable (selected features)
scaler = StandardScaler()

X1 = scaler.fit_transform(selected_features1)


# In[38]:


# defining y variable

y1 = fd['price'].values


# In[39]:


#creating a split (train/test)
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=45)


# In[40]:


from sklearn.ensemble import RandomForestRegressor
optRFR1 = RandomForestRegressor(n_estimators = 300, random_state = 0)
optRFR1 = optRFR.fit(X1_train, y1_train)
y1_predOPT = optRFR1.predict(X1_test)


# In[41]:


expect = y1_test


# In[42]:


#STEP 6 below: predicting house price
dict = {"bedrooms":[3], "bathrooms":[2],"floors":[1],"zipcode":[98028]}


# In[43]:


fd = pd.DataFrame(dict)


# In[44]:


X2 = scaler.transform(fd)


# In[45]:


y_predicting = optRFR1.predict(fd.values)


# In[46]:


print(y_predicting)


# In[ ]:




