#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[12]:


car=pd.read_csv('CAR DETAILS.csv')
car.head()


# In[13]:


#car.shape# shape of the data is 4340 rows and 8 columns


# In[14]:


#car.dtypes# checking data types


# In[15]:


#car.isnull().sum()# there is no null values in the data set


# In[16]:


car.duplicated().sum()# there are 763 duplicates


# In[17]:


car.drop_duplicates(inplace=True)# droping the duplicates


# In[18]:


car.duplicated().sum()


# In[19]:


#car.describe()# decribing the data set


# $Observation$: The dataset is affected by outliers.

# In[51]:


# so filtering outliers using quantile 
min_thresold,max_thresold=car.km_driven.quantile([0.05,0.95])
df=car[(car.km_driven > min_thresold)&(car.km_driven < max_thresold)]
min_sell,max_sell=car.selling_price.quantile([0.05,0.95])
df=car[(car.selling_price > min_sell)&(car.selling_price < max_sell)]
#df.head()


# $LableEncoder$

# In[21]:


from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()


# In[22]:


df['name']=lb.fit_transform(df['name'])
df['seller_type']=lb.fit_transform(df['seller_type'])
df['transmission']=lb.fit_transform(df['transmission'])
df['owner']=lb.fit_transform(df['owner'])
df['fuel']=lb.fit_transform(df['fuel'])


# $Observation$:- Using LabelEncoder the object type columns are transformed into numerical type columns

# In[52]:


from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
sts=['km_driven']
df[sts]=scaler.fit_transform(df[sts])
#df.shape


# In[24]:


#df.head()


# In[25]:


#corr=df.corr()
#sns.heatmap(corr,annot=True)
#plt.show()


# $Observation$:- 
# * From the above heat map Selling Price and Year have moderate postive correlation. <br>
# * All other columns are negtively correlated with Selling Price 

# In[26]:


#df.info


# In[27]:


x=df.drop(['selling_price'],axis=1)
y=df['selling_price']
#print(type(x))
#print(type(y))
#print(x.shape)
#print(y.shape)


# In[28]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.35,train_size=0.65,random_state=42)


# In[29]:


#print(type(y_train),type(y_test))# y are series
#print(type(x_test),type(x_train))# x are DataFrame


# In[30]:


#x_train.head()


# In[31]:


#y_train.head()


# In[32]:


#print(x_train.shape,x_test.shape)# x shape
#print(y_train.shape,y_test.shape)# y shape


# In[33]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import RandomForestClassifier


# ### KNeighborsRegressor()

# In[35]:


knn=KNeighborsRegressor(n_neighbors=10,metric='manhattan')
knn.fit(x_train,y_train)


# ### LinearRegression()

# In[39]:


lg = LinearRegression()
lg.fit(x_train,y_train)


# #### Ridge Regression

# In[43]:


rg = Ridge(alpha=3)  # lambda = alpha
rg.fit(x_train,y_train)


# ### Lasso Regression()

# In[46]:


lasso = Lasso(alpha=0.2) # lambda = alpha = 0.1,0.2
lasso.fit(x_train,y_train)


# ### Saving The best model

# In[49]:


import pickle


# In[50]:


# saving the best model in binary format
# wb-write binary 
filename = 'Final_lg_model.sav'
pickle.dump(knn, open(filename, 'wb'))

