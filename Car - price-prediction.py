#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
warnings.filterwarnings('ignore')


# In[3]:


car=pd.read_csv('CAR DETAILS.csv')
car.head()


# 𝑜𝑏𝑠𝑒𝑟𝑣𝑎𝑡𝑖𝑜𝑛: There are 4340 rows and 8 columns

# In[4]:


car.shape


# $observation$: there is no null values in this data set

# In[5]:


car.isnull().sum()


# $𝑜𝑏𝑠𝑒𝑟𝑣𝑎𝑡𝑖𝑜𝑛$: There are 5 objects and 3 integers datatype in the data set

# In[6]:


car.dtypes


# $Observation$:
# * Name, Fuel, Seller Type, Transmission, Owner are object Data type
# * Year is integer type currently. Since Year is a categorical variable here, let us convert it to category data data type in Python.
# * Remaining Two column variables are numerical and there for their python data types int64 are ok.

# In[7]:


car.duplicated().sum()


# In[8]:


car.drop_duplicates(inplace=True)


# In[9]:


car.duplicated().sum()


# $Conclusion$: There are 763 duplicates now we droped the duplicates.

# In[10]:


car.describe()


# $Obsercation$:-
# * The Selling Price of a car Ranging from 20000 to 8900000 
# * The mean Selling Price is close to the 75% Percentile of the data, indecating right skew.
# * The standard deviation of selling price is high

# $Observations$: There are 763 duplicates values in the Data set

# In[11]:


Car_Brand = car['name'].apply(lambda x : x.split(' ')[0])
car.insert(1,"Car Brand",Car_Brand)
#car.drop(['name'],axis=1,inplace=True)


# In[12]:


car.head()


# ## Uni-Variate EDA
# <b>value counts of categorical column Car Brands on line chart(matplotlib)

# In[98]:


c1=car['Car Brand'].value_counts()
#c2=pd.DataFrame(c1).reset_index()
#c2
print(c1.index)
c1.values


# In[103]:


plt.plot(c1.index,c1.values,color='blue',marker='o',lw=2,markersize=5)
plt.title('value counts of car brand')
plt.xlabel('Car Brands')
plt.ylabel('Car Counts')
plt.xticks(rotation=90)
plt.grid()
plt.show()


# $Conclusion$: There are totally 29 Brands. Maruti, Hyundai, Mahindra are top three cars with Large number of car series.

# <b>value counts of categorical column namely fuels on line chart(matplotlib)

# In[15]:


c1=car['fuel'].value_counts()
print(c1.index)
print(c1.values)


# In[16]:


#plt.figure(figsize=(8,5))
plt.plot(c1.index,c1.values,color='red',marker='d')
plt.xlabel('fuels')
plt.title('types of fuels')
plt.xticks(rotation=90)
plt.grid()
plt.show()


# $Conculsion$:
# * The Count of diesel cars and petrol cars are higer than CNG ,LPG, Electric
# * And there is only one electric car in the data set.

# <b>value counts seller type on pie chart(matplotlib)

# In[104]:


c2=car['seller_type'].value_counts()
c2.index


# In[115]:


plt.pie(c2.values,labels=c2.index,explode=(0,0.1,0),autopct='%.2f%%',shadow=True,)
plt.title('seller type count percentage distribution')
plt.legend()
plt.show()


# $Conclusion$: From the above graph we can see that most people sell their cars individually instead of going to dealers.

# <b>Count values of Transmission on bar chart using matplotlib

# In[19]:


c3=car['transmission'].value_counts()
c3


# In[20]:


plt.figure(figsize=(4,3))
plt.bar(c3.index,c3.values,color='blue',edgecolor='black',label='Transmission counts')
plt.title('Transmission Counts')
plt.ylabel('Counts')
plt.xlabel('Transmissions')
plt.show()


# $Conclusion$: Maximum number of cars are transmitted  manually.

# ### Count values of owner on char using 

# In[21]:


c4=car['owner'].value_counts()
c4


# In[22]:


sns.lineplot(c4.index,c4.values,marker='*',markersize=17)
plt.xticks(rotation=90)
plt.title('car owners')
plt.ylabel('value counts')
plt.xlabel('owners')
plt.grid()
plt.show()


# $Conclusion$: 
# * From the above graph we can conclude that the number of First Owners selling their car is higher.
# * The number of Fourth & Above Owners selling the car is low reason may be due to less profit through selling the cars.
# * Test Drive Car are the lowest count of all other counts.

# In[23]:


sns.pairplot(car,diag_kind='kde')
plt.show()


# In[24]:


corr= car.corr()
print(corr)
#print(corr.index)
#corr.values


# In[25]:


sns.heatmap(corr,annot=True,vmin=-1,vmax=1,fmt=".2f")
plt.show()


# $Conculsion$- 
# * The Km driven and Selling Price have a very low negative correlation.
# * Selling price and year have a moderate postive correlation. 

# ### Normal Distribution

# In[26]:


sns.distplot(np.log(car['selling_price']),color='blue')
plt.show()


# ### Average selling price based on year using bar chart and line plot

# In[105]:


d1=car.groupby('year')['selling_price'].mean().reset_index()
d1=d1.sort_values(by='year',ascending=True)


# In[106]:


plt.figure(figsize=(10,4))
plt.bar(d1['year'],d1['selling_price'],color='cyan',edgecolor='black',label='selling_price')
plt.plot(d1['year'],d1['selling_price'],color='red',lw=2,marker='o')
plt.title('year wise car selling price',fontsize=15,color='Orange')
plt.xlabel('Year')
plt.ylabel('selling price')
plt.xlim(1990,2021)
plt.legend()
plt.grid()
plt.show()


# $Conclusion$:
# * From the above graph we can conclude that the selling price of cars increased from the year 2006 till the year 2019. 
# * But in the year 2020, selling prices dropped slightly. 

# <b> Average selling price and year Correlation based on heat map 

# In[29]:


c1=car.groupby('year')[['selling_price']].mean().reset_index()
corr1=c1.corr()
sns.heatmap(corr1,annot=True,vmin=-1,vmax=1,fmt='.2f')
plt.show()


# $Conclusion$: Average selling price and year have the strong correlation.

# #### Car Brand based selling price

# In[30]:


br=car.groupby('Car Brand')['selling_price'].mean()
bi=br.index.tolist()
bv=br.values.tolist()
print(bi)
print(bv)


# In[31]:


car[car['Car Brand']=='Mercedes-Benz'].mean()


# In[32]:


plt.figure(figsize=(8,4))
plt.bar(bi,bv,color='orange',edgecolor='black')
plt.bar(bi[bv.index(max(bv))],max(bv),color='green')
plt.bar(bi[bv.index(min(bv))],min(bv),color='red')
plt.xlabel('Car Brands')
plt.xticks(rotation=90)
plt.show()


# $Conclusion$: In this Land Car brand has the highest selling price and Daewoo has the lowest selling price

# <b> Fuel based selling price

# In[33]:


car.columns


# In[34]:


d3=car.groupby('fuel')[['selling_price']].mean()
d3.head()


# In[125]:


plt.bar(d3.index,d3['selling_price'],label='selling_price',color='orange',edgecolor='black')
plt.title('fuel wise km driven and selling price')
plt.xlabel('fuel',fontsize=15)
#plt.xticks(rotation=90)
plt.legend()
plt.savefig('fuel wise km driven and selling price')
plt.show()


# $Conclustion$:
# * The Conclustion from the above graph that the <b>LPG</b> car has the mean selling price is less then all other fuels.
# * The Selling Price of the Diesel car is Higher then all other fuel cars.

# <b> Type of fuels used from which year 

# In[57]:


d4=car.groupby(['fuel'])[['year']].min()
d5=car.groupby(['fuel'])[['year']].max()
print(d4)
d5


# In[37]:


plt.bar(d4.index,d4['year'],label='From year',color='cyan',edgecolor='black')
plt.bar(d5.index,d5['year'],label='To year',color='orange',edgecolor='black',alpha=0.5)
plt.ylim(1989,2024)
plt.xlabel('fuel')
plt.ylabel('year')
plt.legend(loc=9)
plt.show()


# $Conculsion$: In the data set petrol cars are from 1992 to 2020 and diesel cars from 1993 to 2020.

# <b> Average selling price based on owner

# In[117]:


# average selling price based on owner
own=car.groupby('owner')[['selling_price']].mean()
own


# In[118]:


# lets plot it in the graph
plt.bar(own.index,own['selling_price'],label='Owners',edgecolor='black')
plt.title('Owner Based Average Selling Price')
plt.legend()
plt.ylabel('Selling price')
plt.xlabel('Owner')
plt.xticks(rotation=90)
plt.show()


# $Observation$: 
# * Test Drive car's selling price are higher than all other cars. Because these cars are directly from the showroom.
# * Fourth & Above-Owner car's selling price are less than all other cars' selling prices. 
# * Because these cars are used by more than four owners, so there may be worse fuel efficiency, old technology are reasons for the drop in selling price.

# <b> Average selling price based seller type

# In[51]:


s1=car['seller_type'].value_counts()
s1=car.groupby('seller_type')['selling_price'].mean()
print(s1.index)
s1.values


# In[122]:


plt.pie(s1.values,labels=s1.index,colors=['red','green','cyan'],explode=(0,0,0.1),autopct='%.2f%%',shadow='True')
plt.title('seller type selling price')
plt.xlabel('seller_type')
plt.ylabel('mean selling price',loc='bottom')
plt.show()


# <b> Average km driven based on seller type

# In[48]:


owndri=car.groupby('seller_type')[['km_driven']].mean()# average km_driven 
owndri.index


# In[49]:


plt.bar(owndri.index,owndri['km_driven'],label='Owners',color='yellow',edgecolor='black')
plt.title('Owner Based Average km driven')
plt.legend()
plt.ylabel('km driven')
plt.xlabel('Owner')
plt.xticks(rotation=90)
plt.show()


# $Conclustion$:
# * We saw in the previous graph the seller_type count of Trustmarker Dealer is very low. But this graph is the opposite, in which the average selling price of the Trustmark Dealer is higher than the other two selling types.
# * The average km driven of Trustmark Dealer is less compare to individual and Dealer

# <b> Average selling price based on Transmission

# In[123]:


tran=car.groupby(['transmission'])[['selling_price']].mean()
tran


# In[124]:


plt.bar(tran.index,tran['selling_price'],label='transmission',color='green',alpha=0.4,edgecolor='black')
plt.title('transmission based selling price')
plt.xlabel('Transmission')
plt.ylabel('selling price')
plt.legend()
plt.show()


# $Conclusion$: 
# * The Automatic transmission based car selling price is higher compare to manual transmission selling price of the car.
# * We seen in the previous graph that the transmission count manual is high and the transmission count automatic is low. 
# * automatic transmissions are easier to use and more comfortable for the driver but the cars more expansive
# * while manual transmission vehicles are less expensive and more involved.

# ### Creating ML model

# In[136]:


# so filtering outliers using quantile 
min_thresold,max_thresold=car.km_driven.quantile([0.05,0.95])
df=car[(car.km_driven > min_thresold)&(car.km_driven < max_thresold)]
min_sell,max_sell=car.selling_price.quantile([0.05,0.95])
df=car[(car.selling_price > min_sell)&(car.selling_price < max_sell)]
#df.head()


# In[137]:


car['Car Brand'].value_counts()


# In[138]:


df= df[(df['Car Brand'] != 'Kia')&(df['Car Brand']!='Daewoo')&(df['Car Brand']!='Force')&(df['Car Brand']!='Isuzu')
       &(df['Car Brand']!='MG')&(df['Car Brand']!='OpelCorsa')
       &(df['Car Brand']!='Jeep')&(df['Car Brand']!='Ambassador')]
df=df[(df['Car Brand']!='Mitsubishi')&(df['Car Brand']!='BMW')&
      (df['Car Brand']!='Mercedes-Benz')&(df['Car Brand']!='Audi')]
df=df[(df['Car Brand']!='Skoda')&(df['Car Brand']!='Fiat')
      &(df['Car Brand']!='Datsun')&(df['Car Brand']!='Nissan')]


# In[139]:


df['Car Brand'].value_counts()


# In[140]:


df['fuel'].value_counts()
df=df[df['fuel'] !='Electric']
df=df[df['fuel']!='LPG']
df['fuel'].value_counts()


# $LableEncoder$

# In[141]:


from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()


# In[142]:


df['name']=lb.fit_transform(df['name'])
df['seller_type']=lb.fit_transform(df['seller_type'])
df['transmission']=lb.fit_transform(df['transmission'])
df['owner']=lb.fit_transform(df['owner'])
df['fuel']=lb.fit_transform(df['fuel'])
df['Car Brand']=lb.fit_transform(df['Car Brand'])


# In[143]:


from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
sts=['km_driven']
df[sts]=scaler.fit_transform(df[sts])
#df.shape


# In[144]:


#df.head()


# In[145]:


#corr=df.corr()
#sns.heatmap(corr,annot=True)
#plt.show()


# $Observation$:- 
# * From the above heat map Selling Price and Year have moderate postive correlation. <br>
# * All other columns are negtively correlated with Selling Price 

# In[146]:


#df.info


# In[147]:


x=df.drop(['selling_price'],axis=1)
y=df['selling_price']
#print(type(x))
#print(type(y))
#print(x.shape)
#print(y.shape)


# In[148]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.35,train_size=0.65,random_state=42)


# In[149]:


#print(type(y_train),type(y_test))# y are series
#print(type(x_test),type(x_train))# x are DataFrame


# In[150]:


x_train.head()


# In[151]:


#y_train.head()


# In[152]:


#print(x_train.shape,x_test.shape)# x shape
#print(y_train.shape,y_test.shape)# y shape


# In[153]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import RandomForestClassifier


# ### KNeighborsRegressor()

# In[154]:


knn=KNeighborsRegressor(n_neighbors=10,metric='manhattan')
knn.fit(x_train,y_train)
y_pred_knn=knn.predict(x_test)
mean_absolute_error(y_test,y_pred_knn)
r2_score(y_test,y_pred_knn)


# In[155]:


knn.score(x_train,y_train)
knn.score(x_test,y_test)


# ### LinearRegression()

# In[156]:


lg = LinearRegression()
lg.fit(x_train,y_train)
y_pred_lg=lg.predict(x_test)
#mean_squared_error(y_test,y_pred_lg)
r2_score(y_test,y_pred_lg)


# #### Ridge Regression

# In[157]:


rg = Ridge(alpha=3)  # lambda = alpha
rg.fit(x_train,y_train)
y_pred_rg=rg.predict(x_test)
#mean_absolute_error(y_test,y_pred_rg)
r2_score(y_test,y_pred_rg)


# ### Lasso Regression()

# In[158]:


lasso = Lasso(alpha=0.2) # lambda = alpha = 0.1,0.2
lasso.fit(x_train,y_train)
y_pred_lo=lasso.predict(x_test)
#mean_absolute_error(y_test,y_pred_rg)
r2_score(y_test,y_pred_lo)


# ### Saving The best model

# In[159]:


import pickle


# In[160]:


# saving the best model in binary format
# wb-write binary 
filename = 'Knn_model.pkl'
pickle.dump(knn, open(filename, 'wb'))


# In[161]:


# saving the best model in binary format 
# wb-write binary
filename= 'df.plk'
pickle.dump(df,open(filename,'wb'))


# In[162]:


#knn=pickle.load(open('knn_model.pkl','rb'))
#y_pred1=knn.predict(x_test)
#r2_score(y_test,y_pred1)

