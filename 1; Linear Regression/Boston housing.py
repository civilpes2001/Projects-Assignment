#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from sklearn import datasets
boston = datasets.load_boston()
boston.feature_names


# In[3]:


print(boston.DESCR)
boston.keys()
boston.data.shape
boston.feature_names
boston.target


# In[4]:


boston_df = pd.DataFrame(boston.data, columns = boston.feature_names)
boston_df.head()
boston_df['House_Price'] = boston.target
boston_df.head()
boston_df.describe()


# In[5]:


x = boston_df.corr()
x
plt.subplots(figsize=(20,20))
sns.heatmap(x, cmap = 'RdYlGn', annot = True)
plt.show()


# In[6]:


x = boston_df.drop('House_Price', axis = 1)
y = boston_df['House_Price']
x.head()
y.head()


# In[7]:


train_x, test_x, train_y, test_y = train_test_split(x,y,test_size = 0.25, random_state = 1)

train_x.shape
test_x.shape
train_y.shape
test_y.shape


# In[8]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm


# In[18]:


lm.fit(train_x, train_y)


# In[10]:


predict_test = lm.predict(test_x)


# In[11]:


print (lm.coef_)
df_m = pd.DataFrame({'features':x.columns, 'coeff':lm.coef_})
df_m = df_m.sort_values(by=['coeff'])
df_m


# In[12]:


df_m.plot(x = 'features', y = 'coeff', kind = 'bar', figsize = (20,15))
plt.show()


# In[13]:


print('RSquare Value for TEST data is-')
np.round(lm.score(test_x,test_y)*100,0)


# In[14]:


print('RSquare Value for TRAIN data is-')
np.round(lm.score(train_x,train_y)*100,0)


# In[15]:


print('Mean Squared Error (MSE) for TEST Data is')
np.round(metrics.mean_squared_error(test_y, predict_test), 0)


# In[16]:


print('Mean Absolute Error (MAE) for TEST Data is')
np.round(metrics.mean_absolute_error(test_y, predict_test), 0)


# In[17]:


fdf = pd.concat([test_x, test_y],1)
fdf['Predicted'] = np.round(predict_test,1)
fdf['Prediction_Error']  = fdf['House_Price'] - fdf['Predicted']
fdf


# In[ ]:




