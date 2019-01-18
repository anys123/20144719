/*
Author :An, Yu Sang, dksdbtkd123@naver.com
Supervisor : Na, In Seop, ypencil@hanmail.net
Starting Project : 2019.1.6
*/

MIT License 
 
Copyright (c) 2019 anys123 

 
Permission is hereby granted, free of charge, to any person obtaining a copy 
of this software and associated documentation files (the "Software"), to deal 
in the Software without restriction, including without limitation the rights 
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
copies of the Software, and to permit persons to whom the Software is 
furnished to do so, subject to the following conditions: 
 
 
The above copyright notice and this permission notice shall be included in all 
copies or substantial portions of the Software. 

 
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
SOFTWARE.

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # CSV file I/O (e.g. pd.read_csv)
import os # reading the input files we have access to

print(os.listdir(r'C:\Users\chosun\Desktop\AN\taxi-1'))


# In[2]:


train_df =  pd.read_csv(r'C:\Users\chosun\Desktop\AN\taxi-1/train.csv', nrows = 60_000_000)
train_df.dtypes


# In[3]:


def add_travel_vector_features(df):
    df['abs_diff_longitude'] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df['abs_diff_latitude'] = (df.dropoff_latitude - df.pickup_latitude).abs()

add_travel_vector_features(train_df)


# In[4]:


print(train_df.isnull().sum())


# In[5]:


print('Old size: %d' % len(train_df))
train_df = train_df.dropna(how = 'any', axis = 'rows')
print('New size: %d' % len(train_df))


# In[6]:


plot = train_df.iloc[:2000].plot.scatter('abs_diff_longitude', 'abs_diff_latitude')


# In[7]:


print('Old size: %d' % len(train_df))
train_df = train_df[(train_df.abs_diff_longitude < 5.0) & (train_df.abs_diff_latitude < 5.0)]
print('New size: %d' % len(train_df))


# In[8]:


def get_input_matrix(df):
    return np.column_stack((df.abs_diff_longitude, df.abs_diff_latitude, np.ones(len(df))))

train_X = get_input_matrix(train_df)
train_y = np.array(train_df['fare_amount'])

print(train_X.shape)
print(train_y.shape)


# In[9]:


(w, _, _, _) = np.linalg.lstsq(train_X, train_y, rcond = None)
print(w)


# In[10]:


w_OLS = np.matmul(np.matmul(np.linalg.inv(np.matmul(train_X.T, train_X)), train_X.T), train_y)
print(w_OLS)


# In[11]:


test_df = pd.read_csv(r'C:\Users\chosun\Desktop\AN\taxi-1/test.csv')
test_df.dtypes


# In[12]:


add_travel_vector_features(test_df)
test_X = get_input_matrix(test_df)
test_y_predictions = np.matmul(test_X, w).round(decimals = 2)

submission = pd.DataFrame(
    {'key': test_df.key, 'fare_amount': test_y_predictions},
    columns = ['key', 'fare_amount'])
submission.to_csv('submission.csv', index = False)

print(os.listdir('.'))


# In[13]:




