#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn import tree
import sklearn.metrics
from sklearn import tree
from io import StringIO
from IPython.display import Image
import pydotplus
import seaborn as sns


# In[2]:


df = pd.read_csv(r"C:\Users\laila\Documents\DecissionTreeOfMathValuebyLailaOktavia.csv")


# In[3]:


df.head(5)


# In[4]:


df.isnull().sum()


# In[5]:


df.dropna()


# In[6]:


df.describe()


# In[7]:


df[['JUMLAH_MURID','RATA_RATA_NILAI_UTS', 'RATA_RATA_NILAI_UAS', 'RATA_RATA_NILAI_TUGAS']].sample(10)


# In[8]:


predictors = df [['RATA_RATA_NILAI_UTS','RATA_RATA_NILAI_UAS','RATA_RATA_NILAI_TUGAS']]
targets = df.JUMLAH_MURID
X_train, X_test, y_train, y_test = train_test_split(predictors, targets, test_size=.2, random_state=0)


# In[9]:


print('X_train =' , X_train.shape )
print('X_test =' , X_test.shape )
print('y_train =' , y_train.shape )  
print('y_test =' , y_test.shape )  


# In[16]:


classifier=DecisionTreeClassifier()
classifier.fit(X_train,y_train)
predictions=classifier.predict(X_test)
confussion_array=sklearn.metrics.confusion_matrix(y_test,predictions)
print(confussion_array)


# In[19]:


print('TN =', confussion_array[0,0])
print('FN =', confussion_array[1,0])
print('TP =', confussion_array[1,1])
print('FP =', confussion_array[0,1])


# In[20]:


sklearn.metrics.accuracy_score(y_test,predictions)


# In[21]:


import os


# In[25]:


out = StringIO()
tree.export_graphviz(classifier, out_file=out)
import pydotplus
graph=pydotplus.graph_from_dot_data(out.getvalue())
Image(graph.create_png())


# In[ ]:




