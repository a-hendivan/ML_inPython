# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 19:51:33 2022

@author: Andivan Ahmad
"""

# Import the libraries and dataset 
from sklearn import datasets 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
plt.style.use('ggplot')


# Data 
iris = datasets.load_iris()
                                
 # First we will explore how the dataset is provided by sklean package #
                                
type(iris) 
    # Out[2]: sklearn.utils.Bunch 
    # Bunch type is similar to a dictionary which had key-value pairs 

# Check keys 
print(iris.keys())
    # dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])
    

iris['target'] # i'll get the labels for this dataset / or the categories in numerical value
iris['target_names'] #i'll get the names of the categories 
    # Out[7]: array(['setosa', 'versicolor', 'virginica'], dtype='<U10')
iris['data'] # i'll get the features values
iris['feature_names'] # i'll get the columns
    # Out[6]: 
    # ['sepal length (cm)',
    # 'sepal width (cm)',
    # 'petal length (cm)',
    # 'petal width (cm)']

iris['DESCR'] # i'll get a description of the dataset 

# Shape 
iris.data.shape
    # Out[9]: (150, 4) 
    # we have 4 features / explanatory vars and 150 observations / records 
iris.target.shape
    # Out[12]: (150,) 
    # Obviously we expect number of labels = number of observations 
    
    
    
                                         # Exploratory Data Analysis # 

# Create a dataframe so we can work with it using sklearn module 

X = iris.data 
y = iris.target 

df= pd.DataFrame ( X , columns = iris.feature_names)

df.head()
df.info()
df.describe().round(2)

df.shape
    # Out[23]: (150, 4)


# Let's visualise our data 

plot = pd.plotting.scatter_matrix(df, c = y , figsize = [8,8] , s =150 ,marker="A")
 
# 
df2 = np.column_stack( (X,y ))
col = iris.feature_names
col.append("species")
df2 = pd.DataFrame ( df2 , columns= col  )


sns.color_palette("hls", 8)

plt.figure() 
sns.countplot( data = df2  , x = "species",palette="magma" )
# palette options : {cubehelix,rocket,magma, Set3, bright, pastel,deep , PRGn,BrBG,RdYlGn , RdYlBu,PiYG }
plt.xticks( [ 0,1,2] , iris.target_names)
plt.xlabel("Species")
plt.ylabel("Total")
plt.title("Distrbution of species in the dataset")
plt.show()

# We can see that this data set is equally distributed among the 3 categories we have 
# which means there is no bias toward any of the categories  



                