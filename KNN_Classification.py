# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 20:32:31 2022

@author: Andivan Ahmad
"""

from Iris_Dataset_DA import *

# Modules for Classification 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.impute import SimpleImputer 
# imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, RandomizedSearchCV,GridSearchCV

# Let's have a look at our data set once again 
print(df2.head())

# As we are going to use sklearn package we need to be sure that :
    # [1] The target viable is numerical
    # [2] There is no missing data 
    # [3] Normalised data 

print("Information about the data \n")
print( df2.info())
print("Summary Statistics\n", "\n" ,df.describe().round(2))



                                 # Split data into train / test #
                     
# Now sklearn requires for the target variable / respose to be a numpy array alone 
# seperatly from the features numpy array 
# in our case the data was already in this form bc it was taken from the package itself
# But this wont be the case for all datasets. Mosat often we would need to split the data
# into to numpy arrays one for features (X) and one for target (y)

# X = np.array(df2.loc[ df2.columns !="species"] )
# y = np.array(df2.loc[ df2.columns =="species"] )

X_train , X_test , y_train, y_test  = train_test_split( X, y , test_size =0.25,random_state=21, stratify=y )   
      

# We will run the model for different k-neighbors ( 3 to 10  ) 
# and then using GridSearchCV we will choose the k that gives better results 

                                # Model Selection # 
     
k_grid = {"n_neighbors": np.arange(3,11)}

knn = KNeighborsClassifier()

# For number of fold for cross-validation technique
# any number between 5-10 offers an optimal balance of the bias-variance trade of 
knn_cv = GridSearchCV(knn , k_grid , cv =6) 

fit = knn_cv.fit(X_train, y_train)

fit.best_params_
        # {'n_neighbors': 4} 
            
# So the best model uses k=4 ( 4 neighboring points)

# MSE (r2) for the best hyperparameter choice ~ here k =4 
fit.best_score_
        # 0.973196881091618
        

# Predict target using the test set 
y_pred = fit.predict(X_test)

# PErcentage of true answers - but very vauge 
sum(y_pred == y_test) / len(y_pred)
        # 0.9736842105263158
  


                            # Visualization # 
                            
import seaborn as sns 
import matplotlib.pyplot as plt 

# df2.columns

        #Index(['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)',
        #    'petal width (cm)', 'species'],
        #    dtype='object')
plt.figure(figsize = (20,10))
plt.subplot (1,2,1)   # plt.subplot ( rows , columns , current subplot )

sns.scatterplot ( X_test[:,2],X_test[:,3] ,  hue = y_test,s=150,palette="deep")

plt.xlabel("Petal Length (cm)" ,fontsize =20)
plt.ylabel ("Petal Width (cm)"  ,fontsize =20)
plt.title("Original Target Classification",fontsize =30)
plt.legend (["setosa","versicolor","virginica"],fontsize = 15)
plt.subplot (1,2,2) 

sns.scatterplot ( X_test[:,2] , X_test[:,3] , hue = y_pred,s =150, palette ="deep")
plt.xlabel("Petal Length (cm)",fontsize =20 )
plt.ylabel ("Petal Width (cm)" ,fontsize =20 )
plt.title("Predicted Target Classification",fontsize =30)

plt.legend (["setosa","versicolor","virginica"],fontsize = 15)
plt.tight_layout()
plt.show()


# import matplotlib.gridspec as gridspec
# gs =gridspec.GridSpec(1,2)
# fig = plt.figure(figsize = (25,10 ))


# ax = plt.subplot (gs[0,0]) 
# ax = sns.scatterplot ( X_test[:,2],X_test[:,3] ,  hue = y_test , palette = "deep")
# plt.xlabel("Petal Length (cm)" )
# plt.ylabel ("Petal Width (cm)"  )
# plt.title("Original Target Classification")

# ax1 = plt.subplot (gs[0,1]) 
# sns.scatterplot ( X_test[:,2] , X_test[:,3] , hue = y_pred,palette = "deep")
# plt.xlabel("Petal Length (cm)" )
# plt.ylabel ("Petal Width (cm)"  )
# plt.title("Predicted Target Classification")

# plt.show()



# plt.figure(figsize = (20,10))
# plt.subplot (1,2,1)   # plt.subplot ( rows , columns , current subplot )

# sns.scatterplot ( X_test[:,0],X_test[:,1] ,  hue = y_test,s=150,palette="deep")

# plt.xlabel("Sepal Length (cm)" ,fontsize =20)
# plt.ylabel ("Sepal Width (cm)"  ,fontsize =20)
# plt.title("Original Target Classification",fontsize =30)
# plt.legend (["setosa","versicolor","virginica"],fontsize = 15)
# plt.subplot (1,2,2) 

# sns.scatterplot ( X_test[:,0] , X_test[:,1] , hue = y_pred,s =150, palette ="deep")
# plt.xlabel("Sepal Length (cm)",fontsize =20 )
# plt.ylabel ("Sepal Width (cm)" ,fontsize =20 )
# plt.title("Predicted Target Classification",fontsize =30)

# plt.legend (["setosa","versicolor","virginica"],fontsize = 15)
# plt.tight_layout()
# plt.show()

                               # Evaluate the model # 
                               
# We will use confusion matrix 

fig , ax = plt.subplots()
plt.axis("off")
plt.axis("tight")#
plt.title("Confussion Matrix")
table = ax.table(cellText =  confusion_matrix(y_test, y_pred) ,colLabels = ["Setosa","Versicolor","Virginica"], loc = "center")

fig.tight_layout() 
plt.show()

# Different metrics to evaluate the Knn Classifier 

fig , ax = plt.subplots()
plt.axis("off")
plt.title("Classification Metrics Report")

# Turn the output of classification_report in a dictionary and then convert to a df
temp =  pd.DataFrame (classification_report( y_test, y_pred,output_dict = True)  )
# Rename Category Column Names 
temp.rename(columns = {"0" :"Setosa", "1":"Versicolor" , "2":"Virginica"} , inplace=True)

# Create the table object 
table = ax.table(cellText = temp.values.round(2) , colLabels = temp.columns , loc = "center")

# Modify the table 
fig.tight_layout() 
table.set_fontsize(14)
table.scale(1,2)
plt.show()
