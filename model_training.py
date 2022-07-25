# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 09:20:37 2022

@author: isaac
"""

import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as ss
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

#%%

def cramers_corrected_stat(confusion_matrix):
    '''
    calculate Cramers V statistic for categorial-categorial association.
    uses correction from Bergsma and Wicher, 
    Journal of the Korean Statistical Society 42 (2013): 323-328

    Parameters
    ----------
    confusion_matrix : DataFrame
        Pandas crosstab product between categorical.

    Returns
    -------
    int
        correlation.

    '''
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))

#%% PATH

CSV_PATH = os.path.join(os.getcwd(), 'datasets', 'heart.csv')
BEST_MODEL_SAVE_PATH = os.path.join(os.getcwd(), 'models','BEST_MODEL.pkl')

#%% Step 1) Data Loading

df = pd.read_csv(CSV_PATH)

#%% Step 2) Data Inspection & Visualization

df.info() # check for dtype
desc = df.describe().T # check for min -> possibility to be category

df.isna().sum() # check for missing data == 0
# however mentioned in the disscussion, thall == 0 is null therefore;
(df.thall==0).sum() # only 2 is detected
df.thall[(df.thall==0)] = np.nan # create nans
df.isna().sum() # double confirm on the create nans


# check and drop duplicates
df.duplicated().sum()
df.drop_duplicates(inplace=True)

# since only oldpeak is in float, lets inspect further
df.oldpeak.describe()

# check for unique and no of unique
df.oldpeak.unique()
df.oldpeak.nunique() # 40 no of unique


categ = []
conti = []
# often, category has a small number of unique
print('- -'*5, '\nNumber of Unique\n', '- -'*5)
for i in df.columns:
    print(f'[{i}] has a [{df[i].nunique()}] of unique')
    
    if df[i].nunique() > 10:
        conti.append(i)
    else:
        categ.append(i)

# lets plots each of the categorical and continuos to see if anything is weird
# plotting for categorical
for i in categ:
    sns.countplot(df[i])
    plt.title(i)
    plt.show()
    
# the target is quite balanced

# plotting for continous
for i in conti:
    sns.displot(df[i], kde=True)
    plt.title(i)
    plt.show()

for i in conti:
    sns.boxplot(y=df[i], orient='v')
    plt.title(i)
    plt.show()

# oldpeak is somewhat concerning
# in boxplot; trtbps, chol and oldpeak shows quite a few numbers of outliers

# lets further check on relationship with target
for i in categ:
    if i != 'output':
        df.groupby(['output', i]).agg({'output': 'count'}).plot(kind='bar')
        plt.title(f'output & {i} count')
        plt.show()
        
# Based on this datasets, the relationship as follows:
# Gender == 1, has higher risk of getting heart attack
# cp == 2, has higher risk of getting heart attack
# fbs == 0, has higher risk of getting heart attack
# rest_ecg == 1, has higher risk of getting heart attack
# exang == 0, has higher risk of getting heart attack
# slp == 2, has higher risk of getting heart attack
# caa == 0, has higher risk of getting heart attack
# thall == 2, has higher risk of getting heart attack
# keep in mind that this statement may changes if better datasets is available


# from https://mode.com/blog/violin-plot-examples/, says that mean and median
# are insufficient to understand the data therefore, countplot were used to
# shows the summary statistic and distribution
for i in conti:
    sns.violinplot(y=df[i], x=df.output)
    plt.title(i)
    plt.show()
    
# output == 1, has normal distribution for all ages
# output == 1 has higher density as well as interquartile range of chol
# output == 1, has higher mean of thalachh
# output == 1, has significant density with lower oldpeak

#%% Step 3) Data Cleaning
df2 = df.copy()
df2.isna().sum()
# since thall is categorical, fill with mode
df2.thall.fillna(df.thall.mode()[0], inplace=True)
df2.isna().sum() # double check for any NaN's

#%% Step 4) Feature Selection
# remember that our target is categorical
sel_features = [] # emtpy template for selected_feature

# the threshold has been tried from 0.4 to 0.6, with 0.5 produced the best
# score of 0.85, although it is possible to create a loop for selecting the
# threshold, it is computationally expensive, therefore this method is 
# preferred.
threshold = 0.5

# Feature selection between categorical & categorical
print('- -'*5, '\nSelection for categorical vs categorical\n', '- -'*5)
for i in categ:
    if i != 'output':
        cm_ = pd.crosstab(df2[i], df2['output']).to_numpy()
        score = cramers_corrected_stat(cm_)
        print(f'{i} : {score}')
        if score > threshold:
            sel_features.append(i)

# Feature selection between categorical & continous
print('- -'*5, '\nSelection for categorical vs continous\n', '- -'*5)
for i in conti:
    lr = LogisticRegression()
    lr.fit(np.expand_dims(df2[i], axis=-1), df2.output)
    score = lr.score(np.expand_dims(df2[i], axis=-1), df2.output)
    print(f'{i} : {score}')
    if score > threshold:
        sel_features.append(i)

print('- -'*5, '\nSelected Features\n', '- -'*5)
print(sel_features)

#%% Step 5) Preprocessing

X = df2[sel_features]
y = df2['output']


X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=42)

#%% Step 5) Pipeline


# create a dictionary for scaler and machine learning
scaler_dict = {
    'Min_Max_Scaler': MinMaxScaler(),
    'Standard_Scaler_': StandardScaler()
}

ml_dict = {
    'Logistic_Classifier': LogisticRegression(),
    'KNN_Classifier': KNeighborsClassifier(),
    'Decision_Tree_Classifier': DecisionTreeClassifier(),
    'SVC_Classifier': SVC(),
    'GBoost_Classifier': GradientBoostingClassifier(),
    'RForest_Classifier': RandomForestClassifier()
}

import re

pipelines_dict = {}

# create a combination between scaler and machine learning
for i in scaler_dict:
    for j in ml_dict:
        name_scaler = re.sub('[^A-Z]', '', i)
        name_ml = re.sub('[^A-Z]', '', j)
        pipelines_dict[name_scaler+'_'+name_ml] = Pipeline([(i,
                                                             scaler_dict[i]),
                                                            (j, ml_dict[j])
                                                            ])
# create an empty template for storing best models
models_score = {}
best_score = 0
best_model = []

# fit the models using pipelines
print('- -'*5, '\nScaler + Models scores\n', '- -'*5)
for pipe in pipelines_dict:
   pipelines_dict[pipe].fit(X_train, y_train)
   models_score[pipe] = pipelines_dict[pipe].score(X_test, y_test)
   print(f'The score for {pipe} : {models_score[pipe]}')
   if models_score[pipe] > best_score:
       best_score = models_score[pipe]
       best_model = pipelines_dict[pipe]

#%% extracting estimator paramater for hyperparameter tuning
# The purpose of this section is to make the program more robust in accepting
# new model or incase of randomness changes the best_pipelines.
# In addition, the program may not suitable for some model as it has not been
# extensively test to all model due to computational limitation
# The main idea of this section is to automatically extract the hyperparameter
# in the best_pipeline and randomly assigned the parameter for float and int
# value only. It is also best practice to changes this section into function
# upon extensively experiment.
best_pipeline = best_model

esti_temp_name = best_pipeline.steps[1][0] + '__'
estimator_params = {}
estimator_params_2 = {}

# extract all estimator paramter keys
for index, key in enumerate(best_pipeline.get_params()):
    if esti_temp_name in key:
        estimator_params[key] = best_pipeline.get_params()[key]

# extract all estimator hyperparamter keys (int&float)
for key in estimator_params:
    if len(estimator_params_2) < 3:
        if isinstance(estimator_params[key], (bool)):
            pass
        elif isinstance(estimator_params[key], (int, float)):
            estimator_params_2[key] = estimator_params[key]            
        
# based on previous loop, select 3 (int&float ) related hyperparameter
# the range of hyperparameter is between defaul value to default value + 1
for key in estimator_params_2:
    if isinstance(estimator_params[key], (int)):
        estimator_params_2[key] = np.arange(estimator_params[key],
                                            estimator_params[key]*2+1,
                                            1)
    elif isinstance(estimator_params[key], (float)):
        estimator_params_2[key] = np.arange(estimator_params[key],
                                            estimator_params[key]*2+1,
                                            0.1)

# To extract the selected model hyperparamter
grid_param = [estimator_params_2]

# Create a GridSearchCV using our model parameter
gridsearch = GridSearchCV(best_pipeline,
                          grid_param,
                          cv=5,
                          verbose=1,
                          n_jobs=-1)

# GridSearchCV using the best model from pipeline
grid = gridsearch.fit(X_train, y_train)

# The score of GridSearchCV using the selected model
gridsearch.score(X_test, y_test)
print(grid.best_params_)

print('The best scaler and classifier for this case is {} with accuracy {}'
      .format(best_pipeline.steps, best_score))

#%% model analysis

y_true = y_test
y_pred = grid.predict(X_test)
print(classification_report(y_true, y_pred))

models_score = pd.DataFrame(models_score, index=[0]).T

#%% Model Saving

with open(BEST_MODEL_SAVE_PATH, 'wb') as file:
    pickle.dump(grid.best_estimator_, file)
    
#%% Conclusion
''' 
Based on the relation between target and features as explored on step 2,
we can deduce the correlation between output and features as a pattern was
identified. Furthermore, more improvement can be made by feature engineering
or applying deep learning method.

In addition, the target is in balanced condition which improve the success
rate of our machine learning implementation.

Although there are some outliers present, scikit-learn provides an pipeline
approach in selecting the best model for classification. Furthermore, 
GridSearchCV helps us to further improve by tuning the hyperparameter.

On rare cases, SVC may be outperformed by DT or RF. Which with improper, 
hyperparameter tuning, may produced an error. Therefore, and extraction
hyperparameter from the class were conduct to make the program as robust as
possible. However, the program may still run into problem as limited test
cases were tested.

Overall, this classification problem recorded a success with accuracy of 0.86
using SVC.

'''






