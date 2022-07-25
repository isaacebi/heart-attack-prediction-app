# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 12:41:17 2022

@author: isaac
"""

#%% import

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score

#%% PATH
CSV_TEST_PATH = os.path.join(os.getcwd(), 'datasets', 'heart_test.csv')
MODEL_PATH = os.path.join(os.getcwd(), 'models', 'BEST_MODEL.pkl')

#%% Model loading
with open(MODEL_PATH, 'rb') as file:
    clf_model = pickle.load(file)
    
#%% DataFrame for display
# our selected features
sel_features = ['cp', 'thall', 'age', 'trtbps', 'chol', 'thalachh', 'oldpeak']

# reading test datasets
df_test = pd.read_csv(CSV_TEST_PATH, delim_whitespace=True)

# split test set into X and y
X_test = df_test[sel_features]
y_test = df_test['True_output']

# prediction using pickled model
y_pred = pd.DataFrame(clf_model.predict(X_test), columns=['Predict_output'])

# accuracy for the model
score = accuracy_score(y_test, y_pred, normalize=True)

# concatenated into a new dataframe
df_show = pd.concat([X_test, y_test, y_pred], axis=1)

#%%
# to create an empty class for collecting data
class structtype():
    pass

#%%
# page config
st.set_page_config(
     page_title="Heart Attack Prediction App",
     page_icon=":hospital:",
     layout="wide",
 )

# create 3 column section
col1, col2, col3 = st.columns(3)
with col1:
    st.container()

with col2:
    st.title(':heartpulse: Heart Attack Prediction App :heartpulse:')
    
with col3:
    st.container()
    
# create 3 column section
col1, col2, col3 = st.columns(3)
placeholder = st.empty()

# info for the our problem
with col1:
    st.markdown("<h1 style='text-align: left; color: black;'>Awareness on heart attack</h1>", unsafe_allow_html=True)
    st.image('data:image/jpeg;base64,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')
    st.markdown('<div style="text-align: justify">A heart attack occurs when\
                the flow of blood to the heart is severely reduced or blocked.\
                The blockage is usually due to a buildup of fat, cholesterol\
                and other substances in the heart (coronary) arteries.\
                The fatty, cholesterol-containing deposits are called plaques. \
                The process of plaque buildup is called atherosclerosis.\
                Sometimes, a plaque can rupture and form a clot that blocks\
                blood flow. A lack of blood flow can damage or destroy part\
                of the heart muscle. A heart attack is also called a \
                myocardial infarction. Prompt treatment is needed for a heart\
                attack to prevent death. Call 911 or emergency medical help\
                if you think you might be having a heart attack.</div>',
                unsafe_allow_html=True)
    
# for user input
with col2:
    placeholder = st.empty()
    with placeholder.container():
        st.markdown("<h1 style='text-align: center; color: black;'>Prediction</h1>", unsafe_allow_html=True)
        # inputs
        age_option = st.slider('Age', min_value=0, max_value=100, step=1, value=20)
        st.markdown("<strong style='text-align: center; color: red;'>More detailed input resulting accurate results</strong>", unsafe_allow_html=True)
        extra = st.expander("Detailed Input")
        oldpeak_option = extra.number_input('ST depression induced by exercise relative to rest, oldpeak', min_value=0.0, max_value=6.2, step=0.1, value=1.0)
        thalach_option = extra.number_input('Maximum heart rate achieved, thalanch', min_value=0, value=150)
        chol_option = extra.number_input('Cholestoral in mg/dl fetched via BMI sensor, chol', min_value=0, value=246)
        trtbps_option = extra.number_input('Resting blood pressure (in mm Hg), trtbps', min_value=0, value=131)

        cp_display = ['Level 0: typical angina',
                      'Level 1: atypical angina',
                      'Level 2: non-anginal pain',
                      'Level 3: asymptomatic']
        cp_value = list(range(len(cp_display)))
        cp_option = extra.selectbox('Chest Pain, cp',
                                    cp_value,
                                    format_func=lambda x: cp_display[x])
        
        thall_display = ['0: Null',
                         '1: Fixed Defect',
                         '2: Normal',
                         '3: Reversable defect']
        thall_value = list(range(len(thall_display)))
        thall_option = extra.selectbox('Thalassemia',
                                       thall_value,
                                       format_func=lambda x: thall_display[x])
        
        # class as data collector
        X = structtype()
        X.cp = cp_option
        X.thall = thall_option
        X.age = age_option
        X.trtbps = trtbps_option
        X.chol = chol_option
        X.thalachh = thalach_option
        X.oldpeak = oldpeak_option
        
        # button for interaction with user
        button = st.button('CLICK ME')
        if button:
            st.write("Results will be shown below")
        else:
            st.write("Click the button after all info is filled")
            
# prevention for our problem info
with col3:
    st.markdown("<h1 style='text-align: left; color: black;'>Lifestyle Changes for Heart Attack Prevention</h1>", unsafe_allow_html=True)
    st.image('https://cdn.dribbble.com/users/609665/screenshots/2992885/heart-dd.gif')
    st.markdown("<h5 style='text-align: left; color: black;'>Stop smoking</h5>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: left; color: black;'>Choose good nutrition</h5>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: left; color: black;'>Be physically active every day</h5>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: left; color: black;'>Limit alcohol</h5>", unsafe_allow_html=True)

# create 3 column section
col1, col2, col3 = st.columns(3)
if button:
    with col1:
        # extract user input to list
        list_input = [X.cp, X.thall, X.age, X.trtbps, X.chol, X.thalachh, X.oldpeak]
        
        # list into dataframe
        _ = pd.DataFrame(np.array(list_input).reshape(1,-1), columns=X_test.columns)
        
        # model prediction
        output = clf_model.predict(np.array(list_input).reshape(1,-1))
        y_pred = pd.DataFrame(output, columns=['Predict_output'])
        
        # concatenate the dataframe - future development for database
        df = pd.concat([_, y_pred], axis=1)
        
        st.write("Results for validation against test data provided - Not related to your test -")
        st.dataframe(df_show)
        st.write(f'The accuracy score for this model is {score*100} %')
            
    # our main results gif
    with col2:
        st.markdown("<h1 style='text-align: center; color: black;'>Your Results</h1>", unsafe_allow_html=True)
        if output == 0:
            st.markdown("<mark style='text-align: center; color: black;'>Good Job Bro! Keep Pi Dap!</mark>", unsafe_allow_html=True)
            st.image('https://c.tenor.com/1ZgeHDHjIQ0AAAAM/healthier-healthy.gif', use_column_width=True)
            
        else:
            st.markdown("<mark style='text-align: center; color: red;'>Hmmmm~ I think you may need to think of your life choices now</mark>", unsafe_allow_html=True)
            st.image('https://i.makeagif.com/media/9-19-2015/oDlTGt.gif', use_column_width=True)
                
    # our side results gif
    with col3:
        if output == 0:
            st.markdown("<h3 style='text-align: center; color: black;'>Good Job Bro! Keep Pi Dap!</h3>", unsafe_allow_html=True)
            st.image('https://i.pinimg.com/originals/e7/95/d3/e795d3bfaa35b8843bf27b83e65a111d.gif', use_column_width=True)
            
        else:
            st.markdown("<mark style='text-align: center; color: blue;'>This might seem to be a good idea now</mark>", unsafe_allow_html=True)
            st.markdown("[![Foo](https://ringgitplus.com/assets/cache_070721/issuers/logo_ALLZ.png)](https://ringgitplus.com/en/health-insurance/critical-illness/Allianz-Prime-Care-Plus.html?filter=Allianz)")
            st.markdown("[![Foo](https://ringgitplus.com/assets/cache_070721/issuers/logo_FWDT.png)](https://ringgitplus.com/en/health-insurance/critical-illness/FWD-Care-Direct.html?filter=FWD%20Takaful)")
            st.markdown("[![Foo](https://ringgitplus.com/assets/cache_070721/issuers/logo_ALLZ.png)](https://ringgitplus.com/en/health-insurance/critical-illness/Gibraltar-BSN-CI-Intense-Shield.html?filter=Gibraltar%20BSN)")

        
