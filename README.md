# :hospital: Heart Attack Prediction App

![Spyder](https://img.shields.io/badge/Spyder-838485?style=for-the-badge&logo=spyder%20ide&logoColor=maroon)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)

Analysis and developed machine learning prediction whether user has higher probability in developing heart attack.

## Description

A Data Science heart attack classification using Machine Learning and Pipelines via Scikit-Learn.

## Getting Started

### Dependencies

* Matplotlib version 3.5*
* Numpy version 1.22*
* Pandas version 1.4*
* Python version 3.8*
* Scikit-learn version 1.0*
* Scipy version 1.7*
* Seaborn 0.11*

### Datasets

* Datasets : [Heart Attack Analysis & Prediction Dataset](https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset)

### Executing model_training.py script

* Code can be clone and run directly using any IDE with the environment listed

* The script was divided into several categories such as:
   1. Data Loading
   2. Data Inspection & Visualization
   3. Data Cleaning
   4. Feature Selection
   5. Data Preprocessing
   6. Model Development
   7. Model Analysis

### Executing app.py script

* Default page

![alt text](static/app_interface/default_layout.png)

* Higher Risk Of Heart Attack

![alt text](static/app_interface/outcome_1.png)

* Lower Risk Of Heart Attack

![alt text](static/app_interface/outcome_0.png)

### Results

* Models score
  - Interestingly, SVC performed the best for this problem

![alt text](static/app_interface/scaler_models_score.png)

* The classification results are as follows:

![alt text](static/app_interface/classification_report.png)

### Why is this possible?

* Lets take a step back and talk on what contribute in the success of this problem. As stated in the 'model_training.py' script, the main success of this project would be the balanced proportion between having a higher risk of heart attack and low risk of heart attack. Furthermore, the data is in the highest quality which we can directly use without affecting the model performances. Finally, kudos to [Rashik Rahman](https://www.kaggle.com/rashikrahmanpritom) for providing such a high quality datasets with detailed explanation.

### Future Works

* Despite the success of this project, there a still minor as well as major improvement can be done. Features engineering is one of them, by reconstruct or creating new features using the unselected features, we may or may not increase the model performances. Therefore, that would be our task in the near future.

* Another things is related to the Streamlit app, althought the app are able to process and formulate either user is having higher risk for getting heart attack or not, there are no database structure on the backend of the app. Therefore, the user input is not recorded hence wasting the resources. Henceforth, this will also be our goal in the future

## Acknowledgments

I would like to express my gratitude [Rashik Rahman](https://www.kaggle.com/rashikrahmanpritom) for providing a high quality datasets as well as [Kaggle](https://www.kaggle.com) which provides an excellent platform for data scientist to improve.

PEACE OUT :love_you_gesture:
