Titanic Survival Prediction: A Machine Learning Approach
Project Overview
This repository contains a comprehensive data science workflow aimed at predicting passenger survival on the Titanic. The project covers the entire machine learning pipeline, from Exploratory Data Analysis (EDA) and feature engineering to model building and ensemble evaluation. It serves as a practical demonstration of handling categorical and continuous data, extracting hidden features, and optimizing classification models.

Tech Stack & Tools
Language: Python

Data Manipulation: Pandas, NumPy

Data Visualization: Seaborn, Matplotlib

Machine Learning: Scikit-learn, XGBoost

Methodology
The project is structured into several key phases:

Exploratory Data Analysis (EDA): Analyzed the distribution of numeric variables (Age, SibSp, Parch, Fare) and visualized survival rates across categorical variables like Ticket, Sex, and Embarked locations using correlation matrices and pivot tables.

Feature Engineering: Created new, highly predictive features from unstructured data. This included extracting passenger titles (Mr., Mrs., Master, etc.) from names, isolating cabin letters to understand deck placements, and creating a cabin_multiple feature to count the number of cabins per passenger.

Data Preprocessing: Imputed missing values using median strategies for continuous data and dropped nulls where appropriate. Applied log normalization to skewed features like Fare. Transformed categorical variables into numerical formats using dummy variables (One-Hot Encoding) and scaled the features using standard scaling.

Model Selection & Evaluation: Evaluated multiple classification algorithms using 5-fold cross-validation to establish performance baselines:

Naive Bayes

Logistic Regression

Decision Tree

K-Nearest Neighbors (KNN)

Random Forest

Support Vector Classifier (SVC)

XGBoost

Ensemble Modeling: Built a Soft Voting Classifier combining Logistic Regression, KNN, Random Forest, Naive Bayes, SVC, and XGBoost to average model confidence and generate highly robust final predictions.

Results
The models were evaluated based on their cross-validation accuracy scores. The Support Vector Classifier (SVC) achieved the highest individual average accuracy at ~82.9%, while the final Soft Voting Ensemble Classifier yielded a highly stable and competitive accuracy of ~82.5%.
