# Titanic Survival Prediction project

In this project, the goal is to train a machine learning model to predict whether a passenger would survive the tragic sinking of the Titanic. The workflow ranges from Exploratory Data Analysis (EDA) to hidden feature extraction and the optimization of classification models.

### Project Structure

* `Titanic project.ipynb`: A comprehensive notebook containing the entire workflow, divided into:
    * **Exploratory Data Analysis (EDA)**: Analysis of variable distributions and survival rates.
    * **Feature Engineering and Preprocessing**: Extraction of titles from names (Mr., Mrs., Master), missing value imputation, log normalization, and categorical variable encoding.
    * **Machine Learning Modeling**: Training and validation (5-fold cross-validation) of baseline models (Naive Bayes, Logistic Regression, Decision Tree, KNN, Random Forest, SVC, XGBoost) and the creation of a final **Soft Voting Ensemble Classifier**.
* `requirements.txt`: List of Python libraries required for the project (Pandas, NumPy, Seaborn, Matplotlib, Scikit-learn, XGBoost).

### Dataset

**Input variables:**
* **Passenger data:**
    * `Age` (numeric): Passenger age in years.
    * `Sex` (categorical): "male", "female".
    * `SibSp` (numeric): Number of siblings, spouses aboard.
    * `Parch` (numeric): Number of parents, children aboard.
    * `Fare` (numeric): Ticket price.
    * `Ticket` (categorical/text): Ticket number.
    * `Cabin` (categorical/text): Cabin number (used to derive deck position and cabin count).
    * `Embarked` (categorical): Port of embarkation ("C", "Q", "S").
    * `Name` (text): Passenger name (used for title extraction).

**Output variable (Target):**
* `Survived`: Whether the passenger survived (binary: 1 = "yes", 0 = "no").

### Results
Models were evaluated based on their cross-validation accuracy scores:
* The **Support Vector Classifier (SVC)** achieved the highest individual average accuracy (~82.9%).
* The final **Soft Voting Ensemble Classifier** yielded a stable and competitive accuracy of **~82.5%**.
