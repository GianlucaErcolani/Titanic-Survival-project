# Titanic Survival Prediction project

In questo progetto, l'obiettivo è addestrare un modello di machine learning per prevedere se un passeggero sopravvivrà o meno al tragico naufragio del Titanic, partendo da un'analisi esplorativa dei dati (EDA) fino all'estrazione di feature nascoste e all'ottimizzazione dei modelli di classificazione.

### Project Structure

* `Titanic project.ipynb`: notebook contenente l'intero workflow, suddiviso in:
    * **Analisi esplorativa del dataset (EDA)**: distribuzione delle variabili e tassi di sopravvivenza.
    * **Feature Engineering e Preprocessing**: estrazione dei titoli dai nomi (Mr., Mrs., Master), gestione dei valori mancanti, log normalization, ed encoding delle variabili categoriche.
    * **Applicazione dei modelli predittivi**: addestramento e validazione (5-fold cross-validation) di modelli base (Naive Bayes, Logistic Regression, Decision Tree, KNN, Random Forest, SVC, XGBoost) e creazione di un Soft Voting Ensemble Classifier finale.
* `requirements.txt`: librerie python che devono essere installate (Pandas, NumPy, Seaborn, Matplotlib, Scikit-learn, XGBoost)

### Dataset

**Input variables:**
* **passenger data:**
    * `Age` (numeric): età del passeggero in anni
    * `Sex` (categorical: "male", "female"): sesso del passeggero
    * `SibSp` (numeric): numero di fratelli, sorelle o coniugi a bordo
    * `Parch` (numeric): numero di genitori o figli a bordo
    * `Fare` (numeric): tariffa pagata per il biglietto
    * `Ticket` (categorical/text): numero del biglietto
    * `Cabin` (categorical/text): numero della cabina (utilizzato per comprendere il posizionamento sui ponti e il numero di cabine per passeggero)
    * `Embarked` (categorical: "C", "Q", "S"): porto di imbarco
    * `Name` (text): nome del passeggero (utilizzato per estrarre i titoli sociali)

**Output variable (desired target):**
* `Survived` - il passeggero è sopravvissuto? (binary: 1 = "yes", 0 = "no")

### Results
I modelli sono stati valutati in base ai punteggi di accuratezza della cross-validation:
* Il **Support Vector Classifier (SVC)** ha ottenuto l'accuratezza media individuale più alta (**~82.9%**).
* Il **Soft Voting Ensemble Classifier** finale ha restituito un'accuratezza altamente stabile e competitiva del **~82.5%**.
