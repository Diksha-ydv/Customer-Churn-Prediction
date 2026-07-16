## Customer Churn Prediction End to End ML Project

## 📌 Project Overview

Customer churn is one of the biggest challenges faced by subscription-based businesses. Predicting which customers are likely to leave helps companies take preventive actions, improve customer retention, and increase revenue.

This project builds an end-to-end **Machine Learning Customer Churn Prediction System** that predicts whether a telecom customer is likely to churn based on customer demographics, services, billing information, and engineered features.

Live Demo
https://customer-churn-prediction-1-3czo.onrender.com/

The project follows a complete **production-level ML pipeline** including:

- Data ingestion
- Data validation
- Data drift detection
- Data transformation
- Handling class imbalance using SMOTE
- Model training
- Model evaluation
- Threshold tuning
- Model saving
- Flask web application deployment


---

# 🚀 Features

✅ Complete ML pipeline architecture  
✅ Automated data ingestion from MongoDB  
✅ Data validation using schema checking  
✅ Data drift detection using statistical testing  
✅ Feature engineering  
✅ Handling imbalanced dataset using SMOTE  
✅ Multiple classification models comparison  
✅ F1-score based model selection  
✅ Probability threshold optimization  
✅ Flask web application  
✅ Saved trained model and preprocessing pipeline  


---

# 📂 Project Structure
Customer-Churn-Prediction/

│
├── Artifacts/
│
├── final model/
│ ├── model.pkl
│ └── preprocessor.pkl
│
├── notebooks/
│
├── src/
│ │
│ ├── components/
│ │ ├── data_ingestion.py
│ │ ├── data_validation.py
│ │ ├── data_transformation.py
│ │ └── model_trainer.py
│ │
│ ├── pipeline/
│ │ └── training_pipeline.py
│ │
│ ├── entity/
│ │
│ ├── utils.py
│ ├── logger.py
│ └── exception.py
│
├── templates/
│ └── index.html
│
├── app.py
├── main.py
├── requirements.txt
└── README.md



---

# 📁 Dataset Information

## Dataset Used

**Telco Customer Churn Dataset**

The dataset contains information about telecom customers including:

- Personal information
- Service subscriptions
- Contract details
- Payment information
- Monthly charges
- Customer churn status


## Original Features

| Feature | Description |
|---|---|
| customerID | Unique customer identifier |
| gender | Customer gender |
| SeniorCitizen | Senior citizen status |
| Partner | Whether customer has partner |
| Dependents | Whether customer has dependents |
| tenure | Number of months with company |
| PhoneService | Phone service availability |
| MultipleLines | Multiple phone lines |
| InternetService | Type of internet service |
| OnlineSecurity | Online security subscription |
| OnlineBackup | Online backup subscription |
| DeviceProtection | Device protection subscription |
| TechSupport | Technical support subscription |
| StreamingTV | Streaming TV subscription |
| StreamingMovies | Streaming movies subscription |
| Contract | Contract duration |
| PaperlessBilling | Billing method |
| PaymentMethod | Payment method |
| MonthlyCharges | Monthly charges |
| TotalCharges | Total amount charged |
| Churn | Target variable |


---

# 🔨 Feature Engineering

To improve model performance, additional features were created.


## 1. NewCustomer

Identifies recently joined customers.

Logic:
Low tenure customers → NewCustomer = Yes

---

## 2. LongContract

Identifies customers having long-term contracts.

Logic:
One year / Two year contract → Yes

---

## 3. FiberOptic

Identifies customers using fiber optic internet.

Logic:
InternetService = Fiber optic → Yes

---

## 4. AutoPayment

Identifies customers using automatic payment methods.

Logic:
Automatic payment methods → Yes

---

## 5. HighMonthlyCharges

Identifies customers with higher monthly charges.

Logic:
MonthlyCharges above threshold → Yes


---

# ⚙️ Machine Learning Pipeline


## 1. Data Ingestion

- Data is imported from MongoDB
- Dataset is divided into training and testing data
- Train and test files are stored as artifacts


---

## 2. Data Validation

Performed validations:

- Schema validation
- Column validation
- Data drift detection


Tools used:

- KS Test


---

## 3. Data Transformation


### Numerical Features

Applied:

- Missing value handling
- Feature scaling


### Categorical Features

Applied:

- One Hot Encoding


### Handling Imbalanced Dataset

Used:
SMOTE to balance churn and non-churn classes.

---

# 🤖 Machine Learning Models Used


The following classification algorithms were implemented:


| Model |
|---|
| Logistic Regression |
| Decision Tree |
| Random Forest |
| Gradient Boosting |
| AdaBoost |
| KNN |
| XGBoost |
| CatBoost |


---

# 📊 Model Evaluation


Models were evaluated using:


## Metrics Used

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score


Because churn datasets are usually imbalanced, **F1 Score** was selected as the primary evaluation metric.


---

# 🎯 Threshold Optimization


Normally classification models use:threshold = 0.5

In this project, probability threshold tuning was performed to improve churn detection.

Process:
Model Probability Output

    ↓

Test different thresholds

    ↓

Select threshold giving maximum F1 Score

    ↓

Final Prediction


Example:


Probability >= tuned threshold

    ↓

Customer will churn



---

# 🌐 Flask Web Application


A Flask-based web interface was created for real-time prediction.


## Workflow



User Input

↓

DataFrame Creation

↓

Preprocessing Pipeline

↓

Machine Learning Model

↓

Threshold Based Prediction

↓

Churn Result



---

# 🛠 Installation


## Clone Repository

```bash
git clone https://github.com/Diksha-ydv/Customer-Churn-Prediction
Navigate to Project Directory
cd Customer-Churn-Prediction
Create Virtual Environment
python -m venv venv
Activate Environment

Mac/Linux:

source venv/bin/activate
Install Dependencies
pip install -r requirements.txt
▶️ Running the Project
Step 1: Train Model

Run:

python main.py

This will execute:

Data ingestion
Validation
Transformation
Model training
Model saving
Step 2: Start Flask Application

Run:

python app.py

Open browser:

http://127.0.0.1:5001/
🧰 Technologies Used
Programming Language
Python
Data Processing
Pandas
NumPy
Machine Learning
Scikit-learn
XGBoost
CatBoost
Imbalanced-learn
Database
MongoDB
Deployment
Flask
Model Serialization
Pickle
📈 Results

The best performing model was selected based on:

Highest F1 Score
Balanced precision and recall
Improved churn detection capability

Threshold tuning further improved the model's ability to identify churn customers.

🔮 Future Improvements
Deploy application on AWS/Azure
Add Docker support
Add MLflow experiment tracking
Add CI/CD pipeline
Add model monitoring
Add automated retraining pipeline
👩‍💻 Author

Diksha Yadav

GitHub:

https://github.com/Diksha-ydv
⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub.








