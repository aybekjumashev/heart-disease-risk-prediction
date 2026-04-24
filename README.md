# 🫀 Heart Disease Risk Prediction

## 📌 Project Overview
This project applies Machine Learning to healthcare analytics, specifically aiming to predict the risk of heart disease based on tabular patient data. Using a classification model, the app can help medical professionals quickly assess whether a patient has a high or low risk of heart disease.

## 🎯 Learning Goals & Features
- **Binary Classification:** Predicting disease vs. no-disease.
- **Model Comparison:** Evaluated Logistic Regression, KNN, SVM, and Random Forest.
- **Model Explainability:** Implemented **SHAP** (SHapley Additive exPlanations) to identify which medical features (e.g., Chest Pain type, Thal, Calcium) contribute most to the model's predictions.
- **Interactive UI:** A simple, easy-to-use **Streamlit** web application for real-time risk prediction.
- **Medical Ethics:** Emphasized the use of public, de-identified data to preserve patient privacy and anonymity.

## 📊 Dataset
The project uses the **UCI Heart Disease Data Set** (processed Cleveland database). 
- **Missing Data:** Handled via median imputation.
- **Scaling:** Applied `StandardScaler` for uniformly scaled features.
- *Note: Some medical features are highly correlated, which was analyzed during the Exploratory Data Analysis (EDA) phase.*

## 🏆 Model Performance
**Random Forest Classifier** outperformed the other models:
- **Accuracy:** ~89%
- **Recall (Sensitivity):** 0.96 (Crucial in medical scenarios to avoid false negatives)
- **AUC (Area Under the ROC Curve):** 0.951

## 🛠️ Tech Stack
- **Language:** Python
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning:** scikit-learn
- **Explainability:** SHAP
- **Data Visualization:** Matplotlib, Seaborn
- **Web App deployment:** Streamlit

## 🚀 How to Run the App Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/aybekjumashev/heart-disease-risk-prediction.git
   cd heart-disease-risk-prediction
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app:**
   ```bash
   cd app
   streamlit run app.py
   ```