import streamlit as st
import pandas as pd
import joblib

# Load the trained model and scaler
model = joblib.load('app/rf_model.pkl')
scaler = joblib.load('app/scaler.pkl')

st.set_page_config(page_title="Heart Disease Predictor", layout="wide")
st.title("🫀 Heart Disease Risk Prediction App")
st.write("Enter patient data below to predict the risk of heart disease.")

# Create 3 columns for better UI layout
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=20, max_value=100, value=50)
    sex = st.selectbox("Sex (1=Male, 0=Female)", [1, 0])
    cp = st.selectbox("Chest Pain Type (1-4)", [1, 2, 3, 4])
    trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
    chol = st.number_input("Serum Cholestoral (mg/dl)", min_value=100, max_value=600, value=200)

with col2:
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1=True, 0=False)", [1, 0])
    restecg = st.selectbox("Resting ECG Results (0, 1, 2)",[0, 1, 2])
    thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
    exang = st.selectbox("Exercise Induced Angina (1=Yes, 0=No)", [1, 0])
    oldpeak = st.number_input("ST depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

with col3:
    slope = st.selectbox("Slope of the peak exercise ST segment (1, 2, 3)", [1, 2, 3])
    ca = st.selectbox("Number of major vessels (0-3)", [0.0, 1.0, 2.0, 3.0])
    thal = st.selectbox("Thal (3=normal, 6=fixed defect, 7=reversable defect)", [3.0, 6.0, 7.0])

# Prediction action
st.markdown("---")
if st.button("Predict Risk"):
    # Create a DataFrame from the inputs
    input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]], 
                              columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                                       'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])
    
    # Scale the input data using the saved scaler
    scaled_data = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(scaled_data)[0]
    probability = model.predict_proba(scaled_data)[0][1]

    # Output the results
    st.subheader("Prediction Result:")
    if prediction == 1:
        st.error(f"🚨 High Risk of Heart Disease! (Probability: {probability:.2f})")
        st.write("Based on the data provided, this patient has a high risk of heart disease. A thorough medical examination is recommended.")
    else:
        st.success(f"✅ Low Risk of Heart Disease. (Probability: {probability:.2f})")
        st.write("Based on the data provided, the risk of heart disease is low.")