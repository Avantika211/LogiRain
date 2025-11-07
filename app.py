import streamlit as st
import joblib
import numpy as np

# Load saved model and label encoder
model = joblib.load('flood_risk_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

st.title("🌧️ Rainfall → Flood Risk Classifier")
st.write("Predict flood risk level based on rainfall amount (in mm).")

rainfall_input = st.number_input("Enter rainfall (mm):", min_value=0.0, max_value=500.0, step=1.0)

if st.button("Predict Flood Risk"):
    risk_class = model.predict(np.array([[rainfall_input]]))[0]
    risk_label = label_encoder.inverse_transform([risk_class])[0]
    st.success(f"Predicted Flood Risk: **{risk_label}**")
