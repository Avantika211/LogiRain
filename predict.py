import joblib
import numpy as np

# Load trained model and encoder
model = joblib.load('flood_risk_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

print("🌧️ Flood Risk Prediction System 🌧️")
print("-----------------------------------")

# Ask user for rainfall input
rainfall = float(input("Enter rainfall amount in mm: "))

# Predict
risk_class = model.predict(np.array([[rainfall]]))[0]
risk_label = label_encoder.inverse_transform([risk_class])[0]

print(f"\nPredicted Flood Risk Level: {risk_label}")
print("-----------------------------------")
