import joblib
import numpy as np

model = joblib.load('flood_risk_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

rainfall = float(input("Enter rainfall in mm: "))
risk_class = model.predict(np.array([[rainfall]]))[0]
risk_label = label_encoder.inverse_transform([risk_class])[0]
print(f"Predicted Flood Risk: {risk_label}")
