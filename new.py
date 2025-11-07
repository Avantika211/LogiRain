# flood_demo.py
# One-file demo: train + predict Flood-Risk from rainfall

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

# -----------------------
# 1.  Mini dataset
# -----------------------
data = {
    "Rainfall_mm": [10, 25, 40, 55, 70, 90, 110, 130, 150, 180, 200, 220],
    "Flood_Risk": [
        "Low", "Low", "Low",
        "Medium", "Medium", "Medium",
        "High", "High", "High", "High", "High", "High"
    ]
}
df = pd.DataFrame(data)

# -----------------------
# 2.  Preprocess + train
# -----------------------
label_encoder = LabelEncoder()
df["Flood_Risk_Label"] = label_encoder.fit_transform(df["Flood_Risk"])

X = df[["Rainfall_mm"]]
y = df["Flood_Risk_Label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model trained successfully (Accuracy: {acc*100:.1f}%)")

# -----------------------
# 3.  Save model (optional)
# -----------------------
joblib.dump(model, "flood_risk_model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

# -----------------------
# 4.  Interactive prediction
# -----------------------
print("\n🌧️ Flood-Risk Prediction Demo 🌧️")
rain = float(input("Enter rainfall amount in mm: "))

risk_class = model.predict(np.array([[rain]]))[0]
risk_label = label_encoder.inverse_transform([risk_class])[0]
print(f"Predicted Flood Risk: {risk_label}")

print("\n(You can re-run this file anytime to test again.)")
