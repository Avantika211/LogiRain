import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
df = pd.read_csv("rainfall_data.csv")

# Encode labels (Low=0, Medium=1, High=2)
label_encoder = LabelEncoder()
df['Flood_Risk_Label'] = label_encoder.fit_transform(df['Flood_Risk'])

# Clean up extra spaces or weird entries
df['Flood_Risk'] = df['Flood_Risk'].str.strip()
df = df[df['Flood_Risk'].isin(['Low', 'Medium', 'High'])]

# Split data
X = df[['Rainfall_mm']]
y = df['Flood_Risk_Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Save model
joblib.dump(model, 'flood_risk_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
print("✅ Model saved successfully!")
