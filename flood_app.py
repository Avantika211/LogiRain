import sys, sqlite3, os
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel,
    QLineEdit, QPushButton, QMessageBox, QTextEdit
)
from PyQt5.QtGui import QFont, QColor
from PyQt5.QtCore import Qt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# ---- Train a quick model if not already saved ----
def train_model():
    data = {
        "Rainfall_mm": [10, 25, 40, 55, 70, 90, 110, 130, 150, 180, 200, 220],
        "Flood_Risk": ["Low", "Low", "Low",
                       "Medium", "Medium", "Medium",
                       "High", "High", "High", "High", "High", "High"]
    }
    df = pd.DataFrame(data)
    le = LabelEncoder()
    df["Label"] = le.fit_transform(df["Flood_Risk"])
    X, y = df[["Rainfall_mm"]], df["Label"]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, "flood_model.pkl")
    joblib.dump(le, "encoder.pkl")

if not (os.path.exists("flood_model.pkl") and os.path.exists("encoder.pkl")):
    train_model()

model = joblib.load("flood_model.pkl")
encoder = joblib.load("encoder.pkl")

# ---- SQLite setup ----
conn = sqlite3.connect("flood_data.db")
cur = conn.cursor()
cur.execute("""
CREATE TABLE IF NOT EXISTS predictions(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rainfall REAL,
    risk TEXT,
    message TEXT
)
""")
conn.commit()

# ---- PyQt5 Interface ----
class FloodApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🌧️ Rainfall → Flood Risk Classifier")
        self.setGeometry(300, 200, 450, 350)

        layout = QVBoxLayout()
        title = QLabel("Rainfall-Based Flood Risk Predictor")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)

        self.label = QLabel("Enter Rainfall (mm):")
        self.input = QLineEdit()
        self.input.setPlaceholderText("e.g. 120")
        self.btn = QPushButton("Predict Flood Risk")
        self.btn.setStyleSheet("background-color:#0078D7;color:white;font-size:14px;padding:6px;")
        self.result = QLabel("")
        self.result.setFont(QFont("Arial", 14))
        self.details = QTextEdit()
        self.details.setReadOnly(True)
        self.details.setStyleSheet("background:#f0f0f0;")

        self.btn.clicked.connect(self.predict)

        layout.addWidget(title)
        layout.addWidget(self.label)
        layout.addWidget(self.input)
        layout.addWidget(self.btn)
        layout.addWidget(self.result)
        layout.addWidget(self.details)
        self.setLayout(layout)

    def predict(self):
        try:
            rainfall = float(self.input.text())
            pred = model.predict(np.array([[rainfall]]))[0]
            risk = encoder.inverse_transform([pred])[0]

            # Add richer information
            if risk == "Low":
                color = "green"
                msg = ("Rainfall is within safe limits.\n"
                       "No immediate flood risk detected.\n"
                       "However, stay alert during extended rainy periods.")
            elif risk == "Medium":
                color = "orange"
                msg = ("Rainfall levels are moderate.\n"
                       "Minor waterlogging may occur in low-lying areas.\n"
                       "Keep drains clear and monitor forecasts.")
            else:
                color = "red"
                msg = ("Heavy rainfall detected!\n"
                       "High flood risk — avoid low areas, secure valuables,\n"
                       "and follow local disaster-management alerts.")

            self.result.setText(f"<b>Predicted Risk: "
                                f"<span style='color:{color}'>{risk}</span></b>")
            self.details.setText(msg)

            cur.execute("INSERT INTO predictions (rainfall, risk, message) VALUES (?,?,?)",
                        (rainfall, risk, msg))
            conn.commit()

        except ValueError:
            QMessageBox.warning(self, "Error", "Please enter a valid number.")

# ---- Run ----
app = QApplication(sys.argv)
window = FloodApp()
window.show()
sys.exit(app.exec_())
