# flood_app_fixed.py
# Minimal and syntactically-safe PyQt5 + SQLite demo

import sys
import os
import sqlite3
import joblib
import numpy as np
import pandas as pd

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox
)
from PyQt5.QtCore import Qt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# -------------------------
# Train tiny model if missing
# -------------------------
MODEL_FILE = "flood_model.pkl"
ENC_FILE = "encoder.pkl"
DB_FILE = "flood_data.db"

def train_and_save_model():
    data = {
        "Rainfall_mm": [10, 30, 60, 90, 120, 160, 200],
        "Flood_Risk": ["Low", "Low", "Medium", "Medium", "High", "High", "High"]
    }
    df = pd.DataFrame(data)
    le = LabelEncoder()
    df["Label"] = le.fit_transform(df["Flood_Risk"])

    X = df[["Rainfall_mm"]]
    y = df["Label"]

    model = RandomForestClassifier(n_estimators=50, random_state=1)
    model.fit(X, y)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(le, ENC_FILE)

if not (os.path.exists(MODEL_FILE) and os.path.exists(ENC_FILE)):
    train_and_save_model()

# load model + encoder
model = joblib.load(MODEL_FILE)
encoder = joblib.load(ENC_FILE)

# -------------------------
# SQLite setup
# -------------------------
conn = sqlite3.connect(DB_FILE)
cur = conn.cursor()
cur.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rainfall REAL NOT NULL,
    risk TEXT NOT NULL,
    ts DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")
conn.commit()

# -------------------------
# PyQt5 GUI
# -------------------------
class FloodApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Flood Risk Demo")
        self.setGeometry(300, 300, 380, 180)

        layout = QVBoxLayout()

        self.title = QLabel("Rainfall → Flood Risk")
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setStyleSheet("font-size:16px; font-weight:bold;")

        self.label = QLabel("Enter rainfall (mm):")
        self.input = QLineEdit()
        self.input.setPlaceholderText("e.g. 120")

        self.button = QPushButton("Predict")
        self.button.clicked.connect(self.on_predict)

        self.result = QLabel("")
        self.result.setWordWrap(True)
        self.result.setStyleSheet("font-size:13px;")

        layout.addWidget(self.title)
        layout.addWidget(self.label)
        layout.addWidget(self.input)
        layout.addWidget(self.button)
        layout.addWidget(self.result)

        self.setLayout(layout)

    def on_predict(self):
        text = self.input.text().strip()
        if text == "":
            QMessageBox.warning(self, "Input required", "Please enter a rainfall value (in mm).")
            return

        try:
            rainfall = float(text)
        except ValueError:
            QMessageBox.warning(self, "Invalid input", "Please enter a valid number (e.g. 120).")
            return

        # predict
        pred_idx = model.predict(np.array([[rainfall]]))[0]
        risk = encoder.inverse_transform([pred_idx])[0]

        # richer message
        if risk == "Low":
            info = "Low risk — rainfall within safe range. Minimal chance of flooding."
            color = "green"
        elif risk == "Medium":
            info = "Medium risk — possible waterlogging in low-lying areas. Stay alert."
            color = "orange"
        else:
            info = "High risk — heavy rainfall. Follow local advisories and avoid low areas."
            color = "red"

        self.result.setText(f"<b>Predicted Risk:</b> <span style='color:{color}; font-weight:bold;'>{risk}</span>\n\n{info}")

        # save to DB
        try:
            cur.execute("INSERT INTO predictions (rainfall, risk) VALUES (?, ?)", (rainfall, risk))
            conn.commit()
        except Exception as e:
            # don't crash GUI on DB error; show message
            QMessageBox.warning(self, "DB error", f"Failed to save prediction: {e}")

# -------------------------
# Run application
# -------------------------
def main():
    app = QApplication(sys.argv)
    window = FloodApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
