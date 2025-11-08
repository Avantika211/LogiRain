# flood_fun_app_fixed.py
# Pretty and readable PyQt5 + SQLite rainfall demo

import sys, os, sqlite3, numpy as np, pandas as pd, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QTextEdit, QMessageBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

MODEL_FILE, ENC_FILE, DB_FILE = "flood_model.pkl", "encoder.pkl", "flood_data.db"

def train_model():
    data = {
        "Rainfall_mm": [10, 30, 60, 90, 120, 160, 200],
        "Flood_Risk": ["Low", "Low", "Medium", "Medium", "High", "High", "High"]
    }
    df = pd.DataFrame(data)
    le = LabelEncoder()
    df["Label"] = le.fit_transform(df["Flood_Risk"])
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(df[["Rainfall_mm"]], df["Label"])
    joblib.dump(model, MODEL_FILE)
    joblib.dump(le, ENC_FILE)

if not os.path.exists(MODEL_FILE):
    train_model()

model = joblib.load(MODEL_FILE)
encoder = joblib.load(ENC_FILE)

# ---- SQLite setup ----
conn = sqlite3.connect(DB_FILE)
cur = conn.cursor()
cur.execute("""CREATE TABLE IF NOT EXISTS predictions
               (id INTEGER PRIMARY KEY, rainfall REAL, risk TEXT, message TEXT,
                ts DATETIME DEFAULT CURRENT_TIMESTAMP)""")
conn.commit()

class FloodApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🌦️ Rainfall → Flood Risk Companion")
        self.setGeometry(350, 200, 550, 450)

        layout = QVBoxLayout()
        title = QLabel("Your Friendly Rainfall Advisor")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Arial", 16, QFont.Bold))

        self.label = QLabel("Enter rainfall (mm):")
        self.input = QLineEdit()
        self.input.setPlaceholderText("e.g. 120")
        self.btn = QPushButton("Predict Flood Risk")
        self.result = QLabel("")
        self.result.setWordWrap(True)
        self.result.setFont(QFont("Arial", 13))

        # Fix visibility of text box
        self.details = QTextEdit()
        self.details.setReadOnly(True)
        self.details.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #f5f5f5;
                border-radius: 8px;
                padding: 10px;
                font-size: 13px;
                line-height: 1.4em;
            }
        """)

        self.btn.setStyleSheet("""
            QPushButton {
                background-color: #0078D7;
                color: white;
                font-weight: bold;
                border-radius: 6px;
                padding: 6px;
            }
            QPushButton:hover {
                background-color: #005bb5;
            }
        """)

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
        except ValueError:
            QMessageBox.warning(self, "Error", "Please enter a valid number.")
            return

        pred_idx = model.predict(np.array([[rainfall]]))[0]
        risk = encoder.inverse_transform([pred_idx])[0]

        if risk == "Low":
            color = "green"
            msg = (
                "🌤️ **Low Flood Risk!**\n"
                "Rainfall is light — perfect for errands or an evening walk.\n\n"
                "**Things to do:**\n"
                "• Go for a short drive or cycle ride 🚲\n"
                "• Try gardening or outdoor games.\n"
                "• Chill with an iced coffee or lemonade 🧋\n"
                "\nStay carefree, but keep an umbrella just in case!"
            )
        elif risk == "Medium":
            color = "orange"
            msg = (
                "🌦️ **Moderate Rainfall.**\n"
                "Some puddles might form — stay cautious while travelling.\n\n"
                "**Safety tips:**\n"
                "• Carry an umbrella ☂️ and wear waterproof shoes.\n"
                "• Avoid parking in low-lying areas.\n"
                "• Watch the weather updates.\n\n"
                "**Fun suggestions:**\n"
                "• Make chai and pakoras ☕🍵\n"
                "• Watch a comfort movie 🎬 or listen to lo-fi rain music 🎧"
            )
        else:
            color = "red"
            msg = (
                "⛈️ **Heavy Rainfall — High Flood Risk!**\n"
                "It’s pouring cats and dogs — stay indoors and safe!\n\n"
                "**Emergency precautions:**\n"
                "• Avoid unnecessary travel; roads might flood.\n"
                "• Keep valuables and electronics above ground level.\n"
                "• Charge phones and power banks 🔋\n"
                "• Stay updated with local weather alerts ⚠️\n\n"
                "**Stay cozy ideas:**\n"
                "• Cook something warm — khichdi, soup, or pancakes 🍲🥞\n"
                "• Watch your favourite show 📺 or read a novel 📖\n"
                "• Text your friends or family and check on them ❤️"
            )

        self.result.setText(
            f"<b>Predicted Risk:</b> "
            f"<span style='color:{color}'>{risk}</span>"
        )
        self.details.setMarkdown(msg)

        # Save to DB
        try:
            cur.execute("INSERT INTO predictions (rainfall, risk, message) VALUES (?,?,?)",
                        (rainfall, risk, msg))
            conn.commit()
        except Exception as e:
            QMessageBox.warning(self, "DB error", str(e))

def main():
    app = QApplication(sys.argv)
    win = FloodApp()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
