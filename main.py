import sys
import pandas as pd
from sklearn.linear_model import Ridge
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QFormLayout, 
                             QPushButton, QLabel, QSpinBox, QComboBox)

class ScryMetricsPoC(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ScryMetrics - R&D Proof of Concept")
        self.resize(400, 200)
        
        # Main Layout
        self.main_layout = QVBoxLayout()
        
        # User Input Form
        self.form_layout = QFormLayout()
        
        self.mana_input = QSpinBox()
        self.mana_input.setRange(0, 16)
        self.mana_input.setValue(4)
        
        self.rarity_input = QComboBox()
        self.rarity_input.addItems(["Common/Uncommon (0)", "Rare/Mythic (1)"])
        
        self.form_layout.addRow("Mana Value:", self.mana_input)
        self.form_layout.addRow("Rarity Level:", self.rarity_input)
        self.main_layout.addLayout(self.form_layout)
        
        # Action Button
        self.predict_btn = QPushButton("Train Model & Predict Price")
        self.predict_btn.setStyleSheet("padding: 10px; font-weight: bold;")
        self.predict_btn.clicked.connect(self.run_ml_pipeline)
        self.main_layout.addWidget(self.predict_btn)
        
        # Output Display
        self.info_label = QLabel("Adjust stats and click predict...")
        self.info_label.setStyleSheet("color: #2c3e50; font-size: 14px; font-weight: bold;")
        self.main_layout.addWidget(self.info_label)
        
        self.setLayout(self.main_layout)

    def run_ml_pipeline(self):
        # 1. Pandas: Load dummy MTG data (Mana Value, Rarity -> Price)
        data = {
            'mana_value': [1, 2, 3, 4, 5, 6, 7, 8],
            'is_rare':    [0, 0, 1, 0, 1, 1, 1, 1],
            'price':      [0.50, 1.25, 15.00, 2.50, 25.00, 18.00, 30.00, 22.00]
        }
        df = pd.DataFrame(data)
        X = df[['mana_value', 'is_rare']]
        y = df['price']
        
        # 2. Scikit-Learn: Train a basic Ridge Regression model
        model = Ridge()
        model.fit(X, y)
        
        # 3. Get user input from the PyQt6 UI
        user_mana = self.mana_input.value()
        user_rarity = 1 if "Rare/Mythic" in self.rarity_input.currentText() else 0
        
        # 4. Predict the price based on user input
        test_card = pd.DataFrame({'mana_value': [user_mana], 'is_rare': [user_rarity]})
        prediction = model.predict(test_card)[0]
        
        # 5. Update UI with result
        self.info_label.setText(f"Model Trained Successfully!\nPredicted Price: ${abs(prediction):.2f}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ScryMetricsPoC()
    window.show()
    sys.exit(app.exec())