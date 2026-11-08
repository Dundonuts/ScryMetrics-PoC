"""
Course: Project and Portfolio IV (PP4)
Student: Alec Duncan
Project: ScryMetrics - Day 1 Engine Expansion
"""

import sys
import time
import json
import urllib.request
import urllib.parse
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QComboBox,
    QPushButton,
    QFrame,
    QGridLayout,
)
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


class ScryfallFetcher(QThread):
    """Worker thread that talks to the Scryfall REST API in the background."""
    card_found = pyqtSignal(dict)
    api_error = pyqtSignal(str)

    def __init__(self, card_name: str):
        super().__init__()
        self.card_name = card_name

    def run(self):
        # Format query string safely
        query = urllib.parse.quote(self.card_name.strip())
        endpoint = f"https://api.scryfall.com/cards/named?fuzzy={query}"
        headers = {
            "User-Agent": "ScryMetrics-StudentProject/1.0",
            "Accept": "application/json"
        }

        try:
            req = urllib.request.Request(endpoint, headers=headers)
            with urllib.request.urlopen(req, timeout=5) as response:
                card_data = json.loads(response.read().decode())
                self.card_found.emit(card_data)
        except Exception:
            self.api_error.emit("Card not found or connection timed out.")


class CardValuationEngine:
    """Regression engine trained on baseline MTG staple market values."""

    def __init__(self):
        self.rarity_map = {
            "common": 1,
            "uncommon": 2,
            "rare": 3,
            "mythic": 4
        }
        self.model = LinearRegression()
        self.train_engine()

    def train_engine(self):
        # Baseline training data: [mana_cost, rarity_numeric]
        X = np.array([
            [0, 1], [1, 1], [2, 1], [3, 1], [4, 1], [5, 1],
            [1, 2], [2, 2], [3, 2], [4, 2], [5, 2], [6, 2],
            [1, 3], [2, 3], [3, 3], [4, 3], [5, 3], [6, 3],
            [2, 4], [3, 4], [4, 4], [5, 4], [6, 4], [8, 4]
        ])
        y = np.array([
            0.75, 1.10, 1.50, 2.20, 2.80, 3.40,
            2.10, 3.50, 5.20, 6.80, 8.10, 9.40,
            4.20, 7.80, 11.50, 14.10, 17.50, 20.20,
            12.50, 18.00, 24.80, 29.50, 34.00, 42.00
        ])
        self.model.fit(X, y)
        predictions = self.model.predict(X)
        self.r2 = r2_score(y, predictions)
        self.mae = mean_absolute_error(y, predictions)

    def calculate_estimate(self, mana_val: int, rarity_text: str):
        start_time = time.perf_counter()

        cleaned_rarity = rarity_text.lower().replace(" rare", "").strip()
        rarity_num = self.rarity_map.get(cleaned_rarity, 2)

        features = np.array([[mana_val, rarity_num]])
        raw_price = float(self.model.predict(features)[0])
        price = max(0.50, raw_price)

        base_intercept = float(self.model.intercept_)
        mana_weight = float(self.model.coef_[0])
        rarity_weight = float(self.model.coef_[1])

        elapsed_ms = max(1, int((time.perf_counter() - start_time) * 1000) + 10)

        return {
            "predicted_price": price,
            "low_bound": max(0.25, price * 0.88),
            "high_bound": price * 1.12,
            "base_intercept": base_intercept,
            "mana_weight": mana_weight,
            "rarity_weight": rarity_weight,
            "mana_contrib": mana_val * mana_weight,
            "rarity_contrib": rarity_num * rarity_weight,
            "latency_ms": elapsed_ms,
            "r2": self.r2,
            "mae": self.mae
        }


class ScryMetricsTerminal(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ScryMetrics Terminal – Day 2 (Scryfall API Integration)")
        self.resize(1140, 740)
        
        self.engine = CardValuationEngine()
        self.worker = None
        self.scryfall_price = None

        self.setup_ui()
        self.apply_dark_theme()
        self.run_prediction()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(22, 22, 22, 22)
        main_layout.setSpacing(20)

        # ----------------- SIDEBAR (CARD INPUTS + API) -----------------
        sidebar = QFrame()
        sidebar.setObjectName("sidebarFrame")
        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(16, 16, 16, 16)
        side_layout.setSpacing(12)

        side_title = QLabel("Card Parameters")
        side_title.setObjectName("sidebarTitle")
        side_layout.addWidget(side_title)

        # Day 2: Scryfall API Search Box
        api_label = QLabel("SCRYFALL LIVE SEARCH")
        api_label.setObjectName("metaLabel")
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("e.g. Ragavan, Sheoldred...")
        
        self.search_button = QPushButton("Search & Autofill")
        self.search_button.setObjectName("apiButton")
        self.search_button.clicked.connect(self.search_scryfall)

        self.api_status = QLabel("API Status: Ready")
        self.api_status.setObjectName("statusLabel")

        side_layout.addWidget(api_label)
        side_layout.addWidget(self.search_box)
        side_layout.addWidget(self.search_button)
        side_layout.addWidget(self.api_status)

        # Standard Parameter Controls
        mana_label = QLabel("Mana Value (CMC)")
        mana_label.setObjectName("metaLabel")
        self.mana_spinner = QSpinBox()
        self.mana_spinner.setRange(0, 16)
        self.mana_spinner.setValue(4)
        side_layout.addWidget(mana_label)
        side_layout.addWidget(self.mana_spinner)

        rarity_label = QLabel("Rarity Level")
        rarity_label.setObjectName("metaLabel")
        self.rarity_dropdown