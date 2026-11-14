"""
Course: Project and Portfolio IV (PP4)
Student: Alec Duncan
Project: ScryMetrics 
"""

import sys
import json
import urllib.request
import urllib.parse
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QStringListModel
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QComboBox,
    QPushButton,
    QFrame,
    QCompleter,
)
import numpy as np
from sklearn.linear_model import LinearRegression

# ---------------- THREAD WORKERS ----------------

class AutocompleteWorker(QThread):
    suggestions_ready = pyqtSignal(list)
    def __init__(self, query: str):
        super().__init__()
        self.query = query
    def run(self):
        encoded = urllib.parse.quote(self.query.strip())
        url = f"https://api.scryfall.com/cards/autocomplete?q={encoded}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ScryMetrics/3.0"})
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode())
                self.suggestions_ready.emit(data.get("data", []))
        except Exception:
            self.suggestions_ready.emit([])

class ScryfallPrintsWorker(QThread):
    """Day 1 Extension: Fetches ALL printings of a card instead of just one."""
    prints_found = pyqtSignal(list)
    fetch_failed = pyqtSignal(str)
    
    def __init__(self, card_name: str):
        super().__init__()
        self.card_name = card_name
        
    def run(self):
        exact_name = urllib.parse.quote(f'!\"{self.card_name.strip()}\"')
        url = f"https://api.scryfall.com/cards/search?q={exact_name}&unique=prints"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ScryMetrics/3.0"})
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode())
                self.prints_found.emit(data.get("data", []))
        except Exception:
            self.fetch_failed.emit("Card not found on Scryfall.")

class ImageDownloadWorker(QThread):
    """Day 1 UX Polish: Downloads card artwork asynchronously to prevent UI freeze."""
    image_ready = pyqtSignal(bytes)
    
    def __init__(self, image_url: str):
        super().__init__()
        self.image_url = image_url
        
    def run(self):
        try:
            req = urllib.request.Request(self.image_url, headers={"User-Agent": "ScryMetrics/3.0"})
            with urllib.request.urlopen(req, timeout=5) as response:
                self.image_ready.emit(response.read())
        except Exception:
            pass 


# ---------------- CUSTOM WIDGETS ----------------

class CardSearchInput(QLineEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setPlaceholderText("Start typing (e.g. Lightning Bolt)...")
        self.completer_model = QStringListModel()
        self.completer = QCompleter(self.completer_model, self)
        self.completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self.setCompleter(self.completer)

        self.debounce_timer = QTimer(self)
        self.debounce_timer.setSingleShot(True)
        self.debounce_timer.setInterval(250)
        self.debounce_timer.timeout.connect(self.fetch)
        self.textChanged.connect(self.on_text)
        self.worker = None

    def on_text(self, text: str):
        if len(text.strip()) >= 2:
            self.debounce_timer.start()
        else:
            self.completer_model.setStringList([])

    def fetch(self):
        query = self.text().strip()
        self.worker = AutocompleteWorker(query)
        self.worker.suggestions_ready.connect(self.update_suggestions)
        self.worker.start()

    def update_suggestions(self, names: list):
        self.completer_model.setStringList(names)
        if names and self.hasFocus():
            self.completer.complete()


# ---------------- ENGINE & UI ----------------

class ValuationEngine:
    def __init__(self):
        self.rarity_map = {"common": 1, "uncommon": 2, "rare": 3, "mythic": 4}
        self.model = LinearRegression()
        self.train_baseline()

    def train_baseline(self):
        X_train = np.array([[0, 1], [1, 1], [2, 1], [3, 1], [4, 1], [5, 1],
                            [1, 2], [2, 2], [3, 2], [4, 2], [5, 2], [6, 2],
                            [1, 3], [2, 3], [3, 3], [4, 3], [5, 3], [6, 3],
                            [2, 4], [3, 4], [4, 4], [5, 4], [6, 4], [8, 4]])
        y_train = np.array([0.75, 1.10, 1.50, 2.20, 2.80, 3.40,
                            2.10, 3.50, 5.20, 6.80, 8.10, 9.40,
                            4.20, 7.80, 11.50, 14.10, 17.50, 20.20,
                            12.50, 18.00, 24.80, 29.50, 34.00, 42.00])
        self.model.fit(X_train, y_train)

    def predict_value(self, mana_cost: int, rarity_text: str):
        valid_mana = max(0, min(16, int(mana_cost))) # Error Prevention
        rarity_num = self.rarity_map.get(rarity_text.lower().replace(" rare", "").strip(), 2)
        est = max(0.50, float(self.model.predict(np.array([[valid_mana, rarity_num]]))[0]))
        return {"est": est}


class ScryMetricsTerminal(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ScryMetrics Terminal – Iteration 3 Polish")
        self.resize(1180, 800)
        self.engine = ValuationEngine()
        self.prints_cache = []  

        self.setup_ui()
        self.apply_dark_theme()

    def setup_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)
        layout.setContentsMargins(22, 22, 22, 22)
        layout.setSpacing(20)

        # ---- LEFT SIDEBAR ----
        sidebar = QFrame()
        sidebar.setObjectName("sideFrame")
        sidebar.setFixedWidth(340)
        side_layout = QVBoxLayout(sidebar)
        side_layout.setSpacing(12)

        side_layout.addWidget(QLabel("RETAIL INTAKE", objectName="panelHeading"))

        self.search_box = CardSearchInput()
        self.search_box.returnPressed.connect(self.start_search)
        self.search_btn = QPushButton("Search Inventory", objectName="apiBtn")
        self.search_btn.clicked.connect(self.start_search)
        self.status_msg = QLabel("API Status: Ready", objectName="statusMeta")
        
        side_layout.addWidget(self.search_box)
        side_layout.addWidget(self.search_btn)
        side_layout.addWidget(self.status_msg)

        # UI Extension: Edition and Finish Dropdowns
        side_layout.addWidget(QLabel("Specific Edition", objectName="labelMeta"))
        self.edition_picker = QComboBox()
        side_layout.addWidget(self.edition_picker)

        side_layout.addWidget(QLabel("Card Finish", objectName="labelMeta"))
        self.finish_picker = QComboBox()
        side_layout.addWidget(self.finish_picker)

        # UX Extension: Card Artwork Display Placeholder
        self.art_label = QLabel()
        self.art_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.art_label.setObjectName("artCanvas")
        self.art_label.setFixedSize(280, 390)
        self.art_label.setText("No artwork loaded")
        side_layout.addWidget(self.art_label)

        side_layout.addStretch()
        layout.addWidget(sidebar)

        # ---- RIGHT WORKSPACE ----
        work_area = QWidget()
        work_layout = QVBoxLayout(work_area)
        work_layout.addWidget(QLabel("WORKSPACE / VALUATION TERMINAL", objectName="dimMeta"))
        
        self.price_card = QFrame(objectName="cardBox")
        pc_box = QVBoxLayout(self.price_card)
        self.active_card_label = QLabel("Active Card: None", objectName="cardHeading")
        self.price_text = QLabel("$0.00", objectName="heroNumber")
        
        pc_box.addWidget(self.active_card_label)
        pc_box.addWidget(QLabel("PREDICTED BASE MARKET VALUE", objectName="labelMeta"))
        pc_box.addWidget(self.price_text)
        
        work_layout.addWidget(self.price_card)
        work_layout.addStretch()
        layout.addWidget(work_area)

    def start_search(self):
        query = self.search_box.text().strip()
        if len(query) < 2: return
        self.status_msg.setText("API Status: Fetching variants...")
        self.prints_worker = ScryfallPrintsWorker(query)
        self.prints_worker.prints_found.connect(self.on_prints_found)
        self.prints_worker.fetch_failed.connect(lambda e: self.status_msg.setText("API Status: Not found"))
        self.prints_worker.start()

    def on_prints_found(self, prints: list):
        self.status_msg.setText(f"API Status: {len(prints)} printings found ✓")
        self.prints_cache = prints
        self.edition_picker.clear()
        
        # Populate the dropdown with all found sets
        for p in prints:
            set_name = p.get("set_name", "Unknown")
            set_code = p.get("set", "???").upper()
            display_name = f"{set_name} ({set_code})"
            self.edition_picker.addItem(display_name)
            
        # Load the first image as a test
        if prints:
            self.active_card_label.setText(f"Active Card: {prints[0].get('name')}")
            try:
                img_url = prints[0]["image_uris"]["normal"]
                self.art_worker = ImageDownloadWorker(img_url)
                self.art_worker.image_ready.connect(self.render_artwork)
                self.art_worker.start()
            except KeyError:
                pass

    def render_artwork(self, img_bytes: bytes):
        img = QImage()
        img.loadFromData(img_bytes)
        pixmap = QPixmap.fromImage(img)
        self.art_label.setPixmap(pixmap.scaled(self.art_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def apply_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #0E1117; }
            QLabel { color: #E6EDF3; font-family: 'Segoe UI', sans-serif; }
            #sideFrame { background-color: #161B22; border: 1px solid #30363D; border-radius: 8px; }
            #panelHeading, #cardHeading { font-size: 15px; font-weight: bold; color: #FFFFFF; }
            #labelMeta { font-size: 11px; font-weight: 600; color: #8B949E; text-transform: uppercase; }
            #statusMeta, #dimMeta { font-size: 10px; color: #6E7681; }
            #cardBox { background-color: #161B22; border: 1px solid #30363D; border-radius: 8px; padding: 18px; }
            #heroNumber { font-size: 48px; font-weight: bold; color: #38D39F; }
            #artCanvas { background-color: #0D1117; border: 1px solid #30363D; border-radius: 8px; }
            QLineEdit, QComboBox { background-color: #0D1117; color: #FFFFFF; border: 1px solid #30363D; border-radius: 6px; padding: 8px; }
            QPushButton#apiBtn { background-color: #238636; color: #FFFFFF; font-weight: 600; padding: 8px; border-radius: 6px; }
        """)

def main():
    app = QApplication(sys.argv)
    terminal = ScryMetricsTerminal()
    terminal.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()