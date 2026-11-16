"""
Course: Project and Portfolio IV (PP4)
Student: Alec Duncan
Project: ScryMetrics - Iteration 3 (UX & Polish Phase)

Day 3 Focus (Final Polish):
- Visual Grid UI Revamp: Transformed the variant list into an IconMode Grid.
- UX Error Prevention: Auto-fill dropdown forces completion prefix to prevent 404s.
- Async ThumbnailWorker progressively loads card art into the grid.
- Collector Modifiers (Release Year, Promos) added to math engine.
"""

import sys
import json
import urllib.request
import urllib.parse
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QStringListModel, QSize
from PyQt6.QtGui import QPixmap, QImage, QIcon
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
    QListWidget,
    QListWidgetItem,
    QAbstractItemView
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
        # 1. Double check the query length just in case the timer glitched
        query_clean = self.query.strip()
        if len(query_clean) < 2:
            self.suggestions_ready.emit([])
            return
            
        # 2. Safely encode the URL parameter
        params = urllib.parse.urlencode({'q': query_clean})
        url = f"https://api.scryfall.com/cards/autocomplete?{params}"
        
        # 3. Explicitly tell Scryfall we expect JSON back
        headers = {
            "User-Agent": "ScryMetrics/3.0",
            "Accept": "application/json"
        }
        
        try:
            # Print the exact URL so we can see what Scryfall is trying to process
            print(f"DEBUG: Autocomplete fetching -> {url}") 
            
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode())
                self.suggestions_ready.emit(data.get("data", []))
                
        except Exception as e:
            # Now we will see the exact URL that caused the 400 Bad Request
            print(f"Autocomplete Error: {e} | URL was: {url}") 
            self.suggestions_ready.emit([])

class ScryfallPrintsWorker(QThread):
    prints_found = pyqtSignal(list)
    fetch_failed = pyqtSignal(str)
    
    def __init__(self, card_name: str):
        super().__init__()
        self.card_name = card_name
        
    def run(self):
        query = f'!"{self.card_name.strip()}"'
        params = urllib.parse.urlencode({'q': query, 'unique': 'prints'})
        url = f"https://api.scryfall.com/cards/search?{params}"
        
        headers = {
            "User-Agent": "ScryMetrics/3.0",
            "Accept": "application/json"
        }
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode())
                self.prints_found.emit(data.get("data", []))
        except Exception as e:
            self.fetch_failed.emit(f"API Error: {str(e)}")

class ImageDownloadWorker(QThread):
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

class ThumbnailWorker(QThread):
    thumbnail_ready = pyqtSignal(int, bytes)

    def __init__(self, prints: list):
        super().__init__()
        self.prints = prints
        self._is_running = True

    def run(self):
        for i, p in enumerate(self.prints):
            if not self._is_running:
                break
            try:
                img_url = p.get("image_uris", {}).get("small")
                if not img_url:
                    img_url = p.get("card_faces", [{}])[0].get("image_uris", {}).get("small", "")
                
                if img_url:
                    req = urllib.request.Request(img_url, headers={"User-Agent": "ScryMetrics/3.0"})
                    with urllib.request.urlopen(req, timeout=5) as response:
                        self.thumbnail_ready.emit(i, response.read())
            except Exception:
                pass 
                
    def stop(self):
        self._is_running = False


# ---------------- CUSTOM WIDGETS ----------------

class CardSearchInput(QLineEdit):
    autofill_triggered = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setPlaceholderText("Start typing (e.g. Jin-Gitaxias)...")
        
        self.completer_model = QStringListModel()
        self.completer = QCompleter(self.completer_model, self)
        self.completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.completer.setFilterMode(Qt.MatchFlag.MatchStartsWith) 
        self.setCompleter(self.completer)

        # FIX: Force the popup to use dark mode colors so it isn't invisible
        self.completer.popup().setStyleSheet("""
            QListView {
                background-color: #161B22;
                color: #FFFFFF;
                border: 1px solid #30363D;
                font-size: 14px;
                outline: none;
            }
            QListView::item { padding: 8px; }
            QListView::item:selected { background-color: #1F6FEB; color: white; }
        """)

        self.completer.activated.connect(self.on_autofill_selected)

        self.debounce_timer = QTimer(self)
        self.debounce_timer.setSingleShot(True)
        self.debounce_timer.setInterval(250)
        self.debounce_timer.timeout.connect(self.fetch)
        self.textChanged.connect(self.on_text)
        self.worker = None

    def on_autofill_selected(self, text: str):
        self.setText(text)
        self.autofill_triggered.emit()

    def on_text(self, text: str):
        if len(text.strip()) >= 2:
            self.debounce_timer.start()
        else:
            self.debounce_timer.stop()
            self.completer_model.setStringList([])
            self.completer.popup().hide()

    def fetch(self):
        query = self.text().strip()
        if len(query) < 2:
            return
        self.worker = AutocompleteWorker(query)
        self.worker.suggestions_ready.connect(self.update_suggestions)
        self.worker.start()

    def update_suggestions(self, names: list):
        self.completer_model.setStringList(names)
        # FIX: Explicitly re-link the model so PyQt6 recognizes the async update
        self.completer.setModel(self.completer_model)
        
        if names and self.hasFocus():
            self.completer.setCompletionPrefix(self.text())
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
        valid_mana = max(0, min(16, int(mana_cost))) 
        rarity_num = self.rarity_map.get(rarity_text.lower().replace(" rare", "").strip(), 2)
        est = max(0.50, float(self.model.predict(np.array([[valid_mana, rarity_num]]))[0]))
        return {"est": est}


class ScryMetricsTerminal(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ScryMetrics Terminal – Iteration 3 Polish")
        self.resize(1250, 850)
        self.engine = ValuationEngine()
        self.prints_cache = []  
        self.thumb_worker = None

        self.setup_ui()
        self.apply_dark_theme()

    def setup_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)
        layout.setContentsMargins(22, 22, 22, 22)
        layout.setSpacing(20)

        # ---- LEFT SIDEBAR (INTAKE & VALUATION) ----
        left_panel = QFrame()
        left_panel.setObjectName("sideFrame")
        left_panel.setFixedWidth(380)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(12)

        left_layout.addWidget(QLabel("1. SEARCH CARD NAME", objectName="panelHeading"))

        self.search_box = CardSearchInput()
        self.search_box.returnPressed.connect(self.start_search)
        self.search_box.autofill_triggered.connect(self.start_search)
        
        self.search_btn = QPushButton("Search Inventory", objectName="apiBtn")
        self.search_btn.clicked.connect(self.start_search)
        self.status_msg = QLabel("API Status: Ready", objectName="statusMeta")
        
        left_layout.addWidget(self.search_box)
        left_layout.addWidget(self.search_btn)
        left_layout.addWidget(self.status_msg)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("background-color: #30363D;")
        left_layout.addWidget(line)

        left_layout.addWidget(QLabel("3. VALUATION & FINISH", objectName="panelHeading"))
        
        self.art_label = QLabel()
        self.art_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.art_label.setObjectName("artCanvas")
        self.art_label.setFixedSize(280, 390)
        self.art_label.setText("Select a printing from the right list.")
        
        img_layout = QHBoxLayout()
        img_layout.addStretch()
        img_layout.addWidget(self.art_label)
        img_layout.addStretch()
        left_layout.addLayout(img_layout)

        left_layout.addWidget(QLabel("Card Finish", objectName="labelMeta"))
        self.finish_picker = QComboBox()
        self.finish_picker.currentIndexChanged.connect(self.recalculate)
        self.finish_picker.setEnabled(False) 
        left_layout.addWidget(self.finish_picker)

        self.price_card = QFrame(objectName="cardBox")
        pc_box = QVBoxLayout(self.price_card)
        self.active_card_label = QLabel("Active Card: None", objectName="cardHeading")
        self.active_card_label.setWordWrap(True)
        self.price_text = QLabel("$0.00", objectName="heroNumber")
        self.market_delta_label = QLabel("TCG Market: $--.-- (Delta: N/A)", objectName="marketBlue")
        
        pc_box.addWidget(self.active_card_label)
        pc_box.addWidget(QLabel("PREDICTED COLLECTOR MARKET VALUE", objectName="labelMeta"))
        pc_box.addWidget(self.price_text)
        pc_box.addWidget(self.market_delta_label)
        
        left_layout.addWidget(self.price_card)
        left_layout.addStretch()
        layout.addWidget(left_panel)

        # ---- RIGHT WORKSPACE (GRID VARIANT LIST) ----
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0,0,0,0)
        
        right_layout.addWidget(QLabel("2. SELECT SPECIFIC PRINTING", objectName="dimMeta"))
        
        self.variant_list = QListWidget()
        self.variant_list.setObjectName("variantList")
        self.variant_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.variant_list.currentRowChanged.connect(self.on_edition_changed)
        
        # Grid Setup
        self.variant_list.setViewMode(QListWidget.ViewMode.IconMode)
        self.variant_list.setIconSize(QSize(146, 204))  
        self.variant_list.setGridSize(QSize(170, 260))  
        self.variant_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.variant_list.setMovement(QListWidget.Movement.Static)
        self.variant_list.setSpacing(10)
        self.variant_list.setWordWrap(True)
        
        right_layout.addWidget(self.variant_list)
        layout.addWidget(right_panel)

    def start_search(self):
        query = self.search_box.text().strip()
        if len(query) < 2: return
        self.status_msg.setText("API Status: Fetching variants...")
        
        if self.thumb_worker:
            self.thumb_worker.stop()
        
        self.variant_list.blockSignals(True)
        self.variant_list.clear()
        self.variant_list.blockSignals(False)
        
        self.finish_picker.setEnabled(False)
        
        self.prints_worker = ScryfallPrintsWorker(query)
        self.prints_worker.prints_found.connect(self.on_prints_found)
        self.prints_worker.fetch_failed.connect(lambda e: self.status_msg.setText(e))
        self.prints_worker.start()

    def on_prints_found(self, prints: list):
        self.status_msg.setText(f"API Status: {len(prints)} printings found. Loading art...")
        self.prints_cache = prints
        
        self.variant_list.blockSignals(True)
        for p in prints:
            set_code = p.get("set", "???").upper()
            year = p.get("released_at", "2024").split("-")[0]
            col_num = p.get("collector_number", "0")
            
            display_str = f"[{year}] {set_code} #{col_num}"
            
            item = QListWidgetItem(display_str)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.variant_list.addItem(item)
            
        self.variant_list.blockSignals(False)
        
        self.thumb_worker = ThumbnailWorker(prints)
        self.thumb_worker.thumbnail_ready.connect(self.apply_thumbnail)
        self.thumb_worker.start()

    def apply_thumbnail(self, index: int, img_bytes: bytes):
        img = QImage()
        img.loadFromData(img_bytes)
        pixmap = QPixmap.fromImage(img)
        icon = QIcon(pixmap)
        
        item = self.variant_list.item(index)
        if item:
            item.setIcon(icon)

    def on_edition_changed(self, index: int):
        if index < 0 or index >= len(self.prints_cache): return
        
        card_data = self.prints_cache[index]
        set_name = card_data.get("set_name", "Unknown Set")
        self.active_card_label.setText(f"Active Card: {card_data.get('name')} ({set_name})")
        
        try:
            img_url = card_data["image_uris"]["normal"]
        except KeyError:
            img_url = card_data.get("card_faces", [{}])[0].get("image_uris", {}).get("normal", "")
            
        if img_url:
            self.art_worker = ImageDownloadWorker(img_url)
            self.art_worker.image_ready.connect(self.render_artwork)
            self.art_worker.start()

        self.finish_picker.blockSignals(True)
        self.finish_picker.clear()
        for f in card_data.get("finishes", ["nonfoil"]):
            self.finish_picker.addItem(f.capitalize())
        self.finish_picker.blockSignals(False)
        self.finish_picker.setEnabled(True) 
        
        self.recalculate()

    def recalculate(self):
        idx = self.variant_list.currentRow()
        if idx < 0 or not self.prints_cache: return
        
        card_data = self.prints_cache[idx]
        finish = self.finish_picker.currentText().lower()
        prices = card_data.get("prices", {})
        
        mana_val = int(card_data.get("cmc", 0))
        rarity_val = card_data.get("rarity", "rare").capitalize()
        set_name = card_data.get("set_name", "")
        
        market_price = None
        if finish == "foil" and prices.get("usd_foil"):
            market_price = float(prices["usd_foil"])
        elif finish == "etched" and prices.get("usd_etched"):
            market_price = float(prices["usd_etched"])
        elif finish == "nonfoil" and prices.get("usd"):
            market_price = float(prices["usd"])

        res = self.engine.predict_value(mana_val, rarity_val)
        predicted = res["est"]
        
        release_year = int(card_data.get("released_at", "2024").split("-")[0])
        frame_effects = card_data.get("frame_effects", [])

        if release_year < 2000:
            predicted *= 35.0  
        if "secret lair" in set_name.lower() or "masterpiece" in set_name.lower():
            predicted *= 15.0  
        if "extendedart" in frame_effects or "showcase" in frame_effects:
            predicted *= 2.5   
            
        if finish in ["foil", "etched"]:
            predicted *= 1.35 
            
        self.price_text.setText(f"${predicted:.2f}")

        if market_price:
            delta = predicted - market_price
            sign = "+" if delta > 0 else ""
            self.market_delta_label.setText(f"{finish.capitalize()} TCG Market: ${market_price:.2f} (Delta: {sign}${delta:.2f})")
        else:
            self.market_delta_label.setText(f"{finish.capitalize()} TCG Market: N/A (No market data)")

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
            #panelHeading, #dimMeta { font-size: 13px; font-weight: bold; color: #58A6FF; text-transform: uppercase; margin-top: 5px; }
            #cardHeading { font-size: 14px; font-weight: bold; color: #FFFFFF; }
            #labelMeta { font-size: 11px; font-weight: 600; color: #8B949E; text-transform: uppercase; }
            #statusMeta { font-size: 10px; color: #6E7681; }
            #cardBox { background-color: #0D1117; border: 1px solid #30363D; border-radius: 8px; padding: 18px; margin-top: 10px; }
            #heroNumber { font-size: 48px; font-weight: bold; color: #38D39F; }
            #marketBlue { font-size: 14px; font-weight: 600; color: #58A6FF; }
            #artCanvas { background-color: #0D1117; border: 1px solid #30363D; border-radius: 8px; }
            
            QLineEdit, QComboBox { background-color: #0D1117; color: #FFFFFF; border: 1px solid #30363D; border-radius: 6px; padding: 8px; }
            QPushButton#apiBtn { background-color: #238636; color: #FFFFFF; font-weight: 600; padding: 8px; border-radius: 6px; }
            
            QListWidget#variantList { 
                background-color: #0E1117; 
                color: #8B949E; 
                border: 1px solid #30363D; 
                border-radius: 8px; 
                padding: 15px;
            }
            QListWidget#variantList::item { 
                background-color: #161B22;
                border: 1px solid #30363D; 
                border-radius: 8px; 
                padding: 8px; 
            }
            QListWidget#variantList::item:hover { 
                background-color: #21262D; 
                border: 1px solid #8B949E;
            }
            QListWidget#variantList::item:selected { 
                background-color: #1F6FEB; 
                color: white; 
                border: 2px solid #58A6FF;
            }
        """)

def main():
    app = QApplication(sys.argv)
    terminal = ScryMetricsTerminal()
    terminal.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()