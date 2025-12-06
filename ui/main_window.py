#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QPushButton, QLineEdit, QFileDialog, QMessageBox, QLabel
)
from PyQt6.QtCore import Qt, QThreadPool
from PyQt6.QtGui import QAction, QKeySequence

# Tabs
from ui.tabs.photos_tab import PhotosTab
from ui.tabs.live_tab import LiveTab
from ui.tabs.duplicates_tab import DuplicatesTab

# ViewModels
from viewmodels.duplicates_vm import DuplicatesViewModel

# Workers
from workers.tasks import DirScanJob


class ImageEditorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Editor Pro (Restored)")
        self.resize(1300, 850)
        self.setAcceptDrops(True)

        self.threadpool = QThreadPool()
        self.current_folder = None

        # --- Initialize ViewModels ---
        self.dupes_vm = DuplicatesViewModel()

        # UI Setup
        central = QWidget()
        self.setCentralWidget(central)
        self.main_layout = QVBoxLayout(central)
        self.main_layout.setContentsMargins(6, 6, 6, 6)

        self._init_top_bar()
        self._init_tabs()
        self._init_shortcuts()

    def _init_top_bar(self):
        top_row = QHBoxLayout()

        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText(
            "Paste folder path here or drop a folder...")
        self.path_edit.returnPressed.connect(self._open_path_from_edit)

        self.open_btn = QPushButton("Open")
        self.open_btn.clicked.connect(self._open_path_from_edit)

        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self._browse_folder)

        top_row.addWidget(self.path_edit, 1)
        top_row.addWidget(self.open_btn)
        top_row.addWidget(self.browse_btn)

        self.main_layout.addLayout(top_row)

    def _init_tabs(self):
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)

        # 1. Photos
        self.tab_photos = PhotosTab()
        self.tabs.addTab(self.tab_photos, "Photos")

        # 2. Live (.mov)
        self.tab_live = LiveTab()
        self.tabs.addTab(self.tab_live, "Live (.mov)")

        # 3. Duplicates
        # Fix: Pass the ViewModel to the Tab
        self.tab_dupes = DuplicatesTab(self.dupes_vm)
        self.tabs.addTab(self.tab_dupes, "Duplicates")

        # Stop video when switching away from Live tab
        self.tabs.currentChanged.connect(self._on_tab_changed)

    def _init_shortcuts(self):
        # Delete Action (Global)
        self.delete_action = QAction("Delete", self)
        self.delete_action.setShortcut(QKeySequence(Qt.Key.Key_Delete))
        self.delete_action.setShortcutContext(
            Qt.ShortcutContext.ApplicationShortcut)
        self.delete_action.triggered.connect(self._handle_global_delete)
        self.addAction(self.delete_action)

        # Navigation (Left/Right) - routed to Photos tab usually
        self.prev_action = QAction("Prev", self)
        self.prev_action.setShortcut(QKeySequence(Qt.Key.Key_Left))
        self.prev_action.triggered.connect(self.tab_photos.go_prev)
        self.addAction(self.prev_action)

        self.next_action = QAction("Next", self)
        self.next_action.setShortcut(QKeySequence(Qt.Key.Key_Right))
        self.next_action.triggered.connect(self.tab_photos.go_next)
        self.addAction(self.next_action)

    # --- Global Logic ---

    def _on_tab_changed(self, index):
        # Stop video if leaving Live tab
        if self.tabs.widget(index) != self.tab_live:
            self.tab_live.stop_video()

    def _handle_global_delete(self):
        """Routes the delete key to the active tab."""
        current = self.tabs.currentWidget()
        # Duplicates tab handles its own shortcuts via widget focus,
        # but if the main window catches it, we can route it manually if needed.
        # However, DuplicatesTab usually has 'focus' when you click items.
        # We'll prioritize the Photos/Live tab for this specific shortcut logic
        # unless we want to make it generic.
        if hasattr(current, "on_delete_request"):
            current.on_delete_request()

    # --- Folder Loading ---

    def _browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self._load_folder_globally(folder)

    def _open_path_from_edit(self):
        path_str = self.path_edit.text().strip()
        if not path_str:
            return
        p = Path(path_str)
        if p.exists() and p.is_dir():
            self._load_folder_globally(str(p))
        else:
            QMessageBox.warning(self, "Error", "Invalid folder path")

    def _load_folder_globally(self, path_str):
        path = Path(path_str)
        self.current_folder = path
        self.path_edit.setText(str(path))

        # Set status in tabs
        self.tab_photos.set_status(f"Scanning {path.name}...")

        # Fix: Update VM with folder directly
        self.dupes_vm.set_folder(path)

        # Start scanning (One scan for everything)
        # We use a dummy ID '1' for the main scan
        job = DirScanJob(1, path)
        job.signals.finished.connect(self._on_scan_finished)
        self.threadpool.start(job)

    def _on_scan_finished(self, jid, images, movs):
        # Populate Photos Tab
        self.tab_photos.populate(images)

        # Populate Live Tab
        self.tab_live.populate(movs)

        # Update Duplicates Tab Status
        # The duplicates tab status is now mostly driven by the VM signals,
        # but we can set an initial text.
        self.tab_dupes.lbl_status.setText(
            f"Folder loaded: {self.current_folder.name}. Found {len(images)} images, {len(movs)} videos."
        )

    # --- Drag & Drop ---
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            path = event.mimeData().urls()[0].toLocalFile()
            if os.path.isdir(path):
                self._load_folder_globally(path)
