#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QListWidget,
    QLabel, QTabWidget, QGroupBox, QPushButton, QRadioButton,
    QButtonGroup, QScrollArea, QComboBox, QMessageBox
)
from PyQt6.QtCore import Qt, QTimer, QThreadPool
from PyQt6.QtGui import QPixmap

from ui.widgets import ImageViewer, DualImageViewer, SliderSpinBox
from core.common import (
    Image, ImageOps, MODE_VIEW, MODE_CROP, FILL_KEEP, FILL_AUTOCROP,
    FILL_STRETCH, LEFT_PANEL_WIDTH, _PIEXIF
)
from core.filters import ALL_FILTERS
from core.geometry import _apply_geometry_perspective
from workers.tasks import PreviewJob


class PhotosTab(QWidget):
    def __init__(self):
        super().__init__()
        self.threadpool = QThreadPool()
        self.image_files = []
        self.current_idx = -1

        # Editing State
        self.original_image_path = None
        self.original_pil = None
        self.preview_source = None
        self.factors = {
            "exposure": 0.0, "brightness": 0.0, "contrast": 0.0, "highlights": 0.0,
            "shadows": 0.0, "blackpoint": 0.0, "saturation": 0.0, "vibrance": 0.0,
            "warmth": 0.0, "tint": 0.0, "sharpness": 0.0, "noise_reduction": 0.0,
            "vignette": 0.0, "brilliance": 0.0
        }
        self.rotation = 0
        self.geom = {"rx": 0.0, "ry": 0.0, "rz": 0.0}
        self.fill_mode = FILL_STRETCH
        self.filter_name = "None"
        self.filter_str = 1.0

        self._preview_timer = QTimer()
        self._preview_timer.setInterval(50)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.timeout.connect(self._run_preview)
        self._preview_job_id = 0

        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        # Left: List
        left = QWidget()
        lv = QVBoxLayout(left)
        self.lbl_status = QLabel("No folder loaded")
        self.lbl_status.setStyleSheet("color:#888;")
        lv.addWidget(self.lbl_status)

        self.list_photos = QListWidget()
        self.list_photos.currentItemChanged.connect(self._on_list_sel)
        lv.addWidget(self.list_photos)
        left.setFixedWidth(LEFT_PANEL_WIDTH)
        splitter.addWidget(left)

        # Right: Viewer + Controls
        right = QWidget()
        rv = QVBoxLayout(right)
        v_split = QSplitter(Qt.Orientation.Vertical)
        rv.addWidget(v_split)

        # Viewer
        self.viewer_stack = QTabWidget()
        self.viewer_stack.setTabBarAutoHide(True)
        self.single_view = ImageViewer()
        self.single_view.editor_ref = self

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.single_view)
        scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.viewer_stack.addTab(scroll, "Single")

        self.dual_view = DualImageViewer()
        self.viewer_stack.addTab(self.dual_view, "Dual")
        v_split.addWidget(self.viewer_stack)

        # Controls
        self.tabs_edit = QTabWidget()
        self.tabs_edit.setFixedHeight(320)
        v_split.addWidget(self.tabs_edit)

        self._build_editors()
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)

    def _build_editors(self):
        # 1. Transform
        t1 = QWidget()
        l1 = QVBoxLayout(t1)

        bg = QGroupBox("Rotate")
        bl = QHBoxLayout(bg)
        b_l = QPushButton("-90")
        b_l.clicked.connect(lambda: self._rotate(-90))
        b_r = QPushButton("+90")
        b_r.clicked.connect(lambda: self._rotate(90))
        bl.addWidget(b_l)
        bl.addWidget(b_r)
        l1.addWidget(bg)

        cg = QGroupBox("Actions")
        cl = QHBoxLayout(cg)
        self.btn_save = QPushButton("Save Overwrite")
        self.btn_save.clicked.connect(self._save)
        self.btn_reset = QPushButton("Reset All")
        self.btn_reset.clicked.connect(self._reset)
        cl.addWidget(self.btn_save)
        cl.addWidget(self.btn_reset)
        l1.addWidget(cg)
        l1.addStretch()
        self.tabs_edit.addTab(t1, "Transform")

        # 2. Tone
        t2 = QWidget()
        l2 = QVBoxLayout(t2)
        self.sliders = {}
        for k, lbl, mi, ma in [
            ("exposure", "Exposure", -2.0, 2.0),
            ("brightness", "Brightness", -100, 100),
            ("contrast", "Contrast", -100, 100),
            ("highlights", "Highlights", -100, 100),
            ("shadows", "Shadows", -100, 100)
        ]:
            s = SliderSpinBox(lbl, mi, ma, 0.0, 1.0)
            s.valueChanged.connect(
                lambda v, key=k: self._on_param_change(key, v))
            l2.addWidget(s)
            self.sliders[k] = s
        self.tabs_edit.addTab(t2, "Tone")

        # 3. Filters
        t3 = QWidget()
        l3 = QVBoxLayout(t3)
        sa = QScrollArea()
        sa.setWidgetResizable(True)
        cw = QWidget()
        gl = QVBoxLayout(cw)
        self.bg_filt = QButtonGroup(self)
        for n in ALL_FILTERS:
            if n == "—":
                continue
            b = QRadioButton(n)
            if n == "None":
                b.setChecked(True)
            self.bg_filt.addButton(b)
            b.toggled.connect(
                lambda c, name=n: self._set_filter(name) if c else None)
            gl.addWidget(b)
        sa.setWidget(cw)
        l3.addWidget(sa)
        self.tabs_edit.addTab(t3, "Filters")

    # --- Logic ---

    def set_status(self, msg):
        self.lbl_status.setText(msg)

    def populate(self, images):
        self.image_files = images
        self.list_photos.clear()
        for p in images:
            self.list_photos.addItem(Path(p).name)
        self.lbl_status.setText(f"{len(images)} images loaded.")
        if images:
            self.list_photos.setCurrentRow(0)

    def go_prev(self):
        if self.current_idx > 0:
            self.list_photos.setCurrentRow(self.current_idx - 1)

    def go_next(self):
        if self.current_idx < len(self.image_files) - 1:
            self.list_photos.setCurrentRow(self.current_idx + 1)

    def on_delete_request(self):
        """Called by Main Window when Delete is pressed."""
        if not self.image_files or self.current_idx < 0:
            return

        path = self.image_files[self.current_idx]
        reply = QMessageBox.question(self, "Delete", f"Move to deleted?\n{path}",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            try:
                p = Path(path)
                trash = p.parent / "deleted"
                trash.mkdir(exist_ok=True)
                import shutil
                shutil.move(str(p), str(trash / p.name))

                # Update UI
                row = self.current_idx
                self.list_photos.takeItem(row)
                self.image_files.pop(row)
                if row >= len(self.image_files):
                    row -= 1
                if row >= 0:
                    self.list_photos.setCurrentRow(row)
                else:
                    self.single_view.clear_pixmap()
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

    def _on_list_sel(self, item):
        if not item:
            return
        idx = self.list_photos.row(item)
        self._load_image(idx)

    def _load_image(self, idx):
        if idx < 0 or idx >= len(self.image_files):
            return
        self.current_idx = idx
        path = self.image_files[idx]
        self.original_image_path = path

        # Reset params
        self._reset_params()

        try:
            with Image.open(path) as im:
                im = ImageOps.exif_transpose(im)
                self.original_pil = im.copy()

            # Create preview source
            w, h = self.original_pil.size
            if max(w, h) > 2048:
                ratio = 2048 / max(w, h)
                self.preview_source = self.original_pil.resize(
                    (int(w*ratio), int(h*ratio)), Image.Resampling.BILINEAR)
            else:
                self.preview_source = self.original_pil.copy()

            self._schedule_preview()
        except:
            self.single_view.setText("Error loading image")

    def _schedule_preview(self):
        self._preview_timer.start()

    def _run_preview(self):
        if not self.preview_source:
            return
        self._preview_job_id += 1

        job = PreviewJob(
            self._preview_job_id, self.preview_source, self.rotation, self.factors,
            (1280, 1280), False, 0, 0, 0, self.filter_name, 1.0, self.fill_mode
        )
        job.signals.done.connect(self._on_preview_done)
        self.threadpool.start(job)

    def _on_preview_done(self, jid, img, mirror):
        if jid != self._preview_job_id:
            return
        self.single_view.set_pixmap(QPixmap.fromImage(img))

    def _on_param_change(self, key, val):
        self.factors[key] = val
        self._schedule_preview()

    def _rotate(self, deg):
        self.rotation = (self.rotation + deg) % 360
        self._schedule_preview()

    def _set_filter(self, name):
        self.filter_name = name
        self._schedule_preview()

    def _reset_params(self):
        self.rotation = 0
        for k in self.factors:
            self.factors[k] = 0.0
        self.filter_name = "None"
        for s in self.sliders.values():
            s.blockSignals(True)
            s.set_value(0)
            s.blockSignals(False)

    def _reset(self):
        self._reset_params()
        self._schedule_preview()

    def _save(self):
        # Placeholder for full save logic
        if self.original_image_path:
            QMessageBox.information(
                self, "Save", "Save functionality to be fully implemented.")
