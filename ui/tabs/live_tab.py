#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import Qt, QTimer, QObject, QRunnable, pyqtSignal, QThreadPool, QEvent
from PyQt6.QtGui import QPixmap, QImage, QKeyEvent
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QListWidget,
    QLabel,
    QPushButton,
    QSplitter,
    QScrollArea,
    QFileDialog,
)

from core.common import _CV2

# --- Async Video Loader ---

class VideoLoaderSignals(QObject):
    loaded = pyqtSignal(str, object, str) # path, QPixmap, info

class VideoLoaderJob(QRunnable):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.signals = VideoLoaderSignals()

    def run(self):
        try:
            # 1. Metadata
            p = Path(self.path)
            stats = p.stat()
            size_mb = stats.st_size / (1024 * 1024)
            date_str = datetime.fromtimestamp(stats.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            
            info_text = (
                f"<b>File:</b> {p.name}<br>"
                f"<b>Folder:</b> {p.parent.name}<br>"
                f"<b>Modified:</b> {date_str}<br>"
                f"<b>Size:</b> {size_mb:.2f} MB"
            )

            # 2. Thumbnail (First Frame)
            pixmap = None
            if _CV2 is not None:
                cap = _CV2.VideoCapture(self.path)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        rgb = _CV2.cvtColor(frame, _CV2.COLOR_BGR2RGB)
                        h, w, ch = rgb.shape
                        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
                        pixmap = QPixmap.fromImage(qimg)
                    cap.release()
            
            self.signals.loaded.emit(self.path, pixmap, info_text)

        except Exception as e:
            self.signals.loaded.emit(self.path, None, f"Error: {e}")


class LiveTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.threadpool = QThreadPool()

        self.mov_files: list[str] = []
        self.cap = None
        self.playing: bool = False
        self.timer = QTimer(self)
        self.timer.setInterval(33)  # ~30 FPS
        self.timer.timeout.connect(self._next_frame)

        self.frame_count: int = 0
        self.current_frame_index: int = 0
        
        self.pending_path = None

        self._init_ui()

    def _init_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        # Left: list
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)

        self.list_movs = QListWidget()
        self.list_movs.currentItemChanged.connect(self._on_movie_selected)
        lv.addWidget(self.list_movs)

        splitter.addWidget(left)
        splitter.setStretchFactor(0, 1)

        # Right: video + controls
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        
        self.lbl_info = QLabel("")
        self.lbl_info.setStyleSheet("color: #ccc; font-size: 11px; padding-bottom: 5px;")
        self.lbl_info.setWordWrap(True)
        rv.addWidget(self.lbl_info)

        self.scroll_video = QScrollArea()
        self.scroll_video.setWidgetResizable(True)
        self.scroll_video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Capture arrow keys
        self.scroll_video.installEventFilter(self)

        self.lbl_video = QLabel("No Video Loaded")
        self.lbl_video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_video.setStyleSheet("background: black; color: white;")
        self.lbl_video.setMinimumSize(320, 240)

        self.scroll_video.setWidget(self.lbl_video)
        rv.addWidget(self.scroll_video, 3)

        controls = QHBoxLayout()
        self.btn_play = QPushButton("Play")
        self.btn_play.setEnabled(False)
        self.btn_play.clicked.connect(self._toggle_play)
        controls.addWidget(self.btn_play)

        self.btn_save_frame = QPushButton("Save Current Frame")
        self.btn_save_frame.setEnabled(False)
        self.btn_save_frame.clicked.connect(self._save_current_frame)
        controls.addWidget(self.btn_save_frame)

        controls.addStretch(1)
        rv.addLayout(controls)

        self.list_frames = QListWidget()
        self.list_frames.currentItemChanged.connect(self._on_frame_selected)
        self.list_frames.setMinimumHeight(120)
        rv.addWidget(self.list_frames, 1)

        splitter.addWidget(right)
        splitter.setStretchFactor(1, 3)
        
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    # --- Navigation ---

    def keyPressEvent(self, event: QKeyEvent):
        if self._handle_navigation(event):
            event.accept()
        else:
            super().keyPressEvent(event)

    def eventFilter(self, source, event):
        if event.type() == QEvent.Type.KeyPress:
            if self._handle_navigation(event):
                return True
        return super().eventFilter(source, event)

    def _handle_navigation(self, event):
        key = event.key()
        if key in (Qt.Key.Key_Up, Qt.Key.Key_Down):
            row = self.list_movs.currentRow()
            count = self.list_movs.count()
            if count == 0: return False
            
            if key == Qt.Key.Key_Up:
                new_row = max(0, row - 1)
            else:
                new_row = min(count - 1, row + 1)
            
            if new_row != row:
                self.list_movs.setCurrentRow(new_row)
            return True
        return False

    # --- Public API ---

    def populate(self, movs: list[str]) -> None:
        self.mov_files = movs
        self.list_movs.clear()
        self.lbl_info.clear()

        for p in movs:
            self.list_movs.addItem(Path(p).name)

        if movs:
            self.list_movs.setCurrentRow(0)
        else:
            self._unload_video()

    def stop_video(self) -> None:
        self.timer.stop()
        self.playing = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.btn_play.setText("Play")

    def on_delete_request(self) -> None:
        pass

    # --- Internal Logic ---

    def _on_movie_selected(self, item) -> None:
        if not item: return
        idx = self.list_movs.row(item)
        if idx < 0 or idx >= len(self.mov_files): return
        
        path = self.mov_files[idx]
        self.pending_path = path
        
        # Stop previous playback immediately
        self.stop_video()
        self.lbl_video.setText("Loading...")
        self.lbl_info.setText("Loading info...")
        self.list_frames.clear()
        self.btn_play.setEnabled(False)
        self.btn_save_frame.setEnabled(False)

        # Fire Async Loader
        loader = VideoLoaderJob(path)
        loader.signals.loaded.connect(self._on_video_loaded)
        self.threadpool.start(loader)

    def _on_video_loaded(self, path, pixmap, info):
        if path != self.pending_path:
            return # Stale
        
        self.lbl_info.setText(info)
        
        if pixmap:
            # Scale to fit
            target = self.scroll_video.viewport().size()
            if target.width() > 0:
                pixmap = pixmap.scaled(target, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.lbl_video.setPixmap(pixmap)
            
            # Enable play button (user must click play to load full video engine)
            self.btn_play.setEnabled(True)
        else:
            self.lbl_video.setText("Could not load preview")

    def _toggle_play(self) -> None:
        # Load the heavy Capture object only when Play is requested
        if not self.cap and self.pending_path:
            self._init_cap_for_playback(self.pending_path)

        if not self.cap:
            return

        if self.playing:
            self.timer.stop()
            self.btn_play.setText("Play")
        else:
            self.timer.start(33)
            self.btn_play.setText("Pause")
        self.playing = not self.playing

    def _init_cap_for_playback(self, path):
        if _CV2 is None: return
        self.cap = _CV2.VideoCapture(path)
        if self.cap.isOpened():
            self.frame_count = int(self.cap.get(_CV2.CAP_PROP_FRAME_COUNT) or 0)
            self.current_frame_index = 0
            self._populate_frame_list()
            self.btn_save_frame.setEnabled(True)
        else:
            self.lbl_video.setText("Failed to load video stream")

    def _populate_frame_list(self) -> None:
        self.list_frames.blockSignals(True)
        self.list_frames.clear()
        if self.frame_count <= 0:
            self.list_frames.blockSignals(False)
            return
        # Limit frame list items if too many
        step = 1 if self.frame_count < 500 else self.frame_count // 500
        for i in range(0, self.frame_count, step):
            self.list_frames.addItem(f"Frame {i}")
        self.list_frames.setCurrentRow(0)
        self.list_frames.blockSignals(False)

    def _next_frame(self) -> None:
        if not self.cap: return
        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(_CV2.CAP_PROP_POS_FRAMES, 0)
            self.current_frame_index = 0
            return

        self.current_frame_index = int(self.cap.get(_CV2.CAP_PROP_POS_FRAMES) or 0)
        self._display_frame(frame)

    def _display_frame(self, frame) -> None:
        if frame is None: return
        frame_rgb = _CV2.cvtColor(frame, _CV2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        qimg = QImage(frame_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        
        target = self.scroll_video.viewport().size()
        if target.width() > 0:
            pix = pix.scaled(target, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.lbl_video.setPixmap(pix)

    def _on_frame_selected(self, item) -> None:
        if not item or self.cap is None: return
        # Parse frame index from text "Frame X"
        txt = item.text()
        try:
            idx = int(txt.split(" ")[1])
        except: return

        self.timer.stop()
        self.playing = False
        self.btn_play.setText("Play")
        self._show_frame(idx)

    def _show_frame(self, index: int) -> None:
        if self.cap is None: return
        self.cap.set(_CV2.CAP_PROP_POS_FRAMES, index)
        ret, frame = self.cap.read()
        if ret:
            self.current_frame_index = index
            self._display_frame(frame)

    def _save_current_frame(self) -> None:
        if self.cap is None: return
        idx = self.current_frame_index
        self.cap.set(_CV2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if ret:
            default_name = f"frame_{idx:05d}.jpg"
            out_path, _ = QFileDialog.getSaveFileName(self, "Save Frame", default_name, "Images (*.jpg)")
            if out_path:
                _CV2.imwrite(out_path, frame)
        self.cap.set(_CV2.CAP_PROP_POS_FRAMES, idx) # Restore

    def _unload_video(self) -> None:
        self.stop_video()
        self.lbl_video.setText("No Video Loaded")
        self.list_frames.clear()
        self.frame_count = 0