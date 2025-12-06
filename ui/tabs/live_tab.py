#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap, QImage
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


class LiveTab(QWidget):
    """
    Live (.mov) tab.

    - Left: list of all detected .mov files in the current folder.
    - Right top: video preview with Play/Pause.
    - Right bottom: list of all frame indices; selecting a frame seeks to it.
    - A "Save Frame" button allows exporting the currently shown frame as an image.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.mov_files: list[str] = []
        self.cap = None
        self.playing: bool = False
        self.timer = QTimer(self)
        self.timer.setInterval(33)  # ~30 FPS
        self.timer.timeout.connect(self._next_frame)

        self.frame_count: int = 0
        self.current_frame_index: int = 0

        self._init_ui()

    # ------------------------------------------------------------------ UI

    def _init_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        # Left: list of .mov files
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)

        self.list_movs = QListWidget()
        self.list_movs.currentItemChanged.connect(self._on_movie_selected)
        lv.addWidget(self.list_movs)

        splitter.addWidget(left)
        splitter.setStretchFactor(0, 1)

        # Right: video + controls + frames list
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)

        # Video area inside a scroll area to avoid the main window resizing
        self.scroll_video = QScrollArea()
        self.scroll_video.setWidgetResizable(True)
        self.scroll_video.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.lbl_video = QLabel("No Video Loaded")
        self.lbl_video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_video.setStyleSheet("background: black; color: white;")
        self.lbl_video.setMinimumSize(320, 240)

        self.scroll_video.setWidget(self.lbl_video)
        rv.addWidget(self.scroll_video, 3)

        # Controls row
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

        # Frames list
        self.list_frames = QListWidget()
        self.list_frames.currentItemChanged.connect(self._on_frame_selected)
        self.list_frames.setMinimumHeight(120)
        rv.addWidget(self.list_frames, 1)

        splitter.addWidget(right)
        splitter.setStretchFactor(1, 3)

    # ------------------------------------------------------------------ Public API used by MainWindow

    def populate(self, movs: list[str]) -> None:
        """Populate .mov files list from a folder scan."""
        self.mov_files = movs
        self.list_movs.clear()

        for p in movs:
            self.list_movs.addItem(Path(p).name)

        # Auto-select the first video
        if movs:
            self.list_movs.setCurrentRow(0)
        else:
            self._unload_video()

    def stop_video(self) -> None:
        """Stop playback and release the capture handle."""
        self.timer.stop()
        self.playing = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.btn_play.setText("Play")

    def on_delete_request(self) -> None:
        """
        Hook for the global Delete key.

        For now we do not delete videos from disk here, but the method is
        implemented so the main window can safely call it.
        """
        # No-op for now (could later implement delete-of-selected-video).
        pass

    # ------------------------------------------------------------------ Internal helpers

    def _on_movie_selected(self, item) -> None:
        if not item:
            return
        idx = self.list_movs.row(item)
        if idx < 0 or idx >= len(self.mov_files):
            return
        path = self.mov_files[idx]
        self._load_video(path)

    def _load_video(self, path: str) -> None:
        self.stop_video()

        if _CV2 is None:
            self.lbl_video.setText("OpenCV (cv2) is not available.")
            self.btn_play.setEnabled(False)
            self.btn_save_frame.setEnabled(False)
            self.list_frames.clear()
            return

        cap = _CV2.VideoCapture(path)
        if not cap or not cap.isOpened():
            self.lbl_video.setText("Failed to open video.")
            self.btn_play.setEnabled(False)
            self.btn_save_frame.setEnabled(False)
            self.list_frames.clear()
            return

        self.cap = cap
        self.frame_count = int(self.cap.get(_CV2.CAP_PROP_FRAME_COUNT) or 0)
        self.current_frame_index = 0

        # Populate frames list (one entry per frame index)
        self._populate_frame_list()

        # Show first frame
        self._show_frame(0)

        self.btn_play.setEnabled(True)
        self.btn_save_frame.setEnabled(True)

    def _populate_frame_list(self) -> None:
        self.list_frames.blockSignals(True)
        self.list_frames.clear()
        if self.frame_count <= 0:
            self.list_frames.blockSignals(False)
            return

        for i in range(self.frame_count):
            self.list_frames.addItem(f"Frame {i}")
        self.list_frames.setCurrentRow(0)
        self.list_frames.blockSignals(False)

    # ------------------------------------------------------------------ Playback and frame navigation

    def _toggle_play(self) -> None:
        if not self.cap:
            return

        if self.playing:
            self.timer.stop()
            self.btn_play.setText("Play")
        else:
            self.timer.start(33)
            self.btn_play.setText("Pause")
        self.playing = not self.playing

    def _next_frame(self) -> None:
        if not self.cap:
            return

        ret, frame = self.cap.read()
        if not ret:
            # Loop back to start
            self.cap.set(_CV2.CAP_PROP_POS_FRAMES, 0)
            self.current_frame_index = 0
            return

        self.current_frame_index = int(
            self.cap.get(_CV2.CAP_PROP_POS_FRAMES) or 0
        )
        self._display_frame(frame)

        # Highlight current frame in the list (if in range)
        if 0 <= self.current_frame_index < self.list_frames.count():
            self.list_frames.blockSignals(True)
            self.list_frames.setCurrentRow(self.current_frame_index)
            self.list_frames.blockSignals(False)

    def _display_frame(self, frame) -> None:
        """Convert a cv2 frame to QPixmap and display it scaled to the viewport."""
        if frame is None:
            return

        frame_rgb = _CV2.cvtColor(frame, _CV2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        qimg = QImage(frame_rgb.data, w, h, ch * w,
                      QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg)

        # Scale to the scroll area's viewport size to avoid the main window growing
        target = self.scroll_video.viewport().size()
        if target.width() > 0 and target.height() > 0:
            pix = pix.scaled(
                target,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )

        self.lbl_video.setPixmap(pix)

    def _on_frame_selected(self, item) -> None:
        if not item or self.cap is None:
            return
        idx = self.list_frames.row(item)
        if idx < 0:
            return

        # When the user manually selects a frame, stop playback and seek
        self.timer.stop()
        self.playing = False
        self.btn_play.setText("Play")

        self._show_frame(idx)

    def _show_frame(self, index: int) -> None:
        if self.cap is None:
            return
        if index < 0:
            index = 0
        if self.frame_count and index >= self.frame_count:
            index = self.frame_count - 1

        self.cap.set(_CV2.CAP_PROP_POS_FRAMES, index)
        ret, frame = self.cap.read()
        if not ret:
            return

        self.current_frame_index = index
        self._display_frame(frame)

        # Keep the frame list selection in sync
        if 0 <= index < self.list_frames.count():
            self.list_frames.blockSignals(True)
            self.list_frames.setCurrentRow(index)
            self.list_frames.blockSignals(False)

    # ------------------------------------------------------------------ Frame saving

    def _save_current_frame(self) -> None:
        if self.cap is None or self.frame_count <= 0:
            return

        idx = self.current_frame_index
        if idx < 0:
            idx = 0

        # Seek to the exact frame we want to save
        self.cap.set(_CV2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if not ret:
            return

        default_name = f"frame_{idx:05d}.jpg"
        out_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Current Frame As",
            default_name,
            "Images (*.png *.jpg *.jpeg *.bmp)",
        )
        if not out_path:
            # Restore capture position to where playback left off
            self.cap.set(_CV2.CAP_PROP_POS_FRAMES, self.current_frame_index)
            return

        # Save in BGR format with OpenCV
        _CV2.imwrite(out_path, frame)

        # Restore the capture position to the current frame index so playback is not disrupted
        self.cap.set(_CV2.CAP_PROP_POS_FRAMES, self.current_frame_index)

    # ------------------------------------------------------------------ Helpers

    def _unload_video(self) -> None:
        self.stop_video()
        self.lbl_video.setText("No Video Loaded")
        self.list_frames.clear()
        self.frame_count = 0
        self.current_frame_index = 0
