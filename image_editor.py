#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import threading
from pathlib import Path
from typing import Optional, Any, Tuple, List, Dict
from datetime import datetime
import shutil

from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QListWidget,
    QPushButton,
    QLabel,
    QSlider,
    QTabWidget,
    QGridLayout,
    QGroupBox,
    QLineEdit,
    QStackedWidget,
    QRadioButton,
    QButtonGroup,
    QFrame,
    QComboBox,
    QScrollArea,
    QSizePolicy,
    QMessageBox,
    QFileDialog,
    QAbstractItemView,
    QAbstractScrollArea,
    QSplitter,
)
from PyQt6.QtGui import (
    QPixmap,
    QImage,
    QAction,
    QPalette,
    QKeySequence,
)
from duplicates import DuplicateFinder, DuplicateRecord
from PyQt6.QtCore import (
    Qt,
    QSize,
    QPoint,
    QRect,
    pyqtSignal,
    QDir,
    QTimer,
    QThreadPool,
    QEvent,
)

from common import (
    Image,
    ImageOps,
    ImageEnhance,
    _CV2,
    _NP,
    _HEIF_PLUGIN,
    _PIEXIF,
    _S2T,
    _MEDIAINFO,
    EXT_TO_FMT,
    HEIF_LIKE_EXTS,
    SUPPORTED_IMAGE_EXTS,
    SUPPORTED_LIVE_EXTS,
    MODE_VIEW,
    MODE_CROP,
    FILL_KEEP,
    FILL_AUTOCROP,
    FILL_STRETCH,
    LEFT_PANEL_WIDTH,
    _fmt_ts_local,
    _sanitize_exif_datetime,
    pil_to_qimage,
    pil_to_qpixmap,
)
from geometry import _apply_geometry_perspective
from filters import ALL_FILTERS, _apply_filter_pipeline
from widgets import ImageViewer, DualImageViewer, SliderSpinBox
from workers import DirScanJob, _PreviewJob


class ImageEditorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced PyQt6 Image Editor + Live Photo Tool")
        self.setGeometry(100, 100, 1300, 870)
        self.setAcceptDrops(True)

        self.current_folder: Optional[Path] = None

        self.duplicate_groups: List[List[DuplicateRecord]] = []

        self.image_files: List[str] = []
        self.current_image_index: int = -1
        self.original_image_path: Optional[str] = None
        self.original_image_pil: Optional[Image.Image] = None
        self.working_image_pil: Optional[Image.Image] = None
        self._orig_format: Optional[str] = None
        self._orig_exif: Optional[bytes] = None
        self.crop_area: Optional[Tuple[int, int, int, int]] = None

        self.rotation_degrees = 0
        self.brightness_factor = 0.0
        self.contrast_factor = 0.0
        self.exposure_factor = 0.0
        self.highlights_factor = 0.0
        self.shadows_factor = 0.0
        self.blackpoint_factor = 0.0
        self.saturation_factor = 0.0
        self.vibrance_factor = 0.0
        self.warmth_factor = 0.0
        self.tint_factor = 0.0
        self.sharpness_factor = 0.0
        self.noise_reduction_factor = 0.0
        self.vignette_factor = 0.0
        self.brilliance_factor = 0.0

        self.geom_rx_deg = 0.0
        self.geom_ry_deg = 0.0
        self.geom_rz_deg = 0.0
        self.fill_mode = FILL_STRETCH
        self._warned_perspective = False
        self._last_changed_name: str = ""

        self.filter_name = "None"
        self.filter_strength = 1.0

        self.mov_files: List[str] = []
        self.current_mov_index: int = -1
        self.cap = None
        self.live_timer = QTimer(self)
        self.live_timer.timeout.connect(self._advance_video_frame)
        self.video_fps = 30.0
        self.total_frames = 0
        self.current_frame_idx = 0
        self.is_playing = False
        self.image_meta: Dict[str, Dict[str, str]] = {}
        self.mov_meta: Dict[str, Dict[str, str]] = {}
        
        self._mirror_on = False

        self._all_sliders: List[QSlider] = []
        self._preview_source: Optional[Image.Image] = None

        self.threadpool = QThreadPool()
        self.threadpool.setMaxThreadCount(2)  # 1 preview + 1 scan
        self._preview_job_id = 0
        self._scan_job_id = 0

        self._preview_timer = QTimer(self)
        self._preview_timer.setInterval(20)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.timeout.connect(self._start_preview_job)
        self._preview_lock = threading.Lock()

        # will be created in _init_image_window
        self.image_window: Optional[QWidget] = None
        self.viewer_stack: Optional[QStackedWidget] = None
        self.single_viewer: Optional[ImageViewer] = None
        self.dual_viewer: Optional[DualImageViewer] = None
        self.single_scroll: Optional[QScrollArea] = None

        self._init_ui()
        self.photo_sort_combo.setCurrentText("Date taken")

        self.set_controls_state(False)

        self.delete_action = QAction("Delete", self)
        self.delete_action.setShortcut(QKeySequence(Qt.Key.Key_Delete))
        self.delete_action.setShortcutContext(
            Qt.ShortcutContext.ApplicationShortcut
        )
        self.delete_action.triggered.connect(self._on_delete_key)
        self.addAction(self.delete_action)

        self.prev_image_action = QAction("Prev image", self)
        self.prev_image_action.setShortcut(QKeySequence(Qt.Key.Key_Left))
        self.prev_image_action.setShortcutContext(
            Qt.ShortcutContext.ApplicationShortcut
        )
        self.prev_image_action.triggered.connect(self._go_prev_image)
        self.addAction(self.prev_image_action)

        self.next_image_action = QAction("Next image", self)
        self.next_image_action.setShortcut(QKeySequence(Qt.Key.Key_Right))
        self.next_image_action.setShortcutContext(
            Qt.ShortcutContext.ApplicationShortcut
        )
        self.next_image_action.triggered.connect(self._go_next_image)
        self.addAction(self.next_image_action)
        
        self.copy_image_action = QAction("Copy image", self)
        self.copy_image_action.setShortcut(QKeySequence.StandardKey.Copy)
        self.copy_image_action.setShortcutContext(
            Qt.ShortcutContext.ApplicationShortcut
        )
        self.copy_image_action.triggered.connect(self._copy_current_image)
        self.addAction(self.copy_image_action)
        
        if _CV2 is None or _NP is None:
            QMessageBox.warning(
                self,
                "Optional Dependencies",
                "Perspective straighten (X/Y) and some filters require opencv-python and numpy.\n\n"
                "pip install opencv-python numpy",
            )

    def _copy_current_image(self):
        """
        Ctrl+C:
        - Copies the absolute path of the current image as text.
        - Also copies the current pixmap (if available) to the clipboard as an image.
        """
        if not self.image_files or self.current_image_index < 0:
            return

        path = self.image_files[self.current_image_index]

        cb = QApplication.clipboard()
        # Copy text path
        cb.setText(path)

        # Copy image bitmap if we have it
        if (
            self.single_viewer is not None
            and getattr(self.single_viewer, "current_qpixmap", None)
        ):
            cb.setPixmap(self.single_viewer.current_qpixmap)


    def _go_prev_image(self):
        if not self.image_files or self.current_image_index <= 0:
            return
        new_idx = self.current_image_index - 1
        self.photo_list.setCurrentRow(new_idx)
        # _on_photo_select will handle loading

    def _go_next_image(self):
        if not self.image_files:
            return
        if self.current_image_index < 0:
            # nothing selected yet -> go to first
            self.photo_list.setCurrentRow(0)
            return
        if self.current_image_index >= len(self.image_files) - 1:
            return
        new_idx = self.current_image_index + 1
        self.photo_list.setCurrentRow(new_idx)
        # _on_photo_select will handle loading

    # ---------- Geometry commit / crop ----------

    def _on_geom_auto_crop(self):
        if not self.working_image_pil:
            QMessageBox.warning(self, "Geometry", "No image loaded.")
            return
        try:
            baked = _apply_geometry_perspective(
                self.working_image_pil,
                self.geom_rx_deg,
                self.geom_ry_deg,
                self.geom_rz_deg,
                self.rotation_degrees,
                FILL_AUTOCROP,
                preview_fast=False,
            )
            self.working_image_pil = baked
            self._preview_source = self._make_preview_source(
                self.working_image_pil
            )
            self.rotation_degrees = 0
            self.geom_rx_deg = self.geom_ry_deg = self.geom_rz_deg = 0.0
            for n in ("geom_rx", "geom_ry", "geom_rz"):
                s = getattr(self, f"{n}_slider", None)
                if s:
                    s.set_value(0.0)
            self._schedule_preview()
        except Exception as e:
            QMessageBox.critical(self, "Geometry", f"Auto crop failed: {e}")

    def _on_geom_commit(self, mode: str):
        if not self.working_image_pil:
            QMessageBox.warning(self, "Geometry", "No image loaded.")
            return
        try:
            baked = _apply_geometry_perspective(
                self.working_image_pil,
                self.geom_rx_deg,
                self.geom_ry_deg,
                self.geom_rz_deg,
                self.rotation_degrees,
                mode,
                preview_fast=False,
            )
            self.working_image_pil = baked
            self._preview_source = self._make_preview_source(
                self.working_image_pil
            )
            self.rotation_degrees = 0
            self.geom_rx_deg = self.geom_ry_deg = self.geom_rz_deg = 0.0
            for n in ("geom_rx", "geom_ry", "geom_rz"):
                s = getattr(self, f"{n}_slider", None)
                if s:
                    s.set_value(0.0)
            self._schedule_preview()
        except Exception as e:
            QMessageBox.critical(self, "Geometry", f"Commit failed: {e}")

    def _on_crop_mode_changed(self, mode: str):
        self.crop_mode_type = mode
        if self.single_viewer is not None:
            self.single_viewer.set_crop_mode_type(mode)

    # ---------- UI build ----------
    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(6, 6, 6, 6)

        # Top quick row: path + open/browse
        quick_row = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText(
            "Paste folder path here or drop a folder anywhere in the window"
        )
        self.open_btn = QPushButton("Open")
        self.open_btn.clicked.connect(self._open_path_from_edit)
        self.photo_load_btn = QPushButton("Browse...")
        self.photo_load_btn.clicked.connect(self._browse_and_load_folder)

        quick_row.addWidget(self.path_edit, 1)
        quick_row.addWidget(self.open_btn)
        quick_row.addWidget(self.photo_load_btn)
        main_layout.addLayout(quick_row)

        # Single tabs widget
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # Photos tab
        self.photos_tab = QWidget()
        self.tabs.addTab(self.photos_tab, "Photos")
        self._build_photos_tab(self.photos_tab)

        # Live .mov tab
        self.live_tab = self._build_live_tab()
        self.tabs.addTab(self.live_tab, "Live (.mov)")

        # Duplicates tab
        self.duplicates_tab = QWidget()
        self.tabs.addTab(self.duplicates_tab, "Duplicates")
        self._build_duplicates_tab(self.duplicates_tab)

        # Tab change handler
        self.tabs.currentChanged.connect(self._on_tab_changed)

        # Separate window for the image viewer(s)
        self._init_image_window()

    def _init_image_window(self):
        """
        Create a separate top-level window that hosts the image viewer(s).
        This window can be freely resized independently of the main editor UI.
        """
        self.image_window = QWidget()
        self.image_window.setWindowTitle("Image Viewer")
        self.image_window.resize(1000, 700)

        layout = QVBoxLayout(self.image_window)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Viewer stack (single / dual)
        self.viewer_stack = QStackedWidget()

        self.single_viewer = ImageViewer()
        self.single_viewer.editor_ref = self

        self.single_scroll = QScrollArea()
        self.single_scroll.setWidgetResizable(True)
        self.single_scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.single_scroll.setWidget(self.single_viewer)

        self.single_scroll.viewport().installEventFilter(self)

        single_holder = QWidget()
        svl = QVBoxLayout(single_holder)
        svl.setContentsMargins(0, 0, 0, 0)
        svl.setSpacing(0)
        svl.addWidget(self.single_scroll)
        self.viewer_stack.addWidget(single_holder)

        self.dual_viewer = DualImageViewer()
        self.viewer_stack.addWidget(self.dual_viewer)

        layout.addWidget(self.viewer_stack)

        # Start with a blank viewer
        self.single_viewer.clear_pixmap("Drop or open a folder to view images")

        self.image_window.show()

    def eventFilter(self, obj, event):
        if (
            self.single_scroll is not None
            and obj is self.single_scroll.viewport()
            and event.type() == QEvent.Type.Resize
        ):
            # viewport changed -> recompute preview to fit new size
            self._schedule_preview()
        return super().eventFilter(obj, event)


    def _build_duplicates_tab(self, tab: QWidget):
        """
        Build the 'Duplicates' tab UI.

        - Top row: status label + 'Scan for duplicates' button
        - Left panel: list of duplicate groups
        - Right panel: scrollable grid of thumbnails for the selected group
        """
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # -------- Top controls row: status + scan button --------
        controls_row = QHBoxLayout()
        controls_row.setContentsMargins(0, 0, 0, 0)
        controls_row.setSpacing(8)

        self.dup_status_lbl = QLabel(
            "Load a folder in Photos tab, then press 'Scan for duplicates'."
        )
        self.dup_status_lbl.setStyleSheet("color:#999;")
        controls_row.addWidget(self.dup_status_lbl, 1)

        self.dup_scan_button = QPushButton("Scan for duplicates")
        self.dup_scan_button.setToolTip(
            "Scan the currently loaded Photos folder for similar images."
        )
        controls_row.addWidget(self.dup_scan_button, 0)

        layout.addLayout(controls_row)

        # -------- Splitter: left groups list / right thumbnails --------
        split = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(split, 1)

        # == Left side: duplicate groups list ==
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(6)

        groups_lbl = QLabel("Duplicate groups:")
        groups_lbl.setStyleSheet("color:#ccc;")
        lv.addWidget(groups_lbl)

        self.dup_groups_list = QListWidget()
        self.dup_groups_list.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.dup_groups_list.setUniformItemSizes(True)
        self.dup_groups_list.setAlternatingRowColors(True)
        self.dup_groups_list.setVerticalScrollMode(
            QAbstractItemView.ScrollMode.ScrollPerPixel
        )
        lv.addWidget(self.dup_groups_list, 1)

        split.addWidget(left)

        # == Right side: thumbnails for selected group ==
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(6)

        thumbs_lbl = QLabel("Thumbnails in selected group:")
        thumbs_lbl.setStyleSheet("color:#ccc;")
        rv.addWidget(thumbs_lbl)

        self.dup_scroll = QScrollArea()
        self.dup_scroll.setWidgetResizable(True)

        thumb_container = QWidget()
        self.dup_thumbs_layout = QGridLayout(thumb_container)
        self.dup_thumbs_layout.setContentsMargins(4, 4, 4, 4)
        self.dup_thumbs_layout.setHorizontalSpacing(10)
        self.dup_thumbs_layout.setVerticalSpacing(10)

        self.dup_scroll.setWidget(thumb_container)
        rv.addWidget(self.dup_scroll, 1)

        split.addWidget(right)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)

        # -------- Handlers --------

        def _on_group_changed(row: int):
            if not (0 <= row < len(self.duplicate_groups)):
                if hasattr(self, "_clear_duplicate_thumbnails"):
                    self._clear_duplicate_thumbnails()
                return
            group = self.duplicate_groups[row]
            self._show_duplicate_group_thumbnails(group)

        self.dup_groups_list.currentRowChanged.connect(_on_group_changed)

        def _run_duplicate_scan():
            if not self.current_folder or not self.image_files:
                QMessageBox.information(
                    self,
                    "Duplicates",
                    "No images loaded.\n\nLoad a folder in the Photos tab first.",
                )
                return

            # Always reset previous results when starting a new scan
            self.duplicate_groups = []
            if hasattr(self, "dup_groups_list"):
                self.dup_groups_list.clear()
            if hasattr(self, "_clear_duplicate_thumbnails"):
                self._clear_duplicate_thumbnails()

            self.dup_status_lbl.setText(
                "Scanning for duplicates... this may take a while."
            )
            self.dup_scan_button.setEnabled(False)
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            QApplication.processEvents()

            try:
                if not hasattr(self, "_duplicate_finder"):
                    self._duplicate_finder = DuplicateFinder()

                groups = self._duplicate_finder.find_duplicates(
                    self.image_files
                )

                self.duplicate_groups = groups or []

                self.dup_groups_list.clear()
                self._clear_duplicate_thumbnails()

                if not self.duplicate_groups:
                    self.dup_status_lbl.setText("No duplicate groups found.")
                else:
                    for idx, group in enumerate(self.duplicate_groups):
                        try:
                            first_path = Path(group[0].path)
                            folder_name = first_path.parent.name
                        except Exception:
                            folder_name = "n/a"
                        item_text = (
                            f"Group {idx + 1}: {len(group)} images "
                            f"(folder: {folder_name})"
                        )
                        self.dup_groups_list.addItem(item_text)

                    self.dup_status_lbl.setText(
                        f"Found {len(self.duplicate_groups)} duplicate groups."
                    )
                    self.dup_groups_list.setCurrentRow(0)

            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Duplicates",
                    f"Error while scanning for duplicates:\n{e}",
                )
                self.dup_status_lbl.setText("Duplicate scan failed.")
            finally:
                QApplication.restoreOverrideCursor()
                self.dup_scan_button.setEnabled(True)

        self.dup_scan_button.clicked.connect(_run_duplicate_scan)

    def _clear_duplicate_thumbnails(self) -> None:
        """
        Remove all widgets from the duplicate thumbnails grid layout.
        Safe to call even before the duplicates tab was fully initialized.
        """
        layout = getattr(self, "dup_thumbs_layout", None)
        if layout is None:
            return

        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()

    def _build_photos_tab(self, tab: QWidget):
        layout = QHBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)

        split = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(split)

        self._in_splitter_handler = False
        self._last_split_sizes = [280, 1020]

        # ----------------- Left panel: list + status -----------------
        left = QWidget()
        lv = QVBoxLayout(left)

        self.scan_status_lbl = QLabel("No folder loaded")
        self.scan_status_lbl.setStyleSheet("color:#999;")
        lv.addWidget(self.scan_status_lbl)

        # --- Sort controls for photos ---
        sort_row = QHBoxLayout()
        sort_row.setContentsMargins(0, 0, 0, 0)
        sort_row.setSpacing(6)

        sort_label = QLabel("Sort images by:")
        self.photo_sort_combo = QComboBox()
        self.photo_sort_combo.addItems(
            ["Name", "Date taken", "Date modified", "Date created"]
        )

        self.photo_sort_order_combo = QComboBox()
        self.photo_sort_order_combo.addItems(["Ascending", "Descending"])

        sort_row.addWidget(sort_label)
        sort_row.addWidget(self.photo_sort_combo)
        sort_row.addWidget(self.photo_sort_order_combo)
        sort_row.addStretch(1)

        lv.addLayout(sort_row)


        self.photo_list = QListWidget()
        self.photo_list.currentItemChanged.connect(self._on_photo_select)
                # Apply sorting whenever user changes mode/order
        self.photo_sort_combo.currentIndexChanged.connect(
            lambda _: self._apply_photo_sort()
        )
        self.photo_sort_order_combo.currentIndexChanged.connect(
            lambda _: self._apply_photo_sort()
        )


        left.setSizePolicy(
            QSizePolicy.Policy.Fixed,
            QSizePolicy.Policy.Expanding,
        )
        left.setFixedWidth(LEFT_PANEL_WIDTH)

        self.photo_list.setSizePolicy(
            QSizePolicy.Policy.Fixed,
            QSizePolicy.Policy.Expanding,
        )
        self.photo_list.setMinimumWidth(LEFT_PANEL_WIDTH)
        self.photo_list.setMaximumWidth(LEFT_PANEL_WIDTH)

        self.photo_list.setSizeAdjustPolicy(
            QAbstractScrollArea.SizeAdjustPolicy.AdjustIgnored
        )
        self.photo_list.setWordWrap(False)
        self.photo_list.setTextElideMode(Qt.TextElideMode.ElideNone)
        self.photo_list.setUniformItemSizes(True)

        self.photo_list.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOn
        )
        self.photo_list.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOn
        )
        self.photo_list.setHorizontalScrollMode(
            QAbstractItemView.ScrollMode.ScrollPerPixel
        )

        lv.addWidget(self.photo_list)
        split.addWidget(left)

        # ----------------- Right panel: editor -----------------
        right = QWidget()
        rv = QVBoxLayout(right)

        viewer_editor_split = QSplitter(Qt.Orientation.Vertical)
        rv.addWidget(viewer_editor_split)

        # Placeholder instead of the viewer; real viewer is in a separate window
        placeholder = QLabel(
            "Image viewer is open in a separate window.\n"
            "Use the Photos list on the left to select an image."
        )
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder.setStyleSheet("color:#aaa;")
        viewer_editor_split.addWidget(placeholder)

        # --- Editor tabs ---
        self.editor_widget = QTabWidget()
        viewer_editor_split.addWidget(self.editor_widget)
        viewer_editor_split.setSizes([200, 640])

        # =====================================================
        # Transform / Save tab
        # =====================================================
        t1 = QWidget()
        t1l = QVBoxLayout(t1)
        t1l.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Rotation group
        rotate_group = QGroupBox("Rotation")
        rgl = QHBoxLayout(rotate_group)
        self.rotate_left_button = QPushButton("Rotate Left (90)")
        self.rotate_left_button.clicked.connect(
            lambda: self.rotate_image(-90)
        )
        self.rotate_right_button = QPushButton("Rotate Right (90)")
        self.rotate_right_button.clicked.connect(
            lambda: self.rotate_image(90)
        )
        rgl.addWidget(self.rotate_left_button)
        rgl.addWidget(self.rotate_right_button)
        t1l.addWidget(rotate_group)

        # Crop / Mirror group
        crop_group = QGroupBox("Crop / Mirror")
        cgl = QHBoxLayout(crop_group)

        self.crop_mode_button = QPushButton("Enable Crop Mode")
        self.crop_mode_button.setCheckable(True)
        self.crop_mode_button.toggled.connect(self._toggle_crop_mode)

        self.apply_crop_button = QPushButton("Apply Crop")
        self.apply_crop_button.setEnabled(False)
        self.apply_crop_button.clicked.connect(self.apply_crop)

        self.mirror_mode_button = QPushButton("Mirror View")
        self.mirror_mode_button.setCheckable(True)
        self.mirror_mode_button.toggled.connect(self._toggle_mirror_mode)

        cgl.addWidget(self.crop_mode_button)
        cgl.addWidget(self.apply_crop_button)
        cgl.addWidget(self.mirror_mode_button)
        t1l.addWidget(crop_group)

        # Save / Reset group
        save_group = QGroupBox("Save & Reset")
        sgl = QHBoxLayout(save_group)

        self.save_button = QPushButton("Save (Overwrite)")
        self.save_button.clicked.connect(self.save_image)

        self.save_as_button = QPushButton("Save As...")
        self.save_as_button.clicked.connect(self.save_image_as)

        self.reset_button = QPushButton("Reset All")
        self.reset_button.clicked.connect(self.reset_edits)

        sgl.addWidget(self.save_button)
        sgl.addWidget(self.save_as_button)
        sgl.addWidget(self.reset_button)
        t1l.addWidget(save_group)

        # Metadata group
        meta_group = QGroupBox("Metadata")
        mgl = QGridLayout(meta_group)

        self.meta_taken_lbl = QLabel("Taken: -")
        self.meta_modified_lbl = QLabel("Modified: -")
        self.meta_note_edit = QLineEdit()
        self.meta_note_edit.setPlaceholderText(
            "Custom note (JPEG/TIFF via EXIF ImageDescription/UserComment)"
        )
        self.meta_write_btn = QPushButton("Write Note to File")
        self.meta_write_btn.clicked.connect(self._write_custom_note_to_file)

        if _PIEXIF is None:
            self.meta_note_edit.setEnabled(False)
            self.meta_write_btn.setEnabled(False)
            self.meta_write_btn.setToolTip(
                "piexif not installed. pip install piexif"
            )

        mgl.addWidget(self.meta_taken_lbl, 0, 0, 1, 2)
        mgl.addWidget(self.meta_modified_lbl, 1, 0, 1, 2)
        mgl.addWidget(self.meta_note_edit, 2, 0, 1, 1)
        mgl.addWidget(self.meta_write_btn, 2, 1, 1, 1)

        t1l.addWidget(meta_group)

        self.editor_widget.addTab(t1, "Transform / Save")

        # =====================================================
        # Tone / Light tab
        # =====================================================
        t2 = QWidget()
        t2l = QVBoxLayout(t2)
        t2l.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.exposure_slider = self._create_slider(
            t2l,
            "Exposure (Stops)",
            -2.0,
            2.0,
            0.01,
            "exposure_factor",
        )
        self.brilliance_slider = self._create_slider(
            t2l,
            "Brilliance",
            -100.0,
            100.0,
            1.0,
            "brilliance_factor",
        )
        self.brightness_slider = self._create_slider(
            t2l,
            "Brightness",
            -100.0,
            100.0,
            1.0,
            "brightness_factor",
        )
        self.contrast_slider = self._create_slider(
            t2l,
            "Contrast",
            -100.0,
            100.0,
            1.0,
            "contrast_factor",
        )
        self.highlights_slider = self._create_slider(
            t2l,
            "Highlights",
            -100.0,
            100.0,
            1.0,
            "highlights_factor",
        )
        self.shadows_slider = self._create_slider(
            t2l,
            "Shadows",
            -100.0,
            100.0,
            1.0,
            "shadows_factor",
        )
        self.blackpoint_slider = self._create_slider(
            t2l,
            "Blackpoint",
            -100.0,
            100.0,
            1.0,
            "blackpoint_factor",
        )

        self.editor_widget.addTab(t2, "Tone / Light")

        # =====================================================
        # Color / Effects tab
        # =====================================================
        t3 = QWidget()
        t3l = QVBoxLayout(t3)
        t3l.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.saturation_slider = self._create_slider(
            t3l,
            "Saturation",
            -100.0,
            100.0,
            1.0,
            "saturation_factor",
        )
        self.vibrance_slider = self._create_slider(
            t3l,
            "Vibrance",
            -100.0,
            100.0,
            1.0,
            "vibrance_factor",
        )
        self.warmth_slider = self._create_slider(
            t3l,
            "Warmth (Temp)",
            -100.0,
            100.0,
            1.0,
            "warmth_factor",
        )
        self.tint_slider = self._create_slider(
            t3l,
            "Tint (G/M)",
            -100.0,
            100.0,
            1.0,
            "tint_factor",
        )
        self.sharpness_slider = self._create_slider(
            t3l,
            "Sharpness",
            -100.0,
            100.0,
            1.0,
            "sharpness_factor",
        )
        self.noise_reduction_slider = self._create_slider(
            t3l,
            "Noise Reduction",
            0.0,
            10.0,
            0.1,
            "noise_reduction_factor",
        )
        self.vignette_slider = self._create_slider(
            t3l,
            "Vignette",
            0.0,
            10.0,
            0.01,
            "vignette_factor",
        )

        self.editor_widget.addTab(t3, "Color / Effects")

        # =====================================================
        # Geometry / Straighten tab
        # =====================================================
        tg = QWidget()
        tgl = QVBoxLayout(tg)
        tgl.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.geom_rx_slider = self._create_slider(
            tgl,
            "Pitch (X) deg",
            -30.0,
            30.0,
            0.1,
            "geom_rx_deg",
        )
        self.geom_ry_slider = self._create_slider(
            tgl,
            "Yaw (Y) deg",
            -30.0,
            30.0,
            0.1,
            "geom_ry_deg",
        )
        self.geom_rz_slider = self._create_slider(
            tgl,
            "Roll (Z) deg",
            -45.0,
            45.0,
            0.1,
            "geom_rz_deg",
        )

        self.editor_widget.addTab(tg, "Geometry / Straighten")
        self._geom_layout = tgl

        # Fill mode radios
        fill_group = QGroupBox("Geometry Fill Mode")
        fgl = QHBoxLayout(fill_group)

        self.fill_keep_rb = QRadioButton("Keep borders")
        self.fill_crop_rb = QRadioButton("Auto crop")
        self.fill_stretch_rb = QRadioButton("Stretch to fill")

        self.fill_group = QButtonGroup(self)
        self.fill_group.addButton(self.fill_keep_rb)
        self.fill_group.addButton(self.fill_crop_rb)
        self.fill_group.addButton(self.fill_stretch_rb)

        self.fill_stretch_rb.setChecked(True)

        fgl.addWidget(self.fill_keep_rb)
        fgl.addWidget(self.fill_crop_rb)
        fgl.addWidget(self.fill_stretch_rb)

        self.fill_keep_rb.toggled.connect(
            lambda _: self._on_fill_mode_changed()
        )
        self.fill_crop_rb.toggled.connect(
            lambda _: self._on_fill_mode_changed()
        )
        self.fill_stretch_rb.toggled.connect(
            lambda _: self._on_fill_mode_changed()
        )

        self._geom_layout.addWidget(fill_group)

        # Geometry commit / crop controls
        geom_crop_group = QGroupBox("Geometry Crop")
        gcl = QHBoxLayout(geom_crop_group)

        self.geom_auto_crop_btn = QPushButton("Auto crop (valid area)")
        self.geom_auto_crop_btn.setToolTip(
            "Apply perspective and crop to the valid area (mask bounding box)."
        )
        self.geom_auto_crop_btn.clicked.connect(self._on_geom_auto_crop)

        self.geom_commit_keep_btn = QPushButton(
            "Commit geometry (keep borders)"
        )
        self.geom_commit_keep_btn.setToolTip(
            "Bake current pitch/yaw/roll into the image, keeping replicated borders."
        )
        self.geom_commit_keep_btn.clicked.connect(
            lambda: self._on_geom_commit(FILL_KEEP)
        )

        self.geom_commit_stretch_btn = QPushButton(
            "Commit geometry (stretch to fill)"
        )
        self.geom_commit_stretch_btn.setToolTip(
            "Bake current pitch/yaw/roll and stretch the valid area back to the canvas."
        )
        self.geom_commit_stretch_btn.clicked.connect(
            lambda: self._on_geom_commit(FILL_STRETCH)
        )

        gcl.addWidget(self.geom_auto_crop_btn)
        gcl.addWidget(self.geom_commit_keep_btn)
        gcl.addWidget(self.geom_commit_stretch_btn)

        self._geom_layout.addWidget(geom_crop_group)

        # Crop mode radios (rect / freeform) - add into crop_group layout
        self.crop_rect_rb = QRadioButton("Rect")
        self.crop_free_rb = QRadioButton("Freeform (4 corners)")

        self.crop_mode_group = QButtonGroup(self)
        self.crop_mode_group.addButton(self.crop_rect_rb)
        self.crop_mode_group.addButton(self.crop_free_rb)

        self.crop_rect_rb.setChecked(True)
        self.crop_rect_rb.toggled.connect(
            lambda checked: self._on_crop_mode_changed(
                "rect" if checked else "free"
            )
        )

        cgl.addWidget(self.crop_rect_rb)
        cgl.addWidget(self.crop_free_rb)

        self.crop_mode_type = "rect"

        # =====================================================
        # Filters tab
        # =====================================================
        tf = QWidget()
        tfl = QVBoxLayout(tf)
        tfl.setAlignment(Qt.AlignmentFlag.AlignTop)

        rb_container = QWidget()
        rb_layout = QGridLayout(rb_container)
        rb_layout.setContentsMargins(0, 0, 0, 0)
        rb_layout.setHorizontalSpacing(12)
        rb_layout.setVerticalSpacing(4)

        self.filter_buttons: Dict[str, QRadioButton] = {}
        self.filter_group_rb = QButtonGroup(self)
        self.filter_group_rb.setExclusive(True)

        def add_filter_rb(name: str, row: int, col: int):
            rb = QRadioButton(name)
            if name == "None":
                rb.setChecked(True)
            self.filter_buttons[name] = rb
            self.filter_group_rb.addButton(rb)
            rb_layout.addWidget(rb, row, col)
            rb.toggled.connect(
                lambda checked, n=name: self._on_filter_radio(n, checked)
            )

        all_names = [n for n in ALL_FILTERS if n != "—"]
        cols = 3
        for i, name in enumerate(all_names):
            r = i // cols
            c = i % cols
            add_filter_rb(name, r, c)

        self.filter_strength_slider = SliderSpinBox(
            "Intensity", 0.0, 1.0, 1.0, 0.01
        )

        rb_scroll = QScrollArea()
        rb_scroll.setWidgetResizable(True)
        rb_scroll.setWidget(rb_container)

        tfl.addWidget(rb_scroll, 1)
        tfl.addWidget(self.filter_strength_slider, 0)

        self.filter_strength_slider.valueChanged.connect(
            self._on_filter_strength
        )

        self.editor_widget.addTab(tf, "Filters")

        # ----------------- finalize splitter -----------------
        split.addWidget(right)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setCollapsible(0, True)
        left.setMinimumWidth(0)
        self._photos_splitter = split
        self._photos_splitter.setCollapsible(0, False)
        self._left_auto_collapse_px = 200

    def _show_duplicate_group_thumbnails(
        self,
        group: List[DuplicateRecord],
    ) -> None:
        self._clear_duplicate_thumbnails()

        max_cols = 4
        row = 0
        col = 0
        thumb_size = 180

        for rec in group:
            pm = QPixmap(rec.path)
            if pm.isNull():
                continue
            pm = pm.scaled(
                thumb_size,
                thumb_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )

            img_label = QLabel()
            img_label.setPixmap(pm)
            img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            img_label.setToolTip(rec.path)

            def _make_click_handler(p: str):
                def handler(event):
                    self._on_duplicate_thumbnail_clicked(p)

                return handler

            img_label.mousePressEvent = _make_click_handler(rec.path)

            text_label = QLabel(Path(rec.path).name)
            text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            text_label.setStyleSheet("color:#ccc;")

            container = QWidget()
            v = QVBoxLayout(container)
            v.setContentsMargins(0, 0, 0, 0)
            v.setSpacing(4)
            v.addWidget(img_label)
            v.addWidget(text_label)

            self.dup_thumbs_layout.addWidget(container, row, col)

            col += 1
            if col >= max_cols:
                col = 0
                row += 1

    def _on_duplicate_thumbnail_clicked(self, path: str) -> None:
        """
        When the user clicks a duplicate thumbnail:
        - Switch to the Photos tab
        - Select that image in the list
        """
        try:
            idx = self.image_files.index(path)
        except ValueError:
            QMessageBox.information(
                self,
                "Duplicates",
                "This image is not in the current Photos list.",
            )
            return

        self.tabs.setCurrentWidget(self.photos_tab)
        self.photo_list.setCurrentRow(idx)
        # _on_photo_select will handle loading it

    def _build_live_tab(self) -> QWidget:
        tab = QWidget()
        root = QHBoxLayout(tab)
        root.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        left_panel = QWidget()
        left_v = QVBoxLayout(left_panel)
        left_v.setContentsMargins(0, 0, 0, 0)
        left_v.setSpacing(6)

        # --- Sort controls for videos ---
        live_sort_row = QHBoxLayout()
        live_sort_row.setContentsMargins(0, 0, 0, 0)
        live_sort_row.setSpacing(6)

        live_sort_label = QLabel("Sort videos by:")
        self.live_sort_combo = QComboBox()
        self.live_sort_combo.addItems(
            ["Name", "Date taken", "Date modified", "Date created"]
        )

        self.live_sort_order_combo = QComboBox()
        self.live_sort_order_combo.addItems(["Ascending", "Descending"])

        live_sort_row.addWidget(live_sort_label)
        live_sort_row.addWidget(self.live_sort_combo)
        live_sort_row.addWidget(self.live_sort_order_combo)
        live_sort_row.addStretch(1)

        left_v.addLayout(live_sort_row)


        self.live_list = QListWidget()
        self.live_list.setMinimumWidth(200)
        self.live_list.setMaximumWidth(420)
        self.live_list.setSizePolicy(
            QSizePolicy.Policy.Fixed,
            QSizePolicy.Policy.Expanding,
        )
        self.live_list.itemSelectionChanged.connect(
            lambda: self._on_live_select(
                self.live_list.currentItem(), None
            )
        )
        left_v.addWidget(self.live_list)

        # Apply sorting whenever user changes mode/order
        self.live_sort_combo.currentIndexChanged.connect(
            lambda _: self._apply_live_sort()
        )
        self.live_sort_order_combo.currentIndexChanged.connect(
            lambda _: self._apply_live_sort()
        )


        self.frames_list = QListWidget()
        self.frames_list.setMinimumWidth(200)
        self.frames_list.setMaximumWidth(420)
        self.frames_list.setSizePolicy(
            QSizePolicy.Policy.Fixed,
            QSizePolicy.Policy.Expanding,
        )
        self.frames_list.itemSelectionChanged.connect(
            lambda: self._on_frame_selected(
                self.frames_list.currentItem(), None
            )
        )
        left_v.addWidget(self.frames_list)

        splitter.addWidget(left_panel)

        right_panel = QWidget()
        right_v = QVBoxLayout(right_panel)
        right_v.setContentsMargins(0, 0, 0, 0)
        right_v.setSpacing(6)

        self.player_label = QLabel("No .mov loaded")
        self.player_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.player_label.setMinimumSize(320, 240)
        self.player_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.player_label.setStyleSheet(
            "background:#000; color:#aaa;"
        )
        right_v.addWidget(self.player_label, 1)

        controls_row = QWidget()
        controls_h = QHBoxLayout(controls_row)
        controls_h.setContentsMargins(0, 0, 0, 0)
        controls_h.setSpacing(8)

        self.play_button = QPushButton("Play")
        self.play_button.setSizePolicy(
            QSizePolicy.Policy.Fixed,
            QSizePolicy.Policy.Fixed,
        )
        self.play_button.setMaximumWidth(110)
        self.play_button.setEnabled(False)
        self.play_button.clicked.connect(self._toggle_play_pause)

        self.save_frame_button = QPushButton("Save Selected Frame")
        self.save_frame_button.setEnabled(False)
        self.save_frame_button.clicked.connect(self._save_selected_frame)

        controls_h.addWidget(self.play_button)
        controls_h.addWidget(self.save_frame_button)
        controls_h.addStretch(1)
        right_v.addWidget(controls_row, 0)

        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([300, 1200])

        return tab

    # ---------- Drag & drop ----------

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                p = Path(url.toLocalFile())
                if p.exists() and p.is_dir():
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            p = Path(url.toLocalFile())
            if p.exists() and p.is_dir():
                self.path_edit.setText(str(p))
                self.load_directory_async(p)
                break

    # ---------- Folder load / scan ----------

    def _open_path_from_edit(self):
        p = self.path_edit.text().strip()
        if not p:
            return
        folder = Path(p)
        if not folder.exists() or not folder.is_dir():
            QMessageBox.warning(
                self, "Folder", f"Folder not found:\n{folder}"
            )
            return
        self.load_directory_async(folder)

    def _browse_and_load_folder(self):
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Select a folder",
            QDir.homePath(),
            options=QFileDialog.Option.ShowDirsOnly,
        )
        if not folder_path:
            return
        self.path_edit.setText(folder_path)
        self.load_directory_async(Path(folder_path))

    def _image_date_taken(self, path: Path) -> Optional[str]:
        try:
            if (
                path.suffix.lower() in HEIF_LIKE_EXTS
                and not _HEIF_PLUGIN
            ):
                return None
            with Image.open(str(path)) as im:
                exif = getattr(im, "getexif", None)
                if not exif:
                    return None
                ex = exif()
                if not ex:
                    return None
                dto = ex.get(36867) or ex.get(306)
                return _sanitize_exif_datetime(dto)
        except Exception:
            return None

    def _mov_date_taken(self, path: Path) -> Optional[str]:
        if _MEDIAINFO is None:
            return None
        try:
            info = _MEDIAINFO.parse(str(path))
            for track in info.tracks:
                if track.track_type == "General":
                    cand = getattr(track, "encoded_date", None) or getattr(
                        track, "tagged_date", None
                    )
                    if not cand:
                        continue
                    s = str(cand).replace("UTC ", "").strip()
                    try:
                        dt = datetime.strptime(
                            s[:19], "%Y-%m-%d %H:%M:%S"
                        )
                        return dt.strftime("%Y-%m-%d %H:%M:%S")
                    except Exception:
                        return s
            return None
        except Exception:
            return None

    def _format_list_item_with_meta(
        self,
        p: Path,
        date_taken: Optional[str],
    ) -> str:
        date_mod = _fmt_ts_local(os.path.getmtime(p))
        taken = date_taken if date_taken else "-"
        return f"{p.name} | taken: {taken} | modified: {date_mod}"

    def load_directory_async(self, folder: Path):
        self._scan_job_id += 1
        jid = self._scan_job_id

        self.scan_status_lbl.setText(f"Scanning {folder} ...")
        self.photo_list.clear()
        self.live_list.clear()
        self.frames_list.clear()
        self.image_files = []
        self.mov_files = []
        self._clear_photos_state(reset_lists_only=True)

        self.image_meta = {}
        self.mov_meta = {}

        job = DirScanJob(jid, folder, self)
        job.signals.started.connect(self._on_scan_started)
        job.signals.found_image.connect(self._on_scan_found_image)
        job.signals.found_mov.connect(self._on_scan_found_mov)
        job.signals.finished.connect(self._on_scan_finished)
        job.signals.error.connect(self._on_scan_error)
        self.threadpool.start(job, 0)

    def _on_scan_started(self, job_id: int, folder: str):
        if job_id != self._scan_job_id:
            return
        self.current_folder = Path(folder)

    def _on_scan_found_image(
        self,
        job_id: int,
        path: str,
        taken: str,
    ):
        if job_id != self._scan_job_id:
            return
        p = Path(path)
        self.image_files.append(path)

        # Cache metadata for sorting
        try:
            date_taken = taken or "-"
        except Exception:
            date_taken = "-"

        try:
            date_modified = _fmt_ts_local(os.path.getmtime(p))
        except Exception:
            date_modified = "-"

        try:
            date_created = _fmt_ts_local(os.path.getctime(p))
        except Exception:
            date_created = "-"

        self.image_meta[path] = {
            "taken": date_taken,
            "modified": date_modified,
            "created": date_created,
        }

        item_text = self._format_list_item_with_meta(p, date_taken)
        self.photo_list.addItem(item_text)
        self.photo_list.item(
            self.photo_list.count() - 1
        ).setToolTip(str(p))
        if (
            self.current_image_index < 0
            and len(self.image_files) == 1
        ):
            self._load_image_by_index(0)
            self.photo_list.setCurrentRow(0)
            self.set_controls_state(True)

        # Reset duplicates state when loading a new folder
        self.duplicate_groups = []
        if hasattr(self, "dup_groups_list"):
            self.dup_groups_list.clear()
        if hasattr(self, "dup_thumbs_layout"):
            self._clear_duplicate_thumbnails()
        if hasattr(self, "dup_status_lbl"):
            self.dup_status_lbl.setText(
                "Load a folder in Photos tab, then scan."
            )

    def _on_scan_found_mov(
        self,
        job_id: int,
        path: str,
        taken: str,
    ):
        if job_id != self._scan_job_id:
            return
        p = Path(path)
        self.mov_files.append(path)

        # Cache metadata for sorting
        try:
            date_taken = taken or self._mov_date_taken(p) or "-"
        except Exception:
            date_taken = "-"

        try:
            date_modified = _fmt_ts_local(os.path.getmtime(p))
        except Exception:
            date_modified = "-"

        try:
            date_created = _fmt_ts_local(os.path.getctime(p))
        except Exception:
            date_created = "-"

        self.mov_meta[path] = {
            "taken": date_taken,
            "modified": date_modified,
            "created": date_created,
        }

    def _on_scan_finished(
        self,
        job_id: int,
        images: List[str],
        movs: List[str],
    ):
        if job_id != self._scan_job_id:
            return
        self.scan_status_lbl.setText(
            f"Loaded {len(images)} images, {len(movs)} videos from {self.current_folder}"
        )

        # Build Live (.mov) list from cached metadata
        self.live_list.clear()
        if self.mov_files:
            for s in self.mov_files:
                p = Path(s)
                meta = self.mov_meta.get(s, {})
                taken = meta.get("taken") or self._mov_date_taken(p)
                item_text = self._format_list_item_with_meta(p, taken)
                self.live_list.addItem(item_text)
                self.live_list.item(
                    self.live_list.count() - 1
                ).setToolTip(str(p))

        # Apply current sort preferences after full lists are loaded
        if hasattr(self, "photo_sort_combo"):
            self._apply_photo_sort()
        if hasattr(self, "live_sort_combo"):
            self._apply_live_sort()

    def _on_scan_error(self, job_id: int, msg: str):
        if job_id != self._scan_job_id:
            return
        QMessageBox.critical(self, "Scan Error", msg)
        self.scan_status_lbl.setText("Scan error")

    def _on_photo_select(self, current, previous):
        if not self.image_files or not current:
            return
        index = self.photo_list.row(current)
        if index != self.current_image_index:
            self._load_image_by_index(index)

    def _load_image_by_index(self, index: int):
        if not (0 <= index < len(self.image_files)):
            return
        self.current_image_index = index
        self._load_image(self.image_files[index])

    def _ensure_heif_plugin_for_path(
        self,
        path: str,
        when: str,
    ) -> bool:
        ext = Path(path).suffix.lower()
        if ext in HEIF_LIKE_EXTS and not _HEIF_PLUGIN:
            QMessageBox.critical(
                self,
                "HEIF/AVIF support missing",
                f"You attempted to {when} a {ext} file but pillow-heif is not installed.\n\n"
                f"pip install pillow-heif\n"
                f"(On some systems, also install libheif.)",
            )
            return False
        return True

    def _infer_format_from_path(self, path: str) -> Optional[str]:
        return EXT_TO_FMT.get(Path(path).suffix.lower())

    def _update_meta_labels_for_image(self, path: Path):
        taken = self._image_date_taken(path) or "-"
        modified = _fmt_ts_local(os.path.getmtime(path))
        self.meta_taken_lbl.setText(f"Taken: {taken}")
        self.meta_modified_lbl.setText(f"Modified: {modified}")

        fmt = (
            (self._orig_format or self._infer_format_from_path(str(path)))
            or ""
        ).upper()
        writable = (
            _PIEXIF is not None and fmt in ("JPEG", "TIFF")
        )
        self.meta_note_edit.setEnabled(writable)
        self.meta_write_btn.setEnabled(writable)
        if not writable:
            if _PIEXIF is None:
                self.meta_write_btn.setToolTip(
                    "piexif not installed. pip install piexif"
                )
            else:
                self.meta_write_btn.setToolTip(
                    "Custom note supported only for JPEG/TIFF via EXIF"
                )

    def _load_image(self, filepath: str):
        self.original_image_path = None
        self.original_image_pil = None
        self.working_image_pil = None
        self._orig_format = None
        self._orig_exif = None
        self.crop_area = None
        self._reset_controls(reset_rotation=True, reset_sliders=True)
        if self.single_viewer is not None:
            self.single_viewer.set_editor_mode(MODE_VIEW)
        self.crop_mode_button.setChecked(False)
        self.mirror_mode_button.setChecked(False)
        self._mirror_on = False

        if not os.path.exists(filepath):
            return
        if not self._ensure_heif_plugin_for_path(filepath, "open"):
            self.set_controls_state(False)
            if self.single_viewer is not None:
                self.single_viewer.clear_pixmap()
            return

        try:
            with Image.open(filepath) as im:
                self._orig_format = (
                    im.format or ""
                ) or self._infer_format_from_path(filepath)
                self._orig_exif = im.info.get("exif")
                im = ImageOps.exif_transpose(im)
                self.original_image_pil = im.copy()

            self.original_image_path = filepath
            self.working_image_pil = self.original_image_pil.copy()
            self._preview_source = self._make_preview_source(
                self.working_image_pil
            )
            self.set_controls_state(True)
            self._update_meta_labels_for_image(Path(filepath))

            # Make sure viewer window is visible when loading an image
            if self.image_window is not None:
                self.image_window.showNormal()
                self.image_window.raise_()
                self.image_window.activateWindow()

            self._schedule_preview()
        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Could not process image: {e}"
            )
            self.original_image_path = None
            if self.single_viewer is not None:
                self.single_viewer.setText(f"Error loading image: {e}")
            self.set_controls_state(False)

    def _clear_photos_state(self, reset_lists_only: bool = False):
        if not reset_lists_only:
            self.original_image_path = None
            self.original_image_pil = None
            self.working_image_pil = None
            self._orig_format = None
            self._orig_exif = None
            self.crop_area = None
            self.current_image_index = -1
            if self.single_viewer is not None:
                self.single_viewer.clear_pixmap()
                self.single_viewer.set_editor_mode(MODE_VIEW)
            self.crop_mode_button.setChecked(False)
            self.mirror_mode_button.setChecked(False)
            self._mirror_on = False
            self._reset_controls(reset_rotation=True, reset_sliders=True)
            self.meta_taken_lbl.setText("Taken: -")
            self.meta_modified_lbl.setText("Modified: -")

    # ---------- Live (.mov) ----------

    def _on_live_select(self, current, previous):
        if not self.mov_files or not current:
            return
        idx = self.live_list.row(current)
        if idx != self.current_mov_index:
            self._select_live_by_index(idx)

    def _select_live_by_index(self, idx: int):
        if not (0 <= idx < len(self.mov_files)):
            return
        self.current_mov_index = idx
        self._open_video(self.mov_files[idx])

    def _open_video(self, path: str):
        self._stop_video()
        self.frames_list.clear()
        self.save_frame_button.setEnabled(False)
        self.player_label.setText("Loading...")
        self.player_label.setPixmap(QPixmap())

        if _CV2 is None:
            QMessageBox.warning(
                self,
                "Live",
                "opencv-python is required for .mov playback.",
            )
            return
        try:
            cap = _CV2.VideoCapture(path)
            if not cap.isOpened():
                raise IOError("Failed to open video file")
            self.cap = cap
            self.video_fps = cap.get(_CV2.CAP_PROP_FPS) or 30.0
            self.total_frames = int(
                cap.get(_CV2.CAP_PROP_FRAME_COUNT) or 0
            )
            self.current_frame_idx = 0

            for i in range(self.total_frames):
                self.frames_list.addItem(f"Frame {i:04d}")

            self.play_button.setEnabled(True)
            self.play_button.setText("Play")
            self.is_playing = False

            self._seek_and_show_frame(0)
            if self.total_frames > 0:
                self.frames_list.setCurrentRow(0)
                self.save_frame_button.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(
                self, "Live", f"Could not process video: {e}"
            )
            self._clear_live_state()

    def _toggle_play_pause(self):
        if not self.cap:
            return
        if self.is_playing:
            self._stop_video()
        else:
            self._start_video()

    def _start_video(self):
        if not self.cap:
            return
        self.is_playing = True
        self.play_button.setText("Pause")
        interval = max(
            10,
            int(1000 / max(1.0, min(60.0, self.video_fps))),
        )
        self.live_timer.start(interval)

    def _stop_video(self):
        self.is_playing = False
        self.live_timer.stop()
        if hasattr(self, "play_button") and self.play_button:
            self.play_button.setText("Play")

    def _advance_video_frame(self):
        if not self.cap:
            return
        ok, frame_bgr = self.cap.read()
        if not ok:
            self.cap.set(_CV2.CAP_PROP_POS_FRAMES, 0)
            ok, frame_bgr = self.cap.read()
            if not ok:
                return
            self.current_frame_idx = 0
        else:
            self.current_frame_idx = int(
                self.cap.get(_CV2.CAP_PROP_POS_FRAMES)
            ) - 1

        self._show_frame_on_label(frame_bgr, self.player_label)

        if 0 <= self.current_frame_idx < self.frames_list.count():
            self.frames_list.blockSignals(True)
            self.frames_list.setCurrentRow(self.current_frame_idx)
            self.frames_list.blockSignals(False)

    def _on_frame_selected(self, current, previous):
        if not current or self.cap is None:
            self.save_frame_button.setEnabled(False)
            return
        self.save_frame_button.setEnabled(True)
        idx = self.frames_list.currentRow()
        self._seek_and_show_frame(idx)

    def _seek_and_show_frame(self, idx: int):
        if not self.cap:
            return
        idx = max(0, min(idx, self.total_frames - 1))
        self.cap.set(_CV2.CAP_PROP_POS_FRAMES, idx)
        ok, frame_bgr = self.cap.read()
        if not ok:
            return
        self.current_frame_idx = idx
        self._show_frame_on_label(frame_bgr, self.player_label)

    def _show_frame_on_label(self, frame_bgr, label: QLabel):
        try:
            frame_rgb = _CV2.cvtColor(frame_bgr, _CV2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(
                frame_rgb.data,
                w,
                h,
                bytes_per_line,
                QImage.Format.Format_RGB888,
            )
            pm = QPixmap.fromImage(qimg)
            target = label.size()
            if target.width() > 0 and target.height() > 0:
                pm = pm.scaled(
                    target,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            label.setPixmap(pm)
            label.setText("")
        except Exception as e:
            label.setText(f"Frame error: {e}")

    def _save_selected_frame(self):
        """
        Save the currently selected frame from the current .mov into a folder
        named after the .mov file.

        Example:
            /videos/clip.mov
                -> /videos/clip/clip_frame_0005.jpg   (when frame index = 5)
        """
        if self.cap is None:
            return

        current = self.frames_list.currentRow()
        if current < 0:
            QMessageBox.warning(
                self, "Save Frame", "Select a frame first."
            )
            return
        if not (0 <= self.current_mov_index < len(self.mov_files)):
            return

        mov_path = Path(self.mov_files[self.current_mov_index])

        # Folder: same parent as .mov, name = mov stem
        out_dir = mov_path.parent / mov_path.stem
        try:
            out_dir.mkdir(exist_ok=True)
        except Exception as e:
            QMessageBox.critical(
                self,
                "Save Frame",
                f"Could not create output folder:\n{out_dir}\n\nError: {e}",
            )
            return

        # File name: <mov_stem>_frame_XXXX.jpg
        frame_name = f"{mov_path.stem}_frame_{current:04d}.jpg"
        out_path = out_dir / frame_name

        try:
            # Seek to the requested frame and read it
            self.cap.set(_CV2.CAP_PROP_POS_FRAMES, current)
            ok, frame_bgr = self.cap.read()
            if not ok:
                QMessageBox.warning(
                    self, "Save Frame", "Could not read frame."
                )
                return

            frame_rgb = _CV2.cvtColor(frame_bgr, _CV2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)

            # Always save as JPEG here (simple, predictable)
            img.save(str(out_path), "JPEG", quality=90)

            QMessageBox.information(
                self,
                "Save Frame",
                f"Saved frame to:\n{out_path}",
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Save Frame",
                f"Error while saving frame:\n{e}",
            )

    def _clear_live_state(self):
        self._stop_video()
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = None
        self.mov_files = []
        self.current_mov_index = -1
        if hasattr(self, "live_list"):
            self.live_list.clear()
        if hasattr(self, "frames_list"):
            self.frames_list.clear()
        self.player_label.setText("No .mov loaded")
        self.player_label.setPixmap(QPixmap())
        self.play_button.setEnabled(False)
        self.save_frame_button.setEnabled(False)

    # ---------- Photo controls ----------

    # ---------- Sorting helpers ----------

    def _apply_photo_sort(self):
        """
        Sort self.image_files and rebuild the Photos list according to
        the selected sort mode and order.
        """
        if not self.image_files:
            return

        # Remember currently selected path
        current_path = None
        if 0 <= self.current_image_index < len(self.image_files):
            current_path = self.image_files[self.current_image_index]

        mode = (
            self.photo_sort_combo.currentText()
            if hasattr(self, "photo_sort_combo")
            else "Name"
        )
        order = (
            self.photo_sort_order_combo.currentText()
            if hasattr(self, "photo_sort_order_combo")
            else "Ascending"
        )
        reverse = (order == "Descending")

        def sort_key(path: str):
            meta = self.image_meta.get(path, {})
            if mode == "Name":
                return Path(path).name.lower()
            elif mode == "Date taken":
                return meta.get("taken") or ""
            elif mode == "Date modified":
                return meta.get("modified") or ""
            elif mode == "Date created":
                return meta.get("created") or ""
            else:
                return Path(path).name.lower()

        # Reorder the internal list
        self.image_files.sort(key=sort_key, reverse=reverse)

        # Rebuild the UI list
        self.photo_list.blockSignals(True)
        self.photo_list.clear()
        for path in self.image_files:
            p = Path(path)
            meta = self.image_meta.get(path, {})
            taken_str = meta.get("taken")
            item_text = self._format_list_item_with_meta(p, taken_str)
            self.photo_list.addItem(item_text)
            self.photo_list.item(
                self.photo_list.count() - 1
            ).setToolTip(str(p))
        self.photo_list.blockSignals(False)

        # Restore selection if possible
        if current_path and current_path in self.image_files:
            new_idx = self.image_files.index(current_path)
            self.current_image_index = new_idx
            self.photo_list.setCurrentRow(new_idx)
        elif self.image_files:
            self.current_image_index = 0
            self.photo_list.setCurrentRow(0)

    def _apply_live_sort(self):
        """
        Sort self.mov_files and rebuild the Live videos list according to
        the selected sort mode and order.
        """
        if not self.mov_files:
            return

        current_mov_path = None
        if 0 <= self.current_mov_index < len(self.mov_files):
            current_mov_path = self.mov_files[self.current_mov_index]

        mode = (
            self.live_sort_combo.currentText()
            if hasattr(self, "live_sort_combo")
            else "Name"
        )
        order = (
            self.live_sort_order_combo.currentText()
            if hasattr(self, "live_sort_order_combo")
            else "Ascending"
        )
        reverse = (order == "Descending")

        def sort_key(path: str):
            meta = self.mov_meta.get(path, {})
            if mode == "Name":
                return Path(path).name.lower()
            elif mode == "Date taken":
                return meta.get("taken") or ""
            elif mode == "Date modified":
                return meta.get("modified") or ""
            elif mode == "Date created":
                return meta.get("created") or ""
            else:
                return Path(path).name.lower()

        self.mov_files.sort(key=sort_key, reverse=reverse)

        # Rebuild Live list UI
        self.live_list.blockSignals(True)
        self.live_list.clear()
        for s in self.mov_files:
            p = Path(s)
            meta = self.mov_meta.get(s, {})
            taken = meta.get("taken") or self._mov_date_taken(p)
            item_text = self._format_list_item_with_meta(p, taken)
            self.live_list.addItem(item_text)
            self.live_list.item(
                self.live_list.count() - 1
            ).setToolTip(str(p))
        self.live_list.blockSignals(False)

        # Restore selection, reload video if needed
        if current_mov_path and current_mov_path in self.mov_files:
            new_idx = self.mov_files.index(current_mov_path)
            self.current_mov_index = new_idx
            self.live_list.setCurrentRow(new_idx)
            self._select_live_by_index(new_idx)
        elif self.mov_files:
            self.current_mov_index = 0
            self.live_list.setCurrentRow(0)
            self._select_live_by_index(0)


    def _make_preview_source(
        self,
        img: Image.Image,
        max_side: int = 2048,
    ) -> Image.Image:
        w, h = img.size
        m = max(w, h)
        if m <= max_side:
            return (
                img.convert("RGB")
                if img.mode != "RGB"
                else img
            )
        ratio = max_side / float(m)
        new_size = (
            max(1, int(w * ratio)),
            max(1, int(h * ratio)),
        )
        return img.convert("RGB").resize(
            new_size, Image.Resampling.BILINEAR
        )

    def _create_slider(
        self,
        parent_layout: QVBoxLayout,
        label: str,
        min_val: float,
        max_val: float,
        step: float,
        attr_name: str,
    ) -> SliderSpinBox:
        default_val = getattr(self, attr_name)
        s = SliderSpinBox(label, min_val, max_val, default_val, step)
        self._all_sliders.append(s.slider)
        s.valueChanged.connect(
            lambda v, name=attr_name: self._on_slider_change(v, name)
        )
        s.slider.sliderReleased.connect(self._schedule_preview)
        setattr(
            self,
            attr_name.replace("_factor", "_slider").replace(
                "_deg", "_slider"
            ),
            s,
        )
        parent_layout.addWidget(s)
        return s

    def set_controls_state(self, enabled: bool):
        self.rotate_left_button.setEnabled(enabled)
        self.rotate_right_button.setEnabled(enabled)
        self.crop_mode_button.setEnabled(enabled)
        self.mirror_mode_button.setEnabled(enabled)
        self.save_button.setEnabled(enabled)
        self.save_as_button.setEnabled(enabled)
        self.reset_button.setEnabled(enabled)
        self.fill_keep_rb.setEnabled(enabled)
        self.fill_crop_rb.setEnabled(enabled)
        self.fill_stretch_rb.setEnabled(enabled)

        slider_groups = [
            "exposure",
            "brilliance",
            "brightness",
            "contrast",
            "highlights",
            "shadows",
            "blackpoint",
            "saturation",
            "vibrance",
            "warmth",
            "tint",
            "sharpness",
            "noise_reduction",
            "vignette",
        ]
        for attr_name in slider_groups:
            slider = getattr(self, f"{attr_name}_slider", None)
            if slider:
                slider.setEnabled(enabled)

        for name in ("geom_rx", "geom_ry", "geom_rz"):
            slider = getattr(self, f"{name}_slider", None)
            if slider:
                if name in ("geom_rx", "geom_ry") and (
                    _CV2 is None or _NP is None
                ):
                    slider.setEnabled(False)
                else:
                    slider.setEnabled(enabled)

        for rb in self.filter_buttons.values():
            rb.setEnabled(enabled)
        if hasattr(self, "filter_strength_slider"):
            self.filter_strength_slider.setEnabled(enabled)

    def _reset_controls(
        self,
        reset_rotation=True,
        reset_sliders=True,
    ):
        if reset_rotation:
            self.rotation_degrees = 0
            self.geom_rx_deg = 0.0
            self.geom_ry_deg = 0.0
            self.geom_rz_deg = 0.0
            for n in ("geom_rx", "geom_ry", "geom_rz"):
                s = getattr(self, f"{n}_slider", None)
                if s:
                    s.set_value(0.0)

        if reset_sliders:
            for name in [
                "brightness",
                "contrast",
                "exposure",
                "highlights",
                "shadows",
                "blackpoint",
                "saturation",
                "vibrance",
                "warmth",
                "tint",
                "sharpness",
                "noise_reduction",
                "vignette",
                "brilliance",
            ]:
                setattr(self, f"{name}_factor", 0.0)
                slider = getattr(self, f"{name}_slider", None)
                if slider:
                    slider.set_value(0.0)
            self.filter_name = "None"
            self.filter_strength = 1.0
            if "None" in self.filter_buttons:
                self.filter_buttons["None"].setChecked(True)
            if hasattr(self, "filter_strength_slider"):
                self.filter_strength_slider.set_value(1.0)

    def reset_edits(self):
        if not self.original_image_pil:
            return
        self.working_image_pil = self.original_image_pil.copy()
        self._preview_source = self._make_preview_source(
            self.working_image_pil
        )
        self.crop_area = None
        self._reset_controls(reset_rotation=True, reset_sliders=True)
        self._schedule_preview()

    def rotate_image(self, degrees: int):
        if not self.working_image_pil:
            return
        self.rotation_degrees = (self.rotation_degrees + degrees) % 360
        self._schedule_preview()

    def _toggle_crop_mode(self, checked: bool):
        if not self.original_image_path:
            self.crop_mode_button.setChecked(False)
            return
        if self.single_viewer is None:
            self.crop_mode_button.setChecked(False)
            return
        if checked:
            self.single_viewer.set_editor_mode(MODE_CROP)
            self.crop_mode_button.setText("Exit Crop Mode")
        else:
            self.single_viewer.set_editor_mode(MODE_VIEW)
            self.crop_mode_button.setText("Enable Crop Mode")
            self.apply_crop_button.setEnabled(False)

    def _toggle_mirror_mode(self, checked: bool):
        self._mirror_on = bool(checked)
        if self.viewer_stack is not None:
            self.viewer_stack.setCurrentIndex(1 if checked else 0)
        self._schedule_preview()

    def _on_fill_mode_changed(self):
        if self.fill_keep_rb.isChecked():
            self.fill_mode = FILL_KEEP
        elif self.fill_crop_rb.isChecked():
            self.fill_mode = FILL_AUTOCROP
        else:
            self.fill_mode = FILL_STRETCH
        self._schedule_preview()

    def apply_crop(self):
        if (
            not self.original_image_path
            or not self.single_viewer
            or not self.single_viewer.current_qpixmap
        ):
            QMessageBox.warning(
                self, "Crop", "No valid crop area selected."
            )
            return

        # freeform crop
        if getattr(self, "crop_mode_type", "rect") == "free":
            if _CV2 is None or _NP is None:
                QMessageBox.warning(
                    self,
                    "Crop",
                    "Freeform crop requires opencv-python and numpy.",
                )
                return
            if len(self.single_viewer.freeform_pts) != 4:
                QMessageBox.warning(
                    self, "Crop", "Please position all four corners."
                )
                return

            img_rect_can = self.single_viewer.get_image_display_rect()
            qpixmap_size = self.single_viewer.current_qpixmap.size()
            w_disp, h_disp = (
                qpixmap_size.width(),
                qpixmap_size.height(),
            )

            temp_rotated = self.working_image_pil.rotate(
                self.rotation_degrees, expand=True
            )
            w_rot, h_rot = temp_rotated.size
            ratio_w = w_rot / float(w_disp) if w_disp else 1.0
            ratio_h = h_rot / float(h_disp) if h_disp else 1.0

            pts = self.single_viewer.freeform_pts
            pts_img = []
            for pt in pts:
                x_img = pt.x() - img_rect_can.left()
                y_img = pt.y() - img_rect_can.top()
                x_img = max(0, min(x_img, w_disp))
                y_img = max(0, min(y_img, h_disp))
                pts_img.append([x_img * ratio_w, y_img * ratio_h])

            pts_np = _NP.array(pts_img, dtype=_NP.float32)
            s = pts_np.sum(axis=1)
            diff = _NP.diff(pts_np, axis=1).ravel()
            tl = pts_np[_NP.argmin(s)]
            br = pts_np[_NP.argmax(s)]
            tr = pts_np[_NP.argmin(diff)]
            bl = pts_np[_NP.argmax(diff)]
            src = _NP.array([tl, tr, br, bl], dtype=_NP.float32)

            def _dist(a, b):
                return float(_NP.linalg.norm(a - b))

            widthA = _dist(br, bl)
            widthB = _dist(tr, tl)
            maxW = int(round(max(widthA, widthB)))
            heightA = _dist(tr, br)
            heightB = _dist(tl, bl)
            maxH = int(round(max(heightA, heightB)))
            maxW = max(1, maxW)
            maxH = max(1, maxH)

            dst = _NP.array(
                [
                    [0, 0],
                    [maxW - 1, 0],
                    [maxW - 1, maxH - 1],
                    [0, maxH - 1],
                ],
                dtype=_NP.float32,
            )

            img_np = _NP.array(temp_rotated.convert("RGB"))
            M = _CV2.getPerspectiveTransform(src, dst)
            warped = _CV2.warpPerspective(
                img_np,
                M,
                (maxW, maxH),
                flags=_CV2.INTER_CUBIC,
            )
            cropped_image = Image.fromarray(warped)

            self.working_image_pil = cropped_image
            self._preview_source = self._make_preview_source(
                self.working_image_pil
            )
            self.rotation_degrees = 0
            self.geom_rx_deg = 0.0
            self.geom_ry_deg = 0.0
            self.geom_rz_deg = 0.0
            self.crop_mode_button.setChecked(False)
            self.single_viewer.set_editor_mode(MODE_VIEW)
            self.apply_crop_button.setEnabled(False)
            self._schedule_preview()
            return

        # rectangular crop
        if (
            self.single_viewer.crop_start_point
            and self.single_viewer.crop_end_point
        ):
            img_rect_can = self.single_viewer.get_image_display_rect()
            qpixmap_size = self.single_viewer.current_qpixmap.size()
            w_disp, h_disp = (
                qpixmap_size.width(),
                qpixmap_size.height(),
            )

            temp_rotated = self.working_image_pil.rotate(
                self.rotation_degrees, expand=True
            )
            w_rot, h_rot = temp_rotated.size
            ratio_w = w_rot / float(w_disp) if w_disp else 1.0
            ratio_h = h_rot / float(h_disp) if h_disp else 1.0

            x1 = max(
                self.single_viewer.crop_start_point.x(),
                img_rect_can.left(),
            )
            y1 = max(
                self.single_viewer.crop_start_point.y(),
                img_rect_can.top(),
            )
            x2 = min(
                self.single_viewer.crop_end_point.x(),
                img_rect_can.right(),
            )
            y2 = min(
                self.single_viewer.crop_end_point.y(),
                img_rect_can.bottom(),
            )
            rect = QRect(QPoint(x1, y1), QPoint(x2, y2)).normalized()

            ix1 = int((rect.left() - img_rect_can.left()) * ratio_w)
            iy1 = int((rect.top() - img_rect_can.top()) * ratio_h)
            ix2 = int((rect.right() - img_rect_can.left()) * ratio_w)
            iy2 = int((rect.bottom() - img_rect_can.top()) * ratio_h)

            ix1 = max(0, min(ix1, w_rot - 1))
            iy1 = max(0, min(iy1, h_rot - 1))
            ix2 = max(ix1 + 1, min(ix2, w_rot))
            iy2 = max(iy1 + 1, min(iy2, h_rot))

            try:
                cropped_image = temp_rotated.crop((ix1, iy1, ix2, iy2))
            except Exception as e:
                QMessageBox.critical(self, "Crop", f"Crop failed: {e}")
                return

            self.working_image_pil = cropped_image
            self._preview_source = self._make_preview_source(
                self.working_image_pil
            )
            self.rotation_degrees = 0
            self.geom_rx_deg = 0.0
            self.geom_ry_deg = 0.0
            self.geom_rz_deg = 0.0
            self.crop_mode_button.setChecked(False)
            self.single_viewer.set_editor_mode(MODE_VIEW)
            self.apply_crop_button.setEnabled(False)
            self._schedule_preview()
            return

        QMessageBox.warning(
            self, "Crop", "No valid crop area selected."
        )

    # ---------- Save ----------

    def save_image(self) -> bool:
        if not self.original_image_path:
            QMessageBox.warning(self, "Save", "No image loaded.")
            return False
        reply = QMessageBox.question(
            self,
            "Confirm Overwrite",
            f"Overwrite original file?\n{self.original_image_path}",
            QMessageBox.StandardButton.Yes
            | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return False
        ok = self._perform_save(self.original_image_path)
        if ok:
            QMessageBox.information(
                self,
                "Save",
                f"Saved:\n{self.original_image_path}",
            )
            self._load_image(self.original_image_path)
        return ok

    def save_image_as(self) -> bool:
        if not self.original_image_path:
            QMessageBox.warning(
                self, "Save As", "No image loaded."
            )
            return False
        default_ext = Path(self.original_image_path).suffix
        filetypes = (
            "JPEG (*.jpg *.jpeg);;PNG (*.png);;TIFF (*.tif *.tiff);;"
            "WEBP (*.webp);;All Files (*.*)"
        )
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Image As",
            str(
                Path(self.original_image_path).parent
                / f"{Path(self.original_image_path).stem}_edited{default_ext}"
            ),
            filetypes,
        )
        if not save_path:
            return False
        ok = self._perform_save(save_path)
        if ok:
            QMessageBox.information(
                self, "Save", f"Saved:\n{save_path}"
            )
        return ok

    def _perform_save(self, save_path: str) -> bool:
        try:
            if not self.working_image_pil:
                return False

            geo_full = _apply_geometry_perspective(
                self.working_image_pil,
                self.geom_rx_deg,
                self.geom_ry_deg,
                self.geom_rz_deg,
                self.rotation_degrees,
                self.fill_mode,
            )

            if (
                self.filter_name
                and self.filter_name not in ("None", "—", "Original")
            ):
                geo_full = _apply_filter_pipeline(
                    geo_full,
                    self.filter_name,
                    max(0.0, min(1.0, self.filter_strength)),
                )

            f = {
                "exposure": self.exposure_factor,
                "brightness": self.brightness_factor,
                "contrast": self.contrast_factor,
                "highlights": self.highlights_factor,
                "shadows": self.shadows_factor,
                "blackpoint": self.blackpoint_factor,
                "saturation": self.saturation_factor,
                "vibrance": self.vibrance_factor,
                "warmth": self.warmth_factor,
                "tint": self.tint_factor,
                "sharpness": self.sharpness_factor,
                "noise_reduction": self.noise_reduction_factor,
                "vignette": self.vignette_factor,
                "brilliance": self.brilliance_factor,
            }
            job = _PreviewJob(
                0,
                geo_full,
                0,
                f,
                geo_full.size,
                False,
                False,
                0.0,
                0.0,
                0.0,
                "None",
                0.0,
                self.fill_mode,
            )
            final_image = job._apply_edits(geo_full, f)

            if self._mirror_on:
                try:
                    flip_const = Image.Transpose.FLIP_LEFT_RIGHT
                except AttributeError:
                    flip_const = Image.FLIP_LEFT_RIGHT
                final_image = final_image.transpose(flip_const)

            p = Path(save_path)
            if not self._ensure_heif_plugin_for_path(save_path, "save"):
                return False

            save_fmt = (
                (self._orig_format or self._infer_format_from_path(save_path))
                or "PNG"
            ).upper()
            if (
                save_fmt == "JPEG"
                and final_image.mode not in ("RGB", "L")
            ):
                final_image = final_image.convert("RGB")

            save_kwargs: Dict[str, Any] = {"format": save_fmt}
            if self._orig_exif and save_fmt in (
                "JPEG",
                "TIFF",
                "WEBP",
                "HEIF",
            ):
                save_kwargs["exif"] = self._orig_exif
            if save_fmt in ("JPEG", "WEBP", "HEIF", "AVIF"):
                save_kwargs.setdefault("quality", 90)

            if p.exists() and save_path == self.original_image_path:
                tmp_path = p.with_name(f".tmp_{p.name}")
                final_image.save(str(tmp_path), **save_kwargs)
                os.replace(str(tmp_path), save_path)
            else:
                final_image.save(str(save_path), **save_kwargs)
            return True
        except Exception as e:
            QMessageBox.critical(
                self, "Save", f"Could not save image: {e}"
            )
            try:
                if "tmp_path" in locals() and Path(tmp_path).exists():
                    Path(tmp_path).unlink()
            except Exception:
                pass
            return False

    # ---------- Filters ----------

    def _on_filter_radio(self, name: str, checked: bool):
        if not checked:
            return
        self.filter_name = name
        self._schedule_preview()

    def _on_filter_strength(self, v: float):
        self.filter_strength = max(0.0, min(1.0, v))
        self._schedule_preview()

    # ---------- Live preview ----------

    def _on_slider_change(self, value: float, attr_name: str):
        self._last_changed_name = attr_name
        setattr(self, attr_name, value)
        if (
            attr_name in ("geom_rx_deg", "geom_ry_deg")
            and (_CV2 is None or _NP is None)
            and not self._warned_perspective
        ):
            self._warned_perspective = True
            QMessageBox.warning(
                self,
                "Perspective",
                "Pitch/Yaw straighten needs opencv-python + numpy. Only Roll (Z) will apply.",
            )
        self._schedule_preview()

    def _schedule_preview(self):
        self._preview_timer.start()

    def _start_preview_job(self):
        if not self.working_image_pil:
            return
        do_mirror = (
            self.viewer_stack is not None
            and self.viewer_stack.currentIndex() == 1
        )
        if do_mirror and self.dual_viewer is not None:
            avail = self.dual_viewer.left_scroll.viewport().size()
        elif self.single_scroll is not None:
            avail = self.single_scroll.viewport().size()
        else:
            return

        cap_w = min(avail.width(), 1280)
        cap_h = min(avail.height(), 1280)
        target_size = (max(1, cap_w), max(1, cap_h))

        interactive = any(sl.isSliderDown() for sl in self._all_sliders)
        geom_sliders_down = any(
            getattr(self, n + "_slider", None)
            and getattr(self, n + "_slider").slider.isSliderDown()
            for n in ("geom_rx", "geom_ry", "geom_rz")
        )
        fast_geom = interactive and geom_sliders_down

        with self._preview_lock:
            self._preview_job_id += 1
            jid = self._preview_job_id

        f = {
            "exposure": self.exposure_factor,
            "brightness": self.brightness_factor,
            "contrast": self.contrast_factor,
            "highlights": self.highlights_factor,
            "shadows": self.shadows_factor,
            "blackpoint": self.blackpoint_factor,
            "saturation": self.saturation_factor,
            "vibrance": self.vibrance_factor,
            "warmth": self.warmth_factor,
            "tint": self.tint_factor,
            "sharpness": self.sharpness_factor,
            "noise_reduction": self.noise_reduction_factor,
            "vignette": self.vignette_factor,
            "brilliance": self.brilliance_factor,
        }
        job = _PreviewJob(
            job_id=jid,
            base_image=(self._preview_source or self.working_image_pil),
            coarse_rotation_degrees=self.rotation_degrees,
            factors=f,
            single_target_size=target_size,
            do_mirror=do_mirror,
            interactive=interactive,
            geom_rx=self.geom_rx_deg,
            geom_ry=self.geom_ry_deg,
            geom_rz=self.geom_rz_deg,
            preset_name=self.filter_name,
            preset_strength=self.filter_strength,
            fill_mode=self.fill_mode,
            fast_geometry_preview=fast_geom,
        )
        job.signals.done.connect(self._on_preview_ready)
        self.threadpool.start(job, 1)

    def _on_preview_ready(
        self,
        job_id: int,
        main_img: QImage,
        mirror_img_obj: object,
    ):
        if job_id != self._preview_job_id:
            # Stale preview result – a newer job already started
            return

        if main_img is None:
            return

        pm_main = QPixmap.fromImage(main_img)

        # Are we in mirror mode?
        do_mirror = (
            self.viewer_stack is not None
            and self.viewer_stack.currentIndex() == 1
        )

        if do_mirror and self.dual_viewer is not None:
            mirror_pm = None
            if isinstance(mirror_img_obj, QImage):
                mirror_pm = QPixmap.fromImage(mirror_img_obj)
            elif isinstance(mirror_img_obj, QPixmap):
                mirror_pm = mirror_img_obj

            # Dual viewer: main on the left, mirror on the right
            try:
                self.dual_viewer.set_pixmaps(pm_main, mirror_pm)
            except AttributeError:
                # Fallback if API is different
                try:
                    self.dual_viewer.set_images(pm_main, mirror_pm)
                except AttributeError:
                    # As a last resort, just show main in left viewer
                    if hasattr(self.dual_viewer, "set_left_pixmap"):
                        self.dual_viewer.set_left_pixmap(pm_main)
        else:
            # Single viewer
            if self.single_viewer is not None:
                try:
                    self.single_viewer.set_pixmap(pm_main)
                except AttributeError:
                    # Fallback for older API
                    if hasattr(self.single_viewer, "set_image"):
                        self.single_viewer.set_image(pm_main)

    # ---------- Metadata note ----------

    def _write_custom_note_to_file(self):
        if not self.original_image_path:
            QMessageBox.warning(
                self,
                "Metadata",
                "No image loaded.",
            )
            return

        if _PIEXIF is None:
            QMessageBox.warning(
                self,
                "Metadata",
                "piexif is not installed.\n\n"
                "Install with: pip install piexif",
            )
            return

        note = self.meta_note_edit.text().strip()
        if not note:
            reply = QMessageBox.question(
                self,
                "Clear note?",
                "Note field is empty.\n\n"
                "Do you want to clear the existing EXIF note (if any)?",
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        path = self.original_image_path
        try:
            import piexif  # type: ignore[import-not-found]

            # Load existing EXIF (if any)
            with Image.open(path) as im:
                exif_bytes = im.info.get("exif", b"")
                if exif_bytes:
                    exif_dict = piexif.load(exif_bytes)
                else:
                    exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}

            # ImageDescription is usually enough (and better supported than UserComment)
            if note:
                exif_dict["0th"][piexif.ImageIFD.ImageDescription] = note.encode(
                    "utf-8", errors="ignore"
                )
            else:
                # Clear the field if it exists
                exif_dict["0th"].pop(piexif.ImageIFD.ImageDescription, None)

            new_exif = piexif.dump(exif_dict)

            # Save back in-place with updated EXIF
            tmp_path = Path(path).with_name(f".tmp_meta_{Path(path).name}")
            with Image.open(path) as im:
                im.save(str(tmp_path), exif=new_exif)
            os.replace(str(tmp_path), path)

            # Remember new EXIF so further saves keep it
            self._orig_exif = new_exif

            QMessageBox.information(
                self,
                "Metadata",
                "Custom note was written to EXIF.",
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Metadata",
                f"Failed to write EXIF note:\n{e}",
            )
            try:
                if "tmp_path" in locals() and tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass

    # ---------- Delete current image (move to deleted/) ----------

    def _on_delete_key(self):
        """
        Handle Delete key pressed (global QAction).

        Instead of actually deleting the file from disk, move it into a
        sibling 'deleted' folder under the same root directory.

        Example:
            /photos/IMG_0001.JPG --> /photos/deleted/IMG_0001.JPG
        """
        if not self.image_files or self.current_image_index < 0:
            return

        idx = self.current_image_index
        path_str = self.image_files[idx]
        path = Path(path_str)

        deleted_dir = path.parent / "deleted"
        deleted_dir_display = str(deleted_dir)

        reply = QMessageBox.question(
            self,
            "Move image to deleted?",
            f"Move this file into:\n{deleted_dir_display}\n\n{path_str}",
            QMessageBox.StandardButton.Yes
            | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        try:
            deleted_dir.mkdir(exist_ok=True)

            dest = deleted_dir / path.name
            if dest.exists():
                # Avoid overwriting: add timestamp suffix
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                dest = deleted_dir / f"{path.stem}_{ts}{path.suffix}"

            shutil.move(str(path), str(dest))
        except Exception as e:
            QMessageBox.critical(
                self,
                "Delete",
                f"Could not move file to deleted folder:\n{path_str}\n\nError: {e}",
            )
            return

        # Remove from list / internal state
        del self.image_files[idx]
        self.photo_list.takeItem(idx)

        # Reset duplicates info – they are stale now
        self.duplicate_groups = []
        if hasattr(self, "dup_groups_list"):
            self.dup_groups_list.clear()
        if hasattr(self, "dup_thumbs_layout"):
            self._clear_duplicate_thumbnails()
        if hasattr(self, "dup_status_lbl"):
            self.dup_status_lbl.setText(
                "Load a folder in Photos tab, then scan."
            )

        if not self.image_files:
            # No images left
            self.current_image_index = -1
            self._clear_photos_state(reset_lists_only=False)
            self.set_controls_state(False)
            return

        # Select next image (or previous if we moved the last one)
        new_idx = min(idx, len(self.image_files) - 1)
        self.photo_list.setCurrentRow(new_idx)
        self._load_image_by_index(new_idx)

    # ---------- Tab / window handling ----------

    def _on_tab_changed(self, index: int):
        current = self.tabs.widget(index)

        # Stop video playback when leaving Live tab
        if current is not self.live_tab:
            self._stop_video()

    def closeEvent(self, event):
        """
        Ensure background resources are cleaned up when the main window closes.
        """
        try:
            self._stop_video()
            if self.cap:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None
        except Exception:
            pass

        # Close the external viewer window as well
        if self.image_window is not None:
            try:
                self.image_window.close()
            except Exception:
                pass

        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)

    # High DPI scaling (safe on modern Qt, no-op on some platforms)
    try:
        QApplication.setAttribute(
            Qt.ApplicationAttribute.AA_EnableHighDpiScaling
        )
    except Exception:
        pass

    win = ImageEditorApp()
    win.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
