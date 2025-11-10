#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
from pathlib import Path
from typing import Optional, Any, Tuple
import math

# PyQt6 Imports
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QListWidget, QPushButton, QLabel, QSlider, QTabWidget, QGridLayout, 
    QDoubleSpinBox, QAbstractSpinBox, QSplitter, QSizePolicy, QScrollArea, 
    QMessageBox, QFileDialog, QToolButton, QGroupBox, QLineEdit
)
import PyQt6.QtGui  as QtGui  # For type hints
from PyQt6.QtGui import (
    QPixmap, QImage, QPainter, QColor, QCursor, QAction,
    QPalette, QResizeEvent, QPen, QPaintEvent, QMouseEvent
)
from PyQt6.QtCore import Qt, QSize, QPoint, QRect, pyqtSignal, QDir
from PyQt6.QtCore import Qt, QSize, QPoint, QRect, pyqtSignal, QDir, QTimer
# PIL Imports (for heavy lifting)
from PIL import Image, ImageTk, ImageOps, ImageEnhance, ImageFilter

# --- Advanced Dependency Check (for Noise Reduction, Complex Color) ---
_CV2 = None
_NP = None
try:
    import cv2
    _CV2 = cv2
    import numpy as np
    _NP = np
except Exception:
    pass

# Optional HEIF/HEIC/AVIF support via pillow-heif
_HEIF_PLUGIN = False
try:
    from pillow_heif import register_heif_opener  # type: ignore
    register_heif_opener()
    _HEIF_PLUGIN = True
except Exception:
    pass

# Extension -> PIL format mapping
EXT_TO_FMT = {
    ".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG", ".bmp": "BMP", 
    ".gif": "GIF", ".tif": "TIFF", ".tiff": "TIFF", ".webp": "WEBP", 
    ".heic": "HEIF", ".heif": "HEIF", ".heics": "HEIF", ".heifs": "HEIF", 
    ".hif": "HEIF", ".avif": "AVIF",
}
HEIF_LIKE_EXTS = {".heic", ".heif", ".heics", ".heifs", ".hif", ".avif"}
SUPPORTED_EXTS = set(EXT_TO_FMT.keys())

# Constants for Cropping Mode
MODE_VIEW = 0
MODE_CROP = 1

# --- Utility Functions for PIL <-> QPixmap Conversion ---

def pil_to_qpixmap(pil_image: Image.Image) -> QPixmap:
    """Converts a PIL Image to QPixmap."""
    # Ensure image is in a format QImage understands (RGBA is universal)
    if pil_image.mode not in ('RGB', 'RGBA', 'L'):
        pil_image = pil_image.convert('RGBA')
    elif pil_image.mode == 'RGB':
        # Add alpha channel for consistent handling
        r, g, b = pil_image.split()
        pil_image = Image.merge('RGBA', (r, g, b, Image.new('L', pil_image.size, 255)))
    
    # Get image data
    data = pil_image.tobytes("raw", "RGBA")
    
    # Create QImage
    qimage = QImage(
        data, 
        pil_image.width, 
        pil_image.height, 
        4 * pil_image.width, # Bytes per line (4 bytes per pixel for RGBA)
        QImage.Format.Format_RGBA8888
    )
    return QPixmap.fromImage(qimage)

# --- Custom Image Viewer Widget with Cropping Overlay ---

class ImageViewer(QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Use background color from parent palette
        self.setBackgroundRole(QPalette.ColorRole.Dark) 
        self.setAutoFillBackground(True)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setScaledContents(False)
        
        self.editor_mode = MODE_VIEW
        self.crop_start_point: Optional[QPoint] = None
        self.crop_end_point: Optional[QPoint] = None
        self.current_qpixmap: Optional[QPixmap] = None
        
        # Reference to the main window's editor for enabling/disabling crop button
        self.editor_ref: Optional[Any] = None 
        
    def set_pixmap(self, qpixmap: QPixmap):
        self.current_qpixmap = qpixmap
        self.setPixmap(qpixmap) # Set the pixmap on the QLabel
        self.setMinimumSize(qpixmap.size())
        self.update()
        
    def clear_pixmap(self):
        self.current_qpixmap = None
        self.setPixmap(QPixmap()) # Clear the pixmap
        self.setMinimumSize(QSize(100, 100))
        self.setText("No Image Loaded")
        self.update()

    def set_editor_mode(self, mode: int):
        self.editor_mode = mode
        if mode == MODE_VIEW:
            self.crop_start_point = None
            self.crop_end_point = None
            self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
        else:
            self.setCursor(QCursor(Qt.CursorShape.CrossCursor))
        self.update()
        
    def get_image_display_rect(self) -> QRect:
        """Calculates the rectangle occupied by the displayed image relative to the QLabel."""
        if not self.current_qpixmap:
            return QRect(0, 0, 0, 0)
            
        w_img = self.current_qpixmap.width()
        h_img = self.current_qpixmap.height()
        w_can = self.width()
        h_can = self.height()

        # Calculate offset to center the image
        x_offset = (w_can - w_img) // 2
        y_offset = (h_can - h_img) // 2
        
        return QRect(x_offset, y_offset, w_img, h_img)
        
    def paintEvent(self, event: QPaintEvent):
        super().paintEvent(event)
        
        # Draw the cropping rectangle if in CROP mode
        if self.editor_mode == MODE_CROP and self.crop_start_point and self.crop_end_point:
            painter = QPainter(self)
            
            # Set pen style for the crop rectangle
            painter.setPen(QPen(QColor(255, 0, 0), 2, Qt.PenStyle.DashLine))

            img_rect = self.get_image_display_rect()
            
            if img_rect.isEmpty():
                return
            
            x1 = self.crop_start_point.x()
            y1 = self.crop_start_point.y()
            x2 = self.crop_end_point.x()
            y2 = self.crop_end_point.y()
            
            # Constrain drawing to the image area
            x1 = max(x1, img_rect.left())
            y1 = max(y1, img_rect.top())
            x2 = min(x2, img_rect.right())
            y2 = min(y2, img_rect.bottom())
            
            crop_rect = QRect(QPoint(x1, y1), QPoint(x2, y2)).normalized()
            painter.drawRect(crop_rect)
            
    def mousePressEvent(self, event: QMouseEvent):
        img_rect = self.get_image_display_rect()
        if self.editor_mode == MODE_CROP and self.current_qpixmap and img_rect.contains(event.position().toPoint()):
            # Constrain start point to image bounds
            x = max(img_rect.left(), min(event.position().x(), img_rect.right()))
            y = max(img_rect.top(), min(event.position().y(), img_rect.bottom()))
            
            self.crop_start_point = QPoint(int(x), int(y))
            self.crop_end_point = QPoint(int(x), int(y))
            
            if self.editor_ref:
                self.editor_ref.apply_crop_button.setEnabled(False)
            
    def mouseMoveEvent(self, event: QMouseEvent):
        if self.editor_mode == MODE_CROP and self.crop_start_point:
            img_rect = self.get_image_display_rect()
            if img_rect.isEmpty(): return

            # Constrain end point to image bounds
            x = max(img_rect.left(), min(event.position().x(), img_rect.right()))
            y = max(img_rect.top(), min(event.position().y(), img_rect.bottom()))
            
            self.crop_end_point = QPoint(int(x), int(y))
            self.update()
            
    def mouseReleaseEvent(self, event: QMouseEvent):
        if self.editor_mode == MODE_CROP and self.crop_start_point and self.crop_end_point:
            rect = QRect(self.crop_start_point, self.crop_end_point).normalized()
            w = rect.width()
            h = rect.height()
            
            # Use the editor_ref to access the button
            if w > 10 and h > 10 and self.editor_ref:
                self.editor_ref.apply_crop_button.setEnabled(True)
            else:
                self.crop_start_point = None
                self.crop_end_point = None
                if self.editor_ref:
                    self.editor_ref.apply_crop_button.setEnabled(False)
                self.update()


# --- Custom Slider and SpinBox Widget ---

class SliderSpinBox(QWidget):
    # Signal emitted when the value changes (used by the main app)
    valueChanged = pyqtSignal(float)

    def __init__(self, label: str, min_val: float, max_val: float, default_val: float, step: float, parent=None):
        super().__init__(parent)
        
        self.min_val = min_val
        self.max_val = max_val
        self.step = step

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Label
        self.label = QLabel(label)
        self.label.setFixedWidth(120)
        layout.addWidget(self.label)
        
        # SpinBox (for precise control)
        self.spinbox = QDoubleSpinBox()
        self.spinbox.setRange(min_val, max_val)
        self.spinbox.setSingleStep(step)
        self.spinbox.setValue(default_val)
        self.spinbox.setDecimals(2)
        # Use QAbstractSpinBox.ButtonSymbols.NoButtons to hide buttons for cleaner look
        self.spinbox.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons) 
        self.spinbox.setFixedWidth(50)
        layout.addWidget(self.spinbox)

        # Slider
        self.slider = QSlider(Qt.Orientation.Horizontal)
        # Scale range by step to use integer slider
        self.slider.setRange(int(min_val / step), int(max_val / step)) 
        self.slider.setValue(int(default_val / step))
        self.slider.setTickInterval(int((max_val - min_val) / step / 10))
        self.slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        layout.addWidget(self.slider)

        # Connections
        # Use the slider's valueChanged signal (int) to update spinbox (float)
        self.slider.valueChanged.connect(self._slider_to_spinbox) 
        # Use the spinbox's valueChanged signal (float) to update slider (int) AND emit public signal
        self.spinbox.valueChanged.connect(self._spinbox_to_slider)
        self.spinbox.valueChanged.connect(self.valueChanged)
        
    def _slider_to_spinbox(self, val: int):
        new_val = val * self.step
        self.spinbox.blockSignals(True)
        self.spinbox.setValue(new_val)
        self.spinbox.blockSignals(False)
        self.valueChanged.emit(new_val)
        
    def _spinbox_to_slider(self, val: float):
        new_val = int(round(val / self.step))
        self.slider.blockSignals(True)
        self.slider.setValue(new_val)
        self.slider.blockSignals(False)

    def set_value(self, val: float):
        self.spinbox.setValue(val)
        
    def get_value(self) -> float:
        return self.spinbox.value()
        
    def setEnabled(self, enabled: bool):
        self.slider.setEnabled(enabled)
        self.spinbox.setEnabled(enabled)
        self.label.setEnabled(enabled)

# --- Main Application ---

class ImageEditorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced PyQt6 Image Editor Tool")
        self.setGeometry(100, 100, 1200, 800)
        
        # --- State (Current Image) ---
        self.current_folder: Optional[Path] = None
        self.image_files: list[str] = []
        self.current_image_index: int = -1
        self.original_image_path: Optional[str] = None
        self.original_image_pil: Optional[Image.Image] = None
        self.working_image_pil: Optional[Image.Image] = None # Base image after crop
        self._orig_format: Optional[str] = None
        self._orig_exif: Optional[bytes] = None
        
        # --- Editing state (non-destructive) ---
        self.rotation_degrees = 0
        self.crop_area: Optional[Tuple[int, int, int, int]] = None 

        # Initialize editing factors (will be updated by controls)
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
        self.brilliance_factor = 0.0 # Placeholder/Unused in current PIL implementation

        self._init_ui()
        self.set_controls_state(False)

        self.update_timer = QTimer(self)
        self.update_timer.setInterval(75) # 75ms delay
        self.update_timer.setSingleShot(True)
        # Connect the timer's timeout to the function that does the work
        self.update_timer.timeout.connect(self._display_scaled)


    def _init_ui(self):
        # Central Widget and Main Layout (QSplitter for resize flexibility)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)

        # --- QSplitter for Left (List) and Right (Viewer + Editor) ---
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # 1. Left Sidebar (File List)
        self._create_sidebar()
        splitter.addWidget(self.sidebar_widget)
        
        # 2. Right Content Area (Viewer and Editor Tabs)
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        
        # Splitter for Viewer and Editor Controls
        viewer_editor_splitter = QSplitter(Qt.Orientation.Vertical)
        content_layout.addWidget(viewer_editor_splitter)

        # Image Viewer (Center)
        self._create_viewer()
        viewer_editor_splitter.addWidget(self.viewer_scroll)
        
        # Editor Tabs (Right Side)
        self._create_editor_tabs()
        viewer_editor_splitter.addWidget(self.editor_widget)

        viewer_editor_splitter.setSizes([700, 300])
        splitter.addWidget(content_widget)
        splitter.setSizes([200, 1000])
        
        # Connect the viewer's editor reference after creation
        self.image_viewer.editor_ref = self

    # --- UI Component Creation ---

    def _create_sidebar(self):
        self.sidebar_widget = QWidget()
        layout = QVBoxLayout(self.sidebar_widget)
        
        self.load_folder_button = QPushButton("Load Folder...")
        self.load_folder_button.clicked.connect(self.load_folder_dialog)
        layout.addWidget(self.load_folder_button)

        self.listbox = QListWidget()
        self.listbox.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.listbox.currentItemChanged.connect(self._on_listbox_select)
        layout.addWidget(self.listbox)

    def _create_viewer(self):
        self.image_viewer = ImageViewer()
        self.viewer_scroll = QScrollArea()
        self.viewer_scroll.setWidgetResizable(True)
        self.viewer_scroll.setWidget(self.image_viewer)
        self.viewer_scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_viewer.clear_pixmap()
        
    def _create_editor_tabs(self):
        self.editor_widget = QTabWidget()
        
        # --- Tab 1: Rotation / Crop / Save ---
        tab1 = QWidget()
        layout1 = QVBoxLayout(tab1)
        layout1.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Rotation Group
        rotate_group = QGroupBox("Rotation")
        rotate_layout = QHBoxLayout(rotate_group)
        self.rotate_left_button = QPushButton("Rotate Left (90°)")
        self.rotate_left_button.clicked.connect(lambda: self.rotate_image(-90))
        self.rotate_right_button = QPushButton("Rotate Right (90°)")
        self.rotate_right_button.clicked.connect(lambda: self.rotate_image(90))
        rotate_layout.addWidget(self.rotate_left_button)
        rotate_layout.addWidget(self.rotate_right_button)
        layout1.addWidget(rotate_group)

        # Cropping Group
        crop_group = QGroupBox("Free-Size Cropping")
        crop_layout = QHBoxLayout(crop_group)
        self.crop_mode_button = QPushButton("Enable Crop Mode")
        self.crop_mode_button.setCheckable(True)
        self.crop_mode_button.toggled.connect(self._toggle_crop_mode)
        self.apply_crop_button = QPushButton("Apply Crop")
        self.apply_crop_button.clicked.connect(self.apply_crop)
        self.apply_crop_button.setEnabled(False)
        crop_layout.addWidget(self.crop_mode_button)
        crop_layout.addWidget(self.apply_crop_button)
        layout1.addWidget(crop_group)
        
        # Save/Reset Group
        save_group = QGroupBox("Save & Reset")
        save_layout = QVBoxLayout(save_group)
        self.save_button = QPushButton("Save (Overwrite Original)")
        self.save_button.clicked.connect(self.save_image)
        self.save_button.setStyleSheet("background-color: #fdd;")
        self.save_as_button = QPushButton("Save As...")
        self.save_as_button.clicked.connect(self.save_image_as)
        self.reset_button = QPushButton("Reset All Edits")
        self.reset_button.clicked.connect(self.reset_edits)
        save_layout.addWidget(self.save_button)
        save_layout.addWidget(self.save_as_button)
        save_layout.addWidget(self.reset_button)
        layout1.addWidget(save_group)
        
        self.editor_widget.addTab(tab1, "Transform / Save")

        # --- Tab 2: Tone / Light ---
        tab2 = QWidget()
        layout2 = QVBoxLayout(tab2)
        layout2.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        self.exposure_slider = self._create_slider(layout2, "Exposure (Stops)", -2.0, 2.0, 0.01, 'exposure_factor')
        # Brilliance is currently non-functional/placeholder in PIL-only code
        self.brilliance_slider = self._create_slider(layout2, "Brilliance", -100.0, 100.0, 1.0, 'brilliance_factor') 
        self.brightness_slider = self._create_slider(layout2, "Brightness", -100.0, 100.0, 1.0, 'brightness_factor')
        self.contrast_slider = self._create_slider(layout2, "Contrast", -100.0, 100.0, 1.0, 'contrast_factor')
        self.highlights_slider = self._create_slider(layout2, "Highlights", -100.0, 100.0, 1.0, 'highlights_factor')
        self.shadows_slider = self._create_slider(layout2, "Shadows", -100.0, 100.0, 1.0, 'shadows_factor')
        self.blackpoint_slider = self._create_slider(layout2, "Blackpoint", -100.0, 100.0, 1.0, 'blackpoint_factor')
        
        self.editor_widget.addTab(tab2, "Tone / Light")

        # --- Tab 3: Color / Effects ---
        tab3 = QWidget()
        layout3 = QVBoxLayout(tab3)
        layout3.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        self.saturation_slider = self._create_slider(layout3, "Saturation", -100.0, 100.0, 1.0, 'saturation_factor')
        self.vibrance_slider = self._create_slider(layout3, "Vibrance", -100.0, 100.0, 1.0, 'vibrance_factor')
        self.warmth_slider = self._create_slider(layout3, "Warmth (Temp)", -100.0, 100.0, 1.0, 'warmth_factor')
        self.tint_slider = self._create_slider(layout3, "Tint (G/M)", -100.0, 100.0, 1.0, 'tint_factor')
        self.sharpness_slider = self._create_slider(layout3, "Sharpness", -100.0, 100.0, 1.0, 'sharpness_factor')
        self.noise_reduction_slider = self._create_slider(layout3, "Noise Reduction", 0.0, 10.0, 0.1, 'noise_reduction_factor')
        self.vignette_slider = self._create_slider(layout3, "Vignette", 0.0, 10.0, 0.01, 'vignette_factor')
        
        self.editor_widget.addTab(tab3, "Color / Effects")

    def _create_slider(self, parent_layout: QVBoxLayout, label: str, min_val: float, max_val: float, step: float, attr_name: str) -> SliderSpinBox:
        """Helper to create and connect a custom slider widget."""
        default_val = getattr(self, attr_name)
        slider = SliderSpinBox(label, min_val, max_val, default_val, step)
        
        # Connect to the general update method
        slider.valueChanged.connect(lambda v: self._on_slider_change(v, attr_name))
        
        # Store a reference to the slider widget in the instance
        setattr(self, attr_name.replace('_factor', '_slider'), slider)
        
        parent_layout.addWidget(slider)
        return slider

    def set_controls_state(self, enabled: bool):
        """Sets the state of all controls."""
        self.rotate_left_button.setEnabled(enabled)
        self.rotate_right_button.setEnabled(enabled)
        self.crop_mode_button.setEnabled(enabled)
        self.save_button.setEnabled(enabled)
        self.save_as_button.setEnabled(enabled)
        self.reset_button.setEnabled(enabled)
        
        
        # Sliders
        for attr_name in ['exposure', 'brilliance', 'brightness', 'contrast', 'highlights', 'shadows', 
                          'blackpoint', 'saturation', 'vibrance', 'warmth', 'tint', 
                          'sharpness', 'noise_reduction', 'vignette']:
            slider = getattr(self, f"{attr_name}_slider", None)
            if slider:
                slider.setEnabled(enabled)

    # ------------------------------- Image Loading/Iteration -------------------------------
    
    def _is_supported_image(self, p: Path) -> bool:
        """Checks if a file path points to a supported image type."""
        return p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
        
    def load_folder_dialog(self):
        """Opens a directory dialog, loads supported files, and displays the first one."""
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select a folder containing images", QDir.homePath()
        )
        if not folder_path:
            return

        self.clear_state()
        
        try:
            folder = Path(folder_path)
            self.current_folder = folder
            # Scan for supported images and sort alphabetically
            self.image_files = sorted(
                [str(p) for p in folder.iterdir() if self._is_supported_image(p)],
                key=str.lower
            )
            
            if not self.image_files:
                QMessageBox.warning(self, "No Images Found", f"No supported images found in: {folder_path}")
                self.clear_state()
                return
            
            self.listbox.clear()
            for filename in self.image_files:
                self.listbox.addItem(Path(filename).name)
                
            # Load the first image
            self.load_image_by_index(0)
            self.listbox.setCurrentRow(0)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not process folder: {e}")
            self.clear_state()

    def _on_listbox_select(self, current, previous):
        """Handles selection change in the listbox."""
        if not self.image_files or not current:
            return
            
        index = self.listbox.row(current)
        if index != self.current_image_index:
            self.load_image_by_index(index)
            
    def navigate_image(self, direction: int):
        """Moves to the previous (-1) or next (+1) image in the file list."""
        if not self.image_files or self.image_viewer.editor_mode == MODE_CROP:
            return

        new_index = self.current_image_index + direction

        if 0 <= new_index < len(self.image_files):
            # Select the item in the listbox 
            self.listbox.setCurrentRow(new_index)
            self.load_image_by_index(new_index)
    
    def load_image_by_index(self, index: int):
        """Loads the image at the specified index."""
        if not (0 <= index < len(self.image_files)):
            return

        filepath = self.image_files[index]
        self.current_image_index = index
            
        # Load the actual image
        self.load_image(filepath)
        
        # Update button states
        self.set_controls_state(True)
        
    def load_image(self, filepath: str):
        """Loads and prepares a new image for editing from the given filepath."""
        
        # Clear image/edit state
        self.original_image_path = None
        self.original_image_pil = None
        self.working_image_pil = None
        self._orig_format = None
        self._orig_exif = None
        self.crop_area = None # Clear crop area when loading a new image
        self.reset_controls(reset_rotation=True, reset_sliders=True)
        self.image_viewer.set_editor_mode(MODE_VIEW)
        self.crop_mode_button.setChecked(False) # Reset button state

        if not os.path.exists(filepath):
            return
        
        if not self._ensure_heif_plugin_for_path(filepath, "open"):
            # If HEIF check fails, stop loading
            self.set_controls_state(False)
            self.image_viewer.clear_pixmap()
            return

        try:
            with Image.open(filepath) as im:
                self._orig_format = (im.format or "") or self._infer_format_from_path(
                    filepath
                )
                self._orig_exif = im.info.get("exif")
                im = ImageOps.exif_transpose(im)
                self.original_image_pil = im.copy()
            
            self.original_image_path = filepath
            self.working_image_pil = self.original_image_pil.copy() 
            
            self._display_scaled()
            self.set_controls_state(True)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not process image: {e}")
            self.original_image_path = None
            self.image_viewer.setText(f"Error loading image: {e}")
            self.set_controls_state(False)

    def clear_state(self):
        """Resets all image and state variables, including folder/iteration state."""
        self.original_image_path = None
        self.original_image_pil = None
        self.working_image_pil = None
        self._orig_format = None
        self._orig_exif = None
        self.crop_area = None
        
        # Reset iteration state
        self.current_folder = None
        self.image_files = []
        self.current_image_index = -1
        self.listbox.clear()
        
        self.image_viewer.clear_pixmap()
        self.image_viewer.set_editor_mode(MODE_VIEW)
        self.crop_mode_button.setChecked(False)
        self.reset_controls(reset_rotation=True, reset_sliders=True)
        self.set_controls_state(False)
        
    def reset_controls(self, reset_rotation=True, reset_sliders=True):
        """Resets editing factors and GUI controls."""
        if reset_rotation:
            self.rotation_degrees = 0
            
        if reset_sliders:
            # Reset Sliders
            for attr_name in ['brightness', 'contrast', 'exposure', 'highlights', 'shadows', 
                              'blackpoint', 'saturation', 'vibrance', 'warmth', 'tint', 
                              'sharpness', 'noise_reduction', 'vignette', 'brilliance']:
                # Reset the underlying factor to 0.0
                setattr(self, f"{attr_name}_factor", 0.0) 
                
                # Reset the slider widget value
                slider = getattr(self, f"{attr_name}_slider", None)
                if slider:
                    slider.set_value(0.0)

    def reset_edits(self):
        """Resets the working image (if cropped/rotated) and controls to the original state."""
        if not self.original_image_pil:
            return
            
        self.working_image_pil = self.original_image_pil.copy()
        self.crop_area = None
        self.reset_controls(reset_rotation=True, reset_sliders=True)
        self._display_scaled()


    # ------------------------------- Cropping Logic -------------------------------
    
    def _toggle_crop_mode(self, checked: bool):
        """Toggles the cropping mode via the button state."""
        if not self.original_image_path:
            self.crop_mode_button.setChecked(False)
            return
        
        if checked:
            self.image_viewer.set_editor_mode(MODE_CROP)
            self.crop_mode_button.setText("Exit Crop Mode")
        else:
            self.image_viewer.set_editor_mode(MODE_VIEW)
            self.crop_mode_button.setText("Enable Crop Mode")
            self.apply_crop_button.setEnabled(False) # Disable apply button when exiting crop mode


    def apply_crop(self):
        """Applies the selected crop to the working image."""
        start_point = self.image_viewer.crop_start_point
        end_point = self.image_viewer.crop_end_point
        
        if not start_point or not end_point or not self.original_image_pil or not self.image_viewer.current_qpixmap:
            QMessageBox.warning(self, "Crop Error", "No valid crop area selected.")
            return

        # Use normalized QRect from the two points
        crop_rect_can = QRect(start_point, end_point).normalized()
        
        # Check if selection is large enough
        if crop_rect_can.width() < 10 or crop_rect_can.height() < 10:
            QMessageBox.warning(self, "Crop Error", "Selection is too small.")
            return

        # 1. Get the current displayed image dimensions (QPixmap size)
        qpixmap_size = self.image_viewer.current_qpixmap.size()
        w_disp, h_disp = qpixmap_size.width(), qpixmap_size.height()
        
        # 2. Get the rect of the image displayed inside the canvas
        img_rect_can = self.image_viewer.get_image_display_rect()

        # 3. Convert canvas coordinates (crop_rect_can) to image coordinates (x,y relative to the scaled image)
        x1_img = crop_rect_can.left() - img_rect_can.left()
        y1_img = crop_rect_can.top() - img_rect_can.top()
        x2_img = crop_rect_can.right() - img_rect_can.left()
        y2_img = crop_rect_can.bottom() - img_rect_can.top()
        
        # Clamp to bounds [0, w_disp] and [0, h_disp]
        x1_img = max(0, min(x1_img, w_disp))
        y1_img = max(0, min(y1_img, h_disp))
        x2_img = max(0, min(x2_img, w_disp))
        y2_img = max(0, min(y2_img, h_disp))
        
        # 4. Get the size of the working image after rotation but before scaling
        temp_rotated_size = self.working_image_pil.rotate(self.rotation_degrees, expand=True).size
        w_rot, h_rot = temp_rotated_size
        
        # The scale ratio is the rotated image size / displayed size
        ratio_w = w_rot / w_disp
        ratio_h = h_rot / h_disp
        
        # 5. Calculate crop box relative to the rotated image size
        crop_box_rotated = (
            int(x1_img * ratio_w),
            int(y1_img * ratio_h),
            int(x2_img * ratio_w),
            int(y2_img * ratio_h)
        )
        
        # 6. Apply the crop to the WORKING image
        try:
            # We first rotate the working image, apply the crop, then reset the rotation to 0
            temp_rotated_image = self.working_image_pil.rotate(self.rotation_degrees, expand=True)
            cropped_image = temp_rotated_image.crop(crop_box_rotated)
            
            self.working_image_pil = cropped_image
            self.rotation_degrees = 0 # Crop is now the new base image, rotation is reset.
            
            # Reset crop mode and update display
            self.crop_mode_button.setChecked(False) # Exits crop mode
            self.image_viewer.set_editor_mode(MODE_VIEW)
            self.apply_crop_button.setEnabled(False)
            self._display_scaled()

        except Exception as e:
            QMessageBox.critical(self, "Crop Error", f"Could not apply crop: {e}")

    # ------------------------------- Image Display & Edits -------------------------------

    def _on_slider_change(self, value: float, attr_name: str):
        """Generic handler for any slider change."""
        setattr(self, attr_name, value)
        self.update_timer.start()

    def _apply_edits(self, pil_image: Image.Image) -> Image.Image:
        """Applies all non-rotation and non-crop edits for display and saving."""
        # Ensure conversion to RGB for consistent manipulation (especially for NumPy/OpenCV)
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
            
        working_img = pil_image.copy()

        # 1. Tonal Adjustments
        
        # Exposure (approximated by multiplying by a factor)
        exposure_adj = 1.0 + (self.exposure_factor / 2.0)
        working_img = ImageEnhance.Brightness(working_img).enhance(exposure_adj)
        
        # Brightness
        # Re-apply brightness factor since it should stack with exposure
        brightness_adj = 1.0 + (self.brightness_factor / 100.0) 
        working_img = ImageEnhance.Brightness(working_img).enhance(brightness_adj)
        
        # Contrast
        contrast_adj = 1.0 + (self.contrast_factor / 100.0)
        working_img = ImageEnhance.Contrast(working_img).enhance(contrast_adj)

        # Highlights/Shadows/Blackpoint (requires NumPy)
        if _NP:
            img_np = _NP.array(working_img).astype(_NP.float32) / 255.0
            
            # Highlights adjustment
            if self.highlights_factor != 0:
                gamma_h = 1.0 - (self.highlights_factor / 200.0) 
                # Apply gamma correction only to tones > 50%
                img_np = _NP.where(img_np > 0.5, img_np**gamma_h, img_np)
                
            # Shadows adjustment
            if self.shadows_factor != 0:
                gamma_s = 1.0 + (self.shadows_factor / 200.0)
                # Apply inverse gamma correction only to tones < 50%
                img_np = _NP.where(img_np < 0.5, img_np**gamma_s, img_np)

            # Blackpoint adjustment
            if self.blackpoint_factor != 0:
                 offset = self.blackpoint_factor / 500.0
                 img_np = _NP.clip(img_np + offset, 0.0, 1.0)

            working_img = Image.fromarray((_NP.clip(img_np, 0.0, 1.0) * 255).astype(_NP.uint8))
            
            if _CV2 and _NP and self.brilliance_factor != 0:
            # Map brilliance factor (-100 to 100) to a clipLimit (1.0 to 8.0)
            # We only use the positive range for this effect
                clip_limit = 1.0 + max(0, self.brilliance_factor) / 100.0 * 7.0 
                
                if clip_limit > 1.1: # Only apply if there's a meaningful change
                    try:
                        # 1. Convert PIL RGB -> OpenCV BGR
                        img_cv_bgr = _CV2.cvtColor(_NP.array(working_img), _CV2.COLOR_RGB2BGR)
                        
                        # 2. Convert BGR -> LAB
                        img_cv_lab = _CV2.cvtColor(img_cv_bgr, _CV2.COLOR_BGR2LAB)
                        
                        # 3. Split L, A, B
                        l_channel, a_channel, b_channel = _CV2.split(img_cv_lab)
                        
                        # 4. Create CLAHE object
                        # tileGridSize controls the "locality". 8x8 is a good default.
                        clahe = _CV2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
                        
                        # 5. Apply CLAHE to L-channel
                        l_channel_clahe = clahe.apply(l_channel)
                        
                        # 6. Merge new L with original A, B
                        merged_lab = _CV2.merge((l_channel_clahe, a_channel, b_channel))
                        
                        # 7. Convert LAB -> BGR
                        final_bgr = _CV2.cvtColor(merged_lab, _CV2.COLOR_LAB2BGR)
                        
                        # 8. Convert OpenCV BGR -> PIL RGB
                        working_img = Image.fromarray(_CV2.cvtColor(final_bgr, _CV2.COLOR_BGR2RGB))
                    
                    except Exception as e:
                        print(f"Warning: Could not apply Brilliance (CLAHE): {e}")
                        # If it fails, just continue with the non-brilliant image
                        pass
        # 2. Color Adjustments 

        # Saturation
        sat_adj = 1.0 + (self.saturation_factor / 100.0)
        working_img = ImageEnhance.Color(working_img).enhance(sat_adj)
        
        # Warmth/Tint/Vibrance (requires NumPy)
        if _NP:
            img_rgb_np = _NP.array(working_img).astype(_NP.float32)
            
            # Vibrance (A simple approximation: selectively boost less saturated colors)
            if self.vibrance_factor != 0:
                img_hsv_np = _NP.array(working_img.convert('HSV')).astype(_NP.float32)
                # Saturation channel is at index 1
                v_adj = self.vibrance_factor / 100.0 
                # Create a mask that is stronger for less saturated pixels
                saturation = img_hsv_np[:,:,1] / 255.0
                vibrance_mask = 1.0 - saturation 
                
                # Apply the boost/reduction based on the vibrance mask
                img_hsv_np[:,:,1] = _NP.clip(img_hsv_np[:,:,1] + (img_hsv_np[:,:,1] * v_adj * vibrance_mask), 0, 255)
                working_img = Image.fromarray(img_hsv_np.astype(_NP.uint8), 'HSV').convert('RGB')
                img_rgb_np = _NP.array(working_img).astype(_NP.float32) # Update RGB array
                
            # Warmth (temp - Red/Blue shift)
            if self.warmth_factor != 0:
                r_gain = 1.0 + (self.warmth_factor / 150.0)
                b_gain = 1.0 - (self.warmth_factor / 150.0)
                
                img_rgb_np[:,:,0] = _NP.clip(img_rgb_np[:,:,0] * r_gain, 0, 255)
                img_rgb_np[:,:,2] = _NP.clip(img_rgb_np[:,:,2] * b_gain, 0, 255)

            # Tint (Green/Magenta shift)
            if self.tint_factor != 0:
                g_gain = 1.0 + (self.tint_factor / 150.0)
                img_rgb_np[:,:,1] = _NP.clip(img_rgb_np[:,:,1] * g_gain, 0, 255)

            working_img = Image.fromarray(img_rgb_np.astype(_NP.uint8))
                
        # 3. Details/Effects

        # Sharpness
        sharp_adj = 1.0 + (self.sharpness_factor / 100.0)
        working_img = ImageEnhance.Sharpness(working_img).enhance(sharp_adj)
        
        # Noise Reduction (requires OpenCV)
        if _CV2 and self.noise_reduction_factor > 0.5:
            # Scale factor 0.5-10.0 to a sensible denoising strength
            nr_strength = int(self.noise_reduction_factor * 8) + 5
            
            img_cv = _CV2.cvtColor(_NP.array(working_img), _CV2.COLOR_RGB2BGR)
            denoised_cv = _CV2.fastNlMeansDenoisingColored(
                img_cv, None, h=nr_strength, hColor=nr_strength, 
                templateWindowSize=7, searchWindowSize=21
            )
            working_img = Image.fromarray(_CV2.cvtColor(denoised_cv, _CV2.COLOR_BGR2RGB))
        
        # Vignette 
        if self.vignette_factor > 0.1:
            working_img = self._apply_vignette(working_img, self.vignette_factor)
        
        return working_img

    def _apply_vignette(self, image: Image.Image, strength: float) -> Image.Image:
        """Applies a circular gradient to simulate a lens vignette."""
        if not _NP: return image
            
        if image.mode != 'RGB': image = image.convert('RGB')
                
        w, h = image.size
        
        # --- START: Vectorized Mask Creation (Replaces the slow for loops) ---
        
        # 1. Create coordinate grids (h, w)
        y_coords, x_coords = _NP.indices((h, w))
        
        # 2. Calculate distance from center
        center_x, center_y = w / 2, h / 2
        radius = min(w, h) / 2
        
        dx = x_coords - center_x
        dy = y_coords - center_y
        dist = _NP.hypot(dx, dy)
        
        # 3. Calculate falloff
        normalized_dist = dist / radius
        # Power curve for smoother falloff, controlled by strength
        falloff = _NP.power(normalized_dist, (1.0 + (strength * 0.5)))
        
        # 4. Calculate vignette effect (0.0 at edge, 1.0 at center)
        # We clip to 0.0-1.0 range, max darkening is 60% (1.0 - 0.6 = 0.4)
        vignette_effect = _NP.clip(1.0 - (falloff * 0.6), 0.0, 1.0)
        
        # This is our new mask, shape (h, w)
        mask_np = vignette_effect
        # --- END: Vectorized Mask Creation ---

        image_np = _NP.array(image).astype(_NP.float32) / 255.0
        
        # Expand mask from (h, w) to (h, w, 1) for broadcasting
        mask_np = mask_np[:, :, _NP.newaxis] 
        
        # Multiply the image by the mask to darken the edges
        final_np = _NP.clip(image_np * mask_np, 0.0, 1.0)
        
        return Image.fromarray((final_np * 255).astype(_NP.uint8))

    def _scale_pixmap(self, qpixmap: QPixmap) -> QPixmap:
        """Scales a QPixmap to fit the ImageViewer."""
        if qpixmap.isNull():
            return qpixmap
            
        # Get actual available size in the QScrollArea's viewport
        avail_size = self.viewer_scroll.viewport().size()
        
        # Scale to fit while maintaining aspect ratio
        return qpixmap.scaled(
            avail_size, 
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation
        )

    def _display_scaled(self):
        """Renders the working image into the main viewer."""
        if not self.working_image_pil:
            return
            
        # --- START: High-Speed Preview Optimization ---
        
        # 1. Get target size for the preview
        viewport_size = self.viewer_scroll.viewport().size()
        
        # If viewport is tiny (e.g., window initializing), don't render
        if viewport_size.width() < 50 or viewport_size.height() < 50:
            return

        # 2. Calculate the "fit" size for the working image
        # We need to get the size of the full-res *rotated* image
        rotated_size = self.working_image_pil.rotate(self.rotation_degrees, expand=True).size
        
        # Use QSize's built-in scaling to find the target preview size
        target_qsize = QSize(rotated_size[0], rotated_size[1])
        target_qsize.scale(viewport_size, Qt.AspectRatioMode.KeepAspectRatio)
        
        target_size = (target_qsize.width(), target_qsize.height())

        # 3. Create a fast, downscaled preview *before* edits
        # We must rotate the *full-res* image first, *then* scale.
        
        rotated_image_full = self.working_image_pil.rotate(
            self.rotation_degrees, 
            expand=True,
            resample=Image.Resampling.NEAREST # Use NEAREST for fast preview rotation
        )
        
        # Now, create the downscaled preview
        preview_image = rotated_image_full.resize(
            target_size, 
            resample=Image.Resampling.BILINEAR # Use BILINEAR for fast preview scaling
        )
        
        # 4. Apply all edits to the *small preview* image
        final_image = self._apply_edits(preview_image)

        # 5. Convert to QPixmap
        # This pixmap is already the correct size for display
        qpixmap_to_display = pil_to_qpixmap(final_image)
        
        # 6. Update the viewer
        # We no longer call _scale_pixmap because it's already scaled.
        self.image_viewer.set_pixmap(qpixmap_to_display)
        
        # --- END: High-Speed Preview Optimization ---

    def resizeEvent(self, event: QtGui.QResizeEvent):
        """Re-scales the image when the main window resizes."""
        super().resizeEvent(event)
        self.update_timer.start()
   
    # ------------------------------- Editing Logic -------------------------------
    
    def rotate_image(self, degrees: int):
        """Updates rotation state and redraws."""
        if not self.working_image_pil:
            return
        
        self.rotation_degrees = (self.rotation_degrees + degrees) % 360
        self._display_scaled()
        
    # --- Saving methods ---

    def _infer_format_from_path(self, path: str) -> Optional[str]:
        return EXT_TO_FMT.get(Path(path).suffix.lower())

    def _ensure_heif_plugin_for_path(self, path: str, when: str) -> bool:
        ext = Path(path).suffix.lower()
        if ext in HEIF_LIKE_EXTS and not _HEIF_PLUGIN:
            QMessageBox.critical(
                self,
                "HEIF/AVIF support missing",
                f"You attempted to {when} a {ext} file but pillow-heif is not installed.\n\n"
                f"Install with:\n"
                f"   pip install pillow-heif\n"
                f"(On some systems, also install libheif.)",
            )
            return False
        return True

    def save_image(self):
        """Saves the final processed image, overwriting the original."""
        if not self.original_image_path:
            QMessageBox.warning(self, "Warning", "No image is currently loaded to save.")
            return False

        reply = QMessageBox.question(
            self,
            "Confirm Overwrite",
            f"Are you sure you want to overwrite the original file:\n{self.original_image_path}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            if self._perform_save(self.original_image_path):
                QMessageBox.information(self, "Save Success", f"Image saved and overwritten:\n{self.original_image_path}")
                # Reload the image to make the saved image the new 'original'
                self.load_image(self.original_image_path)
                return True
        return False
        
    def save_image_as(self):
        """Saves the final processed image to a new file."""
        if not self.original_image_path:
            QMessageBox.warning(self, "Warning", "No image is currently loaded to save.")
            return False

        default_ext = Path(self.original_image_path).suffix
        filetypes = (
            f"JPEG Files (*.jpg *.jpeg);;PNG Files (*.png);;TIFF Files (*.tif *.tiff);;WEBP Files (*.webp);;All Files (*.*)"
        )

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Image As",
            str(Path(self.original_image_path).parent / f"{Path(self.original_image_path).stem}_edited{default_ext}"),
            filetypes
        )
        
        if not save_path:
            return False

        if self._perform_save(save_path):
            QMessageBox.information(self, "Save Success", f"Image saved as:\n{save_path}")
            return True
        return False


    def _perform_save(self, save_path: str) -> bool:
        """Internal function to handle the actual image saving process."""
        try:
            if not self.working_image_pil: return False
            
            # 1. Apply Rotation
            # Use LANCZOS for highest quality resampling during final save
            rotated_image = self.working_image_pil.rotate(
                self.rotation_degrees, 
                expand=True,
                resample=Image.Resampling.LANCZOS
            )
            
            # 2. Apply all Edits
            final_image_to_save = self._apply_edits(rotated_image)

            p = Path(save_path)
            
            if not self._ensure_heif_plugin_for_path(save_path, "save"):
                return False

            save_fmt = (
                self._orig_format 
                or self._infer_format_from_path(save_path) 
                or EXT_TO_FMT.get(p.suffix.lower()) 
                or "PNG"
            ).upper()

            # Handle color mode conversions for specific formats
            if save_fmt == "JPEG" and final_image_to_save.mode not in ("RGB", "L"):
                final_image_to_save = final_image_to_save.convert("RGB")
            
            save_kwargs = {"format": save_fmt}
            
            # Transfer EXIF metadata if available
            if self._orig_exif and save_fmt in ("JPEG", "TIFF", "WEBP", "HEIF"):
                save_kwargs["exif"] = self._orig_exif
                
            # Set quality for JPEG/WEBP
            if save_fmt in ("JPEG", "WEBP"):
                 save_kwargs["quality"] = 90 # Default high quality

            # Use temp file approach for safety if overwriting
            if Path(save_path).exists() and save_path == self.original_image_path:
                tmp_path = p.with_name(f".tmp_{p.name}")
                final_image_to_save.save(str(tmp_path), **save_kwargs)
                os.replace(str(tmp_path), save_path)
            else:
                final_image_to_save.save(str(save_path), **save_kwargs)
            
            return True

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not save image: {e}")
            try:
                if "tmp_path" in locals() and Path(tmp_path).exists():
                    Path(tmp_path).unlink()
            except Exception:
                pass
            return False


# ------------------------------- main -------------------------------

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    if _CV2 is None or _NP is None:
        # Show warning about missing dependencies for full functionality
        QMessageBox.warning(
            None, # Parent is None here since main window isn't created yet
            "Missing Dependencies",
            "Full editing capability (Highlights/Shadows, Noise Reduction, Vibrance) "
            "requires 'opencv-python' and 'numpy'.\n\n"
            "Install with:\n"
            "   pip install opencv-python numpy"
        )

    editor = ImageEditorApp()
    editor.show()
    sys.exit(app.exec())