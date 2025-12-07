#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from pathlib import Path
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QListWidget, QSplitter, QScrollArea, QGridLayout,
    QSizePolicy, QCheckBox, QAbstractItemView, QMessageBox, QFileDialog,
    QMenu, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize, QThreadPool, QTimer, QEvent, QRunnable, QObject
from PyQt6.QtGui import (
    QPixmap, QKeySequence, QAction, QCursor, 
    QPainter, QColor, QPen, QImage, QKeyEvent
)

# Core / Helpers
from core.common import SUPPORTED_LIVE_EXTS, pil_to_qpixmap, Image, ImageOps, _CV2
from viewmodels.duplicates_vm import DuplicatesViewModel
from workers.tasks import ThumbnailLoaderJob

# --- 0. Async Preview Loader ---

class PreviewSignals(QObject):
    loaded = pyqtSignal(str, object) # path, QPixmap (or None)

class PreviewLoaderJob(QRunnable):
    """
    Background job to load the large preview image/video frame.
    This prevents the UI from freezing when clicking items.
    """
    def __init__(self, path: str, target_size: QSize, is_video: bool):
        super().__init__()
        self.path = path
        self.target_size = target_size
        self.is_video = is_video
        self.signals = PreviewSignals()

    def run(self):
        try:
            pixmap = None
            
            if self.is_video:
                if _CV2 is not None:
                    cap = _CV2.VideoCapture(self.path)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        cap.release()
                        if ret and frame is not None:
                            rgb = _CV2.cvtColor(frame, _CV2.COLOR_BGR2RGB)
                            h, w, ch = rgb.shape
                            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
                            pixmap = QPixmap.fromImage(qimg)
            else:
                # Image
                with Image.open(self.path) as im:
                    im = ImageOps.exif_transpose(im)
                    if im.mode != "RGBA":
                        im = im.convert("RGBA")
                    
                    # Pre-scale in PIL for speed if it's huge, but keep quality high
                    # We usually want to fit within the view
                    w, h = im.size
                    # Cap max load size to something reasonable (e.g. 2000px) to save RAM/Time
                    if max(w, h) > 2500:
                        im.thumbnail((2500, 2500), Image.Resampling.LANCZOS)
                    
                    qimg = pil_to_qpixmap(im)
                    pixmap = qimg # Convert to pixmap on main thread usually better, but QImage is safe
            
            self.signals.loaded.emit(self.path, pixmap)
            
        except Exception as e:
            # On error, just emit None
            print(f"Preview load error: {e}")
            self.signals.loaded.emit(self.path, None)


# --- 1. Helper Widget: Individual Thumbnail Item ---

class DuplicateItemWidget(QWidget):
    clicked = pyqtSignal(str)          # Left Click
    right_clicked = pyqtSignal(str)    # Right Click (for context menu)

    def __init__(self, image_path: str, caption: str, is_video: bool = False, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.is_video = is_video
        
        # UI Setup
        self.setFixedSize(200, 240) # Fixed compact size

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("background-color: #222; border-radius: 4px;")
        
        # Ensure image label doesn't disappear
        self.image_label.setMinimumSize(180, 160)

        # Video Badge
        prefix = "[VIDEO] " if is_video else ""

        self.caption_label = QLabel(prefix + caption)
        self.caption_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.caption_label.setWordWrap(True)
        self.caption_label.setStyleSheet("font-size: 10px; color: #bbb; line-height: 120%;")

        vbox = QVBoxLayout(self)
        vbox.setContentsMargins(4, 4, 4, 4)
        vbox.setSpacing(4)
        vbox.addWidget(self.image_label)
        vbox.addWidget(self.caption_label)

        self._selected = False
        self._orig_pixmap = None
        self._cached_scaled_pixmap = None
        self._last_size = QSize(0, 0)

        # Set loading state
        self.image_label.setText("Loading...")
        
    def set_selected(self, selected: bool):
        if self._selected != selected:
            self._selected = selected
            self._update_overlay_only()

    def set_thumbnail(self, qimg):
        """Called by the async loader when the image is ready."""
        if qimg:
            self._orig_pixmap = QPixmap.fromImage(qimg)
            self._update_base_rendering()
        else:
            self.image_label.setText("No Preview")

    def _update_base_rendering(self):
        """Re-scales the base image. Called on load or resize."""
        if not self._orig_pixmap:
            return

        size = self.image_label.size()
        if size.width() <= 10:
            size = QSize(180, 160)
        
        # Only re-scale if size changed significantly
        if self._cached_scaled_pixmap is None or size != self._last_size:
            self._cached_scaled_pixmap = self._orig_pixmap.scaled(
                size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._last_size = size
        
        self._update_overlay_only()

    def _update_overlay_only(self):
        """Draws the blue overlay on top of the cached scaled image. Fast."""
        if not self._cached_scaled_pixmap:
            return

        if self._selected:
            # Copy cached pixmap to draw overlay
            result = QPixmap(self._cached_scaled_pixmap)
            
            painter = QPainter(result)
            # Draw Blue Overlay
            painter.fillRect(result.rect(), QColor(0, 150, 255, 100)) 
            
            # Blue Border
            pen = QPen(QColor(0, 150, 255))
            pen.setWidth(6)
            painter.setPen(pen)
            
            rect = result.rect()
            rect.adjust(3, 3, -3, -3) 
            painter.drawRect(rect)
            
            painter.end()
            self.image_label.setPixmap(result)
            
            self.caption_label.setStyleSheet("font-size: 10px; color: #66ccff; font-weight: bold;")
            self.setStyleSheet("background-color: #2a2a2a; border-radius: 6px;")
        else:
            # Just show cached pixmap
            self.image_label.setPixmap(self._cached_scaled_pixmap)
            self.caption_label.setStyleSheet("font-size: 10px; color: #bbb;")
            self.setStyleSheet("background-color: transparent;")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_base_rendering()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.image_path)
        elif event.button() == Qt.MouseButton.RightButton:
            self.right_clicked.emit(self.image_path)
        else:
            super().mousePressEvent(event)


# --- 2. Grid Container ---

class DupThumbGridContainer(QWidget):
    MIN_ITEM_WIDTH = 210

    def __init__(self, parent=None):
        super().__init__(parent)
        self.grid = QGridLayout(self)
        self.grid.setContentsMargins(10, 10, 10, 10)
        self.grid.setSpacing(10)
        self.grid.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self._widgets = []

    def clear_items(self):
        while self.grid.count():
            item = self.grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._widgets.clear()

    def add_widget(self, widget: QWidget):
        self._widgets.append(widget)
        self._relayout()

    def _relayout(self):
        for i in reversed(range(self.grid.count())):
            self.grid.itemAt(i).widget().setParent(None)

        if not self._widgets:
            return

        width = self.width() if self.width() > 0 else 800
        cols = max(1, width // self.MIN_ITEM_WIDTH)

        for i, w in enumerate(self._widgets):
            row = i // cols
            col = i % cols
            self.grid.addWidget(w, row, col)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._relayout()


# --- 3. Main Duplicates Tab ---

class DuplicatesTab(QWidget):
    def __init__(self, vm: DuplicatesViewModel, parent=None):
        super().__init__(parent)
        self.vm = vm
        self.threadpool = QThreadPool() 

        self._thumb_widgets: dict[str, DuplicateItemWidget] = {}
        
        # Video Playback State
        self.cap = None
        self.video_timer = QTimer()
        self.video_timer.timeout.connect(self._on_video_tick)
        self.is_playing = False
        self.current_preview_path = None
        self.pending_preview_path = None # For async tracking

        self._init_ui()
        self._wire_vm()
        self._wire_actions()

    # ------------------------------------------------------------------ UI setup

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        # Top control bar
        top_bar = QHBoxLayout()
        self.btn_browse = QPushButton("Browse Folder")
        self.cmb_scan_type = QComboBox()
        self.cmb_scan_type.addItems(
            ["Images (.jpg/.png/...)", "Videos (.mov/.mp4/...)"])

        self.chk_recursive = QCheckBox("Recursive")
        self.chk_exact = QCheckBox("Exact match (slow)")
        self.chk_exact.setChecked(True)

        self.btn_scan = QPushButton("Scan Duplicates")
        self.btn_scan.setStyleSheet(
            "background-color: #2A82DA; color: white; font-weight: bold;")

        top_bar.addWidget(self.btn_browse)
        top_bar.addWidget(self.cmb_scan_type)
        top_bar.addWidget(self.chk_exact)
        top_bar.addWidget(self.btn_scan)
        top_bar.addStretch(1)

        layout.addLayout(top_bar)

        # Middle splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter, stretch=1)

        # Left: Groups List
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.lbl_groups = QLabel("Duplicate groups:")
        self.list_groups = QListWidget()
        self.list_groups.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection)
        self.list_groups.currentRowChanged.connect(self._load_group_thumbs)

        left_layout.addWidget(self.lbl_groups)
        left_layout.addWidget(self.list_groups)
        splitter.addWidget(left_panel)

        # Center: Thumbnails Grid
        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(0, 0, 0, 0)

        self.lbl_instruction = QLabel("Select items to DELETE (Blue Overlay)")
        self.lbl_instruction.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_instruction.setStyleSheet("color: #888; font-style: italic;")
        center_layout.addWidget(self.lbl_instruction)

        self.scroll_thumbs = QScrollArea()
        self.scroll_thumbs.setWidgetResizable(True)
        # IMPORTANT: Install event filter to capture arrow keys from the scroll area
        self.scroll_thumbs.installEventFilter(self)
        
        self.grid_container = DupThumbGridContainer()
        self.scroll_thumbs.setWidget(self.grid_container)

        center_layout.addWidget(self.scroll_thumbs)
        splitter.addWidget(center_panel)

        # Right: Preview
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.lbl_preview_title = QLabel("Preview")
        self.lbl_preview_img = QLabel()
        self.lbl_preview_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_preview_img.setMinimumSize(320, 240)
        self.lbl_preview_img.setStyleSheet("background-color: black;")

        self.scroll_preview = QScrollArea()
        self.scroll_preview.setWidgetResizable(True)
        self.scroll_preview.setWidget(self.lbl_preview_img)
        
        # Play/Pause Button for Video
        self.btn_preview_play = QPushButton("Play Video")
        self.btn_preview_play.setCheckable(True)
        self.btn_preview_play.clicked.connect(self._toggle_video_playback)
        self.btn_preview_play.setVisible(False) 

        right_layout.addWidget(self.lbl_preview_title)
        right_layout.addWidget(self.scroll_preview)
        right_layout.addWidget(self.btn_preview_play)
        
        splitter.addWidget(right_panel)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 2)

        # Bottom bar
        bottom_bar = QHBoxLayout()
        
        # Auto-Mark Button
        self.btn_auto_mark = QPushButton("Auto-Mark Same Dir")
        self.btn_auto_mark.setToolTip("Selects files with 'copy' suffixes (e.g. ' 2', '(1)') for deletion if an original exists in the same folder.")
        self.btn_auto_mark.clicked.connect(self._auto_mark_copies)

        self.btn_export = QPushButton("Export List")
        self.btn_delete = QPushButton("Delete Selected")
        self.btn_delete.setStyleSheet("color: #ffcccc; font-weight: bold;")
        self.btn_delete.clicked.connect(self.on_delete_request)

        self.lbl_status = QLabel("Ready.")

        bottom_bar.addWidget(self.btn_auto_mark) 
        bottom_bar.addWidget(self.btn_export)
        bottom_bar.addStretch(1)
        bottom_bar.addWidget(self.lbl_status)
        bottom_bar.addStretch(1)
        bottom_bar.addWidget(self.btn_delete)

        layout.addLayout(bottom_bar)
        
        # Ensure focus policy allows capturing keys
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    # ------------------------------------------------------------------ Keyboard & Event Filter

    def keyPressEvent(self, event: QKeyEvent):
        """Handle Global Key Presses in this tab."""
        self._handle_navigation_keys(event)
        super().keyPressEvent(event)

    def eventFilter(self, source, event):
        """Intercept arrow keys from child widgets (like ScrollArea) to navigate groups."""
        if event.type() == QEvent.Type.KeyPress:
            if self._handle_navigation_keys(event):
                return True # Event consumed
        return super().eventFilter(source, event)

    def _handle_navigation_keys(self, event: QKeyEvent) -> bool:
        key = event.key()
        
        # 1. Navigation (Up/Down) -> Change Group
        if key in (Qt.Key.Key_Up, Qt.Key.Key_Down):
            row = self.list_groups.currentRow()
            count = self.list_groups.count()
            
            if count == 0:
                return False

            if key == Qt.Key.Key_Up:
                new_row = max(0, row - 1)
            else:
                new_row = min(count - 1, row + 1)

            if new_row != row:
                self.list_groups.setCurrentRow(new_row)
            return True # We handled it
            
        # 2. Deletion
        if key == Qt.Key.Key_Delete:
            self.on_delete_request()
            return True

        return False

    # ------------------------------------------------------------------ VM wiring

    def _wire_vm(self):
        self.vm.scan_started.connect(self._on_scan_started)
        self.vm.scan_progress.connect(self._on_scan_progress)
        self.vm.scan_finished.connect(self._on_scan_finished)
        if hasattr(self.vm, 'scan_summary'):
            self.vm.scan_summary.connect(self.lbl_status.setText)
        self.vm.selection_changed.connect(self._update_highlights)

    def _wire_actions(self):
        self.btn_browse.clicked.connect(self._browse_folder)
        self.btn_scan.clicked.connect(self._start_scan)
        self.btn_export.clicked.connect(self._export)

        # Shortcuts
        act_select_all = QAction(self)
        act_select_all.setShortcut(QKeySequence.StandardKey.SelectAll)
        act_select_all.triggered.connect(self._on_select_all_group)
        self.addAction(act_select_all)

    # ------------------------------------------------------------------ Event handlers

    def _on_scan_started(self):
        self.stop_video_preview()
        self.lbl_status.setText("Scanning...")
        self.btn_scan.setEnabled(False)
        self.list_groups.clear()
        self.grid_container.clear_items()

    def _on_scan_progress(self, val):
        if isinstance(val, int):
            self.lbl_status.setText(f"Scanning... {val}%")
        else:
            self.lbl_status.setText(str(val))

    def _on_scan_finished(self, msg: str):
        self._refresh_group_list()
        self.lbl_status.setText(msg)
        self.btn_scan.setEnabled(True)

    def _refresh_group_list(self):
        current_idx = self.list_groups.currentRow()
        
        self.list_groups.clear()
        self.grid_container.clear_items()
        self._thumb_widgets.clear()
        self.lbl_preview_img.clear()
        self.lbl_preview_title.setText("Preview")
        self.stop_video_preview()

        count = len(self.vm.duplicate_groups)
        for i, group in enumerate(self.vm.duplicate_groups):
            name = f"Group {i+1} ({len(group)})"
            try:
                name += f" - {Path(group[0].path).parent.name}"
            except:
                pass
            self.list_groups.addItem(name)

        if count > 0:
            if current_idx >= 0 and current_idx < count:
                self.list_groups.setCurrentRow(current_idx)
            else:
                self.list_groups.setCurrentRow(0)

    def _load_group_thumbs(self, row):
        if row < 0 or row >= len(self.vm.duplicate_groups):
            return
        
        self.stop_video_preview()

        group = self.vm.duplicate_groups[row]
        self.grid_container.clear_items()
        self._thumb_widgets.clear()

        # 1. Create Widgets 
        for rec in group:
            p = Path(rec.path)
            is_video = p.suffix.lower() in SUPPORTED_LIVE_EXTS

            try:
                size_mb = p.stat().st_size / (1024*1024)
                caption = f"[{p.parent.name}]\n{p.name}\n{size_mb:.1f} MB"
            except:
                caption = p.name

            w = DuplicateItemWidget(rec.path, caption, is_video=is_video)
            w.clicked.connect(self._on_thumb_clicked)
            w.right_clicked.connect(self._on_thumb_right_clicked)

            self._thumb_widgets[rec.path] = w
            self.grid_container.add_widget(w)

            # 2. Fire Async Loader
            loader = ThumbnailLoaderJob(rec.path, size=300)
            loader.signals.loaded.connect(self._on_thumb_loaded)
            self.threadpool.start(loader)

        self._update_highlights()

        # Auto-preview first item (Async)
        if group:
            self._request_preview(group[0].path)

    def _on_thumb_loaded(self, path, qimg):
        if path in self._thumb_widgets:
            self._thumb_widgets[path].set_thumbnail(qimg)

    def _on_thumb_clicked(self, path):
        # Immediate UI feedback handled by _update_highlights via VM signal
        # but VM logic is fast. The preview loading is what was slow.
        self.vm.toggle_selection(path)
        self._request_preview(path)

    def _on_thumb_right_clicked(self, path):
        menu = QMenu(self)
        act_keep = QAction("Keep This One (Select Others for Deletion)", self)
        act_keep.triggered.connect(lambda: self._on_keep_this(path))
        menu.addAction(act_keep)
        
        act_open = QAction("Reveal in Explorer/Finder", self)
        act_open.triggered.connect(lambda: self._reveal_in_os(path))
        menu.addAction(act_open)
        menu.exec(QCursor.pos())

    def _on_keep_this(self, keep_path):
        row = self.list_groups.currentRow()
        self.vm.select_all_except(row, keep_path)

    def _reveal_in_os(self, path):
        import subprocess, platform, os
        p = Path(path).parent
        if platform.system() == "Windows":
            os.startfile(p)
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", str(p)])
        else:
            subprocess.Popen(["xdg-open", str(p)])

    # ------------------------------------------------------------------ Async Preview Logic

    def _request_preview(self, path):
        if self.current_preview_path == path:
            return
            
        self.stop_video_preview()
        self.current_preview_path = path
        self.pending_preview_path = path # Track what we want
        
        self.lbl_preview_title.setText(Path(path).name)
        self.lbl_preview_img.setText("Loading Preview...")
        
        # Check Video
        ext = Path(path).suffix.lower()
        if ext in SUPPORTED_LIVE_EXTS:
            self.btn_preview_play.setVisible(True)
            self.btn_preview_play.setText("Play Video")
            self.btn_preview_play.setChecked(False)
            
            # For video, we still use the async loader for the first frame
            job = PreviewLoaderJob(path, self.scroll_preview.viewport().size(), is_video=True)
            job.signals.loaded.connect(self._on_preview_loaded)
            self.threadpool.start(job)
            return

        # Image
        self.btn_preview_play.setVisible(False)
        job = PreviewLoaderJob(path, self.scroll_preview.viewport().size(), is_video=False)
        job.signals.loaded.connect(self._on_preview_loaded)
        self.threadpool.start(job)

    def _on_preview_loaded(self, path, pixmap):
        # Ignore if the user has already clicked another image
        if path != self.pending_preview_path:
            return
            
        if pixmap and not pixmap.isNull():
            # Scale to fit if needed
            target = self.scroll_preview.viewport().size()
            if target.width() > 0:
                scaled = pixmap.scaled(target, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                self.lbl_preview_img.setPixmap(scaled)
            else:
                self.lbl_preview_img.setPixmap(pixmap)
            
            # If video, prepare player (but don't play yet)
            ext = Path(path).suffix.lower()
            if ext in SUPPORTED_LIVE_EXTS:
                 self._prepare_video_player(path)
        else:
            self.lbl_preview_img.setText("Preview Failed")

    def _prepare_video_player(self, path):
        if _CV2 is None: return
        self.cap = _CV2.VideoCapture(path)
        # We already showed the first frame via the async loader

    def _toggle_video_playback(self):
        if not self.cap or not self.cap.isOpened():
            # Try reloading if missing (e.g. if async loader finished but player wasn't ready)
            if self.current_preview_path:
                self._prepare_video_player(self.current_preview_path)
        
        if not self.cap or not self.cap.isOpened():
            return

        if self.is_playing:
            self.video_timer.stop()
            self.btn_preview_play.setText("Play Video")
            self.is_playing = False
        else:
            self.video_timer.start(33) 
            self.btn_preview_play.setText("Pause Video")
            self.is_playing = True

    def _on_video_tick(self):
        if not self.cap: return
        ret, frame = self.cap.read()
        if ret:
            self._display_cv_frame(frame)
        else:
            self.cap.set(_CV2.CAP_PROP_POS_FRAMES, 0)

    def _display_cv_frame(self, frame):
        rgb = _CV2.cvtColor(frame, _CV2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        
        target = self.scroll_preview.viewport().size()
        if target.width() > 0:
            pix = pix.scaled(target, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.lbl_preview_img.setPixmap(pix)

    def stop_video_preview(self):
        self.video_timer.stop()
        self.is_playing = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.btn_preview_play.setVisible(False)

    def _update_highlights(self):
        for path, widget in self._thumb_widgets.items():
            widget.set_selected(path in self.vm.selected_paths)

    # ------------------------------------------------------------------ Actions

    def _browse_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Select Folder")
        if d:
            self.vm.set_folder(d)
            self.lbl_status.setText(f"Selected: {d}")

    def _start_scan(self):
        mode = "Videos" if self.cmb_scan_type.currentIndex() == 1 else "Images"
        exact = self.chk_exact.isChecked()
        self.vm.start_scan(mode, exact)

    def _on_select_all_group(self):
        row = self.list_groups.currentRow()
        self.vm.select_all_in_group(row)
        
    def _auto_mark_copies(self):
        count = self.vm.mark_same_dir_copies()
        if count > 0:
            QMessageBox.information(self, "Auto-Mark", f"Marked {count} copy files for deletion.\n\nPlease review blue items, then click 'Delete Selected' to confirm.")
        else:
            QMessageBox.information(self, "Auto-Mark", "No 'same-directory' copy files found.")

    def on_delete_request(self):
        paths = self.vm.get_paths_to_delete()
        if not paths:
            return

        reply = QMessageBox.question(
            self, "Delete",
            f"Move {len(paths)} files to 'deleted' folder?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            count = 0
            for p in paths:
                try:
                    src = Path(p)
                    dst_dir = src.parent / "deleted"
                    dst_dir.mkdir(exist_ok=True)
                    import shutil
                    shutil.move(str(src), str(dst_dir / src.name))
                    count += 1
                except Exception as e:
                    print(f"Error moving {p}: {e}")

            self.vm.remove_paths_from_data(paths)
            self._refresh_group_list()
            self.lbl_status.setText(f"Moved {count} files to 'deleted'.")

    def _export(self):
        paths = self.vm.get_paths_to_delete()
        if not paths:
            return
        fname, _ = QFileDialog.getSaveFileName(
            self, "Save List", "delete_list.txt")
        if fname:
            with open(fname, "w", encoding="utf-8") as f:
                f.write("\n".join(paths))