#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from pathlib import Path
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QListWidget, QSplitter, QScrollArea, QGridLayout,
    QSizePolicy, QCheckBox, QAbstractItemView, QMessageBox, QFileDialog,
    QMenu
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize, QThreadPool
from PyQt6.QtGui import (
    QPixmap, QKeySequence, QAction, QCursor,
    QPainter, QColor, QPen
)

# Core / Helpers
from core.common import SUPPORTED_LIVE_EXTS, pil_to_qpixmap, Image, ImageOps
from viewmodels.duplicates_vm import DuplicatesViewModel
from workers.tasks import ThumbnailLoaderJob

# --- 1. Helper Widget: Individual Thumbnail Item ---


class DuplicateItemWidget(QWidget):
    clicked = pyqtSignal(str)          # Left Click
    right_clicked = pyqtSignal(str)    # Right Click (for context menu)

    def __init__(self, image_path: str, caption: str, is_video: bool = False, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.is_video = is_video

        # UI Setup
        self.setFixedSize(200, 240)  # Fixed compact size

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet(
            "background-color: #222; border-radius: 4px;")

        # Ensure image label doesn't disappear
        self.image_label.setMinimumSize(180, 160)

        # Video Badge (Text in caption)
        prefix = "[VIDEO] " if is_video else ""

        self.caption_label = QLabel(prefix + caption)
        self.caption_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.caption_label.setWordWrap(True)
        self.caption_label.setStyleSheet(
            "font-size: 10px; color: #bbb; line-height: 120%;")

        vbox = QVBoxLayout(self)
        vbox.setContentsMargins(4, 4, 4, 4)
        vbox.setSpacing(4)
        # Removed stretch to prevent huge vertical gaps
        vbox.addWidget(self.image_label)
        vbox.addWidget(self.caption_label)

        self._selected = False
        self._orig_pixmap = None

        # Set loading state
        self.image_label.setText("Loading...")

    def set_selected(self, selected: bool):
        self._selected = selected
        self._update_rendering()

    def set_thumbnail(self, qimg):
        """Called by the async loader when the image is ready."""
        if qimg:
            self._orig_pixmap = QPixmap.fromImage(qimg)
            self._update_rendering()
        else:
            self.image_label.setText("No Preview")

    # ------------------------------------------------------------------ Rendering

    def _update_rendering(self):
        """Updates the pixmap with selection overlay if needed."""
        if not self._orig_pixmap:
            return

        # 1. Determine size to scale to
        size = self.image_label.size()
        if size.width() <= 10:
            size = QSize(180, 160)  # Default fallback

        # 2. Scale Image
        pm = self._orig_pixmap.scaled(
            size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        # 3. Apply Blue Overlay if selected
        if self._selected:
            result = QPixmap(pm.size())
            result.fill(Qt.GlobalColor.transparent)

            painter = QPainter(result)
            painter.drawPixmap(0, 0, pm)

            # Draw Blue Overlay (R, G, B, Alpha)
            painter.fillRect(result.rect(), QColor(0, 150, 255, 100))

            # Blue Border
            # FIX: Create QPen object correctly
            pen = QPen(QColor(0, 150, 255))
            pen.setWidth(6)  # Thicker border for visibility
            painter.setPen(pen)

            # Draw rect inside the boundary so it doesn't clip
            rect = result.rect()
            rect.adjust(2, 2, -2, -2)
            painter.drawRect(rect)

            painter.end()
            self.image_label.setPixmap(result)

            self.caption_label.setStyleSheet(
                "font-size: 10px; color: #66ccff; font-weight: bold;")
            self.setStyleSheet(
                "background-color: #2a2a2a; border-radius: 6px;")
        else:
            # Normal State
            self.image_label.setPixmap(pm)
            self.caption_label.setStyleSheet("font-size: 10px; color: #bbb;")
            self.setStyleSheet("background-color: transparent;")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_rendering()

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
        self.grid.setAlignment(Qt.AlignmentFlag.AlignTop |
                               Qt.AlignmentFlag.AlignLeft)
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
        # Detach all
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
        self.lbl_preview_img.setMinimumSize(200, 200)
        self.lbl_preview_img.setStyleSheet("background-color: black;")

        self.scroll_preview = QScrollArea()
        self.scroll_preview.setWidgetResizable(True)
        self.scroll_preview.setWidget(self.lbl_preview_img)

        right_layout.addWidget(self.lbl_preview_title)
        right_layout.addWidget(self.scroll_preview)
        splitter.addWidget(right_panel)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 2)

        # Bottom bar
        bottom_bar = QHBoxLayout()
        self.btn_export = QPushButton("Export List")
        self.btn_delete = QPushButton("Delete Selected")
        self.btn_delete.setStyleSheet("color: #ffcccc; font-weight: bold;")
        self.btn_delete.clicked.connect(self.on_delete_request)

        self.lbl_status = QLabel("Ready.")

        bottom_bar.addWidget(self.btn_export)
        bottom_bar.addStretch(1)
        bottom_bar.addWidget(self.lbl_status)
        bottom_bar.addStretch(1)
        bottom_bar.addWidget(self.btn_delete)

        layout.addLayout(bottom_bar)

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
        # Restore selection index to avoid jumping to top
        current_idx = self.list_groups.currentRow()

        self.list_groups.clear()
        self.grid_container.clear_items()
        self._thumb_widgets.clear()
        self.lbl_preview_img.clear()
        self.lbl_preview_title.setText("Preview")

        count = len(self.vm.duplicate_groups)
        for i, group in enumerate(self.vm.duplicate_groups):
            name = f"Group {i+1} ({len(group)})"
            try:
                name += f" - {Path(group[0].path).parent.name}"
            except:
                pass
            self.list_groups.addItem(name)

        if count > 0:
            # Restore previous index if valid, else clamp
            if current_idx >= 0 and current_idx < count:
                self.list_groups.setCurrentRow(current_idx)
            else:
                self.list_groups.setCurrentRow(0)

    def _load_group_thumbs(self, row):
        if row < 0 or row >= len(self.vm.duplicate_groups):
            return

        group = self.vm.duplicate_groups[row]
        self.grid_container.clear_items()
        self._thumb_widgets.clear()

        # 1. Create Widgets
        for rec in group:
            p = Path(rec.path)
            is_video = p.suffix.lower() in SUPPORTED_LIVE_EXTS

            try:
                size_mb = p.stat().st_size / (1024*1024)
                # Show Parent Folder Name
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

        # Auto-preview first item
        if group:
            self._show_preview(group[0].path)

    def _on_thumb_loaded(self, path, qimg):
        if path in self._thumb_widgets:
            self._thumb_widgets[path].set_thumbnail(qimg)

    def _on_thumb_clicked(self, path):
        self.vm.toggle_selection(path)
        self._show_preview(path)

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
        import subprocess
        import platform
        import os
        p = Path(path).parent
        if platform.system() == "Windows":
            os.startfile(p)
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", str(p)])
        else:
            subprocess.Popen(["xdg-open", str(p)])

    def _show_preview(self, path):
        self.lbl_preview_title.setText(Path(path).name)

        # FIX: Use Pillow to load for HEIC support
        try:
            with Image.open(path) as im:
                im = ImageOps.exif_transpose(im)
                # Convert to RGBA for Qt
                if im.mode != "RGBA":
                    im = im.convert("RGBA")

                # Convert to QPixmap
                qimg = pil_to_qpixmap(im)

                # Render to label
                target = self.scroll_preview.viewport().size()
                if target.width() > 0:
                    self.lbl_preview_img.setPixmap(
                        qimg.scaled(target, Qt.AspectRatioMode.KeepAspectRatio,
                                    Qt.TransformationMode.SmoothTransformation)
                    )
                else:
                    self.lbl_preview_img.setPixmap(qimg)
        except Exception as e:
            self.lbl_preview_img.setText(f"Preview Error\n{str(e)}")

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

    def on_delete_request(self):
        """Called by MainWindow via global delete key or local button."""
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
