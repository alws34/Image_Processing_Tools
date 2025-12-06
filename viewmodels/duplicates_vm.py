#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import shutil
from pathlib import Path
from typing import List, Set, Optional, Callable
from PyQt6.QtCore import QObject, pyqtSignal, QThreadPool

# Core & Workers
from core.common import SUPPORTED_IMAGE_EXTS, SUPPORTED_LIVE_EXTS
from core.duplicates_logic import DuplicateRecord
from workers.tasks import ScanDuplicatesWorker, FastDuplicateWorker, ExactDuplicateWorker, VideoDuplicateWorker


class DuplicatesViewModel(QObject):
    # Signals to update the UI
    scan_started = pyqtSignal()
    scan_progress = pyqtSignal(int)
    scan_finished = pyqtSignal(str)  # Status message
    groups_updated = pyqtSignal()   # Data changed
    selection_changed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.threadpool = QThreadPool()
        self.duplicate_groups: List[List[DuplicateRecord]] = []
        self.selected_paths: Set[str] = set()
        self.current_folder: Optional[Path] = None

    def set_folder(self, path: Path):
        if isinstance(path, str):
            path = Path(path)
        self.current_folder = path

    def start_scan(self, mode: str, exact_match: bool):
        """
        Starts the background worker to find duplicates.
        Matches the 'start_scan' call expected by the UI.
        """
        if not self.current_folder or not self.current_folder.exists():
            self.scan_finished.emit("No valid folder selected.")
            return

        self.scan_started.emit()
        self.duplicate_groups.clear()
        self.selected_paths.clear()
        self.groups_updated.emit()

        # 1. Gather Paths (Main thread, fast filesystem walk)
        paths_to_scan = self._gather_paths(mode)

        if not paths_to_scan:
            self.scan_finished.emit(
                f"No {mode.lower()} found in {self.current_folder.name}.")
            return

        # 2. Select Worker based on mode
        # Using the aliases defined in workers/tasks.py for compatibility
        if exact_match:
            worker = ExactDuplicateWorker(paths_to_scan)
        else:
            if mode == "Images":
                worker = FastDuplicateWorker(paths_to_scan)
            else:
                worker = VideoDuplicateWorker(paths_to_scan)

        worker.signals.progress.connect(self.scan_progress.emit)
        worker.signals.finished.connect(self._on_scan_complete)
        self.threadpool.start(worker)

    def _gather_paths(self, mode: str) -> List[str]:
        paths = []
        is_images = (mode == "Images")
        exts = SUPPORTED_IMAGE_EXTS if is_images else SUPPORTED_LIVE_EXTS

        # Recursive walk with filters
        # Use string conversion for robust sorting
        try:
            for p in sorted(self.current_folder.rglob("*"), key=lambda x: str(x).lower()):
                if not p.is_file():
                    continue
                if p.name.startswith("."):
                    continue

                # Skip deleted folders (essential logic from old file)
                parts_lower = [part.lower() for part in p.parts]
                if "deleted" in parts_lower:
                    continue

                if p.suffix.lower() in exts:
                    paths.append(str(p))
        except Exception as e:
            print(f"Error gathering paths: {e}")

        return paths

    def _on_scan_complete(self, groups, msg):
        self.duplicate_groups = groups or []
        self.groups_updated.emit()
        self.scan_finished.emit(msg)

    # --- Selection Logic ---

    def toggle_selection(self, path: str):
        if path in self.selected_paths:
            self.selected_paths.remove(path)
        else:
            self.selected_paths.add(path)
        self.selection_changed.emit()

    def select_all_in_group(self, group_index: int):
        if not (0 <= group_index < len(self.duplicate_groups)):
            return

        group = self.duplicate_groups[group_index]
        changed = False
        for rec in group:
            if rec.path not in self.selected_paths:
                self.selected_paths.add(rec.path)
                changed = True

        if changed:
            self.selection_changed.emit()

    def get_paths_to_delete(self) -> List[str]:
        return sorted(list(self.selected_paths))

    # --- Deletion Logic ---

    def remove_paths_from_data(self, deleted_paths: List[str]):
        """
        Updates internal data structures after file deletion
        so the UI reflects the changes without a re-scan.
        """
        deleted_set = set(deleted_paths)
        new_groups = []

        for group in self.duplicate_groups:
            # Keep records that were NOT deleted
            pruned = [rec for rec in group if rec.path not in deleted_set]
            # Only keep groups that still have potential duplicates (>= 2 items)
            if len(pruned) >= 2:
                new_groups.append(pruned)

        self.duplicate_groups = new_groups
        self.selected_paths -= deleted_set
        self.groups_updated.emit()
        self.selection_changed.emit()
