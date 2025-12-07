#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import shutil
import re
from pathlib import Path
from typing import List, Set, Optional, Callable, Dict
from PyQt6.QtCore import QObject, pyqtSignal, QThreadPool

# Core & Workers
from core.common import SUPPORTED_IMAGE_EXTS, SUPPORTED_LIVE_EXTS
from core.duplicates_logic import DuplicateRecord
from workers.tasks import FastDuplicateWorker, ExactDuplicateWorker
from core.video_duplicates import VideoDuplicateWorker


class DuplicatesViewModel(QObject):
    # Signals to update the UI
    scan_started = pyqtSignal()
    scan_progress = pyqtSignal(int)
    scan_finished = pyqtSignal(str)  # Status message
    scan_summary = pyqtSignal(str)   # New: For Video stats
    groups_updated = pyqtSignal()    # Data changed
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
        if not self.current_folder or not self.current_folder.exists():
            self.scan_finished.emit("No valid folder selected.")
            return

        self.scan_started.emit()
        self.duplicate_groups.clear()
        self.selected_paths.clear()
        self.groups_updated.emit()

        paths_to_scan = self._gather_paths(mode)

        if not paths_to_scan:
            self.scan_finished.emit(
                f"No {mode.lower()} found in {self.current_folder.name}.")
            return

        if mode == "Videos":
            worker = VideoDuplicateWorker(paths_to_scan)
        else:
            if exact_match:
                worker = ExactDuplicateWorker(paths_to_scan)
            else:
                worker = FastDuplicateWorker(paths_to_scan)

        worker.signals.progress.connect(self.scan_progress.emit)
        worker.signals.finished.connect(self._on_scan_complete)
        self.threadpool.start(worker)

    def _gather_paths(self, mode: str) -> List[str]:
        paths = []
        is_images = (mode == "Images")
        exts = SUPPORTED_IMAGE_EXTS if is_images else SUPPORTED_LIVE_EXTS

        try:
            # Recursive walk
            for p in sorted(self.current_folder.rglob("*"), key=lambda x: str(x).lower()):
                if not p.is_file():
                    continue
                if p.name.startswith("."):
                    continue

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
        
        total_dupes = sum(len(g) for g in self.duplicate_groups)
        self.scan_summary.emit(f"Found {len(self.duplicate_groups)} groups ({total_dupes} files).")

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

    def select_all_except(self, group_index: int, keep_path: str):
        if not (0 <= group_index < len(self.duplicate_groups)):
            return

        group = self.duplicate_groups[group_index]
        changed = False
        
        if keep_path in self.selected_paths:
            self.selected_paths.remove(keep_path)
            changed = True
            
        for rec in group:
            if rec.path != keep_path:
                if rec.path not in self.selected_paths:
                    self.selected_paths.add(rec.path)
                    changed = True
        
        if changed:
            self.selection_changed.emit()

    def mark_same_dir_copies(self):
        """
        Scans all groups for duplicates located in the SAME directory.
        Enforces a 'Leave Only One' rule per folder.
        
        Preference for keeping files:
        1. Name starts with 'IMG_'
        2. Shortest filename length (e.g. keep 'file.jpg' over 'file (1).jpg')
        3. Alphabetical order
        """
        changed = False

        for group in self.duplicate_groups:
            # 1. Bucket by parent folder
            dir_map: Dict[str, List[DuplicateRecord]] = {}
            for rec in group:
                parent = str(Path(rec.path).parent)
                if parent not in dir_map:
                    dir_map[parent] = []
                dir_map[parent].append(rec)
            
            # 2. Process folders containing multiple duplicates
            for parent, records in dir_map.items():
                if len(records) < 2:
                    continue
                
                # Define sort key to find the "Best" candidate to keep
                def sort_key(rec):
                    p = Path(rec.path)
                    name = p.name
                    
                    # Priority 1: Starts with "IMG_" (False/0 sorts before True/1, so we invert check)
                    # We want "IMG_" to be 0 (first), others 1
                    is_img = 0 if name.upper().startswith("IMG_") else 1
                    
                    # Priority 2: Shortest filename (heuristic for original vs copy)
                    length = len(name)
                    
                    # Priority 3: Alphabetical (Tie-breaker)
                    return (is_img, length, name)

                # Sort: Best candidate is at index 0
                records.sort(key=sort_key)
                
                best_file = records[0]
                files_to_delete = records[1:]
                
                # Ensure the best file is KEPT (unselected)
                if best_file.path in self.selected_paths:
                    self.selected_paths.remove(best_file.path)
                    changed = True
                
                # Ensure all others are DELETED (selected)
                for rec in files_to_delete:
                    if rec.path not in self.selected_paths:
                        self.selected_paths.add(rec.path)
                        changed = True

        if changed:
            self.selection_changed.emit()
            return len(self.selected_paths)
        return 0

    def get_paths_to_delete(self) -> List[str]:
        return sorted(list(self.selected_paths))

    def remove_paths_from_data(self, deleted_paths: List[str]):
        deleted_set = set(deleted_paths)
        new_groups = []

        for group in self.duplicate_groups:
            pruned = [rec for rec in group if rec.path not in deleted_set]
            if len(pruned) >= 2:
                new_groups.append(pruned)

        self.duplicate_groups = new_groups
        self.selected_paths -= deleted_set
        self.groups_updated.emit()
        self.selection_changed.emit()