#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import os
from pathlib import Path

# Try to import OpenCV and NumPy for video processing
_CV2 = None
_NP = None
try:
    import cv2
    _CV2 = cv2
    import numpy as np
    _NP = np
except Exception:
    _CV2 = None
    _NP = None

SUPPORTED_EXTS = {".mov"}


class LivePhotoApp:
    def __init__(self, master: tk.Tk):
        self.master = master
        master.title("Live Photo (.mov) Frame Tool")
        master.withdraw()  # hide until centered

        # --- State ---
        self.current_folder: Path | None = None
        self.mov_files: list[Path] = []
        self.current_mov_index: int = -1

        self.video_frames: list[Image.Image] = []
        self.video_fps: float = 30.0
        self.current_frame_idx: int = 0
        self.is_playing: bool = False
        self._play_job = None
        self._load_job = None

        self.tk_video_frame: ImageTk.PhotoImage | None = None
        self.tk_preview_frame: ImageTk.PhotoImage | None = None

        # --- Layout ---
        master.rowconfigure(1, weight=1)
        master.columnconfigure(1, weight=1)

        # --- Top Control Frame (Folder Browser) ---
        self.top_frame = tk.Frame(master)
        self.top_frame.grid(row=0, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
        self.top_frame.columnconfigure(1, weight=1)

        self.browse_button = tk.Button(
            self.top_frame, text="Browse Folder...", command=self.load_folder_dialog
        )
        self.browse_button.grid(row=0, column=0, padx=5, sticky="w")

        self.path_entry = tk.Entry(self.top_frame)
        self.path_entry.grid(row=0, column=1, padx=10, sticky="ew")
        self.path_entry.bind("<KeyRelease>", self.load_from_path_event)
        self.path_entry.bind("<FocusOut>", self.load_from_path_event)
        self.path_entry.bind("<Return>", self.load_from_path_event)

        self.status_label = tk.Label(
            self.top_frame, text="No folder loaded", anchor="w"
        )
        self.status_label.grid(row=1, column=0, columnspan=2, padx=10, sticky="ew")

        # --- Left Frame (.mov Listbox) ---
        self.list_frame = tk.Frame(master)
        self.list_frame.grid(row=1, column=0, padx=(10, 5), pady=5, sticky="nsw")
        self.list_frame.rowconfigure(0, weight=1)

        list_scrollbar = tk.Scrollbar(self.list_frame, orient=tk.VERTICAL)
        self.listbox = tk.Listbox(
            self.list_frame,
            width=30,
            yscrollcommand=list_scrollbar.set,
            exportselection=False,
        )
        list_scrollbar.config(command=self.listbox.yview)
        list_scrollbar.grid(row=0, column=1, sticky="ns")
        self.listbox.grid(row=0, column=0, sticky="nsw")
        self.listbox.bind("<<ListboxSelect>>", self.on_mov_list_select)

        # --- Main Viewer Frame ---
        self.viewer_frame = tk.Frame(master, borderwidth=2, relief="groove")
        self.viewer_frame.grid(row=1, column=1, padx=(5, 10), pady=5, sticky="nsew")
        self.viewer_frame.rowconfigure(0, weight=1)
        self.viewer_frame.columnconfigure(0, weight=1)
        self.viewer_frame.columnconfigure(1, weight=1) # Two columns: Player, Frames

        # --- Player Panel (Left side of viewer) ---
        player_panel = tk.Frame(self.viewer_frame)
        player_panel.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        player_panel.rowconfigure(1, weight=1)
        player_panel.columnconfigure(0, weight=1)

        tk.Label(player_panel, text="Video Player").grid(row=0, column=0)
        self.player_label = tk.Label(player_panel, text="No .mov loaded", bg="black")
        self.player_label.grid(row=1, column=0, sticky="nsew")

        self.play_button = tk.Button(
            player_panel, text="Play", command=self.toggle_play_stop, state=tk.DISABLED
        )
        self.play_button.grid(row=2, column=0, pady=5)

        # --- Frames Panel (Right side of viewer) ---
        frames_panel = tk.Frame(self.viewer_frame)
        frames_panel.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        frames_panel.rowconfigure(1, weight=1) # Frame list
        frames_panel.rowconfigure(3, weight=1) # Frame preview
        frames_panel.columnconfigure(0, weight=1)

        tk.Label(frames_panel, text="Frames").grid(row=0, column=0)

        # Frame listbox
        frame_list_frame = tk.Frame(frames_panel)
        frame_list_frame.grid(row=1, column=0, sticky="nsew")
        frame_list_frame.rowconfigure(0, weight=1)
        frame_list_frame.columnconfigure(0, weight=1)
        frame_list_scrollbar = tk.Scrollbar(frame_list_frame, orient=tk.VERTICAL)
        self.frame_listbox = tk.Listbox(
            frame_list_frame,
            height=10,
            yscrollcommand=frame_list_scrollbar.set,
            exportselection=False,
        )
        frame_list_scrollbar.config(command=self.frame_listbox.yview)
        frame_list_scrollbar.grid(row=0, column=1, sticky="ns")
        self.frame_listbox.grid(row=0, column=0, sticky="nsew")
        self.frame_listbox.bind("<<ListboxSelect>>", self.on_frame_select)

        # Frame preview
        tk.Label(frames_panel, text="Preview Selected Frame").grid(row=2, column=0, pady=(10, 2))
        self.frame_preview_label = tk.Label(frames_panel, text="", bg="darkgrey")
        self.frame_preview_label.grid(row=3, column=0, sticky="nsew")

        self.save_frame_button = tk.Button(
            frames_panel, text="Save Selected Frame", command=self.save_selected_frame, state=tk.DISABLED
        )
        self.save_frame_button.grid(row=4, column=0, pady=5)


        # --- Bottom Control Frame (Navigation) ---
        self.control_frame = tk.Frame(master)
        self.control_frame.grid(
            row=2, column=0, columnspan=2, padx=10, pady=5, sticky="ew"
        )
        self.control_frame.columnconfigure(1, weight=1) # Center alignment

        self.prev_button = tk.Button(
            self.control_frame, text="< Prev", command=self.prev_mov, state=tk.DISABLED
        )
        self.prev_button.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.next_button = tk.Button(
            self.control_frame, text="Next >", command=self.next_mov, state=tk.DISABLED
        )
        self.next_button.grid(row=0, column=2, padx=5, pady=5, sticky="e")

        # --- Bindings ---
        master.bind("<Configure>", self._on_resize)
        master.bind("<Left>", self.prev_mov)
        master.bind("<Right>", self.next_mov)

        # --- Center and Show ---
        self._center_window(1200, 700)
        self.check_dependencies()
        master.deiconify()

    # ------------------------------- window/setup -------------------------------

    def _center_window(self, w: int, h: int):
        self.master.update_idletasks()
        sw = self.master.winfo_screenwidth()
        sh = self.master.winfo_screenheight()
        x = int((sw - w) / 2)
        y = int((sh - h) / 2)
        self.master.geometry(f"{w}x{h}+{x}+{y}")
        self.master.minsize(640, 400)

    def check_dependencies(self):
        if _CV2 is None or _NP is None:
            messagebox.showerror(
                "Missing Dependencies",
                "This tool requires 'opencv-python' and 'numpy' to process videos.\n\n"
                "Please install them with:\n"
                "pip install opencv-python numpy"
            )
            self.master.destroy()

    def _on_resize(self, _event=None):
        # Re-scale current frames on resize
        self._display_player_frame()
        self._display_preview_frame()

    # ------------------------------- folder/list loading -------------------------------

    def load_folder_dialog(self):
        directory = filedialog.askdirectory(
            initialdir=os.getcwd(), title="Select a folder with .mov files"
        )
        if not directory:
            return
        self.path_entry.delete(0, tk.END)
        self.path_entry.insert(0, directory)
        self.load_folder(directory)

    def clean_path(self, filepath: str) -> str:
        return filepath.strip().strip('"').strip("'").strip("<>").strip()

    def load_from_path_event(self, _event):
        if self._load_job is not None:
            self.master.after_cancel(self._load_job)

        def _try_load():
            raw = self.path_entry.get()
            filepath = self.clean_path(raw)
            if filepath and os.path.isdir(filepath):
                self.load_folder(filepath)
            elif not filepath:
                self.clear_all_state()

        self._load_job = self.master.after(300, _try_load)

    def load_folder(self, directory: str):
        if not os.path.isdir(directory):
            self.status_label.config(text=f"Folder not found: {directory}")
            return

        self.current_folder = Path(directory)
        self.status_label.config(text=f"Folder: {self.current_folder}")

        if self.path_entry.get() != directory:
            self.path_entry.delete(0, tk.END)
            self.path_entry.insert(0, directory)

        self.mov_files = sorted(
            [p for p in self.current_folder.glob("*") if p.suffix.lower() in SUPPORTED_EXTS]
        )

        self.listbox.delete(0, tk.END)
        for p in self.mov_files:
            self.listbox.insert(tk.END, p.name)

        if self.mov_files:
            self.select_mov_by_index(0)
            self.prev_button.config(state=tk.NORMAL)
            self.next_button.config(state=tk.NORMAL)
        else:
            self.clear_all_state()
            messagebox.showinfo(
                "No .mov Files", "No .mov files found in this folder."
            )

    def on_mov_list_select(self, _event=None):
        sel = self.listbox.curselection()
        if sel:
            self.select_mov_by_index(sel[0])

    def select_mov_by_index(self, index: int):
        if not (0 <= index < len(self.mov_files)):
            return

        self.current_mov_index = index
        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(self.current_mov_index)
        self.listbox.see(self.current_mov_index)

        filepath = self.mov_files[self.current_mov_index]
        self.load_mov_frames(str(filepath))

    def next_mov(self, _event=None):
        if self.current_mov_index < len(self.mov_files) - 1:
            self.select_mov_by_index(self.current_mov_index + 1)

    def prev_mov(self, _event=None):
        if self.current_mov_index > 0:
            self.select_mov_by_index(self.current_mov_index - 1)

    # ------------------------------- .mov loading & playback -------------------------------

    def load_mov_frames(self, filepath: str):
        """Uses OpenCV to read all frames from a .mov file."""
        self.stop_video()
        self.video_frames = []
        self.frame_listbox.delete(0, tk.END)
        self.save_frame_button.config(state=tk.DISABLED)

        if not os.path.exists(filepath):
            messagebox.showerror("Error", f"File not found: {filepath}")
            return

        try:
            cap = _CV2.VideoCapture(filepath)
            if not cap.isOpened():
                raise IOError("Failed to open video file")

            self.video_fps = cap.get(_CV2.CAP_PROP_FPS) or 30.0

            idx = 0
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                frame_rgb = _CV2.cvtColor(frame_bgr, _CV2.COLOR_BGR2RGB)
                pil_im = Image.fromarray(frame_rgb)
                self.video_frames.append(pil_im)
                self.frame_listbox.insert(tk.END, f"Frame {idx:04d}")
                idx += 1

            cap.release()

        except Exception as e:
            messagebox.showerror("Error", f"Could not process video: {e}")
            self.clear_mov_state()
            return

        if self.video_frames:
            self.play_button.config(state=tk.NORMAL)
            self.current_frame_idx = 0
            # Show poster frame (first frame)
            self._display_player_frame()
            # Select first frame in list
            self.frame_listbox.selection_set(0)
            self.on_frame_select()
        else:
            self.clear_mov_state()
            messagebox.showinfo("Info", "Video file contained no frames.")

    def toggle_play_stop(self):
        if self.is_playing:
            self.stop_video()
        else:
            self.play_video()

    def play_video(self):
        if not self.video_frames:
            return
        self.is_playing = True
        self.play_button.config(text="Stop")
        self._video_loop()

    def stop_video(self):
        self.is_playing = False
        if self._play_job:
            self.master.after_cancel(self._play_job)
            self._play_job = None
        self.play_button.config(text="Play")
        # Reset to first frame when stopped
        if self.video_frames:
            self.current_frame_idx = 0
            self._display_player_frame()


    def _video_loop(self):
        if not self.is_playing or not self.video_frames:
            return

        self._display_player_frame()
        self.current_frame_idx += 1
        if self.current_frame_idx >= len(self.video_frames):
            self.current_frame_idx = 0  # Loop

        delay_ms = max(10, int(1000 / self.video_fps))
        self._play_job = self.master.after(delay_ms, self._video_loop)

    # ------------------------------- frame extraction -------------------------------

    def on_frame_select(self, _event=None):
        self._display_preview_frame()
        if self.frame_listbox.curselection():
            self.save_frame_button.config(state=tk.NORMAL)
        else:
            self.save_frame_button.config(state=tk.DISABLED)

    def save_selected_frame(self):
        sel = self.frame_listbox.curselection()
        if not sel:
            messagebox.showwarning("No Frame", "Please select a frame from the list.")
            return
        
        frame_idx = sel[0]
        if not (0 <= frame_idx < len(self.video_frames)):
            return # Should not happen

        # Get original .mov path
        if not (0 <= self.current_mov_index < len(self.mov_files)):
            return
        
        mov_path = self.mov_files[self.current_mov_index]
        mov_stem = mov_path.stem  # e.g., "IMG_1234"
        save_name = f"{mov_stem}_img_{frame_idx:04d}.jpeg"
        save_path = mov_path.parent / save_name

        try:
            img_to_save = self.video_frames[frame_idx]
            if img_to_save.mode != "RGB":
                img_to_save = img_to_save.convert("RGB")
            
            img_to_save.save(str(save_path), "JPEG", quality=90)
            messagebox.showinfo("Frame Saved", f"Saved frame as:\n{save_path}")

        except Exception as e:
            messagebox.showerror("Error", f"Could not save frame: {e}")


    # ------------------------------- display/scaling helpers -------------------------------

    def _scale_image(
        self, pil_image: Image.Image, label: tk.Label
    ) -> ImageTk.PhotoImage | None:
        """Helper to scale a PIL image to fit a label."""
        avail_w = max(1, label.winfo_width() - 4)
        avail_h = max(1, label.winfo_height() - 4)

        if avail_w <= 1 or avail_h <= 1 or not pil_image:
            return None

        w, h = pil_image.size
        ratio = min(avail_w / float(w), avail_h / float(h))
        new_size = (max(1, int(w * ratio)), max(1, int(h * ratio)))
        
        try:
            resample = Image.Resampling.LANCZOS
        except AttributeError:
            resample = Image.LANCZOS  # type: ignore
            
        disp = pil_image.resize(new_size, resample)
        return ImageTk.PhotoImage(disp)

    def _display_player_frame(self):
        """Updates the video player label with the current frame."""
        if not self.video_frames:
            return
        
        pil_image = self.video_frames[self.current_frame_idx]
        self.tk_video_frame = self._scale_image(pil_image, self.player_label)
        if self.tk_video_frame:
            self.player_label.config(image=self.tk_video_frame, text="")
            self.player_label.image = self.tk_video_frame # type: ignore
    
    def _display_preview_frame(self):
        """Updates the preview label with the selected frame."""
        sel = self.frame_listbox.curselection()
        if not sel:
            self.frame_preview_label.config(image="", text="")
            return
        
        frame_idx = sel[0]
        if not (0 <= frame_idx < len(self.video_frames)):
            return

        pil_image = self.video_frames[frame_idx]
        self.tk_preview_frame = self._scale_image(pil_image, self.frame_preview_label)
        if self.tk_preview_frame:
            self.frame_preview_label.config(image=self.tk_preview_frame, text="")
            self.frame_preview_label.image = self.tk_preview_frame # type: ignore

    # ------------------------------- state clearing -------------------------------

    def clear_all_state(self):
        self.listbox.delete(0, tk.END)
        self.mov_files = []
        self.current_mov_index = -1
        self.prev_button.config(state=tk.DISABLED)
        self.next_button.config(state=tk.DISABLED)
        self.status_label.config(text="No folder loaded")
        self.clear_mov_state()

    def clear_mov_state(self):
        self.stop_video()
        self.video_frames = []
        self.video_fps = 30.0
        self.current_frame_idx = 0
        self.frame_listbox.delete(0, tk.END)
        self.player_label.config(image="", text="No .mov loaded")
        self.frame_preview_label.config(image="", text="")
        self.play_button.config(state=tk.DISABLED)
        self.save_frame_button.config(state=tk.DISABLED)


# ------------------------------- main -------------------------------

if __name__ == "__main__":
    if _CV2 is None or _NP is None:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "Missing Dependencies",
            "This tool requires 'opencv-python' and 'numpy' to process videos.\n\n"
            "Please install them with:\n"
            "pip install opencv-python numpy"
        )
    else:
        root = tk.Tk()
        app = LivePhotoApp(root)
        root.mainloop()