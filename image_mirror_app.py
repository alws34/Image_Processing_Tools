#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageOps
import os
from pathlib import Path

# Optional HEIF/HEIC/AVIF support via pillow-heif
_HEIF_PLUGIN = False
try:
    from pillow_heif import register_heif_opener  # type: ignore
    register_heif_opener()
    _HEIF_PLUGIN = True
except Exception:
    _HEIF_PLUGIN = False

# Optional send2trash for delete
_SEND2TRASH = None
try:
    import send2trash
    _SEND2TRASH = send2trash.send2trash
except Exception:
    _SEND2TRASH = None

# Extension -> PIL format mapping
# Removed .mov
EXT_TO_FMT = {
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".png": "PNG",
    ".bmp": "BMP",
    ".gif": "GIF",
    ".tif": "TIFF",
    ".tiff": "TIFF",
    ".webp": "WEBP",
    ".heic": "HEIF",
    ".heif": "HEIF",
    ".heics": "HEIF",
    ".heifs": "HEIF",
    ".hif": "HEIF",
    ".avif": "AVIF",
}

HEIF_LIKE_EXTS = {".heic", ".heif", ".heics", ".heifs", ".hif", ".avif"}
SUPPORTED_EXTS = set(EXT_TO_FMT.keys())


class ImageMirrorApp:
    def __init__(self, master: tk.Tk):
        self.master = master
        master.title("Image Mirror Tool (Folder Edition)")
        master.withdraw()  # hide until centered

        # State
        self.current_folder: Path | None = None
        self.image_files: list[Path] = []
        self.current_index: int = -1

        self.original_image_path: str | None = None
        self.original_image_pil: Image.Image | None = None
        self.mirrored_image_pil: Image.Image | None = None
        self.tk_image_original: ImageTk.PhotoImage | None = None
        self.tk_image_mirrored: ImageTk.PhotoImage | None = None
        self._orig_format: str | None = None
        self._orig_exif: bytes | None = None
        self._load_job = None

        # --- Layout ---
        # Top: Controls
        # Middle: Listbox (left) | Image Frame (right)
        # Bottom: Navigation

        master.rowconfigure(1, weight=1)
        master.columnconfigure(1, weight=1)

        # --- Top Control Frame ---
        self.top_frame = tk.Frame(master)
        self.top_frame.grid(row=0, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
        self.top_frame.columnconfigure(1, weight=1)

        self.browse_button = tk.Button(
            self.top_frame, text="Browse Folder...", command=self.load_folder_dialog
        )
        self.browse_button.grid(row=0, column=0, padx=5, sticky="w")

        self.path_entry = tk.Entry(self.top_frame)
        self.path_entry.grid(row=0, column=1, padx=10, sticky="ew")

        # Debounced load on typing/paste/focus/enter
        self.path_entry.bind("<KeyRelease>", self.load_from_path_event)
        self.path_entry.bind("<FocusOut>", self.load_from_path_event)
        self.path_entry.bind("<Return>", self.load_from_path_event)

        self.status_label = tk.Label(
            self.top_frame, text="No folder loaded", anchor="w"
        )
        self.status_label.grid(row=1, column=0, columnspan=2, padx=10, sticky="ew")

        # --- Left Frame (Listbox) ---
        self.list_frame = tk.Frame(master)
        self.list_frame.grid(row=1, column=0, padx=(10, 5), pady=5, sticky="nsw")
        self.list_frame.rowconfigure(0, weight=1)

        self.list_scrollbar = tk.Scrollbar(self.list_frame, orient=tk.VERTICAL)
        self.listbox = tk.Listbox(
            self.list_frame,
            width=30,
            yscrollcommand=self.list_scrollbar.set,
            exportselection=False,
        )
        self.list_scrollbar.config(command=self.listbox.yview)

        self.list_scrollbar.grid(row=0, column=1, sticky="ns")  # Corrected: "nsv" -> "ns"
        self.listbox.grid(row=0, column=0, sticky="nsw")

        self.listbox.bind("<<ListboxSelect>>", self.on_listbox_select)

        # --- Main Image Frame (Split) ---
        self.image_frame = tk.Frame(master, borderwidth=2, relief="groove")
        self.image_frame.grid(row=1, column=1, padx=(5, 10), pady=5, sticky="nsew")
        self.image_frame.rowconfigure(0, weight=1)
        self.image_frame.columnconfigure(0, weight=1)
        self.image_frame.columnconfigure(1, weight=1)

        # --- Left Panel (Original) ---
        left_panel = tk.Frame(self.image_frame)
        left_panel.grid(row=0, column=0, sticky="nsew")
        left_panel.rowconfigure(1, weight=1)
        left_panel.columnconfigure(0, weight=1)
        tk.Label(left_panel, text="Original").grid(row=0, column=0, pady=2)
        self.original_label = tk.Label(left_panel, text="No Image", bg="lightgrey")
        self.original_label.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        # --- Right Panel (Mirrored) ---
        right_panel = tk.Frame(self.image_frame)
        right_panel.grid(row=0, column=1, sticky="nsew")
        right_panel.rowconfigure(1, weight=1)
        right_panel.columnconfigure(0, weight=1)
        tk.Label(right_panel, text="Mirrored").grid(row=0, column=0, pady=2)
        self.mirrored_label = tk.Label(right_panel, text="No Image", bg="darkgrey")
        self.mirrored_label.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        # Resize handling for dynamic scaling
        self.image_frame.bind("<Configure>", self._on_image_frame_resize)

        # --- Bottom Control Frame ---
        self.control_frame = tk.Frame(master)
        self.control_frame.grid(
            row=2, column=0, columnspan=2, padx=10, pady=5, sticky="ew"
        )
        self.control_frame.columnconfigure(1, weight=1)  # Center buttons

        self.prev_button = tk.Button(
            self.control_frame,
            text="< Prev",
            command=self.prev_image,
            state=tk.DISABLED,
        )
        self.prev_button.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        # --- Center buttons ---
        center_button_frame = tk.Frame(self.control_frame)
        center_button_frame.grid(row=0, column=1)

        self.save_button = tk.Button(
            center_button_frame,
            text="Save Mirrored (Overwrite) [Enter]",
            command=self.save_and_reload,
            state=tk.DISABLED,
        )
        self.save_button.grid(row=0, column=0, padx=5, pady=5)

        self.delete_button = tk.Button(
            center_button_frame,
            text="Delete [Del]",
            command=self.delete_current_image,
            state=tk.DISABLED,
            fg="red",
        )
        self.delete_button.grid(row=0, column=1, padx=5, pady=5)

        self.next_button = tk.Button(
            self.control_frame,
            text="Next >",
            command=self.next_image,
            state=tk.DISABLED,
        )
        self.next_button.grid(row=0, column=2, padx=5, pady=5, sticky="e")

        # --- Keyboard Bindings ---
        master.bind("<Left>", self.prev_image)
        master.bind("<Right>", self.next_image)
        master.bind("<Return>", self.save_and_reload)
        master.bind("<Delete>", self.delete_current_image)

        # Center window and show
        self._center_window(1200, 700)
        master.deiconify()

    # ------------------------------- window behavior -------------------------------

    def _center_window(self, w: int, h: int):
        self.master.update_idletasks()
        sw = self.master.winfo_screenwidth()
        sh = self.master.winfo_screenheight()
        x = int((sw - w) / 2)
        y = int((sh - h) / 2)
        self.master.geometry(f"{w}x{h}+{x}+{y}")
        self.master.minsize(640, 400)

    def _on_image_frame_resize(self, _event=None):
        # When the container resizes, re-render scaled images
        self._display_scaled()

    # ------------------------------- folder/list loading -------------------------------

    def load_folder_dialog(self):
        directory = filedialog.askdirectory(
            initialdir=os.getcwd(), title="Select an image folder"
        )
        if not directory:
            return

        # Update path entry, which will trigger the load event
        self.path_entry.delete(0, tk.END)
        self.path_entry.insert(0, directory)
        # Also trigger load immediately (safer than relying on delayed event)
        self.load_folder(directory)  

    def clean_path(self, filepath: str) -> str:
        return filepath.strip().strip('"').strip("'").strip("<>").strip()

    def load_from_path_event(self, _event):
        if self._load_job is not None:
            try:
                self.master.after_cancel(self._load_job)
            except Exception:
                pass

        def _try_load_folder():
            raw = self.path_entry.get()
            filepath = self.clean_path(raw)
            current_folder_path_str = str(self.current_folder) if self.current_folder else ""
            
            # Fix 2: Refine path loading logic to prevent unnecessary reloads
            if filepath and os.path.isdir(filepath):
                # Only load if the path is different from the currently loaded folder
                if filepath != current_folder_path_str:
                    self.load_folder(filepath)
            else:
                # If the path is empty and we have images loaded, clear the state
                if not filepath and self.image_files:
                    self.clear_image_state()

        self._load_job = self.master.after(300, _try_load_folder)

    def load_folder(self, directory: str):
        """Scans a directory and populates the image list."""
        if not os.path.isdir(directory):
            self.status_label.config(text=f"Folder not found: {directory}")
            return

        self.current_folder = Path(directory)
        self.status_label.config(text=f"Folder: {self.current_folder}")

        # Update path entry if it doesn't match
        if self.path_entry.get() != directory:
            self.path_entry.delete(0, tk.END)
            self.path_entry.insert(0, directory)

        # Scan for supported image files
        self.image_files = sorted(
            [
                p
                for p in self.current_folder.glob("*")
                if p.suffix.lower() in SUPPORTED_EXTS
            ]
        )

        # Populate listbox
        self.listbox.delete(0, tk.END)
        for p in self.image_files:
            self.listbox.insert(tk.END, p.name)

        if self.image_files:
            self.select_image_by_index(0)
            self.prev_button.config(state=tk.NORMAL)
            self.next_button.config(state=tk.NORMAL)
        else:
            self.clear_image_state()
            # Suppress messagebox if the user just cleared the path
            if directory:
                messagebox.showinfo(
                    "No Images", "No supported images found in this folder."
                )

    def on_listbox_select(self, _event=None):
        """Handler for user clicks on the listbox."""
        sel = self.listbox.curselection()
        if sel:
            self.select_image_by_index(sel[0])

    def select_image_by_index(self, index: int):
        """
        Loads the image at the given index.
        Fix 1: Temporarily unbinds the ListboxSelect event to prevent
               recursive calls when programmatically updating the selection.
        """
        if not (0 <= index < len(self.image_files)):
            return
        
        # --- FIX 1: Prevent recursion / index reset ---
        # Temporarily unbind the event to prevent on_listbox_select from firing
        # when we programmatically set the selection below.
        self.listbox.unbind("<<ListboxSelect>>")
        
        self.current_index = index
        # Update listbox selection visual
        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(self.current_index)
        self.listbox.see(self.current_index)
        
        # Re-bind the event immediately after updating the selection
        self.listbox.bind("<<ListboxSelect>>", self.on_listbox_select)
        # --- END FIX 1 ---
        

        filepath = self.image_files[self.current_index]
        self.process_image(str(filepath))

    def next_image(self, _event=None):
        if self.current_index < len(self.image_files) - 1:
            self.select_image_by_index(self.current_index + 1)

    def prev_image(self, _event=None):
        if self.current_index > 0:
            self.select_image_by_index(self.current_index - 1)

    def delete_current_image(self, _event=None):
        if self.current_index == -1 or not self.original_image_path:
            return

        if not _SEND2TRASH:
            messagebox.showerror(
                "send2trash missing",
                "Moving to trash requires the 'send2trash' library.\n\n"
                "Install with:\n"
                "  pip install send2trash",
            )
            return

        current_path = self.original_image_path
        current_index = self.current_index

        try:
            _SEND2TRASH(current_path)
        except Exception as e:
            messagebox.showerror("Error", f"Could not move file to trash: {e}")
            return

        # Success, now update state and UI
        self.image_files.pop(current_index)
        self.listbox.delete(current_index)

        num_remaining = len(self.image_files)
        if num_remaining == 0:
            self.clear_image_state()
            self.status_label.config(text=f"Folder empty: {self.current_folder}")
        else:
            # Select the next image, or the previous one if we deleted the last
            new_index = min(current_index, num_remaining - 1)
            self.select_image_by_index(new_index)

    # ------------------------------- image flow -------------------------------

    def _infer_format_from_path(self, path: str) -> str | None:
        return EXT_TO_FMT.get(Path(path).suffix.lower())

    def _ensure_heif_plugin_for_path(self, path: str, when: str) -> bool:
        ext = Path(path).suffix.lower()
        if ext in HEIF_LIKE_EXTS and not _HEIF_PLUGIN:
            messagebox.showerror(
                "HEIF/AVIF support missing",
                f"You attempted to {when} a {ext} file but pillow-heif is not installed.\n\n"
                f"Install with:\n"
                f"  pip install pillow-heif\n"
                f"(On some systems, also install libheif.)",
            )
            return False
        return True

    def process_image(self, filepath: str):
        """Loads an image from disk, processes it, and triggers a scaled display."""
        if not os.path.exists(filepath):
            messagebox.showerror("Error", f"File not found: {filepath}")
            self.clear_image_state()
            return

        if not self._ensure_heif_plugin_for_path(filepath, "open"):
            return

        try:
            with Image.open(filepath) as im:
                # Capture format + EXIF before transforms
                self._orig_format = (im.format or "") or self._infer_format_from_path(
                    filepath
                )
                self._orig_exif = im.info.get("exif")
                im = ImageOps.exif_transpose(im)
                self.original_image_pil = im.copy()

            self.original_image_path = filepath

            # Mirror horizontally
            try:
                flip_const = Image.Transpose.FLIP_LEFT_RIGHT
            except AttributeError:
                flip_const = Image.FLIP_LEFT_RIGHT  # type: ignore
            self.mirrored_image_pil = self.original_image_pil.transpose(flip_const)

            # Display and enable save
            self._display_scaled()
            self.save_button.config(state=tk.NORMAL)
            self.delete_button.config(state=tk.NORMAL)

        except Exception as e:
            messagebox.showerror("Error", f"Could not process image: {e}")
            self.clear_image_state()

    def _scale_image(
        self, pil_image: Image.Image, avail_w: int, avail_h: int
    ) -> ImageTk.PhotoImage:
        """Helper to scale a PIL image to fit given dimensions."""
        w, h = pil_image.size
        ratio = min(avail_w / float(w), avail_h / float(h))
        new_size = (max(1, int(w * ratio)), max(1, int(h * ratio)))
        try:
            resample = Image.Resampling.LANCZOS
        except AttributeError:
            resample = Image.LANCZOS  # type: ignore
        disp = pil_image.resize(new_size, resample)
        return ImageTk.PhotoImage(disp)

    def _display_scaled(self):
        """Renders both original and mirrored images into their labels."""
        if not self.original_image_pil or not self.mirrored_image_pil:
            return

        # Calculate available size for *one* panel
        avail_w = max(1, (self.image_frame.winfo_width() // 2) - 16)
        avail_h = max(1, self.image_frame.winfo_height() - 16)

        if avail_w <= 1 or avail_h <= 1:
            return  # Frame not rendered yet

        # Scale and display Original (left)
        self.tk_image_original = self._scale_image(
            self.original_image_pil, avail_w, avail_h
        )
        self.original_label.config(image=self.tk_image_original, text="")
        self.original_label.image = self.tk_image_original  # type: ignore

        # Scale and display Mirrored (right)
        self.tk_image_mirrored = self._scale_image(
            self.mirrored_image_pil, avail_w, avail_h
        )
        self.mirrored_label.config(image=self.tk_image_mirrored, text="")
        self.mirrored_label.image = self.tk_image_mirrored  # type: ignore

        self.master.update_idletasks()

    def save_image(self) -> bool:
        """Saves the mirrored image over the original. Returns True on success."""
        if not (self.mirrored_image_pil and self.original_image_path):
            messagebox.showwarning("Warning", "No image is currently processed to save.")
            return False

        if not self._ensure_heif_plugin_for_path(self.original_image_path, "save"):
            return False

        try:
            p = Path(self.original_image_path)
            # Use a temp file in the same dir for atomic replace
            tmp_path = p.with_name(f".tmp_{p.name}")

            save_fmt = (
                self._orig_format
                or self._infer_format_from_path(self.original_image_path)
                or "PNG"
            ).upper()

            img_to_save = self.mirrored_image_pil

            # Handle color mode conversions for specific formats
            if save_fmt == "JPEG" and img_to_save.mode not in ("RGB", "L"):
                img_to_save = img_to_save.convert("RGB")
            if save_fmt in ("HEIF", "AVIF") and img_to_save.mode not in (
                "RGB",
                "RGBA",
                "L",
            ):
                img_to_save = img_to_save.convert("RGB")

            save_kwargs = {"format": save_fmt}
            if self._orig_exif and save_fmt in ("JPEG", "TIFF", "WEBP", "HEIF"):
                save_kwargs["exif"] = self._orig_exif
            if save_fmt in ("HEIF", "AVIF"):
                save_kwargs.setdefault("quality", 90)

            # Save to temp, then replace original
            img_to_save.save(str(tmp_path), **save_kwargs)
            os.replace(str(tmp_path), self.original_image_path)

            return True

        except Exception as e:
            messagebox.showerror("Error", f"Could not save image: {e}")
            try:
                if "tmp_path" in locals() and Path(tmp_path).exists():
                    Path(tmp_path).unlink()
            except Exception:
                pass
            return False

    def save_and_reload(self, _event=None):
        """Saves, then re-processes the (now mirrored) image from disk."""
        if not self.original_image_path:
            return

        current_path = self.original_image_path
        if self.save_image():
            # Success! Reload the image from disk.
            # This will show the new mirrored image on the left,
            # and a mirror of *that* (the original) on the right.
            self.process_image(current_path)

    def clear_image_state(self):
        self.original_image_path = None
        self.original_image_pil = None
        self.mirrored_image_pil = None
        self.tk_image_original = None
        self.tk_image_mirrored = None
        self._orig_format = None
        self._orig_exif = None
        self.current_index = -1

        self.save_button.config(state=tk.DISABLED)
        self.delete_button.config(state=tk.DISABLED)
        self.prev_button.config(state=tk.DISABLED)
        self.next_button.config(state=tk.DISABLED)

        self.original_label.config(image="", text="No Image")
        self.mirrored_label.config(image="", text="No Image")


# ------------------------------- main -------------------------------

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageMirrorApp(root)
    root.mainloop()