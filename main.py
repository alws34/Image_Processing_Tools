#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

from ui.main_window import ImageEditorApp


def main():
    app = QApplication(sys.argv)

    # High DPI scaling settings
    if hasattr(Qt.ApplicationAttribute, 'AA_EnableHighDpiScaling'):
        app.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling)
    if hasattr(Qt.ApplicationAttribute, 'AA_UseHighDpiPixmaps'):
        app.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps)

    # Initialize the correct class
    window = ImageEditorApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
