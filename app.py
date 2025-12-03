import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from ui.main_window import ImageEditorApp

def main():
    app = QApplication(sys.argv)

    # High DPI scaling
    try:
        app.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling)
    except Exception:
        pass

    win = ImageEditorApp()
    win.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()