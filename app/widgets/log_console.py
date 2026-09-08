"""모든 탭이 공유하는 하단 로그 콘솔."""
from __future__ import annotations

from datetime import datetime

from PyQt6.QtWidgets import QPlainTextEdit


class LogConsole(QPlainTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumHeight(140)
        self.setStyleSheet(
            "font-family: Consolas, monospace; font-size: 12px; "
            "background-color: #fafaf7; border-top: 1px solid #ddd;"
        )

    def append_log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.appendPlainText(f"[{timestamp}] {message}")
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())
