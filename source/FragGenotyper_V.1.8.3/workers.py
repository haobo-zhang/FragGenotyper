"""Cooperative background work. No worker directly changes Qt widgets."""
import threading
from PyQt6.QtCore import QThread, pyqtSignal


class AnalysisWorker(QThread):
    progress = pyqtSignal(int, str)
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self, function, parent=None):
        super().__init__(parent)
        self.function = function
        self.cancel_event = threading.Event()

    def cancel(self):
        self.cancel_event.set()

    def run(self):
        def progress(value, message=""):
            self.progress.emit(max(0, min(100, int(value))), str(message))
        try:
            result = self.function(progress, self.cancel_event)
            if self.cancel_event.is_set():
                self.cancelled.emit()
            else:
                self.succeeded.emit(result)
        except InterruptedError:
            self.cancelled.emit()
        except Exception as exc:
            if self.cancel_event.is_set():
                self.cancelled.emit()
            else:
                self.failed.emit(str(exc))

