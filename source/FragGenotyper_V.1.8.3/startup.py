"""Responsive startup; importing this module loads only the standard library and Qt.

Heavy modules are imported in LoaderThread after the progress window is shown.
MainWindow and every widget are constructed on the application's GUI thread.
"""
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import traceback

from PyQt6.QtCore import QObject, QStandardPaths, QThread, QTimer, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication, QDialog, QLabel, QPlainTextEdit, QProgressBar, QPushButton,
    QVBoxLayout, QWidget,
)


VERSION = "1.8.3"
DEFAULT_STAGES = (
    ("NumPy · 数值计算", "numpy"),
    ("pandas · 数据表", "pandas"),
    ("SciPy · 统计分析", "scipy.stats"),
    ("scikit-learn · 聚类", "sklearn.cluster"),
    ("Matplotlib · 绘图库", "matplotlib"),
    ("Matplotlib · 字体缓存", "matplotlib.font_manager"),
    ("Matplotlib · Qt 绘图支持", "matplotlib.backends.backend_qtagg"),
    ("FragGenotyper · 分型模块", "genotype_core"),
    ("FragGenotyper · 表型模块", "haplotype_phenotype"),
)


def _thread_id():
    return int(QThread.currentThreadId())


class LoaderThread(QThread):
    """Import modules and emit data; never create a widget, canvas, or figure."""
    stage_event = pyqtSignal(object)
    succeeded = pyqtSignal()
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self, stages=None, *, module_importer=None, parent=None):
        super().__init__(parent)
        self.stages = tuple(DEFAULT_STAGES if stages is None else stages)
        self.module_importer = module_importer or importlib.import_module
        self.cancel_event = threading.Event()

    def cancel(self):
        self.cancel_event.set()

    def run(self):
        try:
            for index, (label, module_name) in enumerate(self.stages):
                if self.cancel_event.is_set():
                    self.cancelled.emit()
                    return
                began = time.monotonic()
                common = {"index": index, "label": str(label), "module": str(module_name),
                          "thread_id": _thread_id()}
                self.stage_event.emit({**common, "event": "started"})
                self.module_importer(module_name)
                self.stage_event.emit({**common, "event": "finished",
                                       "duration_seconds": time.monotonic() - began})
                if self.cancel_event.is_set():
                    self.cancelled.emit()
                    return
            self.succeeded.emit()
        except BaseException:
            if self.cancel_event.is_set():
                self.cancelled.emit()
            else:
                self.failed.emit(traceback.format_exc())


class LoadingWindow(QDialog):
    """A modeless progress window whose close action requests safe shutdown."""
    close_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"FragGenotyper {VERSION} · 正在加载")
        self.setMinimumWidth(530)
        self.setModal(False)
        self._allow_close = False
        self._started = time.monotonic()
        self._status_text = "准备加载分析组件"
        self._animate = True
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 20)
        self.heading = QLabel("FragGenotyper")
        self.heading.setObjectName("brand")
        layout.addWidget(self.heading)
        layout.addWidget(QLabel(f"{VERSION} · 正在加载分析和绘图组件"))
        self.status_label = QLabel(self._status_text)
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        layout.addWidget(self.progress)
        self.elapsed_label = QLabel("已用时 0.0 秒")
        layout.addWidget(self.elapsed_label)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(115)
        self.log.setPlaceholderText("组件载入进度会显示在这里")
        layout.addWidget(self.log)
        self.error_box = QPlainTextEdit()
        self.error_box.setReadOnly(True)
        self.error_box.setMinimumHeight(150)
        self.error_box.hide()
        layout.addWidget(self.error_box)
        self.close_button = QPushButton("取消启动")
        self.close_button.clicked.connect(lambda: self.close_requested.emit())
        layout.addWidget(self.close_button, alignment=Qt.AlignmentFlag.AlignRight)
        self.clock = QTimer(self)
        self.clock.setInterval(100)
        self.clock.timeout.connect(self.update_elapsed)
        self.clock.start()

    def update_elapsed(self):
        elapsed = time.monotonic() - self._started
        self.elapsed_label.setText(f"已用时 {elapsed:.1f} 秒")
        dots = "." * (int(elapsed * 3) % 4) if self._animate else ""
        self.status_label.setText(self._status_text + dots)

    def set_status(self, text):
        self._status_text = str(text)
        self.update_elapsed()

    def set_cancelling(self):
        self.close_button.setEnabled(False)
        self._animate = True
        self.set_status("正在取消：等待当前组件安全加载结束")

    def show_failure(self, error):
        self._animate = False
        self.setWindowTitle(f"FragGenotyper {VERSION} · 启动失败")
        self.set_status("启动失败。详细错误如下，可复制后用于排查。")
        self.error_box.setPlainText(str(error))
        self.error_box.show()
        self.close_button.setText("关闭")
        self.close_button.setEnabled(True)

    def finish_close(self):
        self._allow_close = True
        self.clock.stop()
        self.close()

    def reject(self):
        # QDialog's Escape shortcut otherwise hides the window immediately.
        self.close_requested.emit()

    def closeEvent(self, event):
        if self._allow_close:
            event.accept()
        else:
            event.ignore()
            self.close_button.setEnabled(False)
            # A nested close() inside closeEvent is ignored by Qt. Defer the
            # request so a finished/failed loader can close the dialog normally.
            QTimer.singleShot(0, self.close_requested.emit)


def _default_window_factory():
    # This function is called only by StartupController on the GUI thread.
    from app_window import MainWindow
    return MainWindow()


class StartupController(QObject):
    """Own the loader until it finishes, then build MainWindow on the GUI thread.

    ``stages`` is a sequence of (display_label, module_name). Tests can inject
    ``module_importer(name)`` and ``window_factory()`` without real dependencies.
    Diagnostics are written only when report_path or its environment variable
    is explicitly set. The ready flag is set on an event-loop turn after show().
    """
    ready = pyqtSignal()
    startup_failed = pyqtSignal(str)
    startup_cancelled = pyqtSignal()

    def __init__(self, app, *, stages=None, module_importer=None, window_factory=None,
                 report_path=None, autoclose_ms=None):
        super().__init__(app)
        if QThread.currentThread() != app.thread():
            raise RuntimeError("StartupController must be created on the GUI thread.")
        self.app = app
        self.app.setApplicationName("FragGenotyper")
        self.window = None
        self.loading_window = LoadingWindow()
        self.loading_window.close_requested.connect(self.request_cancel)
        self.window_factory = window_factory or _default_window_factory
        self.loader = LoaderThread(stages, module_importer=module_importer, parent=self)
        self.loader.stage_event.connect(self._stage_event)
        self.loader.succeeded.connect(self._loaded)
        self.loader.failed.connect(self._failed)
        self.loader.cancelled.connect(self._cancelled)
        self.loader.finished.connect(self._loader_finished)
        self._started = time.monotonic()
        self._scheduled = False
        self._thread_started = False
        self._load_succeeded = False
        self._cancel_requested = False
        self._failure = None
        self._closed = False
        self._main_gui_thread_id = _thread_id()
        self.report = {"schema_version": 1, "version": VERSION, "pid": os.getpid(),
                       "startup_status": "initializing", "startup_elapsed_seconds": 0.0,
                       "gui_thread_id": self._main_gui_thread_id, "stages": [],
                       "main_window_ready": False, "main_window_visible": False,
                       "main_window_constructed_on_gui_thread": False,
                       "main_window_ready_on_gui_thread": False}
        appearance = getattr(app, "_theme_controller", None)
        if appearance is not None:
            self.report.update(appearance_mode=appearance.mode, appearance_theme=appearance.theme)
        self.report.update(_configure_matplotlib_cache())
        diagnostic_path = report_path if report_path is not None else os.environ.get("FRAGGENOTYPER_STARTUP_REPORT")
        self.report_path = Path(diagnostic_path) if diagnostic_path else None
        if self.report_path is not None and not self.report_path.is_absolute():
            self.report["diagnostic_report_error"] = "FRAGGENOTYPER_STARTUP_REPORT must be an absolute JSON path."
            self.report_path = None
        auto_value = autoclose_ms if autoclose_ms is not None else os.environ.get("FRAGGENOTYPER_DIAGNOSTIC_AUTOCLOSE_MS")
        self.autoclose_ms = None
        if auto_value is not None:
            try:
                self.autoclose_ms = int(auto_value)
                if self.autoclose_ms < 0:
                    raise ValueError
            except (TypeError, ValueError):
                self.report["diagnostic_autoclose_error"] = "Diagnostic autoclose must be a nonnegative number of milliseconds."
                self.autoclose_ms = None

    def _save_report(self):
        self.report["startup_elapsed_seconds"] = round(time.monotonic() - self._started, 6)
        if self.report_path is None:
            return
        temporary = self.report_path.with_name(f".{self.report_path.name}.{os.getpid()}.tmp")
        try:
            self.report_path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(json.dumps(self.report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
            os.replace(temporary, self.report_path)
        except OSError as exc:
            # Optional diagnostic output must not break ordinary application use.
            self.report["diagnostic_report_error"] = str(exc)
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    def show(self):
        if self._scheduled or self._closed:
            return
        self._scheduled = True
        self.loading_window.show()
        self.report["startup_status"] = "loading"
        self._save_report()
        # Give Qt a complete event-loop turn to expose the loading window.
        QTimer.singleShot(0, self._start_loader)

    def _start_loader(self):
        if self._closed or self._cancel_requested:
            return
        self._thread_started = True
        self.loader.start()

    def _stage_event(self, event):
        if self._closed:
            return
        index = event["index"]
        if event["event"] == "started":
            record = {"label": event["label"], "module": event["module"], "status": "loading",
                      "import_thread_id": event["thread_id"],
                      "import_thread_is_gui": event["thread_id"] == self._main_gui_thread_id,
                      "started_elapsed_seconds": round(time.monotonic() - self._started, 6)}
            self.report["stages"].append(record)
            if not self._cancel_requested:
                self.loading_window.set_status(f"正在加载 {event['label']}\n{event['module']}")
        else:
            record = self.report["stages"][index]
            record.update(status="loaded", elapsed_seconds=round(event["duration_seconds"], 6))
            self.loading_window.log.appendPlainText(f"✓ {event['label']}  ({event['duration_seconds']:.2f} 秒)")
            total = max(len(self.loader.stages), 1)
            self.loading_window.progress.setValue(int((index + 1) * 90 / total))
        self._save_report()

    def _loaded(self):
        self._load_succeeded = True

    def _failed(self, error):
        self._failure = str(error)
        self.report.update(startup_status="failed", error=str(error))
        for stage in self.report["stages"]:
            if stage["status"] == "loading":
                stage["status"] = "failed"
        self.loading_window.show_failure(error)
        self._save_report()
        self.startup_failed.emit(str(error))

    def _cancelled(self):
        self._cancel_requested = True

    def _loader_finished(self):
        if self._closed:
            return
        if self._cancel_requested:
            self._finish_shutdown()
        elif self._failure is not None:
            # Leave the error and copyable traceback visible until closed.
            self._save_report()
        elif self._load_succeeded:
            self._construct_window()

    def _construct_window(self):
        if self._closed or self._cancel_requested:
            return
        self.report["startup_status"] = "constructing_main_window"
        self.loading_window.set_status("正在准备主窗口")
        self.loading_window.progress.setValue(94)
        self._save_report()
        try:
            on_gui_thread = QThread.currentThread() == self.app.thread()
            if not on_gui_thread:
                raise RuntimeError("MainWindow construction attempted outside the GUI thread.")
            window = self.window_factory()
            if not isinstance(window, QWidget):
                raise TypeError("window_factory must return a QWidget.")
            self.window = window
            self.report["main_window_constructed_on_gui_thread"] = on_gui_thread
            self.report["main_window_construction_thread_id"] = _thread_id()
            self.window.show()
            QTimer.singleShot(0, self._mark_ready)
        except Exception:
            self._failed(traceback.format_exc())

    def _mark_ready(self):
        if self._closed or self._cancel_requested:
            return
        visible = self.window is not None and self.window.isVisible()
        on_gui_thread = QThread.currentThread() == self.app.thread()
        if not visible or not on_gui_thread:
            self._failed("Main window did not become visible on the GUI thread.")
            return
        self.report.update(startup_status="ready", main_window_ready=True,
                           main_window_visible=True, main_window_ready_on_gui_thread=True,
                           main_window_ready_thread_id=_thread_id(),
                           main_window_ready_elapsed_seconds=round(time.monotonic() - self._started, 6),
                           main_window_ready_unix_seconds=time.time())
        appearance = getattr(self.app, "_theme_controller", None)
        if appearance is not None:
            self.report.update(appearance_mode=appearance.mode, appearance_theme=appearance.theme)
        self.loading_window.progress.setValue(100)
        self._save_report()
        self.loading_window.finish_close()
        self.ready.emit()
        if self.autoclose_ms is not None:
            QTimer.singleShot(self.autoclose_ms, self._diagnostic_close)

    def _diagnostic_close(self):
        if self.window is not None:
            self.report["diagnostic_autoclose_requested"] = True
            self._save_report()
            # MainWindow.closeEvent performs cooperative shutdown of its workers.
            self.window.close()

    def request_cancel(self):
        if self._closed or self.report["main_window_ready"]:
            return
        self._cancel_requested = True
        self.loader.cancel()
        if self.loader.isRunning():
            if self._failure is None:
                self.report["startup_status"] = "cancelling"
                self.loading_window.set_cancelling()
            self._save_report()
        else:
            self._finish_shutdown()

    def _finish_shutdown(self):
        if self.loader.isRunning():
            return
        self._closed = True
        if self._failure is None:
            self.report["startup_status"] = "cancelled"
        self.report["loader_running_at_shutdown"] = False
        self._save_report()
        self.loading_window.finish_close()
        self.startup_cancelled.emit()
        self.app.quit()


def _configure_matplotlib_cache():
    """Use a persistent writable cache before importing Matplotlib."""
    # PyInstaller's Matplotlib runtime hook unconditionally replaces this
    # variable with a fresh temporary directory. Our earlier runtime hook
    # remembers the user's original setting so that packaged apps can reuse
    # their font cache too. Matplotlib stores its own bundled fonts relatively.
    packager_cache = None
    if hasattr(sys, "_fraggenotyper_original_mplconfigdir"):
        original = sys._fraggenotyper_original_mplconfigdir
        current = os.environ.get("MPLCONFIGDIR")
        if current != original:
            packager_cache = current
        if original:
            os.environ["MPLCONFIGDIR"] = original
        else:
            os.environ.pop("MPLCONFIGDIR", None)

    def details(path, source):
        resolved = Path(path).resolve() if path else None
        temporary_roots = (Path(tempfile.gettempdir()).resolve(), Path("/tmp").resolve())
        is_temporary = resolved is not None and any(resolved.is_relative_to(root) for root in temporary_roots)
        return {"matplotlib_cache_path": str(path) if path else None,
                "matplotlib_cache_source": source, "matplotlib_cache_temporary": is_temporary,
                "matplotlib_packager_cache_replaced": packager_cache is not None}

    if os.environ.get("MPLCONFIGDIR"):
        return details(os.environ["MPLCONFIGDIR"], "environment")
    cache_root = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.CacheLocation)
    candidates = [(Path(cache_root) / "matplotlib", "application_cache")] if cache_root else []
    candidates.append((Path(tempfile.gettempdir()) / "FragGenotyper" / "matplotlib", "temporary"))
    for directory, source in candidates:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            if os.access(directory, os.W_OK):
                effective = os.environ.setdefault("MPLCONFIGDIR", str(directory))
                return details(effective, source if effective == str(directory) else "environment")
        except OSError:
            continue
    return details(None, "unavailable")


def run_application(argv=None):
    app = QApplication(sys.argv if argv is None else list(argv))
    app.setApplicationName("FragGenotyper")
    app.setApplicationVersion(VERSION)
    from app_style import get_theme_controller
    get_theme_controller(app)
    controller = StartupController(app)
    # Retain the owner until the application and any loader thread have exited.
    app._startup_controller = controller
    controller.show()
    return app.exec()
