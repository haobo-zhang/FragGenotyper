"""Startup stays responsive and constructs all widgets on Qt's GUI thread."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/fraggenotyper-mpl")
MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

import pytest
from PyQt6.QtCore import QThread, QTimer
from PyQt6.QtWidgets import QApplication, QMainWindow
import startup


def wait_until(app, condition, timeout=4):
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() >= deadline:
            pytest.fail("Timed out waiting for startup state.")
        app.processEvents()
        time.sleep(0.003)
    app.processEvents()


@pytest.fixture(scope="module")
def app():
    instance = QApplication.instance() or QApplication([])
    previous = instance.quitOnLastWindowClosed()
    instance.setQuitOnLastWindowClosed(False)
    yield instance
    instance.setQuitOnLastWindowClosed(previous)


@pytest.fixture
def make_controller(app, monkeypatch):
    monkeypatch.delenv("FRAGGENOTYPER_STARTUP_REPORT", raising=False)
    monkeypatch.delenv("FRAGGENOTYPER_DIAGNOSTIC_AUTOCLOSE_MS", raising=False)
    controllers = []

    def make(**kwargs):
        controller = startup.StartupController(app, **kwargs)
        controllers.append(controller)
        return controller

    yield make
    for controller in controllers:
        if controller.loader.isRunning():
            controller.request_cancel()
            wait_until(app, lambda: not controller.loader.isRunning())
        if controller.window is not None:
            controller.window.close()
        controller.loading_window.finish_close()
        controller.loader.wait(1000)
    app.processEvents()


def test_entry_imports_no_application_or_heavy_packages_before_startup():
    probe = """
import runpy, sys
sys.path.insert(0, sys.argv[1])
runpy.run_path(sys.argv[2], run_name='startup_import_probe')
for forbidden in ('app_window', 'numpy', 'pandas', 'matplotlib', 'scipy', 'sklearn'):
    assert forbidden not in sys.modules, forbidden
assert 'startup' in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", probe, str(MODULE_DIR),
         str(MODULE_DIR / "FragGenotyper_V.1.8.3.py")],
        text=True, capture_output=True, timeout=15,
    )
    assert completed.returncode == 0, completed.stderr


def test_slow_imports_keep_heartbeat_and_ready_factory_on_gui_thread(app, make_controller, tmp_path):
    import_threads, construction_threads, heartbeats = [], [], []
    stages = [("Mock one", "mock.one"), ("Mock two", "mock.two"), ("Mock three", "mock.three")]

    def slow_import(name):
        import_threads.append(QThread.currentThread() == app.thread())
        time.sleep(0.06)

    def build_window():
        construction_threads.append((QThread.currentThread() == app.thread(), controller.loader.isRunning()))
        return QMainWindow()

    report_path = tmp_path / "startup.json"
    controller = make_controller(stages=stages, module_importer=slow_import,
                                 window_factory=build_window, report_path=report_path)
    heartbeat = QTimer()
    heartbeat.setInterval(5)
    heartbeat.timeout.connect(lambda: heartbeats.append(time.monotonic()))
    heartbeat.start()
    ready_states = []
    controller.ready.connect(lambda: ready_states.append(QThread.currentThread() == app.thread()))
    controller.show()
    wait_until(app, lambda: controller.report["main_window_ready"])
    heartbeat.stop()

    assert len(heartbeats) >= 12
    assert import_threads == [False, False, False]
    assert construction_threads == [(True, False)]
    assert ready_states == [True]
    assert controller.window.isVisible()
    assert not controller.loading_window.isVisible()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["main_window_ready"]
    assert report["main_window_constructed_on_gui_thread"]
    assert report["main_window_ready_on_gui_thread"]
    assert report["main_window_ready_thread_id"] == report["gui_thread_id"]
    assert report["main_window_ready_elapsed_seconds"] >= 0.15
    assert report["main_window_ready_unix_seconds"] <= time.time()
    assert [stage["module"] for stage in report["stages"]] == [name for _, name in stages]
    assert all(stage["status"] == "loaded" and not stage["import_thread_is_gui"] for stage in report["stages"])
    assert report["matplotlib_cache_source"] == "environment"


@pytest.mark.parametrize("fail_in_factory", [False, True])
def test_failure_stays_visible_with_traceback_and_can_close(app, make_controller, tmp_path, fail_in_factory):
    constructed = []

    def importer(name):
        if not fail_in_factory:
            raise ImportError("mock missing package")

    def build_window():
        constructed.append(True)
        raise RuntimeError("mock failed window")

    report_path = tmp_path / "failure.json"
    controller = make_controller(stages=[("Mock", "mock")], module_importer=importer,
                                 window_factory=build_window, report_path=report_path)
    controller.show()
    wait_until(app, lambda: controller._failure is not None and not controller.loader.isRunning())

    assert controller.loading_window.isVisible()
    assert controller.loading_window.error_box.isVisible()
    expected_error = "mock failed window" if fail_in_factory else "mock missing package"
    assert expected_error in controller.loading_window.error_box.toPlainText()
    assert constructed == ([True] if fail_in_factory else [])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["startup_status"] == "failed"
    assert not report["main_window_ready"]
    controller.loading_window.close()
    wait_until(app, lambda: not controller.loading_window.isVisible())
    assert not controller.loader.isRunning()


def test_closing_while_loading_waits_for_import_and_does_not_construct_main(app, make_controller, tmp_path):
    entered = threading.Event()
    imports, constructed = [], []

    def importer(name):
        imports.append(name)
        entered.set()
        time.sleep(0.15)

    controller = make_controller(
        stages=[("First", "first"), ("Second", "second")], module_importer=importer,
        window_factory=lambda: constructed.append(True) or QMainWindow(),
        report_path=tmp_path / "cancelled.json",
    )
    controller.show()
    wait_until(app, entered.is_set)
    controller.loading_window.close()
    assert controller.loading_window.isVisible()
    assert controller.loader.isRunning()
    assert not controller.loading_window.close_button.isEnabled()
    wait_until(app, lambda: controller._closed)

    assert imports == ["first"]
    assert not constructed
    assert not controller.loader.isRunning()
    assert not controller.loading_window.isVisible()
    report = json.loads((tmp_path / "cancelled.json").read_text(encoding="utf-8"))
    assert report["startup_status"] == "cancelled"
    assert not report["main_window_ready"]
    assert not report["loader_running_at_shutdown"]


def test_cancel_before_thread_start_and_escape_are_safe(app, make_controller):
    imported = []
    controller = make_controller(stages=[("Mock", "mock")], module_importer=imported.append,
                                 window_factory=QMainWindow)
    controller.show()
    controller.loading_window.reject()
    app.processEvents()

    assert not imported
    assert controller._closed
    assert not controller.loader.isRunning()
    assert controller.report["startup_status"] == "cancelled"


def test_diagnostic_autoclose_runs_only_after_ready_and_preserves_ready_marker(app, make_controller, monkeypatch, tmp_path):
    path = tmp_path / "diagnostic.json"
    monkeypatch.setenv("FRAGGENOTYPER_STARTUP_REPORT", str(path))
    monkeypatch.setenv("FRAGGENOTYPER_DIAGNOSTIC_AUTOCLOSE_MS", "20")
    controller = make_controller(stages=[], window_factory=QMainWindow)
    controller.show()
    wait_until(app, lambda: controller.report.get("diagnostic_autoclose_requested", False))

    assert controller.report["main_window_ready"]
    assert not controller.window.isVisible()
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["main_window_ready"]
    assert report["diagnostic_autoclose_requested"]


def test_normal_startup_has_no_automatic_close_or_diagnostic_file(app, make_controller):
    controller = make_controller(stages=[], window_factory=QMainWindow)
    controller.show()
    wait_until(app, lambda: controller.report["main_window_ready"])

    assert controller.autoclose_ms is None
    assert controller.report_path is None
    assert controller.window.isVisible()


def test_cache_respects_existing_setting(monkeypatch, tmp_path):
    explicit = tmp_path / "explicit-font-cache"
    monkeypatch.setenv("MPLCONFIGDIR", str(explicit))
    details = startup._configure_matplotlib_cache()

    assert os.environ["MPLCONFIGDIR"] == str(explicit)
    assert details["matplotlib_cache_path"] == str(explicit)
    assert details["matplotlib_cache_source"] == "environment"


def test_cache_falls_back_to_temp_when_application_cache_is_unavailable(monkeypatch, tmp_path):
    blocked = tmp_path / "file-not-directory"
    blocked.write_text("blocked", encoding="utf-8")
    temporary = tmp_path / "temporary"
    monkeypatch.delenv("MPLCONFIGDIR", raising=False)
    monkeypatch.setattr(startup.QStandardPaths, "writableLocation", lambda location: str(blocked))
    monkeypatch.setattr(startup.tempfile, "gettempdir", lambda: str(temporary))

    details = startup._configure_matplotlib_cache()

    assert details["matplotlib_cache_source"] == "temporary"
    assert details["matplotlib_cache_temporary"]
    assert Path(details["matplotlib_cache_path"]).is_dir()
    assert os.environ["MPLCONFIGDIR"] == details["matplotlib_cache_path"]


def test_packaged_cache_replaces_packager_temp_with_application_cache(monkeypatch, tmp_path):
    packager_cache = tmp_path / "packager-temporary"
    persistent = tmp_path / "application-cache"
    monkeypatch.setenv("MPLCONFIGDIR", str(packager_cache))
    monkeypatch.setattr(startup.sys, "_fraggenotyper_original_mplconfigdir", None, raising=False)
    monkeypatch.setattr(startup.QStandardPaths, "writableLocation", lambda location: str(persistent))

    details = startup._configure_matplotlib_cache()

    assert details["matplotlib_cache_source"] == "application_cache"
    assert details["matplotlib_packager_cache_replaced"]
    assert Path(os.environ["MPLCONFIGDIR"]) == persistent / "matplotlib"
    assert (persistent / "matplotlib").is_dir()


def test_packaged_cache_restores_explicit_user_setting(monkeypatch, tmp_path):
    explicit = tmp_path / "user-font-cache"
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "packager-temporary"))
    monkeypatch.setattr(startup.sys, "_fraggenotyper_original_mplconfigdir", str(explicit), raising=False)

    details = startup._configure_matplotlib_cache()

    assert os.environ["MPLCONFIGDIR"] == str(explicit)
    assert details["matplotlib_cache_source"] == "environment"
    assert details["matplotlib_packager_cache_replaced"]
