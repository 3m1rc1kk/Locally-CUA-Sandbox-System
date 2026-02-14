#!/usr/bin/env python3
"""
CUA Mission Control — Main Window (English UI)
Professional control interface for tasks connected to a Docker sandbox.
"""

from __future__ import annotations
import sys
import time
import threading
import traceback
from typing import Any, Dict, List, Optional, Tuple

from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt6.QtGui import QPixmap, QImage, QPainter, QKeyEvent, QMouseEvent, QWheelEvent, QShortcut, QKeySequence
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel,
    QVBoxLayout, QHBoxLayout, QFrame, QSizePolicy
)

from src.config import cfg
from src.sandbox import Sandbox
from src.llm_client import load_llm, ask_next_action
from src.vision import capture_screen, capture_screen_raw, draw_preview
from src.guards import validate_xy, should_stop_on_repeat
from src.actions import execute_action
from src.design_system import build_stylesheet
from src.panels import TopBar, CommandPanel, InspectorPanel, LogPanel

# ──────────────────────────────────────────────────────────────────────────

def trim_history(history: List[Dict[str, Any]], keep_last: int = 6) -> List[Dict[str, Any]]:
    return history[-keep_last:] if len(history) > keep_last else history

def _center_from_bbox(b: List[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = map(float, b)
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0

def _extract_xy(out: Dict[str, Any]) -> Tuple[float, float]:
    x = out.get("x", 0.5)
    y = out.get("y", 0.5)
    pos = out.get("position", None)
    if pos is not None and isinstance(pos, (list, tuple)):
        if len(pos) == 2 and all(isinstance(t, (int, float)) for t in pos):
            return float(pos[0]), float(pos[1])
        if len(pos) == 4 and all(isinstance(t, (int, float)) for t in pos):
            return _center_from_bbox(list(pos))
    return float(x), float(y)

def pil_to_qpixmap(pil_img) -> QPixmap:
    rgb = pil_img.convert("RGB")
    w, h = rgb.size
    data = rgb.tobytes("raw", "RGB")
    bpl = 3 * w
    qimg = QImage(data, w, h, bpl, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)

# ──────────────────────────────────────────────────────────────────────────

class VMView(QLabel):
    """Display live VM screen and forward mouse/keyboard input."""

    def __init__(self, sandbox: Sandbox, parent=None):
        super().__init__(parent)
        self.sandbox = sandbox
        self._pm: Optional[QPixmap] = None
        self._draw_rect = None
        self.input_enabled = True
        self._pressed_btn = None
        self._last_move_ts: float = 0.0
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setObjectName("vmView")
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def set_frame(self, pm: QPixmap) -> None:
        self._pm = pm
        self.update()

    def _pos_to_norm(self, x: int, y: int):
        if not self._pm or not self._draw_rect:
            return None
        dx, dy, dw, dh = self._draw_rect
        if x < dx or y < dy or x >= dx + dw or y >= dy + dh:
            return None
        return float((x - dx) / dw), float((y - dy) / dh)

    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), Qt.GlobalColor.black)
        if not self._pm:
            p.end()
            return
        scaled = self._pm.scaled(
            self.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)
        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        self._draw_rect = (x, y, scaled.width(), scaled.height())
        p.drawPixmap(x, y, scaled)
        p.end()

    def mousePressEvent(self, e: QMouseEvent):
        if not self.input_enabled:
            return
        self.setFocus()
        mapped = self._pos_to_norm(int(e.position().x()), int(e.position().y()))
        if not mapped:
            return
        nx, ny = mapped
        btn_map = {
            Qt.MouseButton.LeftButton: 1,
            Qt.MouseButton.RightButton: 3,
            Qt.MouseButton.MiddleButton: 2
        }
        btn = btn_map.get(e.button())
        if not btn:
            return
        self._pressed_btn = btn
        self.sandbox.mouse_move_norm(nx, ny)
        self.sandbox.mouse_down(btn)

# ──────────────────────────────────────────────────────────────────────────

class AgentSignals(QObject):
    log = pyqtSignal(str, str)
    busy = pyqtSignal(bool)
    finished = pyqtSignal(str)
    step_update = pyqtSignal(int, str, str)
    action_update = pyqtSignal(dict)
    latency_update = pyqtSignal(float)

# ──────────────────────────────────────────────────────────────────────────

def run_single_command(
    sandbox: Sandbox, llm, objective: str,
    signals: AgentSignals,
    stop_event: Optional[threading.Event] = None,
) -> str:
    history, step = [], 1
    while True:
        if stop_event and stop_event.is_set():
            return "STOPPED"

        signals.log.emit(f"═══ STEP {step} ═══", "info")
        time.sleep(getattr(cfg, "WAIT_BEFORE_SCREENSHOT_SEC", 0.8))

        img = capture_screen(sandbox, cfg.SCREENSHOT_PATH)
        out = ask_next_action(llm, objective, cfg.SCREENSHOT_PATH, trim_history(history))
        action = (out.get("action") or "NOOP").upper()
        signals.log.emit(f"[MODEL] {action}", "model")
        signals.action_update.emit(out)
        signals.step_update.emit(step, action, "")

        if action == "NOOP":
            return "DONE(NOOP)"

        execute_action(sandbox, out)
        history.append(out)
        step += 1
        if step > getattr(cfg, "MAX_STEPS", 30):
            return "DONE(max-steps)"

# ──────────────────────────────────────────────────────────────────────────

class MissionControlWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CUA Mission Control")
        self.resize(1680, 980)
        self.setStyleSheet(build_stylesheet())

        self.sandbox: Optional[Sandbox] = None
        self.llm = None
        self.stop_event = None
        self.worker_thread = None

        self.signals = AgentSignals()
        self.signals.log.connect(self._on_log)
        self.signals.busy.connect(self._on_busy)
        self.signals.finished.connect(self._on_finished)
        self.signals.step_update.connect(self._on_step)
        self.signals.action_update.connect(self._on_action_update)

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        self.top_bar = TopBar()
        layout.addWidget(self.top_bar)

        self.cmd_panel = CommandPanel()
        self.cmd_panel.run_requested.connect(self._on_run)
        self.cmd_panel.stop_requested.connect(self._on_stop)
        layout.addWidget(self.cmd_panel)

        self.vm_display = QLabel("Sandbox connection initializing…")
        self.vm_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.vm_display)

        self.log_panel = LogPanel()
        layout.addWidget(self.log_panel)

        QShortcut(QKeySequence("Ctrl+Return"), self, self._shortcut_run)
        QShortcut(QKeySequence("Escape"), self, self._on_stop)

        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self._refresh_vm)
        self.refresh_timer.start(350)

        self.log_panel.append("Initializing backend…", "info")
        threading.Thread(target=self._init_backend, daemon=True).start()

    def _init_backend(self):
        try:
            self.signals.log.emit("Starting Docker container…", "info")
            self.sandbox = Sandbox(cfg)
            self.sandbox.start()
            self.signals.log.emit("Docker sandbox connected ✓", "success")
        except Exception as e:
            self.signals.log.emit(f"Docker ERROR: {e}", "error")

        try:
            self.signals.log.emit("Loading model…", "info")
            self.llm = load_llm()
            self.signals.log.emit("Model ready ✓", "success")
        except Exception as e:
            self.signals.log.emit(f"Model ERROR: {e}", "error")

    def _refresh_vm(self):
        if not self.sandbox:
            return
        try:
            img = capture_screen_raw(self.sandbox)
            self.vm_display.setPixmap(pil_to_qpixmap(img))
        except Exception:
            pass

    def _shortcut_run(self):
        cmd = self.cmd_panel.cmd_input.text().strip()
        if cmd:
            self._on_run(cmd)

    def _on_run(self, objective: str):
        if not objective:
            self.log_panel.append("Command cannot be empty.", "warn")
            return
        if not self.llm:
            self.log_panel.append("Model not loaded yet!", "error")
            return
        if not self.sandbox:
            self.log_panel.append("No sandbox connection!", "error")
            return

        self.log_panel.append(f"Executing: {objective}", "info")
        self.stop_event = threading.Event()
        self.signals.busy.emit(True)

        def worker():
            try:
                result = run_single_command(
                    self.sandbox, self.llm, objective,
                    self.signals, self.stop_event
                )
                self.signals.finished.emit(f"Result: {result}")
            except Exception:
                self.signals.log.emit("ERROR:\n" + traceback.format_exc(), "error")
            finally:
                self.signals.busy.emit(False)

        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()

    def _on_stop(self):
        if self.stop_event:
            self.stop_event.set()
            self.log_panel.append("Stop signal sent.", "warn")

    def _on_log(self, msg: str, level: str):
        self.log_panel.append(msg, level)

    def _on_busy(self, busy: bool):
        self.cmd_panel.set_busy(busy)

    def _on_finished(self, msg: str):
        self.log_panel.append(msg, "success")

    def _on_step(self, step: int, action: str, detail: str):
        self.log_panel.append(f"Step {step}: {action}", "info")

    def _on_action_update(self, action: dict):
        pass

def main():
    app = QApplication(sys.argv)
    w = MissionControlWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
