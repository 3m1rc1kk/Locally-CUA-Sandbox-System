"""
CUA Mission Control — Panel Widgets
Top bar, Command panel, Inspector, Log panel
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QWidget, QLabel, QFrame, QVBoxLayout, QHBoxLayout,
    QPushButton, QLineEdit, QTextEdit, QListWidget, QListWidgetItem,
    QGridLayout, QSizePolicy, QFileDialog
)

from src.design_system import C


# ═══════════════════════════════════════════
# STATUS DOT helper
# ═══════════════════════════════════════════

def _dot(color: str, text: str) -> str:
    return f'<span style="color:{color}; font-size:14px;">●</span> {text}'


# ═══════════════════════════════════════════
# TOP BAR
# ═══════════════════════════════════════════

class TopBar(QFrame):
    """Docker status, Model status, Step counter, Latency indicator."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("topBar")
        self.setFixedHeight(48)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 0, 16, 0)
        layout.setSpacing(24)

        # Title
        title = QLabel("⚡ CUA Mission Control")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {C.PRIMARY_LIGHT};")
        layout.addWidget(title)

        layout.addStretch()

        # Docker status
        self.docker_status = QLabel(_dot(C.TEXT_MUTED, "Docker: —"))
        self.docker_status.setObjectName("statusDot")
        layout.addWidget(self.docker_status)

        # Model status
        self.model_status = QLabel(_dot(C.TEXT_MUTED, "Model: —"))
        self.model_status.setObjectName("statusDot")
        layout.addWidget(self.model_status)

        # Step counter
        self.step_label = QLabel("Step: 0")
        self.step_label.setStyleSheet(f"color: {C.TEXT_DIM}; font-size: 12px;")
        layout.addWidget(self.step_label)

        # Latency
        self.latency_label = QLabel("Latency: —")
        self.latency_label.setStyleSheet(f"color: {C.TEXT_DIM}; font-size: 12px;")
        layout.addWidget(self.latency_label)

    def set_docker_status(self, connected: bool) -> None:
        color = C.GREEN if connected else C.RED
        text = "Connected" if connected else "Disconnected"
        self.docker_status.setText(_dot(color, f"Docker: {text}"))

    def set_model_status(self, status: str) -> None:
        color_map = {"loading": C.ORANGE, "ready": C.GREEN, "error": C.RED}
        color = color_map.get(status, C.TEXT_MUTED)
        label_map = {"loading": "Loading…", "ready": "Ready", "error": "Error"}
        self.model_status.setText(_dot(color, f"Model: {label_map.get(status, status)}"))

    def set_step(self, n: int) -> None:
        self.step_label.setText(f"Step: {n}")

    def set_latency(self, ms: float) -> None:
        self.latency_label.setText(f"Latency: {ms:.0f}ms")


# ═══════════════════════════════════════════
# COMMAND PANEL (Left)
# ═══════════════════════════════════════════

class CommandPanel(QFrame):
    """Command input, preset commands, history and agent step list."""

    run_requested = pyqtSignal(str)
    stop_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("leftPanel")
        self.setMinimumWidth(280)
        self.setMaximumWidth(360)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # --- Command input ---
        lbl = QLabel("COMMAND")
        lbl.setObjectName("sectionTitle")
        layout.addWidget(lbl)

        self.cmd_input = QLineEdit()
        self.cmd_input.setPlaceholderText("Enter command… (e.g. Open browser)")
        self.cmd_input.returnPressed.connect(self._emit_run)
        layout.addWidget(self.cmd_input)

        btn_row = QHBoxLayout()
        self.run_btn = QPushButton("▶  Run")
        self.run_btn.setObjectName("runBtn")
        self.run_btn.clicked.connect(self._emit_run)
        self.stop_btn = QPushButton("■  Stop")
        self.stop_btn.setObjectName("stopBtn")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_requested.emit)
        btn_row.addWidget(self.run_btn)
        btn_row.addWidget(self.stop_btn)
        layout.addLayout(btn_row)

        # --- Presets ---
        lbl2 = QLabel("PRESETS")
        lbl2.setObjectName("sectionTitle")
        layout.addWidget(lbl2)

        presets = [
            ("🏠 Home", "Click Home file"),
            ("💻 Terminal", "Open terminal"),
            ("🌐 Browser", "Open web browser"),
            ("📝 Notepad", "Open text editor"),
            ("🔍 Wikipedia", 'Open browser, go to Wikipedia, search "LLM"'),
            ("📁 Files", "Open file manager"),
        ]
        grid = QGridLayout()
        grid.setSpacing(4)
        self._preset_buttons: List[QPushButton] = []
        for i, (title, cmd) in enumerate(presets):
            b = QPushButton(title)
            b.setObjectName("presetBtn")
            b.setToolTip(cmd)
            b.clicked.connect(lambda _, c=cmd: self.cmd_input.setText(c))
            grid.addWidget(b, i // 2, i % 2)
            self._preset_buttons.append(b)
        layout.addLayout(grid)

        # --- Agent Steps ---
        lbl3 = QLabel("AGENT STEPS")
        lbl3.setObjectName("sectionTitle")
        layout.addWidget(lbl3)

        self.steps_list = QListWidget()
        self.steps_list.setAlternatingRowColors(False)
        layout.addWidget(self.steps_list, stretch=1)

    def _emit_run(self) -> None:
        text = self.cmd_input.text().strip()
        if text:
            self.run_requested.emit(text)

    def set_busy(self, busy: bool) -> None:
        self.run_btn.setEnabled(not busy)
        self.stop_btn.setEnabled(busy)
        self.cmd_input.setEnabled(not busy)
        for b in self._preset_buttons:
            b.setEnabled(not busy)

    def add_step(self, step_num: int, action: str, detail: str = "") -> None:
        text = f"#{step_num}  {action}"
        if detail:
            text += f" — {detail}"
        item = QListWidgetItem(text)
        self.steps_list.addItem(item)
        self.steps_list.scrollToBottom()

    def clear_steps(self) -> None:
        self.steps_list.clear()


# ═══════════════════════════════════════════
# INSPECTOR (Right)
# ═══════════════════════════════════════════

class InspectorPanel(QFrame):
    """Last action, metrics, settings."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("rightPanel")
        self.setMinimumWidth(260)
        self.setMaximumWidth(340)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # --- Last Action ---
        lbl = QLabel("LAST ACTION")
        lbl.setObjectName("sectionTitle")
        layout.addWidget(lbl)

        self.action_display = QTextEdit()
        self.action_display.setReadOnly(True)
        self.action_display.setMaximumHeight(140)
        self.action_display.setPlaceholderText("No action yet…")
        layout.addWidget(self.action_display)

        # --- Metrics ---
        lbl2 = QLabel("METRICS")
        lbl2.setObjectName("sectionTitle")
        layout.addWidget(lbl2)

        metrics_frame = QFrame()
        metrics_frame.setStyleSheet(f"background: {C.BG_INPUT}; border-radius: 8px; padding: 8px;")
        mg = QGridLayout(metrics_frame)
        mg.setSpacing(8)

        self.metric_labels: Dict[str, tuple] = {}
        for i, (key, label) in enumerate([
            ("steps", "Total Steps"),
            ("clicks", "Clicks"),
            ("types", "Types"),
            ("elapsed", "Elapsed (s)"),
        ]):
            val_lbl = QLabel("0")
            val_lbl.setObjectName("metricValue")
            val_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            name_lbl = QLabel(label)
            name_lbl.setObjectName("metricLabel")
            name_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            mg.addWidget(val_lbl, 0, i)
            mg.addWidget(name_lbl, 1, i)
            self.metric_labels[key] = (val_lbl, name_lbl)

        layout.addWidget(metrics_frame)

        # --- VM Info ---
        lbl3 = QLabel("SANDBOX INFO")
        lbl3.setObjectName("sectionTitle")
        layout.addWidget(lbl3)

        self.vm_info = QLabel("Container: —\nResolution: —\nAPI: —")
        self.vm_info.setStyleSheet(f"color: {C.TEXT_DIM}; font-size: 11px; padding: 8px; background: {C.BG_INPUT}; border-radius: 6px;")
        self.vm_info.setWordWrap(True)
        layout.addWidget(self.vm_info)

        # --- Config ---
        lbl4 = QLabel("CONFIGURATION")
        lbl4.setObjectName("sectionTitle")
        layout.addWidget(lbl4)

        self.config_info = QLabel("")
        self.config_info.setStyleSheet(f"color: {C.TEXT_MUTED}; font-size: 10px; padding: 6px; background: {C.BG_INPUT}; border-radius: 6px;")
        self.config_info.setWordWrap(True)
        layout.addWidget(self.config_info)

        layout.addStretch()

    def set_last_action(self, action_dict: Dict[str, Any]) -> None:
        txt = json.dumps(action_dict, indent=2, ensure_ascii=False)
        self.action_display.setPlainText(txt)

    def set_metrics(self, steps: int = 0, clicks: int = 0, types: int = 0, elapsed: float = 0) -> None:
        data = {"steps": str(steps), "clicks": str(clicks), "types": str(types), "elapsed": f"{elapsed:.1f}"}
        for key, val in data.items():
            if key in self.metric_labels:
                self.metric_labels[key][0].setText(val)

    def set_vm_info(self, container: str, resolution: str, api_url: str) -> None:
        self.vm_info.setText(f"Container: {container}\nResolution: {resolution}\nAPI: {api_url}")

    def set_config(self, cfg) -> None:
        lines = [
            f"Model: {getattr(cfg, 'GGUF_MODEL_FILENAME', '?')}",
            f"Max Steps: {getattr(cfg, 'MAX_STEPS', '?')}",
            f"N_CTX: {getattr(cfg, 'N_CTX', '?')}",
            f"GPU Layers: {getattr(cfg, 'N_GPU_LAYERS', '?')}",
        ]
        self.config_info.setText("\n".join(lines))


# ═══════════════════════════════════════════
# LOG PANEL (Bottom)
# ═══════════════════════════════════════════

class LogPanel(QFrame):
    """Structured logs, error filter, export."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("bottomPanel")
        self.setMinimumHeight(150)
        self.setMaximumHeight(280)

        self._entries: List[Dict[str, str]] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(6)

        header = QHBoxLayout()
        lbl = QLabel("📋 LOGS")
        lbl.setObjectName("sectionTitle")
        header.addWidget(lbl)
        header.addStretch()

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setFixedHeight(28)
        self.clear_btn.clicked.connect(self.clear)
        header.addWidget(self.clear_btn)

        self.export_btn = QPushButton("JSON Export")
        self.export_btn.setFixedHeight(28)
        self.export_btn.clicked.connect(self._export)
        header.addWidget(self.export_btn)

        layout.addLayout(header)

        self.log_box = QTextEdit()
        self.log_box.setObjectName("logBox")
        self.log_box.setReadOnly(True)
        layout.addWidget(self.log_box)

    def append(self, msg: str, level: str = "info") -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        color_map = {"info": C.TEXT_DIM, "warn": C.ORANGE, "error": C.RED, "success": C.GREEN, "model": C.PRIMARY_LIGHT}
        color = color_map.get(level, C.TEXT_DIM)
        self.log_box.append(f'<span style="color:{C.TEXT_MUTED}">[{ts}]</span> <span style="color:{color}">{msg}</span>')
        self._entries.append({"ts": ts, "level": level, "msg": msg})
        sb = self.log_box.verticalScrollBar()
        sb.setValue(sb.maximum())

    def clear(self) -> None:
        self.log_box.clear()
        self._entries.clear()

    def _export(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export Logs", "cua_logs.json", "JSON (*.json)")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._entries, f, ensure_ascii=False, indent=2)
            self.append(f"Logs exported: {path}", "success")
