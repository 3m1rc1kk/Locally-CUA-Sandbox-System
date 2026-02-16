# 🖥️ CUA — Computer Use Agent OS(With Local Sandbox)

A locally-running AI agent that autonomously controls a virtual desktop inside a Docker container using the **Qwen3-VL** vision-language model. The user issues natural language commands; the agent analyzes live screenshots of the VM, plans multi-step actions, and executes mouse/keyboard inputs to accomplish the task — all running on your own hardware with no cloud APIs required.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyQt6](https://img.shields.io/badge/PyQt6-6.10-green)
![CUDA](https://img.shields.io/badge/CUDA-13.0-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🎬 Demo

![Gui](./assets/548197364-27fa7a7a-f725-428d-a3dc-75ba41ec79de.gif)



## 🎯 What Is This?

Most "computer use" demos rely on cloud-hosted models (GPT-4V, Claude, etc.). This project runs the **entire pipeline locally**:

1. A **Docker container** (`trycua/cua-xfce`) provides a full XFCE Linux desktop accessible via VNC and a REST API.
2. A **Qwen3-VL 8B** vision-language model (GGUF format, accelerated on your NVIDIA GPU via `llama-cpp-python`) looks at the VM's screen and decides the next action.
3. A **PyQt6 Mission Control UI** lets you issue commands, watch the agent work in real-time, inspect each step, and intervene when needed.

**Agent loop:** `Screenshot → Model inference → Action (click/type/scroll/hotkey) → Wait → Repeat` — until the objective is complete or a safety guard triggers.

## ✨ Features

- **Qwen3-VL 8B** vision-language model (GGUF, runs locally on GPU)
- **Docker Sandbox** — isolated virtual desktop via `trycua/cua-xfce` container
- **Mission Control UI** — professional 5-panel PyQt6 interface
- **Live VM Screen** — direct mouse/keyboard interaction with the VM
- **Agent Trace** — step-by-step plan visualization, metrics, structured logs
- **Safety Guards** — repeat detection, coordinate validation, step limit
- **Turkish → English Translation** — commands are auto-translated (optional)
- **JSON Log Export** — export structured logs for debugging/analysis

## 📋 Requirements

| Component | Minimum |
|-----------|---------|
| **OS** | Ubuntu 22.04+ |
| **Python** | 3.10 |
| **GPU** | NVIDIA with CUDA support (8 GB+ VRAM recommended) |
| **NVIDIA Driver** | 535+ |
| **Docker** | 24.0+ |
| **RAM** | 16 GB+ recommended |

## 🚀 Installation from Scratch

### 1. NVIDIA Driver

Check if the driver is installed:
```bash
nvidia-smi
```

If not installed:
```bash
sudo apt update
sudo apt install -y nvidia-driver-535
sudo reboot
```

### 2. Docker

```bash
# Install Docker Engine
sudo apt update
sudo apt install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io

# Allow running Docker without sudo (re-login required)
sudo usermod -aG docker $USER
newgrp docker

# Verify
docker run --rm hello-world
```

### 3. Pull the Sandbox Container

```bash
docker pull trycua/cua-xfce:latest
```

### 4. Create Conda Environment

If Miniconda is not installed:
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# Close and reopen your terminal
```

Create and activate the environment:
```bash
conda create -n cua python=3.10 -y
conda activate cua
```

### 5. Install PyTorch (CUDA 13.0)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

> **Note:** For different CUDA versions, visit https://pytorch.org/get-started/locally/

### 6. Install llama-cpp-python (NVIDIA GPU — Prebuilt Wheel)

The standard `pip install llama-cpp-python` **does not include CUDA support**. To run on an NVIDIA GPU, use a prebuilt wheel from the JamePeng fork:

1. Go to https://github.com/JamePeng/llama-cpp-python/releases
2. Download the `.whl` file matching your system:
   - Python version: `cp310` (Python 3.10)
   - Platform: `linux_x86_64`
   - CUDA version: `cu130` (CUDA 13.0) or `cu124` (CUDA 12.4)
   - Example: `llama_cpp_python-0.3.23+cu130-cp310-cp310-linux_x86_64.whl`

3. Install the downloaded wheel:
```bash
conda activate cua
pip install llama_cpp_python-0.3.23+cu130-cp310-cp310-linux_x86_64.whl
```

> **Check your CUDA version:** look at the "CUDA Version" line in `nvidia-smi` output.

### 7. Install Remaining Python Packages

```bash
conda activate cua
pip install -r requirements.txt
```

### 8. Translation Model (Optional — Turkish Command Support)

To auto-translate Turkish commands to English:
```bash
pip install sentencepiece
# The model (Helsinki-NLP/opus-mt-tc-big-tr-en) downloads automatically on first run.
```

## ▶️ Running

### Mission Control UI (Recommended)

```bash
conda activate cua
python gui_mission_control.py
```

Opens a professional 5-panel interface:
- **Top Bar** — Docker/Model status, step counter, latency
- **Left** — Command input, preset commands, agent step trace
- **Center** — Live VM screen (mouse/keyboard active)
- **Right** — Last action detail, metrics, sandbox info, config
- **Bottom** — Structured logs with JSON export

### Classic UI

```bash
conda activate cua
python gui_main.py
```

### Terminal Only (No GUI)

```bash
conda activate cua
python main.py
```

## ⌨️ Keyboard Shortcuts (Mission Control)

| Shortcut | Action |
|----------|--------|
| `Ctrl+Enter` | Run command |
| `Escape` | Stop running command |
| `F11` | Toggle fullscreen |
| `Ctrl+L` | Clear logs |

## 📁 Project Structure

```
CUA-system-running-locally-via-sandbox/
│
├── gui_mission_control.py       # Mission Control UI (recommended)
├── gui_main.py                  # Classic UI
├── main.py                      # Terminal-only agent loop
├── setup.py                     # Package setup
├── requirements.txt             # Python dependencies
├── README.md
│
├── src/                         # Source modules
│   ├── __init__.py
│   ├── config.py                # All configuration parameters
│   ├── sandbox.py               # Docker container REST API wrapper
│   ├── llm_client.py            # Qwen3-VL model loading & inference
│   ├── vision.py                # Screenshot capture, resize, preview
│   ├── actions.py               # Action execution (click, type, scroll)
│   ├── guards.py                # Safety checks (repeat guard, validation)
│   ├── translation.py           # Translation helper
│   ├── design_system.py         # UI design tokens & stylesheet
│   └── panels.py                # UI panel widgets
│
├── assets/                      # Demo videos & media
│   
│
└── img/                         # Runtime screenshots (auto-generated)
    └── (click previews, screen captures)
```

## ⚙️ Configuration

All parameters are in `src/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SANDBOX_IMAGE` | `trycua/cua-xfce:latest` | Docker image for the VM |
| `API_PORT` | `8001` | Container API port (host side) |
| `VNC_RESOLUTION` | `1920x1080` | VM screen resolution |
| `N_GPU_LAYERS` | `-1` (all) | Number of model layers offloaded to GPU |
| `N_CTX` | `2048` | Model context length |
| `MAX_STEPS` | `20` | Maximum steps per command |
| `GGUF_REPO_ID` | `mradermacher/Qwen3-VL-8B...` | HuggingFace model repository |

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: PyQt6` | `conda activate cua && pip install PyQt6` |
| `Docker permission denied` | `sudo usermod -aG docker $USER` + re-login |
| `Sandbox API timeout` | Container startup takes 60–120s, wait for it |
| `CUDA out of memory` | Reduce `N_GPU_LAYERS` in `src/config.py` |
| `llama-cpp CUDA error` | Ensure you installed the wheel matching your CUDA version |
| Slow model download | The first run downloads a ~5 GB GGUF model, be patient |


## 🗺️ Roadmap

> **Status Legend:** ✅ Done · 🔄 In Progress · ⬜ Not Started

| # | Feature | Description | Est. Time | Status |
|---|---------|-------------|-----------|--------|
| 1| **Project Restructuring** | Reorganize files into `src/`, `assets/`, `img/` directories; update all import paths |         | ✅  |
| 2 | **Mission Control UI** | Professional 5-panel PyQt6 interface with live VM view, command panel, inspector, and logs |         | ✅ |
| 3 | **README & Documentation** | Comprehensive README with installation guide, configuration reference, and troubleshooting |        | ✅  |
| 5 | **A model that plans detailed operations.** | An LLM (API with) that performs detailed planning on behalf of the user for more complex operations| 1-2 Week| 🔄  |
| 4 | **Multi-Model Support** | Allow switching between different VLMs (Qwen3-VL, LLaVA, InternVL) via config or UI dropdown | unknown | ⬜  |
| 5 | **Conversation Memory** | Persistent chat history so the agent remembers context across multiple commands in a session | unknown | ⬜  |
| 6 | **Action Undo / Rollback** | Snapshot VM state before each action and allow rollback on failure | unknown | ⬜ |
| 7 | **Multi-Monitor / Multi-VM** | Support controlling multiple Docker containers simultaneously from a single UI | unknown | ⬜ |
| 8 | **Voice Command Input** | Accept voice commands via Whisper (local STT) instead of typing | unknown | ⬜ Not Started |
| 9 | **Windows & macOS Support** | Cross-platform compatibility with native installers and platform-specific sandboxes | unknown | ⬜ |

## 📄 License

MIT
