from setuptools import setup, find_packages

setup(
    name="cuaOS",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "PyQt6>=6.6",
        "requests>=2.31",
        "Pillow>=10.0",
        "numpy>=1.24",
        "opencv-python>=4.8",
        "huggingface_hub>=0.20",
        "transformers>=4.40",
        "sentencepiece>=0.1.99",
        # llama-cpp-python is installed via prebuilt wheel; see README.
    ],
    entry_points={
        "console_scripts": [
            "cua-gui=gui_main:main",
            "cua-mission=gui_mission_control:main",
            "cua-cli=main:main",
        ]
    },
)
