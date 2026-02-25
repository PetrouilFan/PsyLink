# MindWave EEG Python Interface (PsyLink)

A Python interface for the NeuroSky MindWave EEG headset. This project allows you to connect to the MindWave Mobile headset, visualize brain activity in real-time, record data for later analysis, and use Machine Learning (ML) to classify mental states or commands.

## Features

- **Robust Connectivity**: Connect to the MindWave Mobile headset via serial connection with automatic reconnection logic.
- **Real-time Visualization**: View raw EEG data, attention levels, and meditation levels in real-time.
- **Data Recording**: Record EEG streams to CSV files with timestamping for offline analysis.
- **Signal Processing**: Apply high-pass, low-pass, and notch filters (via SciPy) to clean raw signals.
- **Machine Learning**: Extract time-domain, frequency-domain, and dynamic features, and train/run an MLP classifier using PyTorch.
- **Centralized Configuration**: Easily adjust hyperparameters like window size, batch size, and network architecture via a unified `config.json`.

## Components

The project consists of the following main modules:
- `mindwave.py`: Core driver for communication with the MindWave headset, with `Enum` bytecodes and robust error handling.
- `eeg_buffer.py`: Thread-safe buffer implementation for storing EEG sliding windows.
- `demo_mindwave.py`: Demonstration script with real-time visualization and filtering.
- `mindwave_recorder.py`: Application to record EEG data to CSV files with GUI interactions.
- `filters.py`: Signal processing utilities for noise attenuation and data normalization.
- `feature_extraction.py`: Functions to extract statistical and frequency-band power features from EEG epochs.
- `train_mlp_model.py`: Script to train a PyTorch Multilayer Perceptron on recorded datasets.
- `inference_mlp_model.py`: Real-time inference script to classify live brainwaves into commands.
- `config.py`: Centralized configuration manager.

## Installation

This project uses [`uv`](https://github.com/astral-sh/uv) directly for fast and deterministic dependency management.

1. Install `uv` if you haven't already.
2. Clone the directory and sync dependencies:

```bash
uv sync
```

## Usage

You can run any script securely within the managed environment using `uv run`.

### Basic Demo
Run the real-time visualization (specify the `--port` if needed, e.g., `--port COM4` or `--port /dev/ttyUSB0`):
```bash
uv run python demo_mindwave.py --port COM4
```

### Recording Data
To record EEG data to a CSV file for training an ML model:
```bash
uv run python mindwave_recorder.py --port COM4
```

### Training the Machine Learning Model
Assuming you've organized standard recordings into a folder mapping classes (e.g., `datasets/commands/up`, `datasets/commands/down`):
```bash
uv run python train_mlp_model.py --dataset datasets/commands
```

### Real-Time Inference
Once trained, run the live model to see predictions:
```bash
uv run python inference_mlp_model.py --port COM4 --model_dir models
```

## Configuration

Most overarching parameters (like window size, ML learning rates, network architecture sizes) are editable in `config.json` after the scripts auto-generate it.

## Hardware Requirements

- NeuroSky MindWave Mobile headset
- Bluetooth adapter or built-in Bluetooth on your computer

## Credits

The core MindWave driver (`mindwave.py`) is based on the [python-mindwave](https://github.com/faturita/python-mindwave) project by [faturita](https://github.com/faturita).

A significant portion of this codebase was enhanced and optimized with the assistance of AI tools.

## License

This project is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/). This means you are free to:

- Share — copy and redistribute the material in any medium or format
- Adapt — remix, transform, and build upon the material

Under the following terms:
- Attribution — You must give appropriate credit, provide a link to the license, and indicate if changes were made.
- NonCommercial — You may not use the material for commercial purposes.
