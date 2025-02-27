# MindWave EEG Python Interface

A Python interface for the NeuroSky MindWave EEG headset. This project allows you to connect to the MindWave Mobile headset, visualize brain activity in real-time, and record data for later analysis.

## Features

- Connect to MindWave Mobile headset via serial connection
- Real-time visualization of EEG signals including:
  - Raw EEG data
  - Attention levels
  - Meditation levels
- Record data to CSV files for later analysis
- Apply noise filtering to raw EEG signals
- Replay previously recorded sessions

## Components

The project consists of the following main components:

- mindwave.py - Core driver for communication with the MindWave headset
- eeg_buffer.py - Buffer implementation for storing EEG data in windows
- demo_mindwave.py - Demonstration script with real-time visualization and filtering
- mindwave_recorder.py - Application to record EEG data to CSV files

## Usage

### Basic Demo

Run the demo visualization:

```bash
python demo_mindwave.py
```

### Recording Data

To record EEG data to a CSV file:

```bash
python mindwave_recorder.py
```

## Hardware Requirements

- NeuroSky MindWave Mobile headset
- Bluetooth adapter or built-in Bluetooth on your computer

## Credits

The core MindWave driver (`mindwave.py`) is based on the [python-mindwave](https://github.com/faturita/python-mindwave) project by [faturita](https://github.com/faturita).

A significant portion of this codebase was developed with the assistance of AI tools.

## License

This project is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/). This means you are free to:

- Share — copy and redistribute the material in any medium or format
- Adapt — remix, transform, and build upon the material

Under the following terms:

- Attribution — You must give appropriate credit, provide a link to the license, and indicate if changes were made.
- NonCommercial — You may not use the material for commercial purposes.
