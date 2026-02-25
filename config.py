import json
import os

CONFIG_FILE = os.path.join(os.path.dirname(__file__), "config.json")

def _load_config():
    if not os.path.exists(CONFIG_FILE):
        return {
            "model": {
                "hidden_sizes": [128, 64, 32],
                "dropout_rate": 0.2,
                "learning_rate": 0.001,
                "batch_size": 32,
                "patience": 20,
            },
            "features": {
                "window_size": 100,
                "window_overlap": 0.50,
                "sampling_rate": 250
            }
        }
    with open(CONFIG_FILE, 'r') as f:
        return json.load(f)

def _save_config(config_dict):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config_dict, f, indent=4)

CONFIG = _load_config()

# Ensure file exists physically for tracking parity
if not os.path.exists(CONFIG_FILE):
    _save_config(CONFIG)
