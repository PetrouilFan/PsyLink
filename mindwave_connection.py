import time
import threading
import mindwave
from eeg_buffer import EEGBuffer

class MindwaveConnection:
    """
    Manages the connection between a Mindwave headset and an EEGBuffer.
    Handles the data flow and provides methods to access the buffered data.
    """
    def __init__(self, device_path):
        """
        Initialize the connection.
        
        Args:
            device_path: Path to the Mindwave device (e.g., COM3 on Windows, /dev/ttyUSB0 on Linux)
        """
        self.device_path = device_path
        self.headset = None
        self.buffer = EEGBuffer()
        self.running = False
    
    def connect(self):
        """Connect to the Mindwave headset and set up handlers."""
        try:
            self.headset = mindwave.Headset(self.device_path)
            
            # Register handlers for different data types
            self.headset.raw_value_handlers.append(self._handle_raw)
            self.headset.attention_handlers.append(self._handle_attention)
            self.headset.meditation_handlers.append(self._handle_meditation)
            
            print(f"Connected to Mindwave headset at {self.device_path}")
            return True
        except Exception as e:
            print(f"Failed to connect to headset: {e}")
            return False
    
    def _handle_raw(self, headset, raw_value):
        """Handler for raw EEG values."""
        self.buffer.on_value_callback(raw=raw_value)
    
    def _handle_attention(self, headset, attention):
        """Handler for attention values."""
        self.buffer.on_value_callback(attention=attention)
    
    def _handle_meditation(self, headset, meditation):
        """Handler for meditation values."""
        self.buffer.on_value_callback(meditation=meditation)
    
    def start(self):
        """Start processing data and changing windows."""
        if not self.headset:
            if not self.connect():
                return False
        
        self.running = True
        
        print("Started processing Mindwave data")
        return True
    
    def stop(self):
        """Stop processing data."""
        self.running = False
        
        if self.headset:
            self.headset.stop()
            print("Disconnected from Mindwave headset")
    
    def get_current_window(self):
        """Get the data from the current window."""
        return self.buffer.current_window
    
    def get_buffer(self):
        """Get all data from the buffer."""
        return self.buffer.buffer
    
    def is_connected(self):
        """Check if the headset is connected and running."""
        return self.headset is not None and self.running
