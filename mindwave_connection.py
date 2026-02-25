import time
import threading
import mindwave
import logging
from eeg_buffer import EEGBuffer

logger = logging.getLogger(__name__)

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
            
            logger.info(f"Connected to Mindwave headset at {self.device_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to headset: {e}")
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
        
        logger.info("Started processing Mindwave data")
        return True
    
    def stop(self):
        """Stop processing data."""
        self.running = False
        
        if self.headset:
            self.headset.stop()
            logger.info("Disconnected from Mindwave headset")
    
    def get_current_window(self):
        """Get the data from the current window safely."""
        return self.buffer.get_current_window_copy()
    
    def get_buffer(self):
        """Get all data from the buffer safely."""
        return self.buffer.get_buffer_copy()
    
    def is_connected(self):
        """Check if the headset is connected and running."""
        return self.headset is not None and self.running
