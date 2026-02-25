import mindwave
import time
import numpy as np
import threading

class EEGBuffer:
    '''
    Buffer will have ATTENTION_COUNT windows
    Each window will have all the raw values as well as the attention and meditation values
    The window will change every time change_window is called
    The number of samples (raw, attention, meditation) in each window is automatically determined every time a new window is created
    '''
    def __init__(self, size=4):
        self.size = size
        self.current_window = []
        self.buffer = []
        self.last_value = 0
        self.last_attention = 0
        self.last_meditation = 0
        self.window_counter = 0
        self.lock = threading.Lock()
    
    def is_ready(self):
        with self.lock:
            return len(self.buffer) >= self.size
    
    def get_current_values(self):
        with self.lock:
            return self.last_value, self.last_attention, self.last_meditation
    
    def get_all_values(self):
        with self.lock:
            return [item for _, window in self.buffer for item in window] + self.current_window
            
    def get_buffer_copy(self):
        with self.lock:
            # return shallow copy to prevent iteration crashes in other threads
            return list(self.buffer)
            
    def get_current_window_copy(self):
        with self.lock:
            return list(self.current_window)
    
    def on_value_callback(self, raw=None, attention=None, meditation=None):
        with self.lock:
            if raw is not None:
                self.last_value = raw
            if attention is not None:
                self._change_window_internal()
                self.last_attention = attention
            if meditation is not None:
                self.last_meditation = meditation
            self.current_window.append([self.last_value, self.last_attention, self.last_meditation])

    def change_window(self):
        with self.lock:
            self._change_window_internal()
            
    def _change_window_internal(self):
        self.buffer.append((self.window_counter, self.current_window))
        self.window_counter += 1
        self.current_window = []
        if len(self.buffer) > self.size:
            self.buffer.pop(0)
