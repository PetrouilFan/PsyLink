import mindwave
import time
import numpy as np

BUFFER_SIZE = 4 # how many windows to keep in the buffer

class EEGBuffer:
    '''
    Buffer will have BUFFER_SIZE windows
    Each window will have all the raw values as well as the attention and meditation values
    The window will change every time change_window is called
    The number of samples (raw, attention, meditation) in each window is automatically determined every time a new window is created
    '''
    def __init__(self):
        self.size = BUFFER_SIZE
        self.current_window = []
        self.buffer = []
        self.last_value = 0
        self.last_attention = 0
        self.last_meditation = 0

    def on_value_callback(self, raw=None, attention=None, meditation=None):
        '''
        Add the raw, attention, and meditation values to the current window
        '''
        if raw is not None:
            self.last_value = raw
        if attention is not None:
            self.change_window()
            self.last_attention = attention
        if meditation is not None:
            self.last_meditation = meditation
        self.current_window.append([self.last_value, self.last_attention, self.last_meditation])

    def change_window(self):
        '''
        Change the current window and add it to the buffer
        '''
        self.buffer.append(self.current_window)
        self.current_window = []
        if len(self.buffer) > self.size:
            self.buffer.pop(0)

    
