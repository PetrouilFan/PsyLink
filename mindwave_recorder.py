import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
import tkinter as tk
from tkinter import filedialog
import csv
import os
import argparse
import logging
from datetime import datetime
from mindwave_connection import MindwaveConnection

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PREVIEW_SECONDS = 5
MAX_HISTORY_SECONDS = PREVIEW_SECONDS + 2

class MindwaveRecorder:
    def __init__(self, device_path):
        self.device_path = device_path
        
        # Data storage
        self.all_raw_data = []
        self.all_attention_data = []
        self.all_meditation_data = []
        self.time_points = []
        self.processed_windows = set()
        
        # Recording state
        self.recording = False
        self.recording_file = None
        self.csv_writer = None
        self.csv_file = None
        self.start_time = None
        
        # Initialize the connection
        self.connection = MindwaveConnection(self.device_path)
        
        # Create the GUI
        self.setup_gui()
    
    def setup_gui(self):
        # Set up the figure and axes for plotting with space for buttons
        self.fig = plt.figure(figsize=(10, 10))
        gs = self.fig.add_gridspec(5, 1)
        self.ax1 = self.fig.add_subplot(gs[0, 0])
        self.ax2 = self.fig.add_subplot(gs[1, 0])
        self.ax3 = self.fig.add_subplot(gs[2, 0])
        
        # Add buttons
        record_ax = self.fig.add_subplot(gs[3, 0])
        stop_ax = self.fig.add_subplot(gs[4, 0])
        
        self.record_button = Button(record_ax, 'Record', color='limegreen')
        self.record_button.on_clicked(self.start_recording)
        
        self.stop_button = Button(stop_ax, 'Stop Recording', color='tomato')
        self.stop_button.on_clicked(self.stop_recording)
        
        self.fig.suptitle('Mindwave Recorder')
        
        # Initialize plot lines
        self.raw_line, = self.ax1.plot([], [], 'b-')
        self.attention_line, = self.ax2.plot([], [], 'g-')
        self.meditation_line, = self.ax3.plot([], [], 'r-')
        
        # Set up axes
        self.ax1.set_title('Raw EEG Values')
        self.ax1.set_ylabel('Amplitude')
        self.ax2.set_title('Attention Values')
        self.ax2.set_ylabel('Attention')
        self.ax3.set_title('Meditation Values')
        self.ax3.set_xlabel('Time (s)')
        self.ax3.set_ylabel('Meditation')
        
        # Setup initial plot limits
        self.ax1.set_ylim(-2000, 2000)
        self.ax1.set_xlim(0, PREVIEW_SECONDS)
        self.ax2.set_ylim(0, 100)
        self.ax2.set_xlim(0, PREVIEW_SECONDS)
        self.ax3.set_ylim(0, 100)
        self.ax3.set_xlim(0, PREVIEW_SECONDS)
        
        # Adjust layout
        self.fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    def connect(self):
        """Connect to the Mindwave headset"""
        logger.info("Connecting to Mindwave headset...")
        if not self.connection.connect():
            logger.error("Failed to connect to headset. Please check the device path and try again.")
            return False
        logger.info("Connected to headset.")
        self.connection.start()
        self.start_time = time.time()
        return True
    
    def select_file(self):
        """Open a dialog to select where to save the recording"""
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        
        # Get current date/time for default filename
        now = datetime.now()
        default_filename = f"mindwave_recording_{now.strftime('%Y%m%d_%H%M%S')}.csv"
        
        # Open file dialog
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=default_filename
        )
        
        root.destroy()
        return file_path
    
    def start_recording(self, event):
        """Start recording data to a CSV file"""
        if self.recording:
            logger.info("Already recording!")
            return
        
        # Select file to save recording
        file_path = self.select_file()
        if not file_path:
            logger.info("Recording cancelled")
            return
        
        self.recording_file = file_path
        self.recording = True
        
        # Create CSV file and writer
        self.csv_file = open(self.recording_file, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        
        # Write header
        self.csv_writer.writerow(['timestamp', 'raw', 'attention', 'meditation'])
        
        logger.info(f"Recording started. Saving to {self.recording_file}")
        
        # Add recording indicator to the plot title
        self.fig.suptitle(f'Mindwave Recorder - RECORDING to {os.path.basename(self.recording_file)}')
    
    def stop_recording(self, event):
        """Stop recording data"""
        if not self.recording:
            logger.info("Not currently recording!")
            return
        
        self.recording = False
        
        # Close the CSV file
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None
        
        logger.info(f"Recording stopped. Data saved to {self.recording_file}")
        
        # Update the plot title
        self.fig.suptitle('Mindwave Recorder - NOT RECORDING')
    
    def update_plot(self, frame):
        """Update the plot and record data if recording is active"""
        current_time = time.time() - self.start_time
        
        # Get all buffer windows and the current window
        buffer = self.connection.get_buffer()
        current_window = self.connection.get_current_window()
        
        # Process any new complete windows from the buffer
        for i, (window_id, window) in enumerate(buffer):
            if window_id not in self.processed_windows:
                # This is a new complete window we haven't processed yet
                window_time = current_time - (len(buffer) - i) * 0.5  # Approximate time offset
                time_step = 0.5 / len(window) if window else 0.01  # Distribute samples within the window
                
                for j, sample in enumerate(window):
                    sample_time = window_time + (j * time_step)
                    raw_value, attention_value, meditation_value = sample
                    
                    # Record data if we're recording
                    if self.recording and self.csv_writer:
                        self.csv_writer.writerow([sample_time, raw_value, attention_value, meditation_value])
                    
                    self.time_points.append(sample_time)
                    self.all_raw_data.append(raw_value)
                    self.all_attention_data.append(attention_value)
                    self.all_meditation_data.append(meditation_value)
                
                self.processed_windows.add(window_id)
                # Keep set small
                if len(self.processed_windows) > 100:
                    self.processed_windows.remove(min(self.processed_windows))
        
        # Process current window (in progress)
        if current_window and len(current_window) > 0:
            # Calculate time for each sample in current window
            time_step = 0.5 / len(current_window) if current_window else 0.01
            
            for j, sample in enumerate(current_window):
                # Place current window samples right before current time
                sample_time = current_time - (1.0 - j * time_step)
                raw_value, attention_value, meditation_value = sample
                
                # Record data if we're recording
                if self.recording and self.csv_writer:
                    self.csv_writer.writerow([sample_time, raw_value, attention_value, meditation_value])
                
                # Only add if we don't already have data at this time
                if not self.time_points or abs(self.time_points[-1] - sample_time) > 0.001:
                    self.time_points.append(sample_time)
                    self.all_raw_data.append(raw_value)
                    self.all_attention_data.append(attention_value)
                    self.all_meditation_data.append(meditation_value)
        
        # Sort data by time to ensure correct plotting order
        if self.time_points:
            sorted_indices = np.argsort(self.time_points)
            self.time_points[:] = [self.time_points[i] for i in sorted_indices]
            self.all_raw_data[:] = [self.all_raw_data[i] for i in sorted_indices]
            self.all_attention_data[:] = [self.all_attention_data[i] for i in sorted_indices]
            self.all_meditation_data[:] = [self.all_meditation_data[i] for i in sorted_indices]
            
            # Prevent memory leak by truncating lists
            truncate_time = current_time - MAX_HISTORY_SECONDS
            trunc_index = 0
            for i, t in enumerate(self.time_points):
                if t >= truncate_time:
                    trunc_index = i
                    break
                    
            if trunc_index > 0:
                self.time_points[:] = self.time_points[trunc_index:]
                self.all_raw_data[:] = self.all_raw_data[trunc_index:]
                self.all_attention_data[:] = self.all_attention_data[trunc_index:]
                self.all_meditation_data[:] = self.all_meditation_data[trunc_index:]

            # Limit data to last PREVIEW_SECONDS seconds for plotting
            cutoff_time = current_time - PREVIEW_SECONDS
            cutoff_index = 0
            for i, t in enumerate(self.time_points):
                if t >= cutoff_time:
                    cutoff_index = i
                    break
            
            display_times = self.time_points[cutoff_index:]
            display_raw = self.all_raw_data[cutoff_index:]
            display_attention = self.all_attention_data[cutoff_index:]
            display_meditation = self.all_meditation_data[cutoff_index:]
            
            # Update plot data
            self.raw_line.set_data(display_times, display_raw)
            self.attention_line.set_data(display_times, display_attention)
            self.meditation_line.set_data(display_times, display_meditation)
            
            # Adjust x-axis limits to show latest data
            if display_times:
                min_time = max(0, current_time - PREVIEW_SECONDS)
                max_time = current_time + 0.1  # Small padding
                self.ax1.set_xlim(min_time, max_time)
                self.ax2.set_xlim(min_time, max_time)
                self.ax3.set_xlim(min_time, max_time)
                
                # Adjust y-axis limits for raw data if needed
                if len(display_raw) > 0:
                    min_raw = min(display_raw)
                    max_raw = max(display_raw)
                    padding = (max_raw - min_raw) * 0.1 if max_raw > min_raw else 100
                    self.ax1.set_ylim(min_raw - padding, max_raw + padding)
        
        return self.raw_line, self.attention_line, self.meditation_line
    
    def run(self):
        """Run the recorder application"""
        if not self.connect():
            return
        
        # Set initial title
        self.fig.suptitle('Mindwave Recorder - NOT RECORDING')
        
        # Create animation
        self.ani = FuncAnimation(
            self.fig, self.update_plot,
            interval=50, blit=True, cache_frame_data=False
        )
        
        try:
            plt.show()
        except KeyboardInterrupt:
            logger.info("Recorder stopped by user.")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        if self.recording:
            self.recording = False
            if self.csv_file:
                self.csv_file.close()
        
        self.connection.stop()
        logger.info("Disconnected from headset.")

def main():
    parser = argparse.ArgumentParser(description='Mindwave Recorder')
    parser.add_argument('--port', type=str, default='COM4', help='Serial port for Mindwave headset')
    args = parser.parse_args()

    recorder = MindwaveRecorder(device_path=args.port)
    recorder.run()

if __name__ == "__main__":
    main()
