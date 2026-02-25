import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
import argparse
import logging
from mindwave_connection import MindwaveConnection
from filters import smooth_signal

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PREVIEW_SECONDS = 5
MAX_HISTORY_SECONDS = PREVIEW_SECONDS + 2 # Keep a small buffer to avoid memory leaks

def main():
    parser = argparse.ArgumentParser(description='Mindwave Real-time Demo')
    parser.add_argument('--port', type=str, default='COM4', help='Serial port for Mindwave headset (e.g. COM4, /dev/ttyUSB0)')
    args = parser.parse_args()
    
    device_path = args.port
    
    # Create connection
    connection = MindwaveConnection(device_path)
    
    logger.info("Connecting to Mindwave headset...")
    if not connection.connect():
        logger.error("Failed to connect to headset. Please check the device path and try again.")
        return
    
    logger.info("Starting data collection. Close plot window to stop.")
    connection.start()
    
    # Set up the figure and axes for plotting with space for slider
    fig = plt.figure(figsize=(10, 9))
    gs = fig.add_gridspec(4, 1)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0])
    slider_ax = fig.add_subplot(gs[3, 0])  # Add axis for slider
    
    fig.suptitle('Real-time EEG Data')
    
    # Add filter slider
    filter_slider = Slider(
        ax=slider_ax,
        label='Filter Strength',
        valmin=1,
        valmax=50,
        valinit=1,
        valstep=1
    )
    slider_ax.set_title("Noise Filtering (Moving Average Window Size)")
    
    # Initialize empty lists for data
    all_raw_data = []
    all_attention_data = []
    all_meditation_data = []
    time_points = []
    
    # Track which windows we've already processed
    processed_windows = set()
    
    # Plot lines that will be updated
    raw_line, = ax1.plot([], [], 'b-')
    attention_line, = ax2.plot([], [], 'g-')
    meditation_line, = ax3.plot([], [], 'r-')
    
    # Store running state to gracefully exit
    is_running = True
    
    def on_close(event):
        nonlocal is_running
        is_running = False
        logger.info("Plot window closed.")
        
    fig.canvas.mpl_connect('close_event', on_close)
    
    # Set up axes
    ax1.set_title('Raw EEG Values')
    ax1.set_ylabel('Amplitude')
    ax2.set_title('Attention Values')
    ax2.set_ylabel('Attention')
    ax3.set_title('Meditation Values')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Meditation')
    
    # Initialize time counter
    start_time = time.time()
    
    def init_plot():
        ax1.set_ylim(-2000, 2000)  # Adjust based on your typical raw EEG range
        ax1.set_xlim(0, PREVIEW_SECONDS)
        ax2.set_ylim(0, 100)       # Attention range is 0-100
        ax2.set_xlim(0, PREVIEW_SECONDS)
        ax3.set_ylim(0, 100)       # Meditation range is 0-100
        ax3.set_xlim(0, PREVIEW_SECONDS)
        return raw_line, attention_line, meditation_line
    
    def update_plot(frame):
        nonlocal processed_windows, time_points, all_raw_data, all_attention_data, all_meditation_data
        current_time = time.time() - start_time
        
        # Get all buffer windows and the current window
        buffer = connection.get_buffer()
        current_window = connection.get_current_window()
        
        # Process any new complete windows from the buffer
        # buffer items are now tuples: (window_id, window)
        for i, (window_id, window) in enumerate(buffer):
            if window_id not in processed_windows:
                # Calculate True Time Offset:
                # 512 raw bytes per second vs ~1 attention byte per second
                # Time distributed over entire duration
                window_time = current_time - (len(buffer) - i) * 1.0  
                time_step = 1.0 / len(window) if window else 0.01  
                
                for j, sample in enumerate(window):
                    sample_time = window_time + (j * time_step)
                    time_points.append(sample_time)
                    all_raw_data.append(sample[0])
                    all_attention_data.append(sample[1])
                    all_meditation_data.append(sample[2])
                
                processed_windows.add(window_id)
                # Keep processed_windows small to prevent memory leak
                if len(processed_windows) > 100:
                    processed_windows.remove(min(processed_windows))
        
        # Process current window (in progress)
        if current_window and len(current_window) > 0:
            # Calculate time for each sample in current window
            time_step = 1.0 / len(current_window) if current_window else 0.01
            
            for j, sample in enumerate(current_window):
                # Place current window samples right before current time
                sample_time = current_time - (1.0 - j * time_step)
                
                # Only add if we don't already have data at this time
                if not time_points or abs(time_points[-1] - sample_time) > 0.001:
                    time_points.append(sample_time)
                    all_raw_data.append(sample[0])
                    all_attention_data.append(sample[1])
                    all_meditation_data.append(sample[2])
        
        # Sort data by time to ensure correct plotting order
        if time_points:
            sorted_indices = np.argsort(time_points)
            time_points[:] = [time_points[idx] for idx in sorted_indices]
            all_raw_data[:] = [all_raw_data[idx] for idx in sorted_indices]
            all_attention_data[:] = [all_attention_data[idx] for idx in sorted_indices]
            all_meditation_data[:] = [all_meditation_data[idx] for idx in sorted_indices]
            
            # Prevent memory leak by truncating lists
            truncate_time = current_time - MAX_HISTORY_SECONDS
            trunc_index = 0
            for i, t in enumerate(time_points):
                if t >= truncate_time:
                    trunc_index = i
                    break
                    
            if trunc_index > 0:
                time_points[:] = time_points[trunc_index:]
                all_raw_data[:] = all_raw_data[trunc_index:]
                all_attention_data[:] = all_attention_data[trunc_index:]
                all_meditation_data[:] = all_meditation_data[trunc_index:]

            # Limit data to last PREVIEW_SECONDS seconds for display
            cutoff_time = current_time - PREVIEW_SECONDS
            cutoff_index = 0
            for i, t in enumerate(time_points):
                if t >= cutoff_time:
                    cutoff_index = i
                    break
            
            display_times = time_points[cutoff_index:]
            display_raw = all_raw_data[cutoff_index:]
            display_attention = all_attention_data[cutoff_index:]
            display_meditation = all_meditation_data[cutoff_index:]
            
            # Apply optimized filter to the raw data using filters.py
            filter_window_size = int(filter_slider.val)
            if filter_window_size > 1 and len(display_raw) > 0:
                filtered_raw = smooth_signal(np.array(display_raw), filter_window_size)
            else:
                filtered_raw = np.array(display_raw)
            
            # Update plot data
            raw_line.set_data(display_times, filtered_raw)
            attention_line.set_data(display_times, display_attention)
            meditation_line.set_data(display_times, display_meditation)
            
            # Adjust x-axis limits to show latest data
            if display_times:
                min_time = max(0, current_time - PREVIEW_SECONDS)
                max_time = current_time + 0.1  # Small padding
                ax1.set_xlim(min_time, max_time)
                ax2.set_xlim(min_time, max_time)
                ax3.set_xlim(min_time, max_time)
                
                # Adjust y-axis limits for raw data if needed
                if len(filtered_raw) > 0:
                    min_raw = min(filtered_raw)
                    max_raw = max(filtered_raw)
                    padding = (max_raw - min_raw) * 0.1 if max_raw > min_raw else 100
                    ax1.set_ylim(min_raw - padding, max_raw + padding)
        
        return raw_line, attention_line, meditation_line
    
    # Create animation with faster refresh rate
    ani = FuncAnimation(fig, update_plot, init_func=init_plot,
                        interval=50, blit=True, cache_frame_data=False)
    
    # Make sure figure layout is adjusted to accommodate all elements
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    try:
        # Instead of blocking forever with plt.show(block=True),
        # use a loop that respects the window close event
        plt.show(block=False)
        while is_running and plt.fignum_exists(fig.number):
            plt.pause(0.1)
    except KeyboardInterrupt:
        logger.info("Data collection stopped by user.")
    finally:
        connection.stop()
        logger.info("Disconnected from headset.")
        plt.close(fig)

if __name__ == "__main__":
    main()
