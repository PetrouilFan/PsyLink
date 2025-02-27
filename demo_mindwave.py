import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from mindwave_connection import MindwaveConnection

PREVIEW_SECONDS = 5

def apply_filter(data, window_size):
    """Apply a simple moving average filter to the data"""
    if window_size <= 1:
        return data
    
    window_size = int(window_size)
    filtered_data = np.copy(data)
    
    # Apply moving average filter
    for i in range(len(data)):
        # Determine window boundaries
        left = max(0, i - window_size // 2)
        right = min(len(data), i + window_size // 2 + 1)
        # Calculate average of values within window
        if right > left:
            filtered_data[i] = np.mean(data[left:right])
    
    return filtered_data

def main():
    # Replace with your actual device path
    device_path = "COM4"  # Windows example
    # device_path = "/dev/ttyUSB0"  # Linux example
    
    # Create connection with 2-second windows
    connection = MindwaveConnection(device_path)
    
    print("Connecting to Mindwave headset...")
    if not connection.connect():
        print("Failed to connect to headset. Please check the device path and try again.")
        return
    
    print("Starting data collection. Press Ctrl+C to stop.")
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
        nonlocal processed_windows
        current_time = time.time() - start_time
        
        # Get all buffer windows and the current window
        buffer = connection.get_buffer()
        current_window = connection.get_current_window()
        
        # Process any new complete windows from the buffer
        for i, window in enumerate(buffer):
            if i not in processed_windows:
                # This is a new complete window we haven't processed yet
                window_time = current_time - (len(buffer) - i) * 0.5  # Approximate time offset (adjust as needed)
                time_step = 0.5 / len(window) if window else 0.01  # Distribute samples within the window
                
                for j, sample in enumerate(window):
                    sample_time = window_time + (j * time_step)
                    time_points.append(sample_time)
                    all_raw_data.append(sample[0])
                    all_attention_data.append(sample[1])
                    all_meditation_data.append(sample[2])
                
                processed_windows.add(i)
        
        # Process current window (in progress)
        if current_window and len(current_window) > 0:
            # Calculate time for each sample in current window
            time_step = 0.5 / len(current_window) if current_window else 0.01
            
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
            time_points_sorted = [time_points[i] for i in sorted_indices]
            raw_data_sorted = [all_raw_data[i] for i in sorted_indices]
            attention_data_sorted = [all_attention_data[i] for i in sorted_indices]
            meditation_data_sorted = [all_meditation_data[i] for i in sorted_indices]
            
            # Limit data to last PREVIEW_SECONDS seconds
            cutoff_time = current_time - PREVIEW_SECONDS
            cutoff_index = 0
            for i, t in enumerate(time_points_sorted):
                if t >= cutoff_time:
                    cutoff_index = i
                    break
            
            display_times = time_points_sorted[cutoff_index:]
            display_raw = raw_data_sorted[cutoff_index:]
            display_attention = attention_data_sorted[cutoff_index:]
            display_meditation = meditation_data_sorted[cutoff_index:]
            
            # Apply filter to the raw data based on slider value
            filter_window_size = filter_slider.val
            filtered_raw = apply_filter(display_raw, filter_window_size)
            
            # Update plot data - use filtered data for raw values
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
                if len(filtered_raw) > 0:  # Fix: check length instead of truthiness
                    min_raw = min(filtered_raw)
                    max_raw = max(filtered_raw)
                    padding = (max_raw - min_raw) * 0.1 if max_raw > min_raw else 100
                    ax1.set_ylim(min_raw - padding, max_raw + padding)
        
        return raw_line, attention_line, meditation_line
    
    # Create animation with faster refresh rate - fix the warning by adding cache_frame_data=False
    ani = FuncAnimation(fig, update_plot, init_func=init_plot,
                        interval=50, blit=True, cache_frame_data=False)  # 50ms refresh for smoother plotting
    
    # Make sure figure layout is adjusted to accommodate all elements
    fig.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust to make room for title and slider
    
    try:
        # Show the plot and keep it updating
        plt.show(block=True)  # Using block=True for better performance
            
    except KeyboardInterrupt:
        print("Data collection stopped by user.")
    finally:
        connection.stop()
        print("Disconnected from headset.")
        print("Closing connection...")
        plt.close(fig)

if __name__ == "__main__":
    main()
