import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from mindwave_connection import MindwaveConnection
from feature_extraction import extract_features_with_windows, DEFAULT_LOWPASS_CUTOFF
from eeg_buffer import EEGBuffer
import os
import pandas as pd
import traceback  # Add for detailed error tracing
import torch.nn.functional as F
# Model constants - must match training parameters
HIDDEN_SIZES = [128, 64, 32]
DROPOUT_RATE = 0.2
CLASSES = ['up', 'down', 'left', 'right']
NUM_CLASSES = len(CLASSES)

# Window parameters - must match training parameters
WINDOW_SIZE = 100  # Reduced from 100 to 50 samples for faster inference
PREVIEW_SECONDS = 5
BUFFERSIZE = 100
# Filtering settings
DEFAULT_FILTER_WINDOW = 0  # No filtering by default

# Minimum required data points before attempting prediction
MIN_DATA_POINTS_FOR_PREDICTION = WINDOW_SIZE * 2

# Define sampling rate to be slightly higher than cutoff*2 to avoid filter error
# With DEFAULT_LOWPASS_CUTOFF at 50Hz, we need sampling_rate > 100Hz
SAMPLING_RATE = 110  # Increased to avoid digital filter critical frequencies error

class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, dropout_rate=0.2):
        super(MLPModel, self).__init__()
        
        # Create layers
        layers = []
        
        # First hidden layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.BatchNorm1d(hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # Additional hidden layers
        for i in range(len(hidden_sizes)-1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.BatchNorm1d(hidden_sizes[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_sizes[-1], num_classes)
        
    def forward(self, x, apply_softmax=False):
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        
        # Apply softmax for prediction but not during training
        # (CrossEntropyLoss expects raw logits)
        if apply_softmax:
            x = F.softmax(x, dim=1)
        
        return x

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

def load_model_and_scaler(model_dir="models"):
    """Load the trained model and scaler parameters"""
    model_path = os.path.join(model_dir, "mlp_model.pth")
    scaler_path = os.path.join(model_dir, "scaler_params.npy")
    
    # Check if files exist
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Model or scaler file not found in {model_dir}. Make sure to train the model first.")
    
    # Load scaler parameters
    scaler_params = np.load(scaler_path, allow_pickle=True).item()
    
    # Get input size from scaler means
    input_size = len(scaler_params['mean'])
    
    # Create model with same architecture as during training
    model = MLPModel(input_size, HIDDEN_SIZES, NUM_CLASSES, dropout_rate=DROPOUT_RATE)
    
    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set to evaluation mode
    
    print(f"Model loaded successfully - Input size: {input_size}, Output classes: {NUM_CLASSES}")
    print(f"Model structure: {model}")
    
    # Verify scaler parameters
    print(f"Scaler mean shape: {scaler_params['mean'].shape}")
    print(f"Scaler scale shape: {scaler_params['scale'].shape}")
    
    return model, scaler_params

# Add a debug function to test model prediction independently
def test_model_prediction(model, scaler_params, sample_data):
    """Test model prediction with sample data"""
    try:
        print("\nTesting model prediction...")
        print(f"Sample data shape: {sample_data.shape}")
        
        # Standardize features
        standardized_features = (sample_data - scaler_params['mean']) / scaler_params['scale']
        standardized_features = np.nan_to_num(standardized_features, nan=0.0)
        print(f"Standardized features shape: {standardized_features.shape}")
        
        # Check for NaN or infinity values
        if np.isnan(standardized_features).any() or np.isinf(standardized_features).any():
            print("WARNING: NaN or infinity values present after standardization")
        
        # Convert to tensor and run inference
        features_tensor = torch.FloatTensor(standardized_features).unsqueeze(0)
        print(f"Input tensor shape: {features_tensor.shape}")
        
        with torch.no_grad():
            # Update to use apply_softmax=True
            outputs = model(features_tensor, apply_softmax=True)
            print(f"Model output (with softmax): {outputs}")
            
            probabilities = outputs[0]  # Softmax already applied
            print(f"Probabilities: {probabilities}")
            
            confidence, class_idx = torch.max(probabilities, 0)
            prediction_class = CLASSES[class_idx.item()]
            print(f"Prediction: {prediction_class} ({confidence.item()*100:.1f}%)")
            
        return True
    except Exception as e:
        print(f"Error in test prediction: {e}")
        traceback.print_exc()  # Print detailed stack trace
        return False

def main():
    # Load model and scaler
    print("Loading model and scaler...")
    try:
        model, scaler_params = load_model_and_scaler()
        print("Model loaded successfully!")
        
        # Create a small random sample to verify the model works
        feature_size = len(scaler_params['mean'])
        test_features = np.random.rand(feature_size)
        test_model_prediction(model, scaler_params, test_features)
        
    except Exception as e:
        print(f"Error loading model: {e}")
        traceback.print_exc()
        return
    
    # Create device connection
    device_path = "COM4"  # Windows example, change as needed
    # device_path = "/dev/ttyUSB0"  # Linux example
    
    # Create connection
    connection = MindwaveConnection(device_path)
    
    print("Connecting to Mindwave headset...")
    if not connection.connect():
        print("Failed to connect to headset. Please check the device path and try again.")
        return
    
    print("Starting data collection. Press Ctrl+C to stop.")
    connection.start()
    
    # Set up the figure and axes for plotting
    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(6, 1)
    
    # Create plots for EEG, attention/meditation, and prediction
    ax1 = fig.add_subplot(gs[0, 0])  # Raw EEG
    ax2 = fig.add_subplot(gs[1, 0])  # Attention
    ax3 = fig.add_subplot(gs[2, 0])  # Meditation
    ax4 = fig.add_subplot(gs[3:5, 0])  # Prediction bars
    slider_ax = fig.add_subplot(gs[5, 0])  # Add axis for slider
    
    fig.suptitle('Real-time EEG Classification')
    
    # Add filter slider
    filter_slider = Slider(
        ax=slider_ax,
        label='Filter Strength',
        valmin=1,
        valmax=50,
        valinit=DEFAULT_FILTER_WINDOW,
        valstep=1
    )
    slider_ax.set_title("Noise Filtering (Moving Average Window Size)")
    
    # Initialize EEG buffer for collecting data for feature extraction
    eeg_buffer = EEGBuffer(size=BUFFERSIZE)  # Increased buffer size for better data collection
    
    # Initialize empty lists for plotting data
    all_raw_data = []
    all_attention_data = []
    all_meditation_data = []
    time_points = []
    
    # Initialize prediction data
    prediction_confidences = [0] * NUM_CLASSES  # Current confidence for each class
    
    # Track which windows we've already processed
    processed_windows = set()
    
    # Plot lines that will be updated
    raw_line, = ax1.plot([], [], 'b-')
    attention_line, = ax2.plot([], [], 'g-')
    meditation_line, = ax3.plot([], [], 'r-')
    
    # Set up prediction visualization with vertical bars
    bar_width = 0.6
    bar_positions = np.arange(NUM_CLASSES)
    colors = ['#FF6347', '#4CAF50', '#3498DB', '#9B59B6']  # Red, green, blue, purple
    bars = ax4.bar(bar_positions, [0] * NUM_CLASSES, bar_width, color=colors)
    
    # Add labels to the bars
    ax4.set_xticks(bar_positions)
    ax4.set_xticklabels(CLASSES)
    ax4.set_ylim(0, 100)
    
    # Set up axes
    ax1.set_title('Raw EEG Values')
    ax1.set_ylabel('Amplitude')
    ax2.set_title('Attention Values')
    ax2.set_ylabel('Attention')
    ax3.set_title('Meditation Values')
    ax3.set_ylabel('Meditation')
    ax4.set_title('Prediction Confidence')
    ax4.set_xlabel('Class')
    ax4.set_ylabel('Confidence (%)')
    
    # Add text for current prediction and buffer status
    current_prediction_text = ax4.text(0.02, 0.92, '', transform=ax4.transAxes, 
                                      fontsize=14, fontweight='bold')
    buffer_status_text = ax1.text(0.02, 0.92, 'Data collection: 0%', transform=ax1.transAxes,
                                   fontsize=12, color='red')
    
    # Initialize time counter
    start_time = time.time()
    last_prediction_time = 0
    total_collected_samples = 0
    
    def init_plot():
        ax1.set_ylim(-2000, 2000)  # Adjust based on typical raw EEG range
        ax1.set_xlim(0, PREVIEW_SECONDS)
        ax2.set_ylim(0, 100)       # Attention range is 0-100
        ax2.set_xlim(0, PREVIEW_SECONDS)
        ax3.set_ylim(0, 100)       # Meditation range is 0-100
        ax3.set_xlim(0, PREVIEW_SECONDS)
        # ax4 is already set up for bar chart
        return (raw_line, attention_line, meditation_line, 
                *bars, current_prediction_text, buffer_status_text)
    
    def update_plot(frame):
        nonlocal processed_windows, last_prediction_time, total_collected_samples, prediction_confidences
        current_time = time.time() - start_time
        
        # Get all buffer windows and the current window
        buffer = connection.get_buffer()
        current_window = connection.get_current_window()
        
        # Get the current filter window size
        filter_window_size = filter_slider.val
        
        # Process any new complete windows from the buffer
        new_samples_added = False
        for i, window in enumerate(buffer):
            if i not in processed_windows and window:
                # This is a new complete window we haven't processed yet
                window_time = current_time - (len(buffer) - i) * 0.5  # Approximate time offset
                time_step = 0.5 / len(window) if window else 0.01  # Distribute samples within the window
                
                for j, sample in enumerate(window):
                    sample_time = window_time + (j * time_step)
                    time_points.append(sample_time)
                    all_raw_data.append(sample[0])
                    all_attention_data.append(sample[1])
                    all_meditation_data.append(sample[2])
                    
                    # Add to EEG buffer for feature extraction using on_value_callback
                    eeg_buffer.on_value_callback(raw=sample[0], attention=sample[1], meditation=sample[2])
                    total_collected_samples += 1
                    new_samples_added = True
                
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
                    
                    # Add to EEG buffer for feature extraction using on_value_callback
                    eeg_buffer.on_value_callback(raw=sample[0], attention=sample[1], meditation=sample[2])
                    total_collected_samples += 1
                    new_samples_added = True
        
        # Update buffer status text
        buffer_percentage = min(100, (total_collected_samples / MIN_DATA_POINTS_FOR_PREDICTION) * 100)
        buffer_color = 'green' if total_collected_samples >= MIN_DATA_POINTS_FOR_PREDICTION else 'red'
        buffer_status_text.set_text(f'Data collection: {buffer_percentage:.0f}%')
        buffer_status_text.set_color(buffer_color)
        
        # Make a prediction if we have enough data and enough time has passed
        prediction_interval = 0.5  # Make a prediction every 0.5 second
        enough_data = total_collected_samples >= MIN_DATA_POINTS_FOR_PREDICTION
        enough_time = current_time - last_prediction_time >= prediction_interval
        
        if enough_data and enough_time:
            try:
                # Get the current buffer data using get_all_values method
                all_values = eeg_buffer.get_all_values()
                
                # Skip prediction if not enough data in buffer yet
                if len(all_values) < WINDOW_SIZE:
                    # Not enough data yet, just skip prediction
                    print(f"Skipping prediction - need {WINDOW_SIZE} samples, but only have {len(all_values)}")
                else:
                    print(f"\n--- Making prediction at time {current_time:.2f}s ---")
                    print(f"Buffer has {len(all_values)} samples")
                    
                    # Convert the format to match what extract_features_with_windows expects
                    buffer_data = np.array(all_values)
                    print(f"Buffer data shape: {buffer_data.shape}")
                    
                    # Create a pandas DataFrame for feature extraction
                    df_data = pd.DataFrame(buffer_data, columns=['raw', 'attention', 'meditation'])
                    
                    # Create a format compatible with extract_features_with_windows
                    mock_data_with_labels = [(df_data, 0)]  # Label doesn't matter for inference
                    
                    # Extract features using the same function as in training
                    # Use slightly higher sampling rate to avoid filter error
                    with np.errstate(divide='ignore', invalid='ignore'):  # Suppress potential warning messages
                        print(f"Extracting features with sampling_rate={SAMPLING_RATE}Hz (cutoff={DEFAULT_LOWPASS_CUTOFF}Hz)")
                        features, _ = extract_features_with_windows(
                            mock_data_with_labels,
                            window_size=WINDOW_SIZE,
                            overlap=0,
                            sampling_rate=SAMPLING_RATE  # Using our adjusted sampling_rate
                        )
                    
                    print(f"Extracted features count: {len(features)}")
                    
                    if len(features) == 0:
                        print("Warning: No features extracted, skipping prediction")
                    else:
                        print(f"Feature vector shape: {features[0].shape}")
                        
                        # Check feature vector for NaN or inf values before standardization
                        if np.isnan(features[0]).any() or np.isinf(features[0]).any():
                            print("WARNING: Feature vector contains NaN or inf values")
                            
                        # Standardize features using the scaler parameters
                        standardized_features = (features[0] - scaler_params['mean']) / scaler_params['scale']
                        
                        # Verify shapes match
                        if standardized_features.shape != features[0].shape:
                            print(f"WARNING: Shape mismatch after standardization. Original: {features[0].shape}, After: {standardized_features.shape}")
                            
                        # Handle any NaN values that might have occurred
                        standardized_features = np.nan_to_num(standardized_features, nan=0.0)
                        
                        # Convert features to tensor
                        features_tensor = torch.FloatTensor(standardized_features).unsqueeze(0)  # Add batch dimension
                        print(f"Input tensor shape: {features_tensor.shape}")
                        
                        # Get prediction
                        with torch.no_grad():
                            try:
                                # Update to use apply_softmax=True
                                outputs = model(features_tensor, apply_softmax=True)
                                
                                # Probabilities are now directly in outputs (softmax already applied)
                                probabilities = outputs[0]
                                
                                # Print raw model outputs for debugging
                                print(f"Model output (with softmax): {outputs.numpy()}")
                                print(f"Probabilities: {probabilities.numpy()}")
                                
                                # Get predicted class and confidences
                                confidence, class_idx = torch.max(probabilities, 0)
                                prediction_class = CLASSES[class_idx.item()]
                                
                                print(f"Prediction result: {prediction_class} with {confidence.item()*100:.1f}% confidence")
                                
                                # Update prediction confidences for bars visualization
                                prediction_confidences = [prob.item() * 100 for prob in probabilities]
                                
                                # Update current prediction text
                                current_prediction_text.set_text(f"Current: {prediction_class} ({confidence.item()*100:.1f}%)")
                                
                                # Update bar heights for visualization
                                for bar, height in zip(bars, prediction_confidences):
                                    bar.set_height(height)
                                
                                # Update last prediction time
                                last_prediction_time = current_time
                            except Exception as e:
                                print(f"Error during model forward pass: {e}")
                                traceback.print_exc()
                
            except Exception as e:
                print(f"Error during prediction: {e}")
                traceback.print_exc()  # More detailed error information
                # Keep going without crashing the visualization
        
        # Sort data by time to ensure correct plotting order
        if time_points:
            sorted_indices = np.argsort(time_points)
            time_points_sorted = [time_points[i] for i in sorted_indices]
            raw_data_sorted = [all_raw_data[i] for i in sorted_indices]
            attention_data_sorted = [all_attention_data[i] for i in sorted_indices]
            meditation_data_sorted = [all_meditation_data[i] for i in sorted_indices]
            
            # Limit data to last PREVIEW_SECONDS
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
            
            # Apply filter to the raw data
            filtered_raw = apply_filter(display_raw, filter_window_size)
            
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
        
        return (raw_line, attention_line, meditation_line, 
                *bars, current_prediction_text, buffer_status_text)
    
    # Create animation with faster refresh rate
    ani = FuncAnimation(fig, update_plot, init_func=init_plot,
                        interval=50, blit=True, cache_frame_data=False)
    
    # Make sure figure layout is adjusted to accommodate all elements
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    try:
        # Show the plot and keep it updating
        plt.show(block=True)
    except KeyboardInterrupt:
        print("Data collection stopped by user.")
    finally:
        connection.stop()
        print("Disconnected from headset.")
        print("Closing connection...")
        plt.close(fig)

if __name__ == "__main__":
    main()