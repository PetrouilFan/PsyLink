import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from mindwave_connection import MindwaveConnection
from feature_extraction import extract_features_with_windows, DEFAULT_LOWPASS_CUTOFF
from filters import smooth_signal
from eeg_buffer import EEGBuffer
import os
import pandas as pd
import traceback
import torch.nn.functional as F
import argparse
import joblib
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model constants - must match training parameters
HIDDEN_SIZES = [128, 64, 32]
DROPOUT_RATE = 0.2
CLASSES = ['up', 'down', 'left', 'right']
NUM_CLASSES = len(CLASSES)

# Window parameters
WINDOW_SIZE = 100  # Reduced from 100 to 50 samples for faster inference
PREVIEW_SECONDS = 5
MAX_HISTORY_SECONDS = PREVIEW_SECONDS + 2
BUFFERSIZE = 100
DEFAULT_FILTER_WINDOW = 0

MIN_DATA_POINTS_FOR_PREDICTION = WINDOW_SIZE * 2
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
        
        if apply_softmax:
            x = F.softmax(x, dim=1)
        
        return x

def load_model_and_scaler(model_dir="models"):
    """Load the trained model and scaler parameters"""
    model_path = os.path.join(model_dir, "mlp_model.pth")
    scaler_path = os.path.join(model_dir, "scaler.joblib")
    
    # Check if files exist
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Model or scaler file not found in {model_dir}. Make sure to train the model first.")
    
    # Load scaler
    scaler = joblib.load(scaler_path)
    
    # Get input size from scaler
    input_size = scaler.n_features_in_
    
    # Create model with same architecture as during training
    model = MLPModel(input_size, HIDDEN_SIZES, NUM_CLASSES, dropout_rate=DROPOUT_RATE)
    
    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set to evaluation mode
    
    logger.info(f"Model loaded successfully - Input size: {input_size}, Output classes: {NUM_CLASSES}")
    return model, scaler

def test_model_prediction(model, scaler, sample_data):
    """Test model prediction with sample data"""
    try:
        logger.info("\nTesting model prediction...")
        
        # Standardize features using standard sklearn API
        standardized_features = scaler.transform(sample_data.reshape(1, -1))
        standardized_features = np.nan_to_num(standardized_features, nan=0.0)
        
        # Convert to tensor and run inference
        features_tensor = torch.FloatTensor(standardized_features)
        
        with torch.no_grad():
            outputs = model(features_tensor, apply_softmax=True)
            probabilities = outputs[0]  # Softmax already applied
            
            confidence, class_idx = torch.max(probabilities, 0)
            prediction_class = CLASSES[class_idx.item()]
            logger.info(f"Prediction: {prediction_class} ({confidence.item()*100:.1f}%)")
            
        return True
    except Exception as e:
        logger.error(f"Error in test prediction: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Mindwave Inference')
    parser.add_argument('--port', type=str, default='COM4', help='Serial port for Mindwave headset')
    parser.add_argument('--model_dir', type=str, default='models', help='Directory containing the trained model and scaler')
    args = parser.parse_args()

    # Load model and scaler
    logger.info("Loading model and scaler...")
    try:
        model, scaler = load_model_and_scaler(args.model_dir)
        logger.info("Model loaded successfully!")
        
        # Create a small random sample to verify the model works
        feature_size = scaler.n_features_in_
        test_features = np.random.rand(feature_size)
        test_model_prediction(model, scaler, test_features)
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return
    
    # Create connection
    connection = MindwaveConnection(args.port)
    
    logger.info("Connecting to Mindwave headset...")
    if not connection.connect():
        logger.error("Failed to connect to headset.")
        return
    
    logger.info("Starting data collection. Close plot window to stop.")
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
    
    eeg_buffer = EEGBuffer(size=BUFFERSIZE)
    
    all_raw_data = []
    all_attention_data = []
    all_meditation_data = []
    time_points = []
    
    prediction_confidences = [0] * NUM_CLASSES
    processed_windows = set()
    
    raw_line, = ax1.plot([], [], 'b-')
    attention_line, = ax2.plot([], [], 'g-')
    meditation_line, = ax3.plot([], [], 'r-')
    
    bar_width = 0.6
    bar_positions = np.arange(NUM_CLASSES)
    colors = ['#FF6347', '#4CAF50', '#3498DB', '#9B59B6']
    bars = ax4.bar(bar_positions, [0] * NUM_CLASSES, bar_width, color=colors)
    
    ax4.set_xticks(bar_positions)
    ax4.set_xticklabels(CLASSES)
    ax4.set_ylim(0, 100)
    
    ax1.set_title('Raw EEG Values')
    ax1.set_ylabel('Amplitude')
    ax2.set_title('Attention Values')
    ax2.set_ylabel('Attention')
    ax3.set_title('Meditation Values')
    ax3.set_ylabel('Meditation')
    ax4.set_title('Prediction Confidence')
    ax4.set_xlabel('Class')
    ax4.set_ylabel('Confidence (%)')
    
    current_prediction_text = ax4.text(0.02, 0.92, '', transform=ax4.transAxes, 
                                      fontsize=14, fontweight='bold')
    buffer_status_text = ax1.text(0.02, 0.92, 'Data collection: 0%', transform=ax1.transAxes,
                                   fontsize=12, color='red')
    
    start_time = time.time()
    last_prediction_time = 0
    total_collected_samples = 0
    
    def init_plot():
        ax1.set_ylim(-2000, 2000)
        ax1.set_xlim(0, PREVIEW_SECONDS)
        ax2.set_ylim(0, 100)
        ax2.set_xlim(0, PREVIEW_SECONDS)
        ax3.set_ylim(0, 100)
        ax3.set_xlim(0, PREVIEW_SECONDS)
        return (raw_line, attention_line, meditation_line, 
                *bars, current_prediction_text, buffer_status_text)
    
    def update_plot(frame):
        nonlocal processed_windows, last_prediction_time, total_collected_samples, prediction_confidences
        nonlocal time_points, all_raw_data, all_attention_data, all_meditation_data
        current_time = time.time() - start_time
        
        buffer = connection.get_buffer()
        current_window = connection.get_current_window()
        
        filter_window_size = filter_slider.val
        
        for i, (window_id, window) in enumerate(buffer):
            if window_id not in processed_windows and window:
                window_time = current_time - (len(buffer) - i) * 0.5
                time_step = 0.5 / len(window) if window else 0.01
                
                for j, sample in enumerate(window):
                    sample_time = window_time + (j * time_step)
                    time_points.append(sample_time)
                    all_raw_data.append(sample[0])
                    all_attention_data.append(sample[1])
                    all_meditation_data.append(sample[2])
                    
                    eeg_buffer.on_value_callback(raw=sample[0], attention=sample[1], meditation=sample[2])
                    total_collected_samples += 1
                
                processed_windows.add(window_id)
                if len(processed_windows) > 100:
                    processed_windows.remove(min(processed_windows))
        
        if current_window and len(current_window) > 0:
            time_step = 0.5 / len(current_window) if current_window else 0.01
            
            for j, sample in enumerate(current_window):
                sample_time = current_time - (1.0 - j * time_step)
                
                if not time_points or abs(time_points[-1] - sample_time) > 0.001:
                    time_points.append(sample_time)
                    all_raw_data.append(sample[0])
                    all_attention_data.append(sample[1])
                    all_meditation_data.append(sample[2])
                    
                    eeg_buffer.on_value_callback(raw=sample[0], attention=sample[1], meditation=sample[2])
                    total_collected_samples += 1
        
        buffer_percentage = min(100, (total_collected_samples / MIN_DATA_POINTS_FOR_PREDICTION) * 100)
        buffer_color = 'green' if total_collected_samples >= MIN_DATA_POINTS_FOR_PREDICTION else 'red'
        buffer_status_text.set_text(f'Data collection: {buffer_percentage:.0f}%')
        buffer_status_text.set_color(buffer_color)
        
        prediction_interval = 0.5
        enough_data = total_collected_samples >= MIN_DATA_POINTS_FOR_PREDICTION
        enough_time = current_time - last_prediction_time >= prediction_interval
        
        if enough_data and enough_time:
            try:
                all_values = eeg_buffer.get_all_values()
                
                if len(all_values) < WINDOW_SIZE:
                    pass
                else:
                    buffer_data = np.array(all_values)
                    df_data = pd.DataFrame(buffer_data, columns=['raw', 'attention', 'meditation'])
                    mock_data_with_labels = [(df_data, 0)]
                    
                    with np.errstate(divide='ignore', invalid='ignore'):
                        features, _ = extract_features_with_windows(
                            mock_data_with_labels,
                            window_size=WINDOW_SIZE,
                            overlap=0,
                            sampling_rate=SAMPLING_RATE
                        )
                    
                    if len(features) > 0:
                        # Standardize features using the scaler parameters via joblib standard
                        standardized_features = scaler.transform(features[0].reshape(1, -1))
                        standardized_features = np.nan_to_num(standardized_features, nan=0.0)
                        
                        features_tensor = torch.FloatTensor(standardized_features)
                        
                        with torch.no_grad():
                            try:
                                outputs = model(features_tensor, apply_softmax=True)
                                probabilities = outputs[0]
                                
                                confidence, class_idx = torch.max(probabilities, 0)
                                prediction_class = CLASSES[class_idx.item()]
                                
                                prediction_confidences = [prob.item() * 100 for prob in probabilities]
                                
                                current_prediction_text.set_text(f"Current: {prediction_class} ({confidence.item()*100:.1f}%)")
                                
                                for bar, height in zip(bars, prediction_confidences):
                                    bar.set_height(height)
                                
                                last_prediction_time = current_time
                            except Exception as e:
                                logger.error(f"Error during model forward pass: {e}")
                
            except Exception as e:
                logger.error(f"Error during prediction: {e}")
        
        if time_points:
            sorted_indices = np.argsort(time_points)
            time_points[:] = [time_points[idx] for idx in sorted_indices]
            all_raw_data[:] = [all_raw_data[idx] for idx in sorted_indices]
            all_attention_data[:] = [all_attention_data[idx] for idx in sorted_indices]
            all_meditation_data[:] = [all_meditation_data[idx] for idx in sorted_indices]
            
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
            
            filter_window_size = int(filter_slider.val)
            if filter_window_size > 1 and len(display_raw) > 0:
                filtered_raw = smooth_signal(np.array(display_raw), filter_window_size)
            else:
                filtered_raw = np.array(display_raw)
            
            raw_line.set_data(display_times, filtered_raw)
            attention_line.set_data(display_times, display_attention)
            meditation_line.set_data(display_times, display_meditation)
            
            if display_times:
                min_time = max(0, current_time - PREVIEW_SECONDS)
                max_time = current_time + 0.1
                ax1.set_xlim(min_time, max_time)
                ax2.set_xlim(min_time, max_time)
                ax3.set_xlim(min_time, max_time)
                
                if len(filtered_raw) > 0:
                    min_raw = min(filtered_raw)
                    max_raw = max(filtered_raw)
                    padding = (max_raw - min_raw) * 0.1 if max_raw > min_raw else 100
                    ax1.set_ylim(min_raw - padding, max_raw + padding)
        
        return (raw_line, attention_line, meditation_line, 
                *bars, current_prediction_text, buffer_status_text)
    
    ani = FuncAnimation(fig, update_plot, init_func=init_plot,
                        interval=50, blit=True, cache_frame_data=False)
    
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    try:
        plt.show(block=True)
    except KeyboardInterrupt:
        logger.info("Data collection stopped by user.")
    finally:
        connection.stop()
        plt.close(fig)

if __name__ == "__main__":
    main()