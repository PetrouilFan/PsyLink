import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt

# Default signal processing parameters
DEFAULT_SAMPLING_RATE = 250
DEFAULT_HIGHPASS_CUTOFF = 1.0
DEFAULT_LOWPASS_CUTOFF = 50.0
DEFAULT_NOTCH_FREQ = 50.0  # For power line interference

def extract_features(eeg_data, sampling_rate=DEFAULT_SAMPLING_RATE):
    """
    Extract features from EEG data.
    
    Args:
        eeg_data: List of EEG records or single record
        sampling_rate: Sampling rate of the data (Hz)
    
    Returns:
        features: Array of extracted features
    """
    all_features = []
    
    # Check if eeg_data is a list or a single record
    if not isinstance(eeg_data, list):
        eeg_data = [eeg_data]
    
    for record in eeg_data:
        # Apply preprocessing
        filtered_data = preprocess_eeg(record, sampling_rate)
        
        # Extract time-domain features
        time_features = extract_time_domain_features(filtered_data)
        
        # Extract frequency-domain features
        freq_features = extract_frequency_domain_features(filtered_data, sampling_rate)
        
        # Extract attention/meditation features (if available)
        att_med_features = extract_attention_meditation_features(filtered_data)
        
        # Extract dynamic/sequence-based features
        dynamic_features = extract_dynamic_features(filtered_data)
        
        # Combine all features
        combined_features = np.concatenate([
            time_features, 
            freq_features, 
            att_med_features, 
            dynamic_features
        ])
        
        all_features.append(combined_features)
    
    return np.array(all_features)

def preprocess_eeg(eeg_record, sampling_rate=DEFAULT_SAMPLING_RATE):
    """
    Apply preprocessing filters to the EEG record.
    
    Args:
        eeg_record: DataFrame containing EEG data
        sampling_rate: Sampling rate of the data (Hz)
    
    Returns:
        filtered_data: Preprocessed EEG data
    """
    # Extract signal columns (assuming non-signal columns are 'time', 'label', etc.)
    signal_columns = [col for col in eeg_record.columns if col not in ['time', 'label', 'timestamp']]
    
    filtered_data = eeg_record.copy()
    
    # Apply filters 
    for col in signal_columns:
        signal_data = eeg_record[col].values
        
        # Apply filters
        signal_data = apply_highpass_filter(signal_data, cutoff=DEFAULT_HIGHPASS_CUTOFF, fs=sampling_rate)
        signal_data = apply_lowpass_filter(signal_data, cutoff=DEFAULT_LOWPASS_CUTOFF, fs=sampling_rate)
        signal_data = apply_notch_filter(signal_data, notch_freq=DEFAULT_NOTCH_FREQ, fs=sampling_rate)
        
        # Normalize
        signal_data = normalize_signal(signal_data)
        
        filtered_data[col] = signal_data
        
    return filtered_data

def apply_highpass_filter(signal_data, cutoff=DEFAULT_HIGHPASS_CUTOFF, fs=DEFAULT_SAMPLING_RATE):
    """Apply high-pass filter to remove low frequency drift."""
    b, a = signal.butter(4, cutoff/(fs/2), 'highpass')
    return signal.filtfilt(b, a, signal_data)

def apply_lowpass_filter(signal_data, cutoff=DEFAULT_LOWPASS_CUTOFF, fs=DEFAULT_SAMPLING_RATE):
    """Apply low-pass filter to remove high frequency noise."""
    b, a = signal.butter(4, cutoff/(fs/2), 'lowpass')
    return signal.filtfilt(b, a, signal_data)

def apply_notch_filter(signal_data, notch_freq=DEFAULT_NOTCH_FREQ, fs=DEFAULT_SAMPLING_RATE):
    """Apply notch filter to remove power line interference."""
    q = 30.0  # Quality factor
    b, a = signal.iirnotch(notch_freq, q, fs)
    return signal.filtfilt(b, a, signal_data)

def normalize_signal(signal_data):
    """Normalize signal to have zero mean and unit variance."""
    mean = np.mean(signal_data)
    std = np.std(signal_data)
    if std > 0:
        return (signal_data - mean) / std
    else:
        return signal_data - mean

def extract_time_domain_features(filtered_data):
    """
    Extract time-domain features from filtered EEG data.
    
    Features:
    - Mean
    - Standard Deviation
    - Range (maximum - minimum)
    - Peak-to-Peak Amplitude
    - Root Mean Square (RMS)
    - Zero-Crossing Rate (ZCR)
    - Slope Sign Changes (SSC)
    """
    # Select only signal columns
    signal_columns = [col for col in filtered_data.columns if col not in ['time', 'label', 'timestamp']]
    features = []
    
    for col in signal_columns:
        signal_data = filtered_data[col].values
        
        # Basic statistical features
        mean_val = np.mean(signal_data)
        std_val = np.std(signal_data)
        max_val = np.max(signal_data)
        min_val = np.min(signal_data)
        range_val = max_val - min_val
        
        # Peak-to-Peak Amplitude
        # First, find peaks
        peaks, _ = signal.find_peaks(signal_data)
        troughs, _ = signal.find_peaks(-signal_data)
        
        if len(peaks) > 0 and len(troughs) > 0:
            peak_values = signal_data[peaks]
            trough_values = signal_data[troughs]
            peak_to_peak = np.max(peak_values) - np.min(trough_values)
        else:
            peak_to_peak = range_val
        
        # Root Mean Square (RMS)
        rms_val = np.sqrt(np.mean(np.square(signal_data)))
        
        # Zero-Crossing Rate
        zero_crossings = np.sum(np.diff(np.signbit(signal_data))) / len(signal_data)
        
        # Slope Sign Changes (SSC)
        # Detect changes in the slope sign
        diff_signal = np.diff(signal_data)
        slope_changes = np.diff(np.signbit(diff_signal))
        ssc = np.sum(slope_changes != 0) / len(signal_data)
        
        # Append all time-domain features
        features.extend([
            mean_val, std_val, max_val, min_val, range_val,
            peak_to_peak, rms_val, zero_crossings, ssc
        ])
    
    return np.array(features)

def extract_frequency_domain_features(filtered_data, sampling_rate=DEFAULT_SAMPLING_RATE):
    """
    Extract frequency-domain features from filtered EEG data.
    
    Features:
    - Band Power (Delta, Theta, Alpha, Beta, Gamma)
    - Dominant Frequency
    - Spectral Entropy
    - Relative Power Ratios
    """
    # Select only signal columns
    signal_columns = [col for col in filtered_data.columns if col not in ['time', 'label', 'timestamp']]
    features = []
    
    # Define frequency bands (Hz)
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 12),
        'beta': (12, 30),
        'gamma': (30, 50)
    }
    
    for col in signal_columns:
        signal_data = filtered_data[col].values
        
        # Calculate PSD using Welch's method
        f, psd = signal.welch(signal_data, sampling_rate, nperseg=min(256, len(signal_data)))
        
        # Extract band powers
        band_powers = {}
        total_power = np.sum(psd)
        
        for band, (low, high) in bands.items():
            # Find the indices that correspond to the frequency band
            idx_band = np.logical_and(f >= low, f <= high)
            # Calculate the power in this frequency band
            band_power = np.sum(psd[idx_band])
            band_powers[band] = band_power
            
            # Also add normalized/relative band power
            if total_power > 0:
                band_powers[f"{band}_rel"] = band_power / total_power
            else:
                band_powers[f"{band}_rel"] = 0
        
        # Calculate dominant frequency (frequency with maximum power)
        if len(psd) > 0:
            dominant_freq_idx = np.argmax(psd)
            dominant_freq = f[dominant_freq_idx]
        else:
            dominant_freq = 0
        
        # Calculate spectral entropy
        norm_psd = psd / np.sum(psd) if np.sum(psd) > 0 else psd
        spectral_entropy = -np.sum(norm_psd * np.log2(norm_psd + 1e-10))
        
        # Calculate relative power ratios
        delta_theta_ratio = band_powers['delta'] / band_powers['theta'] if band_powers['theta'] > 0 else 0
        beta_alpha_ratio = band_powers['beta'] / band_powers['alpha'] if band_powers['alpha'] > 0 else 0
        
        # Add all frequency-domain features
        features.extend([
            *band_powers.values(),
            dominant_freq,
            spectral_entropy,
            delta_theta_ratio,
            beta_alpha_ratio
        ])
    
    return np.array(features)

def extract_attention_meditation_features(filtered_data):
    """
    Extract attention and meditation features if available.
    
    Features:
    - Average Attention Value
    - Average Meditation Value
    - Attention Trend (if multiple data points available)
    - Meditation Trend (if multiple data points available)
    """
    features = []
    
    # Check if attention and meditation columns exist
    if 'attention' in filtered_data.columns:
        att_values = filtered_data['attention'].values
        att_mean = np.mean(att_values)
        att_std = np.std(att_values)
        
        # Calculate trend (slope of linear fit)
        if len(att_values) > 1:
            att_trend = np.polyfit(np.arange(len(att_values)), att_values, 1)[0]
        else:
            att_trend = 0
            
        features.extend([att_mean, att_std, att_trend])
    else:
        # Add zeros if attention data not available
        features.extend([0, 0, 0])
    
    if 'meditation' in filtered_data.columns:
        med_values = filtered_data['meditation'].values
        med_mean = np.mean(med_values)
        med_std = np.std(med_values)
        
        # Calculate trend (slope of linear fit)
        if len(med_values) > 1:
            med_trend = np.polyfit(np.arange(len(med_values)), med_values, 1)[0]
        else:
            med_trend = 0
            
        features.extend([med_mean, med_std, med_trend])
    else:
        # Add zeros if meditation data not available
        features.extend([0, 0, 0])
    
    return np.array(features)

def extract_dynamic_features(filtered_data):
    """
    Extract dynamic/sequence-based features.
    
    Features:
    - Inter-window differences (if multiple data points)
    - Trends over windows
    - Coefficient of variation over time
    """
    # Select only signal columns
    signal_columns = [col for col in filtered_data.columns if col not in ['time', 'label', 'timestamp']]
    features = []
    
    for col in signal_columns:
        signal_data = filtered_data[col].values
        
        # We can only calculate dynamic features if we have enough data points
        if len(signal_data) > 10:  # Arbitrary threshold
            # Divide the signal into segments
            num_segments = min(4, len(signal_data) // 5)  # At most 4 segments, each with at least 5 points
            segments = np.array_split(signal_data, num_segments)
            
            # Calculate features for each segment
            segment_means = [np.mean(seg) for seg in segments]
            segment_stds = [np.std(seg) for seg in segments]
            
            # Calculate inter-segment differences
            mean_diffs = np.diff(segment_means)
            std_diffs = np.diff(segment_stds)
            
            # Calculate trend across segments
            if len(segment_means) > 1:
                mean_trend = np.polyfit(np.arange(len(segment_means)), segment_means, 1)[0]
            else:
                mean_trend = 0
                
            # Coefficient of variation over time
            cv = np.std(segment_means) / np.mean(segment_means) if np.mean(segment_means) != 0 else 0
            
            # Add all dynamic features
            features.extend([
                *mean_diffs,
                *std_diffs,
                mean_trend,
                cv
            ])
            
            # Padding for consistent feature vector size
            pad_size = 10 - len(mean_diffs) - len(std_diffs) - 2  # -2 for mean_trend and cv
            if pad_size > 0:
                features.extend([0] * pad_size)
        else:
            # Not enough data points for dynamic analysis, add zeros
            features.extend([0] * 10)  # Placeholder values
    
    return np.array(features)

def visualize_features(features, labels):
    """
    Visualize extracted features.
    
    Args:
        features: Array of features
        labels: Array of labels
    """
    # PCA for dimensionality reduction
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=2)
    features_reduced = pca.fit_transform(features)
    
    # Plot
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    for label in unique_labels:
        mask = labels == label
        plt.scatter(features_reduced[mask, 0], features_reduced[mask, 1], label=f"Class {label}")
    
    plt.title("PCA of Extracted Features")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.show()

def get_feature_names():
    """
    Return the list of feature names.
    
    Returns:
        list: Names of all features extracted
    """
    # This is just an example - update with actual feature names
    # Time-domain features per channel
    time_features = ["Mean", "STD", "Max", "Min", "Range", "Peak-to-Peak", "RMS", "ZCR", "SSC"]
    
    # Frequency-domain features per channel
    freq_band_features = ["Delta", "Theta", "Alpha", "Beta", "Gamma", 
                          "Delta_rel", "Theta_rel", "Alpha_rel", "Beta_rel", "Gamma_rel"]
    other_freq_features = ["Dominant Frequency", "Spectral Entropy", "Delta/Theta", "Beta/Alpha"]
    freq_features = freq_band_features + other_freq_features
    
    # Attention and meditation features
    att_med_features = ["Attention Mean", "Attention STD", "Attention Trend",
                        "Meditation Mean", "Meditation STD", "Meditation Trend"]
    
    # Dynamic features (simplified for readability)
    dynamic_features = ["Dynamic Features (various)"]
    
    return time_features + freq_features + att_med_features + dynamic_features
