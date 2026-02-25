import numpy as np
from scipy import signal

def apply_highpass_filter(data, cutoff, fs, order=5):
    """
    Apply high-pass filter to the data.
    
    Args:
        data: 1D array, the input signal
        cutoff: High-pass cutoff frequency in Hz
        fs: Sampling frequency in Hz
        order: Filter order
    
    Returns:
        filtered_data: Filtered signal
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data

def apply_lowpass_filter(data, cutoff, fs, order=5):
    """
    Apply low-pass filter to the data.
    
    Args:
        data: 1D array, the input signal
        cutoff: Low-pass cutoff frequency in Hz
        fs: Sampling frequency in Hz
        order: Filter order
    
    Returns:
        filtered_data: Filtered signal
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data

def apply_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Apply band-pass filter to the data.
    
    Args:
        data: 1D array, the input signal
        lowcut: Low cutoff frequency in Hz
        highcut: High cutoff frequency in Hz
        fs: Sampling frequency in Hz
        order: Filter order
    
    Returns:
        filtered_data: Filtered signal
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data

def apply_notch_filter(data, notch_freq, fs, quality_factor=30):
    """
    Apply notch filter to remove a specific frequency component (e.g., 50/60 Hz power line interference).
    
    Args:
        data: 1D array, the input signal
        notch_freq: Frequency to remove in Hz
        fs: Sampling frequency in Hz
        quality_factor: Quality factor of the notch filter
    
    Returns:
        filtered_data: Filtered signal
    """
    nyq = 0.5 * fs
    freq = notch_freq / nyq
    b, a = signal.iirnotch(freq, quality_factor)
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data

def normalize_signal(data, method='minmax'):
    """
    Normalize signal data.
    
    Args:
        data: 1D array, the input signal
        method: Normalization method ('minmax', 'zscore', or 'robust')
    
    Returns:
        normalized_data: Normalized signal
    """
    if method == 'minmax':
        # Min-Max normalization to [0, 1]
        min_val = np.min(data)
        max_val = np.max(data)
        if max_val == min_val:
            return np.zeros_like(data)
        normalized_data = (data - min_val) / (max_val - min_val)
        
    elif method == 'zscore':
        # Z-score normalization
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return np.zeros_like(data)
        normalized_data = (data - mean) / std
        
    elif method == 'robust':
        # Use median and Interquartile Range (IQR) for robustness against outliers
        median = np.median(data)
        q75, q25 = np.percentile(data, [75, 25])
        iqr = q75 - q25
        if iqr == 0:
            return data - median
        return (data - median) / iqr
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized_data

def smooth_signal(data, window_size=5):
    """
    Apply smoothing to the signal using moving average.
    
    Args:
        data: 1D array, the input signal
        window_size: Size of the moving average window
    
    Returns:
        smoothed_data: Smoothed signal
    """
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

def compute_band_power(data, fs, band):
    """
    Compute power in a specific frequency band.
    
    Args:
        data: 1D array, the input signal
        fs: Sampling frequency in Hz
        band: Tuple with (low_freq, high_freq) defining the band
    
    Returns:
        band_power: Power in the specified frequency band
    """
    low, high = band
    
    # Compute the PSD
    freqs, psd = signal.welch(data, fs, nperseg=min(256, len(data)))
    
    # Find indices of frequencies in the band
    idx = np.logical_and(freqs >= low, freqs <= high)
    
    # Calculate power
    band_power = np.sum(psd[idx])
    
    return band_power
