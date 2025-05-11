#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EEG Data Processing Module for Dream Content Platform

This module provides the core functionality for processing EEG data
collected during sleep, extracting relevant features, and identifying 
dream-related patterns.
"""

import os
import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import entropy, kurtosis, skew
import mne
from abc import ABC, abstractmethod

class EEGProcessor:
    """
    Base processor for EEG data from various sources.
    Handles filtering, artifact removal, and feature extraction.
    """
    
    def __init__(self, sampling_rate=1000, channels=None):
        """
        Initialize the EEG Processor.
        
        Args:
            sampling_rate (int): Sampling rate in Hz
            channels (list): List of channel names
        """
        self.sampling_rate = sampling_rate
        self.channels = channels or ['Fp1', 'Fp2']
        self.filters = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45)
        }
        
    def preprocess(self, raw_data):
        """
        Preprocess the raw EEG data.
        
        Args:
            raw_data (numpy.ndarray): Raw EEG data with shape (channels, samples)
            
        Returns:
            numpy.ndarray: Preprocessed EEG data
        """
        # Apply bandpass filter (1-45 Hz)
        data_filtered = self.apply_bandpass_filter(raw_data, 1, 45)
        
        # Remove artifacts
        data_clean = self.remove_artifacts(data_filtered)
        
        # Normalize data
        data_norm = self.normalize_data(data_clean)
        
        return data_norm
    
    def apply_bandpass_filter(self, data, low_freq, high_freq):
        """
        Apply bandpass filter to the data.
        
        Args:
            data (numpy.ndarray): EEG data
            low_freq (float): Lower cutoff frequency
            high_freq (float): Upper cutoff frequency
            
        Returns:
            numpy.ndarray: Filtered data
        """
        nyquist = 0.5 * self.sampling_rate
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        # Design filter
        b, a = signal.butter(4, [low, high], btype='bandpass')
        
        # Apply filter to each channel
        filtered_data = np.zeros_like(data)
        for i in range(data.shape[0]):
            filtered_data[i] = signal.filtfilt(b, a, data[i])
        
        return filtered_data
    
    def remove_artifacts(self, data, threshold=3.0):
        """
        Remove artifacts from EEG data using statistical thresholding.
        
        Args:
            data (numpy.ndarray): EEG data
            threshold (float): Number of standard deviations for outlier detection
            
        Returns:
            numpy.ndarray: Clean data
        """
        clean_data = data.copy()
        
        for i in range(data.shape[0]):
            channel_data = data[i]
            
            # Calculate statistics
            mean_val = np.mean(channel_data)
            std_val = np.std(channel_data)
            
            # Find outliers
            outlier_mask = np.abs(channel_data - mean_val) > threshold * std_val
            
            # Replace outliers with interpolated values
            outlier_indices = np.where(outlier_mask)[0]
            for idx in outlier_indices:
                # Get nearest non-outlier values
                left_idx = idx - 1
                while left_idx >= 0 and outlier_mask[left_idx]:
                    left_idx -= 1
                
                right_idx = idx + 1
                while right_idx < len(channel_data) and outlier_mask[right_idx]:
                    right_idx += 1
                
                # Interpolate
                if left_idx >= 0 and right_idx < len(channel_data):
                    clean_data[i, idx] = (channel_data[left_idx] + channel_data[right_idx]) / 2
                elif left_idx >= 0:
                    clean_data[i, idx] = channel_data[left_idx]
                elif right_idx < len(channel_data):
                    clean_data[i, idx] = channel_data[right_idx]
                else:
                    clean_data[i, idx] = mean_val
        
        return clean_data
    
    def normalize_data(self, data):
        """
        Normalize the data to zero mean and unit variance.
        
        Args:
            data (numpy.ndarray): EEG data
            
        Returns:
            numpy.ndarray: Normalized data
        """
        norm_data = np.zeros_like(data)
        
        for i in range(data.shape[0]):
            channel_mean = np.mean(data[i])
            channel_std = np.std(data[i])
            
            if channel_std > 0:
                norm_data[i] = (data[i] - channel_mean) / channel_std
            else:
                norm_data[i] = data[i] - channel_mean
        
        return norm_data
    
    def extract_frequency_bands(self, data, window_sec=30, overlap=0.5):
        """
        Extract power in different frequency bands.
        
        Args:
            data (numpy.ndarray): EEG data with shape (channels, samples)
            window_sec (float): Window size in seconds
            overlap (float): Overlap ratio between windows
            
        Returns:
            dict: Power in different frequency bands for each channel
        """
        window_size = int(window_sec * self.sampling_rate)
        hop_size = int(window_size * (1 - overlap))
        
        num_channels = data.shape[0]
        num_windows = max(1, (data.shape[1] - window_size) // hop_size + 1)
        
        # Initialize results
        band_powers = {band: np.zeros((num_channels, num_windows)) 
                       for band in self.filters.keys()}
        
        for ch in range(num_channels):
            for win in range(num_windows):
                start_idx = win * hop_size
                end_idx = start_idx + window_size
                
                if end_idx > data.shape[1]:
                    break
                
                segment = data[ch, start_idx:end_idx]
                
                # Calculate PSD
                freqs, psd = signal.welch(segment, fs=self.sampling_rate, 
                                         nperseg=min(1024, len(segment)))
                
                # Extract band powers
                for band, (low, high) in self.filters.items():
                    freq_mask = (freqs >= low) & (freqs <= high)
                    if np.any(freq_mask):
                        band_powers[band][ch, win] = np.mean(psd[freq_mask])
        
        return band_powers
    
    def detect_sleep_stages(self, data, window_sec=30):
        """
        Detect sleep stages from EEG data.
        
        Args:
            data (numpy.ndarray): EEG data with shape (channels, samples)
            window_sec (float): Window size in seconds
            
        Returns:
            list: Sleep stages for each window ('Wake', 'N1', 'N2', 'N3', 'REM')
        """
        # Extract band powers
        band_powers = self.extract_frequency_bands(data, window_sec=window_sec)
        
        # Calculate features for classification
        num_windows = band_powers['delta'].shape[1]
        stages = []
        
        for win in range(num_windows):
            # Average across channels
            delta = np.mean(band_powers['delta'][:, win])
            theta = np.mean(band_powers['theta'][:, win])
            alpha = np.mean(band_powers['alpha'][:, win])
            beta = np.mean(band_powers['beta'][:, win])
            gamma = np.mean(band_powers['gamma'][:, win])
            
            # Calculate ratios
            theta_delta_ratio = theta / delta if delta > 0 else 0
            alpha_delta_ratio = alpha / delta if delta > 0 else 0
            beta_delta_ratio = beta / delta if delta > 0 else 0
            
            # Apply simple rule-based classification
            # Note: This is a simplified approximation; a real system would use a
            # trained classifier based on labeled sleep stage data
            if alpha_delta_ratio > 1.0 and beta > 0.3 * alpha:
                stage = 'Wake'
            elif theta_delta_ratio > 0.7 and alpha > 0.5 * theta:
                stage = 'N1'
            elif (delta < 0.5 * (theta + alpha)) and (beta < 0.5 * alpha):
                stage = 'N2'
            elif delta > 0.6 * (theta + alpha + beta):
                stage = 'N3'
            elif (theta_delta_ratio > 0.8) and (beta_delta_ratio > 0.4) and (alpha < 0.8 * theta):
                stage = 'REM'
            else:
                stage = 'Uncertain'
            
            stages.append(stage)
        
        return stages
    
    def detect_rem_periods(self, data, window_sec=5, min_duration_sec=60):
        """
        Detect REM sleep periods from EEG data.
        
        Args:
            data (numpy.ndarray): EEG data with shape (channels, samples)
            window_sec (float): Window size for analysis in seconds
            min_duration_sec (float): Minimum duration for a period to be considered REM
            
        Returns:
            list: List of REM periods as (start_sec, end_sec) tuples
        """
        # Get sleep stages with higher resolution
        stages = self.detect_sleep_stages(data, window_sec=window_sec)
        
        # Find consecutive REM windows
        rem_periods = []
        current_start = None
        
        for i, stage in enumerate(stages):
            if stage == 'REM' and current_start is None:
                current_start = i
            elif stage != 'REM' and current_start is not None:
                # End of a REM period
                duration_windows = i - current_start
                duration_sec = duration_windows * window_sec
                
                if duration_sec >= min_duration_sec:
                    rem_periods.append((
                        current_start * window_sec,
                        i * window_sec
                    ))
                
                current_start = None
        
        # Handle case where REM continues until the end
        if current_start is not None:
            duration_windows = len(stages) - current_start
            duration_sec = duration_windows * window_sec
            
            if duration_sec >= min_duration_sec:
                rem_periods.append((
                    current_start * window_sec,
                    len(stages) * window_sec
                ))
        
        return rem_periods
    
    def extract_dream_features(self, data, rem_period=None):
        """
        Extract features relevant to dream content from a REM sleep segment.
        
        Args:
            data (numpy.ndarray): EEG data with shape (channels, samples)
            rem_period (tuple): Start and end time of REM period in seconds
            
        Returns:
            dict: Dream features including emotional valence, complexity, etc.
        """
        if rem_period is not None:
            start_sample = int(rem_period[0] * self.sampling_rate)
            end_sample = min(int(rem_period[1] * self.sampling_rate), data.shape[1])
            segment = data[:, start_sample:end_sample]
        else:
            segment = data
        
        # Preprocess segment
        processed_segment = self.preprocess(segment)
        
        # Extract band powers
        band_powers = self.extract_frequency_bands(processed_segment, window_sec=5, overlap=0.8)
        
        # Average across windows
        avg_powers = {band: np.mean(powers) for band, powers in band_powers.items()}
        
        # Calculate frontal alpha asymmetry as a proxy for emotional valence
        # Positive values suggest positive emotions, negative values suggest negative emotions
        if len(self.channels) >= 2 and 'Fp1' in self.channels and 'Fp2' in self.channels:
            fp1_idx = self.channels.index('Fp1')
            fp2_idx = self.channels.index('Fp2')
            
            left_alpha = band_powers['alpha'][fp1_idx].mean()
            right_alpha = band_powers['alpha'][fp2_idx].mean()
            
            if left_alpha > 0 and right_alpha > 0:
                alpha_asymmetry = (right_alpha - left_alpha) / (right_alpha + left_alpha)
            else:
                alpha_asymmetry = 0
        else:
            alpha_asymmetry = 0
        
        # Calculate theta/beta ratio as a proxy for emotional arousal
        theta_beta_ratio = avg_powers['theta'] / avg_powers['beta'] if avg_powers['beta'] > 0 else 1
        emotional_arousal = 1 / (1 + np.exp(-(theta_beta_ratio - 2.5)))  # Sigmoid transform
        
        # Calculate dream complexity based on signal entropy and frequency band diversity
        signal_entropy = np.mean([entropy(np.abs(np.fft.rfft(processed_segment[ch]))) 
                                 for ch in range(processed_segment.shape[0])])
        
        # Normalize entropy
        max_entropy = np.log(processed_segment.shape[1] // 2 + 1)  # Max possible entropy
        normalized_entropy = signal_entropy / max_entropy if max_entropy > 0 else 0
        
        # Band power variability
        band_variability = np.std(list(avg_powers.values())) / np.mean(list(avg_powers.values())) \
                          if np.mean(list(avg_powers.values())) > 0 else 0
        
        # Combined complexity metric
        complexity = 0.6 * normalized_entropy + 0.4 * band_variability
        complexity = max(0, min(1, complexity))  # Clamp to [0, 1]
        
        # Identify potential thematic elements based on frequency patterns
        thematic_elements = []
        
        if avg_powers['delta'] > 1.5 * avg_powers['theta']:
            thematic_elements.append("deep imagery")
        
        if avg_powers['theta'] > 1.3 * avg_powers['alpha']:
            thematic_elements.append("memory integration")
        
        if avg_powers['alpha'] > 1.2 * avg_powers['beta']:
            thematic_elements.append("visual scenes")
        
        if avg_powers['beta'] > 0.8 * avg_powers['alpha']:
            thematic_elements.append("active scenarios")
        
        if avg_powers['gamma'] > 0.4 * avg_powers['beta']:
            thematic_elements.append("vivid perceptions")
        
        if alpha_asymmetry > 0.2:
            thematic_elements.append("positive interactions")
        
        if alpha_asymmetry < -0.2:
            thematic_elements.append("conflict or negative situations")
        
        if emotional_arousal > 0.7:
            thematic_elements.append("high intensity experiences")
        
        if complexity > 0.8:
            thematic_elements.append("complex narrative")
        
        return {
            "frequency_bands": {band: float(power) for band, power in avg_powers.items()},
            "emotional_tone": {
                "valence": float(alpha_asymmetry),
                "arousal": float(emotional_arousal)
            },
            "complexity": float(complexity),
            "thematic_elements": thematic_elements
        }


class DeviceInterface(ABC):
    """Abstract base class for EEG device interfaces."""
    
    @abstractmethod
    def connect(self):
        """Connect to the device."""
        pass
    
    @abstractmethod
    def start_recording(self):
        """Start recording data."""
        pass
    
    @abstractmethod
    def stop_recording(self):
        """Stop recording data."""
        pass
    
    @abstractmethod
    def get_data(self):
        """Get recorded data."""
        pass


class DreamBandDevice(DeviceInterface):
    """Interface for the DreamBand EEG headband."""
    
    def __init__(self, device_id=None, sampling_rate=1000):
        """
        Initialize DreamBand device interface.
        
        Args:
            device_id (str): Device identifier or None for auto-detection
            sampling_rate (int): Sampling rate in Hz
        """
        self.device_id = device_id
        self.sampling_rate = sampling_rate
        self.connected = False
        self.recording = False
        self.data_buffer = []
    
    def connect(self):
        """Connect to the DreamBand device."""
        # Implementation would use device-specific library
        print(f"[DreamBand] Connecting to device {self.device_id or 'auto-detected'}...")
        
        # Simulate connection to device
        self.connected = True
        print("[DreamBand] Connection successful")
        
        return self.connected
    
    def start_recording(self):
        """Start recording data from DreamBand."""
        if not self.connected:
            raise RuntimeError("Device not connected. Call connect() first.")
        
        print("[DreamBand] Starting recording...")
        self.recording = True
        self.data_buffer = []
        
        # In a real implementation, this would start a background thread
        # to continuously read data from the device
        
        return True
    
    def stop_recording(self):
        """Stop recording data."""
        if not self.recording:
            return False
        
        print("[DreamBand] Stopping recording...")
        self.recording = False
        
        return True
    
    def get_data(self):
        """
        Get recorded data.
        
        Returns:
            numpy.ndarray: EEG data with shape (channels, samples)
        """
        if len(self.data_buffer) == 0:
            return np.array([])
        
        # Convert buffer to numpy array
        return np.array(self.data_buffer)
    
    def save_data(self, filename):
        """
        Save recorded data to a file.
        
        Args:
            filename (str): Output filename
        
        Returns:
            bool: Success status
        """
        if len(self.data_buffer) == 0:
            return False
        
        data = self.get_data()
        
        # Determine file format based on extension
        _, ext = os.path.splitext(filename)
        
        if ext.lower() == '.csv':
            pd.DataFrame(data.T).to_csv(filename, index=False)
        elif ext.lower() == '.npy':
            np.save(filename, data)
        elif ext.lower() in ['.edf', '.bdf']:
            # Create MNE raw object
            info = mne.create_info(
                ch_names=['EEG{}'.format(i+1) for i in range(data.shape[0])],
                sfreq=self.sampling_rate,
                ch_types=['eeg'] * data.shape[0]
            )
            raw = mne.io.RawArray(data, info)
            
            # Save to EDF/BDF
            raw.export(filename)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
        
        print(f"[DreamBand] Data saved to {filename}")
        return True


# Simplified example of usage:
if __name__ == "__main__":
    # Create device interface
    device = DreamBandDevice(sampling_rate=1000)
    
    # Connect to device
    if device.connect():
        try:
            # Start recording
            device.start_recording()
            
            # In a real application, we would wait for recording to complete
            # or implement a background recording thread
            print("Recording... (press Ctrl+C to stop)")
            import time
            time.sleep(5)  # Simulate 5 seconds of recording
            
            # Stop recording
            device.stop_recording()
            
            # Get data
            data = device.get_data()
            
            # Process data
            processor = EEGProcessor(sampling_rate=1000)
            processed_data = processor.preprocess(data)
            
            # Detect REM periods
            rem_periods = processor.detect_rem_periods(processed_data)
            
            # Extract dream features
            if rem_periods:
                dream_features = processor.extract_dream_features(
                    processed_data, rem_periods[0]
                )
                print("\nDream Features:")
                print(f"Emotional valence: {dream_features['emotional_tone']['valence']:.2f}")
                print(f"Emotional arousal: {dream_features['emotional_tone']['arousal']:.2f}")
                print(f"Dream complexity: {dream_features['complexity']:.2f}")
                print(f"Thematic elements: {', '.join(dream_features['thematic_elements'])}")
            else:
                print("No REM periods detected")
            
            # Save data
            device.save_data("eeg_recording.edf")
            
        except KeyboardInterrupt:
            print("\nRecording interrupted")
            device.stop_recording()
