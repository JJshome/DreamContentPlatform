#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dream Analysis Module

This module provides AI-powered analysis of sleep and dream data.
It extracts dream features and patterns that are then used for content generation.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

@dataclass
class DreamFeatures:
    """Data class for dream features extracted from EEG data."""
    
    frequency_bands: Dict[str, float]
    emotional_tone: Dict[str, float]
    complexity: float
    thematic_elements: List[str]
    dream_intensity: float = 0.0
    narrative_structure: Optional[Dict[str, Any]] = None
    visual_patterns: Optional[Dict[str, Any]] = None
    auditory_patterns: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "frequency_bands": self.frequency_bands,
            "emotional_tone": self.emotional_tone,
            "complexity": self.complexity,
            "thematic_elements": self.thematic_elements,
            "dream_intensity": self.dream_intensity,
            "narrative_structure": self.narrative_structure,
            "visual_patterns": self.visual_patterns,
            "auditory_patterns": self.auditory_patterns
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DreamFeatures':
        """Create from dictionary."""
        return cls(
            frequency_bands=data.get("frequency_bands", {}),
            emotional_tone=data.get("emotional_tone", {}),
            complexity=data.get("complexity", 0.0),
            thematic_elements=data.get("thematic_elements", []),
            dream_intensity=data.get("dream_intensity", 0.0),
            narrative_structure=data.get("narrative_structure"),
            visual_patterns=data.get("visual_patterns"),
            auditory_patterns=data.get("auditory_patterns")
        )
    
    def save(self, filepath: str) -> None:
        """Save features to a JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'DreamFeatures':
        """Load features from a JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)


class CNNLSTMModel(nn.Module):
    """
    CNN-LSTM hybrid model for dream analysis.
    Processes EEG data to extract relevant features.
    """
    
    def __init__(
        self,
        input_channels: int = 2,
        sequence_length: int = 1000,
        num_filters: int = 64,
        lstm_hidden_size: int = 128,
        num_lstm_layers: int = 2,
        output_features: int = 64
    ):
        """
        Initialize the CNN-LSTM model.
        
        Args:
            input_channels: Number of EEG channels
            sequence_length: Length of input sequence
            num_filters: Number of CNN filters
            lstm_hidden_size: LSTM hidden layer size
            num_lstm_layers: Number of LSTM layers
            output_features: Size of output feature vector
        """
        super(CNNLSTMModel, self).__init__()
        
        # CNN layers
        self.conv1 = nn.Conv1d(input_channels, num_filters, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(num_filters, num_filters*2, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(num_filters*2)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        # Calculate size after convolutions and pooling
        cnn_output_size = sequence_length // 4  # After two pooling layers with kernel size 2
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=num_filters*2,
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # Fully connected output layer
        self.fc = nn.Linear(lstm_hidden_size*2, output_features)  # *2 for bidirectional
        
    def forward(self, x):
        """Forward pass through the network."""
        # x shape: (batch_size, channels, sequence_length)
        
        # CNN feature extraction
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)
        
        # Reshape for LSTM: (batch_size, sequence_length, features)
        x = x.permute(0, 2, 1)
        
        # LSTM sequence processing
        x, _ = self.lstm(x)
        
        # Use the last time step output
        x = x[:, -1, :]
        
        # Final feature vector
        x = self.fc(x)
        
        return x


class DreamAnalyzer:
    """
    Main class for dream content analysis using AI.
    Extracts meaningful features from sleep data for content generation.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize the Dream Analyzer.
        
        Args:
            model_path: Path to pre-trained model weights
            device: Computation device ('cuda' or 'cpu')
        """
        self.device = device
        
        # Initialize the CNN-LSTM model
        self.model = CNNLSTMModel().to(device)
        
        # Load pre-trained weights if provided
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            self.model.eval()
        
        # Thematic mapping from embedding space to dream themes
        self.theme_categories = [
            "water", "flying", "falling", "chasing", "being_chased",
            "school", "work", "family", "friends", "romantic",
            "adventure", "discovery", "conflict", "resolution", "transformation",
            "nature", "urban", "indoor", "surreal", "nostalgic",
            "future", "past", "journey", "challenge", "achievement",
            "mystery", "revelation", "loss", "reunion", "celebration"
        ]
        
    def preprocess_eeg_data(self, eeg_data: np.ndarray) -> torch.Tensor:
        """
        Preprocess EEG data for the model.
        
        Args:
            eeg_data: EEG data with shape (channels, samples)
            
        Returns:
            torch.Tensor: Processed data ready for the model
        """
        # Convert to tensor
        tensor_data = torch.tensor(eeg_data, dtype=torch.float32)
        
        # Add batch dimension if not present
        if tensor_data.dim() == 2:
            tensor_data = tensor_data.unsqueeze(0)
        
        return tensor_data.to(self.device)
    
    def analyze_dream(self, eeg_data: np.ndarray) -> DreamFeatures:
        """
        Analyze dream data to extract features.
        
        Args:
            eeg_data: EEG data with shape (channels, samples)
            
        Returns:
            DreamFeatures: Extracted dream features
        """
        # Preprocess data
        tensor_data = self.preprocess_eeg_data(eeg_data)
        
        # Extract features using the model
        with torch.no_grad():
            feature_vector = self.model(tensor_data)
        
        # Convert to numpy for further processing
        features = feature_vector.cpu().numpy()[0]
        
        # Extract frequency band information (simplified simulation)
        # In a real implementation, this would come from a more sophisticated analysis
        frequency_bands = {
            "delta": float(np.abs(features[0])),
            "theta": float(np.abs(features[1])),
            "alpha": float(np.abs(features[2])),
            "beta": float(np.abs(features[3])),
            "gamma": float(np.abs(features[4]))
        }
        
        # Normalize band powers
        total_power = sum(frequency_bands.values())
        if total_power > 0:
            frequency_bands = {k: v/total_power for k, v in frequency_bands.items()}
        
        # Extract emotional tone
        # In a real implementation, this would use a more sophisticated model
        valence = float(np.tanh(features[5]))  # -1 to 1
        arousal = float(np.clip(sigmoid(features[6]), 0, 1))  # 0 to 1
        
        emotional_tone = {
            "valence": valence,
            "arousal": arousal
        }
        
        # Extract complexity
        complexity = float(np.clip(sigmoid(features[7]), 0, 1))
        
        # Extract thematic elements using cosine similarity
        theme_embeddings = features[8:38]  # Use part of feature vector for themes
        theme_scores = theme_embeddings / (np.linalg.norm(theme_embeddings) + 1e-8)
        
        # Select top themes
        top_indices = np.argsort(theme_scores)[-5:]  # Top 5 themes
        thematic_elements = [self.theme_categories[i] for i in top_indices if theme_scores[i] > 0.2]
        
        # Calculate dream intensity
        dream_intensity = float(np.clip(sigmoid(features[38]), 0, 1))
        
        # Extract narrative structure (simplified)
        narrative_structure = {
            "linearity": float(np.clip(sigmoid(features[40]), 0, 1)),
            "coherence": float(np.clip(sigmoid(features[41]), 0, 1)),
            "character_presence": float(np.clip(sigmoid(features[42]), 0, 1))
        }
        
        # Extract visual patterns (simplified)
        visual_patterns = {
            "color_intensity": float(np.clip(sigmoid(features[43]), 0, 1)),
            "spatial_complexity": float(np.clip(sigmoid(features[44]), 0, 1)),
            "movement_level": float(np.clip(sigmoid(features[45]), 0, 1))
        }
        
        # Extract auditory patterns (simplified)
        auditory_patterns = {
            "presence": float(np.clip(sigmoid(features[46]), 0, 1)),
            "rhythm": float(np.clip(sigmoid(features[47]), 0, 1)),
            "harmony": float(np.clip(sigmoid(features[48]), 0, 1))
        }
        
        # Create DreamFeatures object
        dream_features = DreamFeatures(
            frequency_bands=frequency_bands,
            emotional_tone=emotional_tone,
            complexity=complexity,
            thematic_elements=thematic_elements,
            dream_intensity=dream_intensity,
            narrative_structure=narrative_structure,
            visual_patterns=visual_patterns,
            auditory_patterns=auditory_patterns
        )
        
        return dream_features
    
    def analyze_dream_batch(self, eeg_data_batch: List[np.ndarray]) -> List[DreamFeatures]:
        """
        Analyze multiple dream data samples in batch.
        
        Args:
            eeg_data_batch: List of EEG data arrays
            
        Returns:
            List[DreamFeatures]: List of extracted dream features
        """
        results = []
        for eeg_data in eeg_data_batch:
            dream_features = self.analyze_dream(eeg_data)
            results.append(dream_features)
        return results
    
    def extract_rem_segments(
        self,
        eeg_data: np.ndarray,
        sampling_rate: int = 1000,
        min_duration_sec: int = 60
    ) -> List[Tuple[int, int]]:
        """
        Extract REM sleep segments from EEG data.
        
        Args:
            eeg_data: EEG data with shape (channels, samples)
            sampling_rate: Sampling rate in Hz
            min_duration_sec: Minimum duration for a valid REM segment
            
        Returns:
            List[Tuple[int, int]]: List of (start_idx, end_idx) tuples
        """
        # Preprocess data
        tensor_data = self.preprocess_eeg_data(eeg_data)
        
        # Use the model to predict REM probability for sliding windows
        window_size = sampling_rate * 30  # 30-second window
        hop_size = sampling_rate * 5      # 5-second hop
        
        rem_probs = []
        
        for i in range(0, tensor_data.shape[2] - window_size, hop_size):
            window = tensor_data[:, :, i:i+window_size]
            with torch.no_grad():
                feature_vector = self.model(window)
            
            # Assume the 50th feature is REM probability
            rem_prob = sigmoid(feature_vector.cpu().numpy()[0, 49])
            rem_probs.append(rem_prob)
        
        # Find continuous segments of high REM probability
        rem_segments = []
        in_rem = False
        start_idx = 0
        
        threshold = 0.7  # REM probability threshold
        
        for i, prob in enumerate(rem_probs):
            if prob > threshold and not in_rem:
                # Start of REM segment
                in_rem = True
                start_idx = i * hop_size
            elif prob <= threshold and in_rem:
                # End of REM segment
                in_rem = False
                end_idx = i * hop_size
                
                # Check if segment is long enough
                if (end_idx - start_idx) >= (min_duration_sec * sampling_rate):
                    rem_segments.append((start_idx, end_idx))
        
        # Handle case where REM continues until the end
        if in_rem:
            end_idx = len(rem_probs) * hop_size
            if (end_idx - start_idx) >= (min_duration_sec * sampling_rate):
                rem_segments.append((start_idx, end_idx))
        
        return rem_segments


def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))


if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt
    
    # Create a sample EEG data (2 channels, 30 seconds at 1000Hz)
    sample_rate = 1000
    duration_sec = 30
    num_samples = sample_rate * duration_sec
    
    # Generate synthetic EEG data
    t = np.arange(num_samples) / sample_rate
    
    # Channel 1: Mix of alpha (10Hz) and theta (6Hz) waves
    ch1 = 0.5 * np.sin(2 * np.pi * 10 * t) + 0.8 * np.sin(2 * np.pi * 6 * t)
    
    # Channel 2: Mix of alpha (8Hz) and beta (20Hz) waves
    ch2 = 0.4 * np.sin(2 * np.pi * 8 * t) + 0.3 * np.sin(2 * np.pi * 20 * t)
    
    # Add noise
    ch1 += 0.2 * np.random.randn(num_samples)
    ch2 += 0.2 * np.random.randn(num_samples)
    
    # Combine channels
    eeg_data = np.vstack([ch1, ch2])
    
    # Initialize Dream Analyzer
    analyzer = DreamAnalyzer()
    
    # Analyze dream
    dream_features = analyzer.analyze_dream(eeg_data)
    
    # Print results
    print("Dream Analysis Results:")
    print(f"Frequency Bands: {dream_features.frequency_bands}")
    print(f"Emotional Tone: {dream_features.emotional_tone}")
    print(f"Complexity: {dream_features.complexity}")
    print(f"Thematic Elements: {dream_features.thematic_elements}")
    
    # Visualize EEG data
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t[:1000], ch1[:1000])
    plt.title('Channel 1 (First 1 second)')
    plt.ylabel('Amplitude (µV)')
    
    plt.subplot(2, 1, 2)
    plt.plot(t[:1000], ch2[:1000])
    plt.title('Channel 2 (First 1 second)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (µV)')
    
    plt.tight_layout()
    plt.savefig('eeg_sample.png')
    plt.close()
    
    # Save dream features
    dream_features.save('dream_features.json')
    print("Dream features saved to dream_features.json")
