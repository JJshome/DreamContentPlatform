"""
Sleep Data Collection and Processing Module

This module provides functionality for collecting, processing, and analyzing
sleep data, particularly EEG signals during REM sleep for dream content
extraction.
"""

from .eeg_processor import EEGProcessor, DeviceInterface, DreamBandDevice

__all__ = ['EEGProcessor', 'DeviceInterface', 'DreamBandDevice']
