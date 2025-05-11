#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dream Content Platform Simulator

This script simulates the process of collecting sleep data, analyzing it,
and generating content based on dream patterns.
"""

import os
import sys
import time
import json
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Ensure the script can be run from any directory
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, "..", ".."))

# Function definitions for different simulation components

def simulate_eeg_signal(duration_seconds=300, sampling_rate=256, rem_intensity=0.7):
    """
    Simulate an EEG signal with REM-like characteristics.
    
    Args:
        duration_seconds (int): Duration of the signal in seconds
        sampling_rate (int): Sampling rate in Hz
        rem_intensity (float): Intensity of REM patterns (0.0 to 1.0)
        
    Returns:
        numpy.ndarray: Simulated EEG signal
    """
    # Number of samples
    n_samples = duration_seconds * sampling_rate
    
    # Time array
    t = np.arange(n_samples) / sampling_rate
    
    # Base signal (pink noise)
    base_signal = np.random.normal(0, 1, n_samples)
    
    # Add frequency components
    # Alpha waves (8-13 Hz) - reduced during REM
    alpha = (0.3 - 0.2 * rem_intensity) * np.sin(2 * np.pi * 10 * t)
    
    # Theta waves (4-8 Hz) - prominent during REM
    theta = 0.7 * rem_intensity * np.sin(2 * np.pi * 6 * t)
    
    # Beta waves (13-30 Hz) - present during REM
    beta = 0.4 * rem_intensity * np.sin(2 * np.pi * 20 * t)
    
    # Delta waves (1-4 Hz) - reduced during REM
    delta = (0.5 - 0.4 * rem_intensity) * np.sin(2 * np.pi * 2 * t)
    
    # Combine all components
    signal = base_signal + alpha + theta + beta + delta
    
    # Add random eye movement artifacts during REM
    if rem_intensity > 0.5:
        for _ in range(int(duration_seconds * rem_intensity / 10)):
            # Random eye movement artifact
            artifact_pos = random.randint(0, n_samples - int(sampling_rate/2))
            artifact_len = int(sampling_rate * random.uniform(0.1, 0.5))
            artifact_shape = np.hanning(artifact_len)
            artifact_amplitude = random.uniform(2, 5) * rem_intensity
            signal[artifact_pos:artifact_pos+artifact_len] += artifact_amplitude * artifact_shape
    
    return signal

def extract_dream_features(eeg_signal, sampling_rate=256):
    """
    Extract dream-related features from EEG signal.
    
    Args:
        eeg_signal (numpy.ndarray): EEG signal
        sampling_rate (int): Sampling rate in Hz
        
    Returns:
        dict: Extracted features
    """
    # Simulate feature extraction
    
    # Calculate frequency bands
    from scipy import signal
    
    # Calculate power in different frequency bands
    freqs, psd = signal.welch(eeg_signal, fs=sampling_rate, nperseg=sampling_rate)
    
    # Extract band powers
    delta_idx = np.logical_and(freqs >= 1, freqs <= 4)
    theta_idx = np.logical_and(freqs >= 4, freqs <= 8)
    alpha_idx = np.logical_and(freqs >= 8, freqs <= 13)
    beta_idx = np.logical_and(freqs >= 13, freqs <= 30)
    gamma_idx = np.logical_and(freqs >= 30, freqs <= 45)
    
    delta_power = np.mean(psd[delta_idx])
    theta_power = np.mean(psd[theta_idx])
    alpha_power = np.mean(psd[alpha_idx])
    beta_power = np.mean(psd[beta_idx])
    gamma_power = np.mean(psd[gamma_idx])
    
    # Calculate REM probability based on frequency band ratios
    # High theta and beta with low delta is characteristic of REM
    theta_delta_ratio = theta_power / delta_power if delta_power > 0 else 1
    rem_probability = min(1.0, theta_delta_ratio / 2)
    
    # Emotional tone based on frontal asymmetry (simulated)
    emotional_valence = random.uniform(-1.0, 1.0)  # -1 (negative) to 1 (positive)
    emotional_arousal = random.uniform(0.0, 1.0)   # 0 (calm) to 1 (excited)
    
    # Dream complexity based on signal entropy (simulated)
    from scipy.stats import entropy
    
    # Calculate histogram for entropy approximation
    hist, _ = np.histogram(eeg_signal, bins=50)
    complexity = entropy(hist + 1e-10)  # Add small constant to avoid log(0)
    normalized_complexity = min(1.0, complexity / 5.0)  # Normalize to 0-1 scale
    
    # Thematic elements based on signal patterns (simulated)
    # In a real implementation, this would use machine learning models
    themes = []
    
    if theta_power > 0.7 * np.mean(psd):
        themes.append("water")
    if beta_power > 0.6 * np.mean(psd):
        themes.append("movement")
    if gamma_power > 0.5 * np.mean(psd):
        themes.append("vivid colors")
    if emotional_valence > 0.7:
        themes.append("positive interaction")
    if emotional_valence < -0.7:
        themes.append("conflict")
    if emotional_arousal > 0.8:
        themes.append("adventure")
    if normalized_complexity > 0.8:
        themes.append("complex narrative")
    if len(themes) < 2:
        # Ensure at least two themes
        possible_themes = ["nature", "urban", "people", "flying", "falling", "searching"]
        additional_themes = random.sample(possible_themes, 2 - len(themes))
        themes.extend(additional_themes)
    
    return {
        "frequency_bands": {
            "delta": float(delta_power),
            "theta": float(theta_power),
            "alpha": float(alpha_power),
            "beta": float(beta_power),
            "gamma": float(gamma_power)
        },
        "rem_probability": float(rem_probability),
        "emotional_tone": {
            "valence": float(emotional_valence),
            "arousal": float(emotional_arousal)
        },
        "complexity": float(normalized_complexity),
        "thematic_elements": themes
    }

def generate_dream_image(dream_features, output_path=None):
    """
    Generate an abstract image representation of dream features.
    This is a simplified visualization rather than actual AI-generated art.
    
    Args:
        dream_features (dict): Dream features
        output_path (str): Path to save the image (optional)
        
    Returns:
        str: Path to saved image
    """
    # Create a new figure
    plt.figure(figsize=(10, 10))
    
    # Set background color based on emotional valence
    valence = dream_features["emotional_tone"]["valence"]
    arousal = dream_features["emotional_tone"]["arousal"]
    complexity = dream_features["complexity"]
    
    # Background color based on valence (red to blue)
    bg_color = (
        max(0, -valence/2 + 0.5),  # Red channel
        0.5,                        # Green channel
        max(0, valence/2 + 0.5)     # Blue channel
    )
    
    # Create background
    plt.fill_between([-1, 1], [-1, -1], [1, 1], color=bg_color, alpha=0.5)
    
    # Generate shapes based on frequency bands
    bands = dream_features["frequency_bands"]
    theta_power = bands["theta"]
    beta_power = bands["beta"]
    gamma_power = bands["gamma"]
    
    # Number of elements based on complexity
    n_elements = int(complexity * 50) + 10
    
    # Generate random shapes
    for _ in range(n_elements):
        shape_type = random.choice(["circle", "line", "arc"])
        size = random.uniform(0.05, 0.2)
        x = random.uniform(-0.9, 0.9)
        y = random.uniform(-0.9, 0.9)
        
        # Color based on emotional tone
        r = max(0, min(1, 0.5 + valence/2 + random.uniform(-0.2, 0.2)))
        g = max(0, min(1, 0.5 + arousal/2 + random.uniform(-0.2, 0.2)))
        b = max(0, min(1, 0.7 + (complexity-0.5)/2 + random.uniform(-0.2, 0.2)))
        color = (r, g, b, random.uniform(0.3, 0.9))  # RGBA
        
        if shape_type == "circle":
            circle = plt.Circle((x, y), size, color=color)
            plt.gca().add_patch(circle)
        elif shape_type == "line":
            x2 = x + random.uniform(-0.4, 0.4)
            y2 = y + random.uniform(-0.4, 0.4)
            plt.plot([x, x2], [y, y2], color=color, 
                     linewidth=size*20, alpha=color[3])
        elif shape_type == "arc":
            theta1 = random.uniform(0, 360)
            theta2 = theta1 + random.uniform(30, 180)
            arc = plt.matplotlib.patches.Arc((x, y), size*2, size*2, 
                                           theta1=theta1, theta2=theta2,
                                           color=color, linewidth=size*10, alpha=color[3])
            plt.gca().add_patch(arc)
    
    # Add thematic elements as text
    themes = dream_features["thematic_elements"]
    for i, theme in enumerate(themes):
        x = random.uniform(-0.8, 0.8)
        y = random.uniform(-0.8, 0.8)
        rotation = random.uniform(-45, 45)
        alpha = random.uniform(0.2, 0.5)
        plt.text(x, y, theme, rotation=rotation, alpha=alpha,
                fontsize=12 + random.uniform(0, 10),
                ha='center', va='center', color='white')
    
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.axis('off')
    plt.tight_layout()
    
    # Save image if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return output_path
    else:
        plt.show()
        plt.close()
        return None

def generate_dream_text(dream_features):
    """
    Generate a simple narrative text based on dream features.
    This is a template-based approach, not actual AI generation.
    
    Args:
        dream_features (dict): Dream features
        
    Returns:
        str: Generated narrative
    """
    # Extract features
    valence = dream_features["emotional_tone"]["valence"]
    arousal = dream_features["emotional_tone"]["arousal"]
    complexity = dream_features["complexity"]
    themes = dream_features["thematic_elements"]
    
    # Setting templates
    settings = [
        "a vast {adj} landscape",
        "a strange {adj} city",
        "an abandoned {adj} building",
        "a mysterious {adj} forest",
        "a {adj} beach at twilight",
        "a {adj} mountain peak",
        "a {adj} underground chamber",
        "{adj} clouds floating in the sky",
        "a {adj} room that seemed to shift and change",
        "a {adj} path that led into darkness"
    ]
    
    # Action templates
    actions = [
        "I was walking through",
        "I found myself in",
        "I was floating above",
        "I discovered",
        "I was running through",
        "I was observing",
        "I entered",
        "I was searching through",
        "I was flying over",
        "I was surrounded by"
    ]
    
    # Event templates
    events = [
        "Suddenly, {event_desc}.",
        "Without warning, {event_desc}.",
        "As I continued, {event_desc}.",
        "Then {event_desc}.",
        "In the distance, I noticed {event_desc}.",
        "I realized that {event_desc}.",
        "To my surprise, {event_desc}.",
        "The scene shifted, and {event_desc}.",
        "A voice called out, and {event_desc}.",
        "The atmosphere changed as {event_desc}."
    ]
    
    # Ending templates
    endings = [
        "I woke up feeling {end_emotion}.",
        "The dream faded as I {end_action}.",
        "Everything dissolved into {end_desc}.",
        "I tried to continue but {end_action}.",
        "The last thing I remember is {end_desc}.",
        "I found myself {end_action} as the dream ended.",
        "The scene transformed into {end_desc} before I awoke.",
        "A sense of {end_emotion} washed over me as I woke up.",
        "I was left with a lingering feeling of {end_emotion}.",
        "As I returned to consciousness, I felt {end_emotion}."
    ]
    
    # Adjectives based on emotional tone
    positive_adj = ["beautiful", "vibrant", "peaceful", "enchanted", "luminous", "serene"]
    negative_adj = ["dark", "ominous", "unsettling", "strange", "foreboding", "twisted"]
    high_arousal_adj = ["intense", "chaotic", "wild", "dramatic", "electric", "vivid"]
    low_arousal_adj = ["quiet", "still", "muted", "gentle", "hazy", "soft"]
    
    # Select adjectives based on emotional tone
    adjectives = []
    if valence > 0.3:
        adjectives.extend(positive_adj)
    elif valence < -0.3:
        adjectives.extend(negative_adj)
        
    if arousal > 0.6:
        adjectives.extend(high_arousal_adj)
    elif arousal < 0.4:
        adjectives.extend(low_arousal_adj)
        
    if not adjectives:
        adjectives = ["mysterious", "unusual", "dream-like", "curious"]
    
    # Event descriptions based on themes
    event_descriptions = []
    for theme in themes:
        if theme == "water":
            event_descriptions.append("water began to rise around me")
            event_descriptions.append("I saw a vast ocean stretching out")
        elif theme == "movement":
            event_descriptions.append("everything started to move rapidly")
            event_descriptions.append("I felt myself being carried forward")
        elif theme == "vivid colors":
            event_descriptions.append("the world exploded into brilliant colors")
            event_descriptions.append("patterns of light danced before my eyes")
        elif theme == "positive interaction":
            event_descriptions.append("I met someone who seemed familiar and welcoming")
            event_descriptions.append("a friendly presence guided me forward")
        elif theme == "conflict":
            event_descriptions.append("I sensed a threatening presence")
            event_descriptions.append("I found myself trying to escape from danger")
        elif theme == "adventure":
            event_descriptions.append("I embarked on a journey through uncharted territory")
            event_descriptions.append("I accepted a challenge that appeared before me")
        elif theme == "complex narrative":
            event_descriptions.append("I became part of an intricate story unfolding around me")
            event_descriptions.append("multiple threads of reality seemed to interweave")
        elif theme == "nature":
            event_descriptions.append("natural elements came alive with their own consciousness")
            event_descriptions.append("the landscape transformed into a living entity")
        elif theme == "urban":
            event_descriptions.append("city streets twisted into impossible geometries")
            event_descriptions.append("buildings shifted and rearranged themselves")
        elif theme == "people":
            event_descriptions.append("figures appeared and disappeared in the periphery")
            event_descriptions.append("I recognized faces that kept changing")
        elif theme == "flying":
            event_descriptions.append("I realized I could lift off the ground")
            event_descriptions.append("gravity seemed to lose its hold on me")
        elif theme == "falling":
            event_descriptions.append("the ground gave way beneath me")
            event_descriptions.append("I felt myself plummeting through space")
        elif theme == "searching":
            event_descriptions.append("I was looking for something important")
            event_descriptions.append("I knew I needed to find a key object")
    
    if not event_descriptions:
        event_descriptions = [
            "the scene transformed unexpectedly",
            "I noticed unexpected details emerging",
            "reality seemed to shift around me",
            "my perspective suddenly changed"
        ]
    
    # End emotions based on valence
    if valence > 0.5:
        end_emotions = ["peaceful", "content", "joyful", "refreshed", "inspired"]
    elif valence > 0:
        end_emotions = ["curious", "intrigued", "thoughtful", "calm", "satisfied"]
    elif valence > -0.5:
        end_emotions = ["uncertain", "confused", "unsettled", "pensive", "restless"]
    else:
        end_emotions = ["anxious", "troubled", "disturbed", "uneasy", "drained"]
    
    # End actions and descriptions
    end_actions = [
        "tried to remember the details",
        "reached out to touch something that vanished",
        "began to realize it was a dream",
        "heard sounds from the waking world",
        "felt my consciousness shifting"
    ]
    
    end_descs = [
        "fragments of memory",
        "a swirl of colors",
        "mist and shadow",
        "a single point of light",
        "whispers that faded away"
    ]
    
    # Generate narrative
    paragraphs = []
    
    # Introduction
    setting = random.choice(settings).format(adj=random.choice(adjectives))
    action = random.choice(actions)
    intro = f"{action} {setting}."
    paragraphs.append(intro)
    
    # Middle - number of paragraphs based on complexity
    n_paragraphs = max(1, int(complexity * 4))
    
    for _ in range(n_paragraphs):
        # Select random themes for this paragraph
        para_themes = random.sample(themes, min(2, len(themes)))
        
        # Create relevant event descriptions
        relevant_events = [e for e in event_descriptions 
                           if any(t in e for t in para_themes)]
        
        if not relevant_events:
            relevant_events = event_descriptions
            
        event = random.choice(events).format(
            event_desc=random.choice(relevant_events)
        )
        
        # Additional detail based on complexity
        if complexity > 0.6 and random.random() < 0.7:
            detail = random.choice([
                "The air felt {adj} and {adj}.",
                "I noticed {adj} details that seemed important.",
                "There was a {adj} quality to everything around me.",
                "Time seemed to move in a {adj} way.",
                "My emotions felt {adj} and somewhat {adj}."
            ]).format(
                adj=random.choice(adjectives),
                adj2=random.choice(adjectives)
            )
            event += " " + detail
        
        paragraphs.append(event)
    
    # Conclusion
    ending_template = random.choice(endings)
    if "{end_emotion}" in ending_template:
        ending = ending_template.format(end_emotion=random.choice(end_emotions))
    elif "{end_action}" in ending_template:
        ending = ending_template.format(end_action=random.choice(end_actions))
    else:
        ending = ending_template.format(end_desc=random.choice(end_descs))
    
    paragraphs.append(ending)
    
    return "\n\n".join(paragraphs)

def simulate_full_workflow(output_dir, rem_intensity=0.7):
    """
    Simulate the full workflow from sleep data to content generation.
    
    Args:
        output_dir (str): Directory to save outputs
        rem_intensity (float): Intensity of REM patterns (0.0 to 1.0)
        
    Returns:
        dict: Simulation results
    """
    print("Dream Content Platform Simulation")
    print("=================================")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Generate sleep data
    print("\n1. Collecting sleep data...")
    time.sleep(1)  # Simulate processing time
    
    # Generate sleep data for a 5-minute REM period
    eeg_signal = simulate_eeg_signal(
        duration_seconds=300,
        sampling_rate=256,
        rem_intensity=rem_intensity
    )
    
    print(f"   - Generated {len(eeg_signal)} samples of EEG data")
    
    # Save EEG data visualization
    eeg_file = os.path.join(output_dir, "eeg_data.png")
    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(len(eeg_signal)) / 256, eeg_signal)
    plt.title("Simulated EEG Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (µV)")
    plt.tight_layout()
    plt.savefig(eeg_file)
    plt.close()
    print(f"   - Saved EEG visualization to {eeg_file}")
    
    # Step 2: Extract dream features
    print("\n2. Analyzing dream patterns...")
    time.sleep(1.5)  # Simulate processing time
    
    dream_features = extract_dream_features(eeg_signal)
    
    print("   - Dream features extracted:")
    print(f"     - REM probability: {dream_features['rem_probability']:.2f}")
    print(f"     - Emotional valence: {dream_features['emotional_tone']['valence']:.2f}")
    print(f"     - Emotional arousal: {dream_features['emotional_tone']['arousal']:.2f}")
    print(f"     - Dream complexity: {dream_features['complexity']:.2f}")
    print(f"     - Thematic elements: {', '.join(dream_features['thematic_elements'])}")
    
    # Save features to JSON
    features_file = os.path.join(output_dir, "dream_features.json")
    with open(features_file, 'w') as f:
        json.dump(dream_features, f, indent=2)
    print(f"   - Saved dream features to {features_file}")
    
    # Step 3: Generate content
    print("\n3. Generating dream content...")
    
    # Generate image
    print("   - Creating visual representation...")
    time.sleep(2)  # Simulate processing time
    image_file = os.path.join(output_dir, "dream_image.png")
    generate_dream_image(dream_features, image_file)
    print(f"   - Saved dream image to {image_file}")
    
    # Generate narrative
    print("   - Creating narrative...")
    time.sleep(1.5)  # Simulate processing time
    narrative = generate_dream_text(dream_features)
    narrative_file = os.path.join(output_dir, "dream_narrative.txt")
    with open(narrative_file, 'w') as f:
        f.write(narrative)
    print(f"   - Saved dream narrative to {narrative_file}")
    
    # Step 4: Simulate NFT creation (just for demonstration)
    print("\n4. Preparing NFT metadata...")
    time.sleep(1)  # Simulate processing time
    
    # Create timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Generate NFT metadata
    nft_metadata = {
        "name": f"Dream #{random.randint(1000, 9999)}",
        "description": f"Generated dream content based on REM sleep patterns. {dream_features['thematic_elements'][0].capitalize()} theme with {dream_features['complexity']:.2f} complexity.",
        "created_at": timestamp,
        "creator": "Dream Content Platform Simulator",
        "content_type": "mixed_media",
        "attributes": [
            {"trait_type": "Dream Intensity", "value": rem_intensity},
            {"trait_type": "Emotional Valence", "value": dream_features["emotional_tone"]["valence"]},
            {"trait_type": "Emotional Arousal", "value": dream_features["emotional_tone"]["arousal"]},
            {"trait_type": "Complexity", "value": dream_features["complexity"]},
            {"trait_type": "Primary Theme", "value": dream_features["thematic_elements"][0].capitalize()},
            {"trait_type": "Secondary Theme", "value": dream_features["thematic_elements"][1].capitalize() if len(dream_features["thematic_elements"]) > 1 else "None"}
        ],
        "dream_features": dream_features
    }
    
    # Save NFT metadata
    nft_file = os.path.join(output_dir, "nft_metadata.json")
    with open(nft_file, 'w') as f:
        json.dump(nft_metadata, f, indent=2)
    print(f"   - Saved NFT metadata to {nft_file}")
    
    print("\nSimulation complete! All outputs saved to {output_dir}")
    
    return {
        "eeg_signal": eeg_signal,
        "dream_features": dream_features,
        "output_files": {
            "eeg": eeg_file,
            "features": features_file,
            "image": image_file,
            "narrative": narrative_file,
            "nft": nft_file
        }
    }

def main():
    """Main function to run the simulation from command line."""
    parser = argparse.ArgumentParser(description="Dream Content Platform Simulator")
    parser.add_argument("--output", "-o", type=str, default="./simulation_output",
                        help="Directory to save simulation outputs")
    parser.add_argument("--intensity", "-i", type=float, default=0.7,
                        help="REM intensity (0.0 to 1.0)")
    
    args = parser.parse_args()
    
    simulate_full_workflow(args.output, args.intensity)

if __name__ == "__main__":
    main()
