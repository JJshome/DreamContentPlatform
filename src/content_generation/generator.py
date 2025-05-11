#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dream Content Generator

This module provides functionality for generating content from dream features
extracted by the AI analysis module. It supports generating images, narratives,
and music based on dream data.
"""

import os
import json
import uuid
import numpy as np
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
import matplotlib.pyplot as plt

# Assuming dream_features is from the ai_analysis module
from src.ai_analysis.dream_analyzer import DreamFeatures

@dataclass
class DreamContent:
    """Data class for generated dream content."""
    
    content_id: str
    content_type: str  # "image", "narrative", "music", "mixed"
    metadata: Dict[str, Any]
    creation_params: Dict[str, Any]
    
    # Content paths or data
    image_path: Optional[str] = None
    narrative_text: Optional[str] = None
    music_path: Optional[str] = None
    
    # Source dream features
    dream_features: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content_id": self.content_id,
            "content_type": self.content_type,
            "metadata": self.metadata,
            "creation_params": self.creation_params,
            "image_path": self.image_path,
            "narrative_text": self.narrative_text,
            "music_path": self.music_path,
            "dream_features": self.dream_features
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DreamContent':
        """Create from dictionary."""
        return cls(
            content_id=data.get("content_id", str(uuid.uuid4())),
            content_type=data.get("content_type", "mixed"),
            metadata=data.get("metadata", {}),
            creation_params=data.get("creation_params", {}),
            image_path=data.get("image_path"),
            narrative_text=data.get("narrative_text"),
            music_path=data.get("music_path"),
            dream_features=data.get("dream_features")
        )
    
    def save(self, filepath: str) -> None:
        """Save content metadata to a JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'DreamContent':
        """Load content from a JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)


class ContentGeneratorBase(ABC):
    """Abstract base class for content generators."""
    
    @abstractmethod
    def generate(self, dream_features: DreamFeatures, params: Dict[str, Any] = None) -> Any:
        """Generate content from dream features."""
        pass


class ImageGenerator(ContentGeneratorBase):
    """
    Generates visual art from dream features using StyleGAN3.
    In a real implementation, this would use actual StyleGAN3 model.
    This is a simplified version for demonstration.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize the Image Generator.
        
        Args:
            model_path: Path to pre-trained StyleGAN3 model
            device: Computation device ('cuda' or 'cpu')
        """
        self.device = device
        self.model_path = model_path
        
        # In a real implementation, this would load the StyleGAN3 model
        # For this demo, we'll simulate image generation
        
    def generate(
        self,
        dream_features: DreamFeatures,
        params: Dict[str, Any] = None
    ) -> str:
        """
        Generate a dream-inspired image.
        
        Args:
            dream_features: Dream features from analyzer
            params: Generation parameters
                - resolution: Image resolution (default: 512)
                - style: Visual style (default: "abstract")
                - truncation: Truncation parameter (default: 0.7)
                - output_dir: Directory to save the image
                
        Returns:
            str: Path to the generated image
        """
        # Default parameters
        params = params or {}
        resolution = params.get("resolution", 512)
        style = params.get("style", "abstract")
        truncation = params.get("truncation", 0.7)
        output_dir = params.get("output_dir", "output")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # In a real implementation, this would use StyleGAN3 to generate an image
        # For this demo, we'll create a synthetic image based on dream features
        
        # Generate a unique filename
        image_id = str(uuid.uuid4())[:8]
        image_filename = f"dream_image_{image_id}.png"
        image_path = os.path.join(output_dir, image_filename)
        
        # Create a synthetic image based on dream features
        self._create_synthetic_image(dream_features, image_path, resolution, style)
        
        return image_path
    
    def _create_synthetic_image(
        self,
        dream_features: DreamFeatures,
        output_path: str,
        resolution: int = 512,
        style: str = "abstract"
    ) -> None:
        """
        Create a synthetic image based on dream features.
        
        Args:
            dream_features: Dream features
            output_path: Path to save the image
            resolution: Image resolution
            style: Visual style
        """
        # Extract relevant features
        valence = dream_features.emotional_tone.get("valence", 0)
        arousal = dream_features.emotional_tone.get("arousal", 0.5)
        complexity = dream_features.complexity
        
        # Create figure
        plt.figure(figsize=(resolution/100, resolution/100), dpi=100)
        
        # Set background color based on emotional valence
        bg_color = (
            max(0, -valence/2 + 0.5),  # Red channel
            0.5,                        # Green channel
            max(0, valence/2 + 0.5)     # Blue channel
        )
        
        # Create background
        plt.fill_between([-1, 1], [-1, -1], [1, 1], color=bg_color, alpha=0.5)
        
        # Number of elements based on complexity
        n_elements = int(complexity * 50) + 10
        
        # Generate random shapes
        for _ in range(n_elements):
            shape_type = np.random.choice(["circle", "line", "arc"])
            size = np.random.uniform(0.05, 0.2)
            x = np.random.uniform(-0.9, 0.9)
            y = np.random.uniform(-0.9, 0.9)
            
            # Color based on emotional tone
            r = max(0, min(1, 0.5 + valence/2 + np.random.uniform(-0.2, 0.2)))
            g = max(0, min(1, 0.5 + arousal/2 + np.random.uniform(-0.2, 0.2)))
            b = max(0, min(1, 0.7 + (complexity-0.5)/2 + np.random.uniform(-0.2, 0.2)))
            color = (r, g, b, np.random.uniform(0.3, 0.9))  # RGBA
            
            if shape_type == "circle":
                circle = plt.Circle((x, y), size, color=color)
                plt.gca().add_patch(circle)
            elif shape_type == "line":
                x2 = x + np.random.uniform(-0.4, 0.4)
                y2 = y + np.random.uniform(-0.4, 0.4)
                plt.plot([x, x2], [y, y2], color=color, 
                         linewidth=size*20, alpha=color[3])
            elif shape_type == "arc":
                theta1 = np.random.uniform(0, 360)
                theta2 = theta1 + np.random.uniform(30, 180)
                arc = plt.matplotlib.patches.Arc((x, y), size*2, size*2, 
                                              theta1=theta1, theta2=theta2,
                                              color=color, linewidth=size*10, alpha=color[3])
                plt.gca().add_patch(arc)
        
        # Add thematic elements as text
        themes = dream_features.thematic_elements
        for i, theme in enumerate(themes):
            x = np.random.uniform(-0.8, 0.8)
            y = np.random.uniform(-0.8, 0.8)
            rotation = np.random.uniform(-45, 45)
            alpha = np.random.uniform(0.2, 0.5)
            plt.text(x, y, theme, rotation=rotation, alpha=alpha,
                    fontsize=12 + np.random.uniform(0, 10),
                    ha='center', va='center', color='white')
        
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.axis('off')
        plt.tight_layout()
        
        # Save image
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()


class NarrativeGenerator(ContentGeneratorBase):
    """
    Generates narrative text from dream features using GPT-based model.
    In a real implementation, this would use an actual GPT model.
    This is a simplified version for demonstration.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the Narrative Generator.
        
        Args:
            model_path: Path to pre-trained language model
        """
        self.model_path = model_path
        
        # In a real implementation, this would load a language model
        # For this demo, we'll simulate text generation
        
        # Setting templates
        self.settings = [
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
        self.actions = [
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
        self.events = [
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
        self.endings = [
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
        self.positive_adj = ["beautiful", "vibrant", "peaceful", "enchanted", "luminous", "serene"]
        self.negative_adj = ["dark", "ominous", "unsettling", "strange", "foreboding", "twisted"]
        self.high_arousal_adj = ["intense", "chaotic", "wild", "dramatic", "electric", "vivid"]
        self.low_arousal_adj = ["quiet", "still", "muted", "gentle", "hazy", "soft"]
        
    def generate(
        self,
        dream_features: DreamFeatures,
        params: Dict[str, Any] = None
    ) -> str:
        """
        Generate a narrative based on dream features.
        
        Args:
            dream_features: Dream features from analyzer
            params: Generation parameters
                - length: Desired narrative length (default: "medium")
                - style: Writing style (default: "descriptive")
                - perspective: Narrative perspective (default: "first_person")
                
        Returns:
            str: Generated narrative text
        """
        # Default parameters
        params = params or {}
        length = params.get("length", "medium")
        style = params.get("style", "descriptive")
        perspective = params.get("perspective", "first_person")
        
        # Map length to number of paragraphs
        length_map = {
            "short": (2, 3),
            "medium": (4, 6),
            "long": (7, 10)
        }
        n_paragraphs_range = length_map.get(length, (4, 6))
        
        # In a real implementation, this would use a language model
        # For this demo, we'll create a template-based narrative
        
        # Extract features
        valence = dream_features.emotional_tone.get("valence", 0)
        arousal = dream_features.emotional_tone.get("arousal", 0.5)
        complexity = dream_features.complexity
        themes = dream_features.thematic_elements
        
        # Select adjectives based on emotional tone
        adjectives = []
        if valence > 0.3:
            adjectives.extend(self.positive_adj)
        elif valence < -0.3:
            adjectives.extend(self.negative_adj)
            
        if arousal > 0.6:
            adjectives.extend(self.high_arousal_adj)
        elif arousal < 0.4:
            adjectives.extend(self.low_arousal_adj)
            
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
            elif theme == "flying":
                event_descriptions.append("I realized I could lift off the ground effortlessly")
                event_descriptions.append("gravity seemed to lose its hold on me")
            elif theme == "falling":
                event_descriptions.append("the ground gave way beneath my feet")
                event_descriptions.append("I felt myself plummeting through endless space")
            elif theme in ["chasing", "being_chased"]:
                event_descriptions.append("I sensed something pursuing me through the shadows")
                event_descriptions.append("I was racing to catch something just ahead")
            elif theme in ["school", "work"]:
                event_descriptions.append("I was late for an important deadline")
                event_descriptions.append("faces of colleagues or classmates surrounded me")
            elif theme in ["family", "friends", "romantic"]:
                event_descriptions.append("I recognized familiar faces that kept shifting")
                event_descriptions.append("someone important was trying to tell me something")
            elif theme in ["adventure", "discovery"]:
                event_descriptions.append("I found a hidden passageway that beckoned me forward")
                event_descriptions.append("a map appeared, showing an unexplored territory")
            elif theme in ["conflict", "resolution"]:
                event_descriptions.append("I confronted a shadowy figure blocking my path")
                event_descriptions.append("opposing forces suddenly found harmony")
            elif theme == "transformation":
                event_descriptions.append("my body began to change into something else")
                event_descriptions.append("the world around me morphed into a new reality")
            elif theme == "nature":
                event_descriptions.append("plants and trees seemed alive with consciousness")
                event_descriptions.append("animals gathered around me with knowing eyes")
            elif theme == "urban":
                event_descriptions.append("city streets twisted into impossible geometries")
                event_descriptions.append("buildings shifted and rearranged themselves")
        
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
        setting = np.random.choice(self.settings).format(adj=np.random.choice(adjectives))
        action = np.random.choice(self.actions)
        intro = f"{action} {setting}."
        paragraphs.append(intro)
        
        # Middle paragraphs
        n_paragraphs = np.random.randint(*n_paragraphs_range)
        n_paragraphs = max(1, min(int(complexity * 8), n_paragraphs))  # Adjust based on complexity
        
        for _ in range(n_paragraphs):
            # Select random themes for this paragraph
            para_themes = np.random.choice(themes, min(2, len(themes)), replace=False).tolist()
            
            # Create relevant event descriptions
            relevant_events = [e for e in event_descriptions 
                            if any(t in e for t in para_themes)]
            
            if not relevant_events:
                relevant_events = event_descriptions
                
            event = np.random.choice(self.events).format(
                event_desc=np.random.choice(relevant_events)
            )
            
            # Additional detail based on complexity
            if complexity > 0.6 and np.random.random() < 0.7:
                detail = np.random.choice([
                    "The air felt {adj} and {adj2}.",
                    "I noticed {adj} details that seemed important.",
                    "There was a {adj} quality to everything around me.",
                    "Time seemed to move in a {adj} way.",
                    "My emotions felt {adj} and somewhat {adj2}."
                ]).format(
                    adj=np.random.choice(adjectives),
                    adj2=np.random.choice(adjectives)
                )
                event += " " + detail
            
            paragraphs.append(event)
        
        # Conclusion
        ending_template = np.random.choice(self.endings)
        if "{end_emotion}" in ending_template:
            ending = ending_template.format(end_emotion=np.random.choice(end_emotions))
        elif "{end_action}" in ending_template:
            ending = ending_template.format(end_action=np.random.choice(end_actions))
        else:
            ending = ending_template.format(end_desc=np.random.choice(end_descs))
        
        paragraphs.append(ending)
        
        # Combine paragraphs into final narrative
        narrative = "\n\n".join(paragraphs)
        
        # Apply style modifications if needed
        if style == "poetic":
            # Add more metaphors and poetic language
            narrative = narrative.replace(".", ".\n")
            narrative = "~ " + narrative + " ~"
        elif style == "analytical":
            # Add more analytical framing
            narrative = "Dream Analysis:\n\n" + narrative + "\n\nKeywords: " + ", ".join(themes)
        
        return narrative


class MusicGenerator(ContentGeneratorBase):
    """
    Generates music from dream features.
    In a real implementation, this would use a MIDI-based AI model.
    This is a simplified version that outputs a stub of what would be generated.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the Music Generator.
        
        Args:
            model_path: Path to pre-trained music generation model
        """
        self.model_path = model_path
        
        # In a real implementation, this would load a music generation model
        
    def generate(
        self,
        dream_features: DreamFeatures,
        params: Dict[str, Any] = None
    ) -> str:
        """
        Generate music based on dream features.
        
        Args:
            dream_features: Dream features from analyzer
            params: Generation parameters
                - duration: Duration in seconds (default: 60)
                - genre: Musical genre (default: "ambient")
                - output_dir: Directory to save the music file
                
        Returns:
            str: Path to the generated music file
        """
        # Default parameters
        params = params or {}
        duration = params.get("duration", 60)
        genre = params.get("genre", "ambient")
        output_dir = params.get("output_dir", "output")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # In a real implementation, this would generate actual music
        # For this demo, we'll create a placeholder file with metadata
        
        # Generate a unique filename
        music_id = str(uuid.uuid4())[:8]
        music_filename = f"dream_music_{music_id}.midi"
        music_path = os.path.join(output_dir, music_filename)
        
        # Create a placeholder file with metadata
        metadata = {
            "dream_features": dream_features.to_dict(),
            "generation_params": {
                "duration": duration,
                "genre": genre
            },
            "note": "This is a placeholder for music generation. In a real implementation, this would be a MIDI file."
        }
        
        with open(music_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        return music_path


class ContentGenerator:
    """
    Main content generator class that orchestrates different content types.
    It can generate visual art, narratives, and music from dream features.
    """
    
    def __init__(
        self,
        image_model_path: Optional[str] = None,
        narrative_model_path: Optional[str] = None,
        music_model_path: Optional[str] = None,
        output_dir: str = "output"
    ):
        """
        Initialize the Content Generator.
        
        Args:
            image_model_path: Path to pre-trained image generation model
            narrative_model_path: Path to pre-trained narrative generation model
            music_model_path: Path to pre-trained music generation model
            output_dir: Directory to save generated content
        """
        self.output_dir = output_dir
        
        # Initialize individual generators
        self.image_generator = ImageGenerator(image_model_path)
        self.narrative_generator = NarrativeGenerator(narrative_model_path)
        self.music_generator = MusicGenerator(music_model_path)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
    def generate_content(
        self,
        dream_features: DreamFeatures,
        content_types: List[str] = None,
        params: Dict[str, Any] = None
    ) -> DreamContent:
        """
        Generate content from dream features.
        
        Args:
            dream_features: Dream features from analyzer
            content_types: List of content types to generate (default: ["image", "narrative"])
            params: Generation parameters for each content type
            
        Returns:
            DreamContent: Generated content object
        """
        # Default content types and parameters
        content_types = content_types or ["image", "narrative"]
        params = params or {}
        
        # Content paths
        image_path = None
        narrative_text = None
        music_path = None
        
        # Generate content based on requested types
        if "image" in content_types:
            image_params = params.get("image", {})
            image_params["output_dir"] = os.path.join(self.output_dir, "images")
            image_path = self.image_generator.generate(dream_features, image_params)
        
        if "narrative" in content_types:
            narrative_params = params.get("narrative", {})
            narrative_text = self.narrative_generator.generate(dream_features, narrative_params)
            
            # Save narrative text to file
            if narrative_text:
                narrative_id = str(uuid.uuid4())[:8]
                narrative_filename = f"dream_narrative_{narrative_id}.txt"
                narrative_path = os.path.join(self.output_dir, "narratives", narrative_filename)
                
                # Create narratives directory if it doesn't exist
                os.makedirs(os.path.dirname(narrative_path), exist_ok=True)
                
                with open(narrative_path, 'w', encoding='utf-8') as f:
                    f.write(narrative_text)
        
        if "music" in content_types:
            music_params = params.get("music", {})
            music_params["output_dir"] = os.path.join(self.output_dir, "music")
            music_path = self.music_generator.generate(dream_features, music_params)
        
        # Determine content type
        if len(content_types) > 1:
            content_type = "mixed"
        else:
            content_type = content_types[0]
        
        # Create metadata
        metadata = {
            "created_at": np.datetime_as_string(np.datetime64('now')),
            "content_types": content_types,
            "emotional_tone": dream_features.emotional_tone,
            "thematic_elements": dream_features.thematic_elements,
            "complexity": dream_features.complexity
        }
        
        # Create dream content object
        dream_content = DreamContent(
            content_id=str(uuid.uuid4()),
            content_type=content_type,
            metadata=metadata,
            creation_params=params,
            image_path=image_path,
            narrative_text=narrative_text,
            music_path=music_path,
            dream_features=dream_features.to_dict()
        )
        
        # Save content metadata
        content_path = os.path.join(self.output_dir, f"dream_content_{dream_content.content_id[:8]}.json")
        dream_content.save(content_path)
        
        return dream_content


if __name__ == "__main__":
    # Example usage
    from src.ai_analysis.dream_analyzer import DreamFeatures
    
    # Create sample dream features
    dream_features = DreamFeatures(
        frequency_bands={
            "delta": 0.2,
            "theta": 0.4,
            "alpha": 0.2,
            "beta": 0.1,
            "gamma": 0.1
        },
        emotional_tone={
            "valence": 0.3,  # Slightly positive
            "arousal": 0.7   # Fairly aroused
        },
        complexity=0.8,  # High complexity
        thematic_elements=["flying", "water", "transformation"],
        dream_intensity=0.9,
        narrative_structure={
            "linearity": 0.3,
            "coherence": 0.7,
            "character_presence": 0.8
        },
        visual_patterns={
            "color_intensity": 0.9,
            "spatial_complexity": 0.7,
            "movement_level": 0.8
        },
        auditory_patterns={
            "presence": 0.6,
            "rhythm": 0.4,
            "harmony": 0.5
        }
    )
    
    # Initialize content generator
    generator = ContentGenerator(output_dir="output")
    
    # Generate all content types
    content = generator.generate_content(
        dream_features,
        content_types=["image", "narrative", "music"],
        params={
            "image": {
                "resolution": 512,
                "style": "abstract"
            },
            "narrative": {
                "length": "medium",
                "style": "descriptive"
            },
            "music": {
                "duration": 60,
                "genre": "ambient"
            }
        }
    )
    
    print("Generated Content:")
    print(f"Content ID: {content.content_id}")
    print(f"Content Type: {content.content_type}")
    
    if content.image_path:
        print(f"Image Path: {content.image_path}")
    
    if content.narrative_text:
        print(f"Narrative Text Preview: {content.narrative_text[:100]}...")
        
    if content.music_path:
        print(f"Music Path: {content.music_path}")
