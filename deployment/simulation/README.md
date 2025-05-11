# Dream Content Simulator

This simulator demonstrates the core functionality of the Dream Content Platform by simulating the process of collecting sleep data, analyzing dream patterns, and generating content based on those patterns.

## Overview

The simulator performs the following steps:

1. **Sleep Data Collection**: Generates synthetic EEG data that mimics REM sleep patterns
2. **Dream Analysis**: Extracts features from the EEG data, including emotional tone and thematic elements
3. **Content Generation**: Creates visual representations and narrative text based on the dream features
4. **NFT Preparation**: Builds metadata suitable for NFT creation

## Requirements

- Python 3.8+
- NumPy
- Matplotlib
- SciPy

Install dependencies:

```bash
pip install numpy matplotlib scipy
```

## Usage

### Basic Usage

Run the simulator with default settings:

```bash
python dream_content_simulator.py
```

This will:
- Generate simulated dream data
- Process and analyze the data
- Create visual and narrative content
- Save all outputs to `./simulation_output`

### Advanced Options

```bash
python dream_content_simulator.py --output ./my_dream_output --intensity 0.9
```

Parameters:
- `--output`, `-o`: Directory to save simulation outputs (default: `./simulation_output`)
- `--intensity`, `-i`: REM intensity from 0.0 to 1.0 (default: 0.7)

Higher intensity values will create more vivid and emotionally charged dream content.

## Output Files

The simulator generates the following files:

- `eeg_data.png`: Visualization of the simulated EEG data
- `dream_features.json`: Extracted features from dream analysis
- `dream_image.png`: Visual representation of the dream
- `dream_narrative.txt`: Text narrative generated from dream patterns
- `nft_metadata.json`: Metadata suitable for NFT creation

## Integration

This simulator is designed to demonstrate the basic concepts behind the Dream Content Platform. In a production environment, these functions would interface with:

1. Real EEG hardware and drivers
2. Advanced AI models for dream analysis
3. Sophisticated generative models (StyleGAN3, GPT, etc.)
4. Actual blockchain integration for NFT minting

## Notes

- The visual representations are simplified abstractions and not actual AI-generated artwork
- The narrative generation uses templates rather than advanced language models
- Real dream analysis would involve significantly more sophisticated algorithms and neural networks

## Example

Here's an example of how to use the simulator in your code:

```python
from dream_content_simulator import simulate_full_workflow

# Run simulation with high REM intensity
results = simulate_full_workflow(
    output_dir="./my_dream_test",
    rem_intensity=0.85
)

# Access the results
eeg_signal = results["eeg_signal"]
dream_features = results["dream_features"]
output_files = results["output_files"]

print(f"Generated dream image: {output_files['image']}")
print(f"Generated dream narrative: {output_files['narrative']}")
```
