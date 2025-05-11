# Installation Guide

This guide provides step-by-step instructions for setting up the Dream Content Platform components.

## Prerequisites

Before installing the platform, ensure you have the following prerequisites:

- Python 3.9+
- Node.js 16+
- Git
- Docker and Docker Compose
- Ethereum wallet (MetaMask recommended)
- GPU with CUDA support (for AI components)

## Environment Setup

### 1. Clone the Repository

```bash
git clone https://github.com/JJshome/DreamContentPlatform.git
cd DreamContentPlatform
```

### 2. Set Up Python Environment

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### 3. Install JavaScript Dependencies

```bash
# Navigate to web application directory
cd src/web
npm install
```

### 4. Configure Environment Variables

Create a `.env` file in the root directory and add the following variables:

```
# Database Configuration
DB_HOST=localhost
DB_PORT=27017
DB_NAME=dreamcontent
DB_USER=admin
DB_PASSWORD=your_secure_password

# Blockchain Configuration
ETH_NETWORK=goerli  # Use 'mainnet' for production
ETH_PROVIDER_URL=https://goerli.infura.io/v3/your_infura_project_id
CONTRACT_ADDRESS=0x...  # NFT contract address after deployment

# AI Configuration
MODEL_PATH=./models
GPU_DEVICE=0  # Set to -1 for CPU only

# Security
JWT_SECRET=your_secret_key
AES_KEY=your_encryption_key
```

## Component Installation

### Sleep Data Collection Module

```bash
# Install specialized libraries
pip install mne-python pyedflib pywavelets

# Setup device configurations
python src/setup/configure_device.py
```

### AI Analysis and Generation Modules

```bash
# Download pre-trained models
python src/setup/download_models.py --all

# Alternative: Download specific models
python src/setup/download_models.py --cnn-lstm --stylegan3 --gpt
```

### Blockchain Components

```bash
# Install Truffle globally
npm install -g truffle

# Compile and migrate smart contracts
cd src/blockchain
truffle compile
truffle migrate --network goerli  # Replace with desired network
```

### Web and Mobile Applications

```bash
# Build web application
cd src/web
npm run build

# Prepare mobile app (requires additional setup for iOS/Android)
cd ../mobile
npm install
```

## Docker Deployment (Recommended for Production)

```bash
# Build and start all containers
docker-compose up -d

# Check container status
docker-compose ps
```

## Individual Component Testing

### Test Sleep Data Processor

```bash
python src/tests/test_sleep_data.py
```

### Test AI Models

```bash
python src/tests/test_ai_models.py
```

### Test Smart Contracts

```bash
cd src/blockchain
truffle test
```

### Test Web Application

```bash
cd src/web
npm test
```

## Setting Up EEG Hardware

1. **Calibration**:
   ```bash
   python src/tools/calibrate_eeg.py --device-id your_device_id
   ```

2. **Testing Connection**:
   ```bash
   python src/tools/test_eeg_connection.py
   ```

3. **Sample Recording**:
   ```bash
   python src/tools/sample_recording.py --duration 60
   ```

## Running the Platform

### Development Mode

```bash
# Start the backend API
cd src/backend
python app.py

# In a new terminal, start the frontend
cd src/web
npm run dev
```

### Production Mode

```bash
# Start all services using Docker
docker-compose -f docker-compose.prod.yml up -d
```

Access the platform at `http://localhost:3000` (by default).

## Troubleshooting

### Common Issues

1. **GPU Not Detected**:
   Ensure CUDA drivers are installed and configured correctly.
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Smart Contract Deployment Failures**:
   Check your Ethereum wallet has sufficient test ETH and the network is correctly configured.

3. **Database Connection Issues**:
   ```bash
   docker-compose logs mongodb
   ```

4. **EEG Device Connection Problems**:
   Ensure Bluetooth is enabled and the device is charged. Try:
   ```bash
   python src/tools/device_diagnostics.py
   ```

### Getting Help

For additional support:
- Check the [Troubleshooting Guide](troubleshooting.md)
- Submit an issue on GitHub
- Consult the [Developer Documentation](developer_docs.md)

## Post-Installation

After successful installation:

1. Create an administrator account
2. Configure platform settings
3. Test the full workflow from sleep data collection to NFT generation
4. Set up backup and monitoring systems

## Updating the Platform

```bash
# Pull latest changes
git pull

# Update dependencies
pip install -r requirements.txt
cd src/web && npm install

# Run database migrations
python src/backend/manage.py migrate

# Rebuild and restart containers
docker-compose up -d --build
```
