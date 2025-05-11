# Developer Documentation

This guide provides technical details for developers working on or extending the Dream Content Platform.

## Project Structure

```
DreamContentPlatform/
│
├── src/                         # Source code
│   ├── sleep_data/              # Sleep data collection and processing
│   ├── ai_analysis/             # AI analysis of sleep/dream data
│   ├── content_generation/      # AI content generation modules
│   ├── blockchain/              # Blockchain and NFT functionality
│   ├── marketplace/             # Digital marketplace implementation
│   ├── web/                     # Web frontend
│   ├── mobile/                  # Mobile applications
│   └── api/                     # RESTful API implementation
│
├── models/                      # Pre-trained AI models
│
├── doc/                         # Documentation
│   └── images/                  # Documentation images and diagrams
│
├── deployment/                  # Deployment configurations
│   └── simulation/              # Simulation tools
│
└── Scientific_papers/           # Related research papers
```

## Architecture Overview

The Dream Content Platform follows a modular microservices architecture pattern with the following key components:

### Core Services

1. **Sleep Data Service**
   - Responsible for device communication, data acquisition, and storage
   - Implements real-time signal processing and filtering
   - Handles data encryption and privacy safeguards

2. **AI Processing Service**
   - Manages neural network models and inference pipelines
   - Orchestrates content generation across different media types
   - Maintains model versioning and performance optimization

3. **Blockchain Service**
   - Manages interactions with Ethereum network
   - Handles NFT minting, metadata storage, and smart contract operations
   - Provides transaction monitoring and event processing

4. **Marketplace Service**
   - Implements content discoverability and recommendation algorithms
   - Manages user profiles, favoriting, and social features
   - Handles payments, transactions, and royalty distribution

### Data Flow

```
[Sleep Device] → [Data Collection] → [AI Analysis] → [Content Generation] 
                                                            ↓
[User Marketplace] ← [Trading Platform] ← [NFT Creation] ← [Content Approval]
```

## Key Technologies and Implementation Details

### Sleep Data Collection

#### EEG Signal Processing
```python
def process_eeg_signal(raw_signal, sampling_rate=1000):
    """Process EEG signal to extract relevant features.
    
    Args:
        raw_signal (numpy.ndarray): Raw EEG signal
        sampling_rate (int): Sampling rate in Hz
        
    Returns:
        dict: Dictionary of extracted features
    """
    # Bandpass filter (1-45 Hz)
    filtered = bandpass_filter(raw_signal, 1, 45, sampling_rate)
    
    # Extract frequency bands
    bands = {
        'delta': bandpower(filtered, sampling_rate, [1, 4]),
        'theta': bandpower(filtered, sampling_rate, [4, 8]),
        'alpha': bandpower(filtered, sampling_rate, [8, 13]),
        'beta': bandpower(filtered, sampling_rate, [13, 30]),
        'gamma': bandpower(filtered, sampling_rate, [30, 45])
    }
    
    # REM detection
    rem_probability = detect_rem_state(filtered, sampling_rate)
    
    return {
        'bands': bands,
        'rem_probability': rem_probability,
        'entropy': compute_entropy(filtered),
        'complexity': compute_complexity(filtered)
    }
```

#### Sleep Stage Classification
```python
class SleepStageClassifier:
    """CNN-LSTM model for sleep stage classification."""
    
    def __init__(self, model_path):
        self.model = load_model(model_path)
        
    def predict(self, eeg_segment, eog_segment, emg_segment):
        """Predict sleep stage from multimodal data."""
        features = self._extract_features(eeg_segment, eog_segment, emg_segment)
        prediction = self.model.predict(features)
        return self._decode_prediction(prediction)
        
    def _extract_features(self, eeg, eog, emg):
        # Feature extraction implementation
        pass
        
    def _decode_prediction(self, prediction):
        stages = ['Wake', 'N1', 'N2', 'N3', 'REM']
        return stages[np.argmax(prediction)]
```

### AI Content Generation

#### Image Generation (StyleGAN3)
```python
class DreamImageGenerator:
    """Generate images from dream data using StyleGAN3."""
    
    def __init__(self, model_path, device='cuda'):
        self.model = load_stylegan3_model(model_path, device)
        self.device = device
        
    def generate_from_features(self, dream_features, num_images=1, truncation=0.7):
        """Generate images based on dream features.
        
        Args:
            dream_features (dict): Features extracted from dream data
            num_images (int): Number of images to generate
            truncation (float): Truncation psi parameter (affects diversity)
            
        Returns:
            list: Generated images as tensors
        """
        # Map dream features to latent space
        latent_codes = self._map_features_to_latents(dream_features, num_images)
        
        # Generate images
        with torch.no_grad():
            images = self.model.synthesis(latent_codes, truncation_psi=truncation)
            
        return images
        
    def _map_features_to_latents(self, features, num_images):
        # Implementation of mapping dream features to StyleGAN latent space
        pass
```

#### Narrative Generation (GPT)
```python
class DreamNarrativeGenerator:
    """Generate narrative text from dream data using GPT."""
    
    def __init__(self, model_name="dreamgpt-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
    def generate_narrative(self, dream_features, max_length=1000, temperature=0.8):
        """Generate narrative based on dream features."""
        # Create prompt from dream features
        prompt = self._create_prompt(dream_features)
        
        # Generate text
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        output = self.model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            top_p=0.9,
            no_repeat_ngram_size=3
        )
        
        return self.tokenizer.decode(output[0], skip_special_tokens=True)
        
    def _create_prompt(self, features):
        # Implementation of mapping dream features to text prompt
        pass
```

### Blockchain Integration

#### NFT Minting
```javascript
// Smart contract for Dream NFTs
contract DreamContentNFT is ERC721URIStorage, Ownable {
    using Counters for Counters.Counter;
    Counters.Counter private _tokenIds;
    
    // Events
    event DreamContentCreated(uint256 tokenId, address creator, string contentType);
    
    // Mapping from token ID to creator address
    mapping(uint256 => address) public creators;
    
    // Mapping from token ID to royalty percentage (in basis points, 1% = 100)
    mapping(uint256 => uint256) public royaltyBasisPoints;
    
    constructor() ERC721("DreamContent", "DREAM") {}
    
    function mintDreamContent(
        address to,
        string memory tokenURI,
        string memory contentType,
        uint256 royaltyBasisPoints
    ) 
        public 
        returns (uint256) 
    {
        require(royaltyBasisPoints <= 1000, "Royalty cannot exceed 10%");
        
        _tokenIds.increment();
        uint256 newTokenId = _tokenIds.current();
        
        _safeMint(to, newTokenId);
        _setTokenURI(newTokenId, tokenURI);
        
        creators[newTokenId] = msg.sender;
        royaltyBasisPoints[newTokenId] = royaltyBasisPoints;
        
        emit DreamContentCreated(newTokenId, msg.sender, contentType);
        
        return newTokenId;
    }
    
    function getRoyaltyInfo(uint256 tokenId, uint256 salePrice) 
        external
        view
        returns (address receiver, uint256 royaltyAmount)
    {
        address creator = creators[tokenId];
        uint256 bps = royaltyBasisPoints[tokenId];
        
        return (creator, (salePrice * bps) / 10000);
    }
}
```

#### JavaScript Integration
```javascript
// NFT Service for Frontend Integration
class NFTService {
  constructor(web3Provider, contractAddress) {
    this.web3 = new Web3(web3Provider);
    this.contract = new this.web3.eth.Contract(
      DreamContentNFTABI,
      contractAddress
    );
  }

  async mintDreamContent(dreamData, accountAddress) {
    try {
      // Upload metadata to IPFS
      const metadataHash = await this.uploadMetadata(dreamData);
      const tokenURI = `ipfs://${metadataHash}`;
      
      // Mint NFT
      const tx = await this.contract.methods.mintDreamContent(
        accountAddress,
        tokenURI,
        dreamData.contentType,
        dreamData.royaltyBasisPoints
      ).send({ from: accountAddress, gas: 3000000 });
      
      return {
        success: true,
        tokenId: tx.events.DreamContentCreated.returnValues.tokenId,
        transactionHash: tx.transactionHash
      };
    } catch (error) {
      console.error('Error minting dream content:', error);
      return { success: false, error: error.message };
    }
  }
  
  async uploadMetadata(dreamData) {
    // Implementation of IPFS metadata upload
    // Returns CID hash
  }
}
```

### Marketplace Implementation

#### Recommendation Engine
```python
class DreamContentRecommender:
    """Hybrid recommendation system for dream content."""
    
    def __init__(self, db_connection):
        self.db = db_connection
        self.content_features = self._load_content_features()
        self.user_interactions = self._load_user_interactions()
        
    def recommend_for_user(self, user_id, n=10):
        """Get top N recommendations for a specific user."""
        # Collaborative filtering component
        cf_recommendations = self._collaborative_filtering(user_id)
        
        # Content-based component
        cb_recommendations = self._content_based_filtering(user_id)
        
        # Hybrid blending
        recommendations = self._hybrid_blend(cf_recommendations, cb_recommendations)
        
        return recommendations[:n]
    
    def _collaborative_filtering(self, user_id):
        # Implementation of collaborative filtering
        pass
        
    def _content_based_filtering(self, user_id):
        # Implementation of content-based filtering
        pass
        
    def _hybrid_blend(self, cf_recs, cb_recs):
        # Implementation of recommendation blending strategy
        pass
```

## API Reference

### RESTful API Endpoints

#### Sleep Data API
- `POST /api/sleep-data/upload`: Upload sleep recording data
- `GET /api/sleep-data/{id}`: Retrieve processed sleep data
- `GET /api/sleep-data/user/{user_id}`: Get user's sleep data history

#### Content Generation API
- `POST /api/content/generate`: Generate content from dream data
- `GET /api/content/{id}`: Retrieve generated content
- `PUT /api/content/{id}/refine`: Refine generated content with user feedback

#### NFT API
- `POST /api/nft/mint`: Mint new dream content as NFT
- `GET /api/nft/{id}`: Get NFT metadata and status
- `GET /api/nft/user/{user_id}`: List user's NFT collection

#### Marketplace API
- `GET /api/marketplace/listings`: Get active marketplace listings
- `POST /api/marketplace/listings`: Create new listing
- `POST /api/marketplace/offer`: Make an offer for listed content
- `GET /api/marketplace/recommendations`: Get personalized recommendations

### GraphQL API

Example schema for content queries:

```graphql
type DreamContent {
  id: ID!
  title: String!
  description: String
  contentType: ContentType!
  mediaSources: [MediaSource!]!
  creator: User!
  createdAt: DateTime!
  dreamData: DreamData
  nftData: NFTData
}

type DreamData {
  sleepStages: [SleepStage!]!
  emotionalTone: EmotionalTone
  thematicElements: [String!]
  complexity: Float
}

type NFTData {
  tokenId: ID
  contractAddress: String
  mintedAt: DateTime
  ownerAddress: String
  lastPrice: Float
  currency: String
}

enum ContentType {
  IMAGE
  NARRATIVE
  MUSIC
  MIXED
}

type Query {
  dreamContent(id: ID!): DreamContent
  dreamContentByCreator(creatorId: ID!): [DreamContent!]!
  trendingDreamContent(limit: Int = 10): [DreamContent!]!
}
```

## Database Schema

### MongoDB Collections

#### Users
```json
{
  "_id": "ObjectId",
  "username": "String",
  "email": "String",
  "walletAddress": "String",
  "preferences": {
    "contentTypes": ["Array<String>"],
    "thematicPreferences": ["Array<String>"],
    "privacySettings": "Object"
  },
  "createdAt": "Date",
  "lastActive": "Date"
}
```

#### SleepData
```json
{
  "_id": "ObjectId",
  "userId": "ObjectId",
  "recordingDate": "Date",
  "deviceId": "String",
  "data": {
    "eegData": "String", // Base64 encoded or file reference
    "eogData": "String",
    "ecgData": "String"
  },
  "processedData": {
    "sleepStages": ["Array<Object>"],
    "remPeriods": ["Array<Object>"],
    "metrics": "Object"
  },
  "contentGenerated": "Boolean",
  "createdAt": "Date"
}
```

#### DreamContent
```json
{
  "_id": "ObjectId",
  "userId": "ObjectId",
  "sleepDataId": "ObjectId",
  "title": "String",
  "description": "String",
  "contentType": "String",
  "mediaFiles": ["Array<Object>"],
  "generationParams": "Object",
  "userFeedback": "Object",
  "isPublic": "Boolean",
  "isNFT": "Boolean",
  "nftData": {
    "tokenId": "String",
    "contractAddress": "String",
    "transactionHash": "String",
    "mintedAt": "Date"
  },
  "createdAt": "Date",
  "updatedAt": "Date"
}
```

#### MarketplaceListings
```json
{
  "_id": "ObjectId",
  "contentId": "ObjectId",
  "sellerId": "ObjectId",
  "price": "Number",
  "currency": "String",
  "status": "String", // 'active', 'sold', 'cancelled'
  "listedAt": "Date",
  "expiresAt": "Date",
  "transactions": ["Array<Object>"]
}
```

## Error Handling

The platform uses a standardized error handling approach:

```javascript
// Example of API error response format
{
  "error": {
    "code": "DREAM_ANALYSIS_FAILED",
    "message": "Failed to analyze dream data due to insufficient REM sleep data",
    "details": {
      "requiredMinutes": 15,
      "actualMinutes": 7,
      "sleepDataId": "60d21b4667d0d8992e610c8e"
    }
  }
}
```

## Security Considerations

### Data Encryption

All sleep and biometric data is encrypted:
- Data at rest: AES-256 encryption
- Data in transit: TLS 1.3
- User-specific encryption keys stored in secure key management system

### Access Control

Role-based access control (RBAC) implementation:
- Admin: Full platform access
- Content Creator: Sleep data and content management
- Collector: Marketplace and collection management
- Guest: Public content browsing only

### Smart Contract Security

All smart contracts undergo:
- Static analysis with tools like Slither
- Formal verification where applicable
- External security audit
- Thorough test coverage

## Testing Guidelines

### Unit Testing

Example of unit test for dream feature extraction:

```python
def test_dream_feature_extraction():
    # Load test data
    test_data = load_test_eeg_data('test_rem_sample.edf')
    
    # Process data
    features = process_eeg_signal(test_data, sampling_rate=1000)
    
    # Test assertions
    assert 'bands' in features
    assert 'rem_probability' in features
    assert features['rem_probability'] > 0.7  # High probability for REM test data
    
    # Test frequency bands
    assert 4 <= features['bands']['theta'] <= 8
    assert features['bands']['delta'] < features['bands']['theta'] # Expected for REM
```

### Integration Testing

All services should have integration tests covering:
- API endpoints and responses
- Service communication
- Database interactions
- Smart contract interactions

### End-to-End Testing

E2E tests should cover complete workflows:
- Sleep data upload → Analysis → Content generation → NFT minting → Marketplace listing

## Performance Optimization

### GPU Acceleration

For AI model inference:
- Use TensorRT for optimized inference
- ONNX for model portability
- Batch prediction when possible

### Blockchain Optimization

To reduce gas costs:
- Use batch minting when appropriate
- Optimize smart contract code
- Consider layer-2 solutions for high-volume operations

## Contribution Guidelines

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit changes: `git commit -am 'Add new feature'`
4. Push to the branch: `git push origin feature/your-feature-name`
5. Submit a pull request

### Code Style

- Python: Follow PEP 8
- JavaScript: Use ESLint with Airbnb config
- Solidity: Follow Solidity style guide

## Deployment

Recommended deployment approach:
- Containerize all services with Docker
- Orchestrate with Kubernetes
- Use CI/CD pipeline for automated testing and deployment
- Monitor with Prometheus and Grafana
- Use blue-green deployment for zero-downtime updates
