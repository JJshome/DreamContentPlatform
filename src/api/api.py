#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dream Content Platform API

This module provides a RESTful API for interacting with the Dream Content Platform.
"""

import os
import json
import uuid
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Import modules from other parts of the platform
from src.sleep_data.eeg_processor import EEGProcessor, DreamBandDevice
from src.ai_analysis.dream_analyzer import DreamAnalyzer, DreamFeatures
from src.content_generation.generator import ContentGenerator, DreamContent
from src.marketplace.marketplace import MarketplaceManager, BlockchainInterface, ContentListing

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_app(config=None):
    """
    Create and configure the Flask application.
    
    Args:
        config: Configuration dictionary or path to config file
        
    Returns:
        Flask: Configured Flask application
    """
    app = Flask(__name__)
    CORS(app)
    
    # Load configuration
    if config is None:
        config = {}
    
    if isinstance(config, str) and os.path.exists(config):
        with open(config, 'r', encoding='utf-8') as f:
            config = json.load(f)
    
    app.config.update(config)
    
    # Initialize components
    eeg_processor = EEGProcessor(
        sampling_rate=app.config.get('EEG_SAMPLING_RATE', 1000),
        channels=app.config.get('EEG_CHANNELS', ['Fp1', 'Fp2'])
    )
    
    dream_analyzer = DreamAnalyzer(
        model_path=app.config.get('AI_MODEL_PATH'),
        device=app.config.get('AI_DEVICE', 'cpu')
    )
    
    content_generator = ContentGenerator(
        image_model_path=app.config.get('IMAGE_MODEL_PATH'),
        narrative_model_path=app.config.get('NARRATIVE_MODEL_PATH'),
        music_model_path=app.config.get('MUSIC_MODEL_PATH'),
        output_dir=app.config.get('CONTENT_OUTPUT_DIR', 'output')
    )
    
    blockchain_interface = None
    if app.config.get('BLOCKCHAIN_ENABLED', False):
        blockchain_interface = BlockchainInterface(
            provider_url=app.config.get('BLOCKCHAIN_PROVIDER_URL'),
            contract_address=app.config.get('NFT_CONTRACT_ADDRESS'),
            contract_abi=app.config.get('NFT_CONTRACT_ABI')
        )
    
    marketplace_manager = MarketplaceManager(
        blockchain_interface=blockchain_interface,
        data_dir=app.config.get('MARKETPLACE_DATA_DIR', 'marketplace_data')
    )
    
    # Create upload directory if it doesn't exist
    upload_dir = app.config.get('UPLOAD_DIR', 'uploads')
    os.makedirs(upload_dir, exist_ok=True)
    
    # Register API routes
    
    @app.route('/api/health', methods=['GET'])
    def health_check():
        """Health check endpoint."""
        return jsonify({
            'status': 'ok',
            'timestamp': datetime.now().isoformat(),
            'components': {
                'eeg_processor': 'ok',
                'dream_analyzer': 'ok',
                'content_generator': 'ok',
                'blockchain': 'ok' if blockchain_interface and blockchain_interface.is_connected() else 'inactive',
                'marketplace': 'ok'
            }
        })
    
    #
    # Sleep Data Endpoints
    #
    
    @app.route('/api/sleep-data/upload', methods=['POST'])
    def upload_sleep_data():
        """
        Upload sleep recording data.
        
        Expected request format:
        - Multipart/form-data with 'file' field containing EEG data file
        - JSON metadata in 'metadata' field
        
        Returns:
            JSON: Processing result with data ID
        """
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        # Get metadata
        metadata = {}
        if 'metadata' in request.form:
            try:
                metadata = json.loads(request.form['metadata'])
            except json.JSONDecodeError:
                return jsonify({'error': 'Invalid metadata format'}), 400
        
        # Generate unique ID for the recording
        recording_id = str(uuid.uuid4())
        
        # Save file
        filename = secure_filename(file.filename)
        file_ext = os.path.splitext(filename)[1].lower()
        save_path = os.path.join(upload_dir, f"{recording_id}{file_ext}")
        file.save(save_path)
        
        # Save metadata
        metadata_path = os.path.join(upload_dir, f"{recording_id}_metadata.json")
        metadata['recording_id'] = recording_id
        metadata['filename'] = filename
        metadata['upload_time'] = datetime.now().isoformat()
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        # Return recording ID for later retrieval
        return jsonify({
            'status': 'success',
            'recording_id': recording_id,
            'message': 'Sleep data uploaded successfully'
        })
    
    @app.route('/api/sleep-data/<recording_id>', methods=['GET'])
    def get_sleep_data(recording_id):
        """
        Get processed sleep data.
        
        Args:
            recording_id: Recording ID
            
        Returns:
            JSON: Processed sleep data
        """
        # Check if metadata exists
        metadata_path = os.path.join(upload_dir, f"{recording_id}_metadata.json")
        if not os.path.exists(metadata_path):
            return jsonify({'error': 'Recording not found'}), 404
        
        # Load metadata
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Find actual file
        file_candidates = [
            os.path.join(upload_dir, f"{recording_id}.edf"),
            os.path.join(upload_dir, f"{recording_id}.csv"),
            os.path.join(upload_dir, f"{recording_id}.npy")
        ]
        
        file_path = None
        for candidate in file_candidates:
            if os.path.exists(candidate):
                file_path = candidate
                break
        
        if not file_path:
            return jsonify({'error': 'Recording file not found'}), 404
        
        # Process data based on file type
        try:
            # In a real implementation, this would process the file
            # For this demo, we'll return basic information
            
            return jsonify({
                'recording_id': recording_id,
                'metadata': metadata,
                'status': 'processed',
                'file_type': os.path.splitext(file_path)[1][1:],
                'processed_at': datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Error processing sleep data: {e}")
            return jsonify({'error': 'Error processing sleep data'}), 500
    
    @app.route('/api/sleep-data/user/<user_id>', methods=['GET'])
    def get_user_sleep_data(user_id):
        """
        Get user's sleep data history.
        
        Args:
            user_id: User ID
            
        Returns:
            JSON: List of sleep recordings
        """
        # In a real implementation, this would query a database
        # For this demo, we'll scan the upload directory
        
        recordings = []
        
        for filename in os.listdir(upload_dir):
            if filename.endswith('_metadata.json'):
                metadata_path = os.path.join(upload_dir, filename)
                
                try:
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    
                    if metadata.get('user_id') == user_id:
                        recordings.append({
                            'recording_id': metadata.get('recording_id'),
                            'upload_time': metadata.get('upload_time'),
                            'duration': metadata.get('duration'),
                            'device': metadata.get('device')
                        })
                except (json.JSONDecodeError, IOError) as e:
                    logger.error(f"Error reading metadata {filename}: {e}")
        
        return jsonify({
            'user_id': user_id,
            'recordings': recordings
        })
    
    #
    # Dream Analysis Endpoints
    #
    
    @app.route('/api/dreams/analyze', methods=['POST'])
    def analyze_dream():
        """
        Analyze dream data from a sleep recording.
        
        Expected request format:
        - JSON with recording_id
        
        Returns:
            JSON: Dream analysis results
        """
        data = request.json
        if not data or 'recording_id' not in data:
            return jsonify({'error': 'Recording ID is required'}), 400
        
        recording_id = data['recording_id']
        
        # Check if metadata exists
        metadata_path = os.path.join(upload_dir, f"{recording_id}_metadata.json")
        if not os.path.exists(metadata_path):
            return jsonify({'error': 'Recording not found'}), 404
        
        # Find actual file
        file_candidates = [
            os.path.join(upload_dir, f"{recording_id}.edf"),
            os.path.join(upload_dir, f"{recording_id}.csv"),
            os.path.join(upload_dir, f"{recording_id}.npy")
        ]
        
        file_path = None
        for candidate in file_candidates:
            if os.path.exists(candidate):
                file_path = candidate
                break
        
        if not file_path:
            return jsonify({'error': 'Recording file not found'}), 404
        
        # In a real implementation, this would load and analyze the file
        # For this demo, we'll create synthetic dream features
        
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
        
        # Save dream features
        features_id = str(uuid.uuid4())
        features_path = os.path.join(upload_dir, f"{features_id}_dream_features.json")
        
        with open(features_path, 'w', encoding='utf-8') as f:
            json.dump(dream_features.to_dict(), f, indent=2)
        
        return jsonify({
            'status': 'success',
            'features_id': features_id,
            'dream_features': dream_features.to_dict()
        })
    
    @app.route('/api/dreams/<features_id>', methods=['GET'])
    def get_dream_features(features_id):
        """
        Get dream features by ID.
        
        Args:
            features_id: Features ID
            
        Returns:
            JSON: Dream features
        """
        features_path = os.path.join(upload_dir, f"{features_id}_dream_features.json")
        if not os.path.exists(features_path):
            return jsonify({'error': 'Dream features not found'}), 404
        
        try:
            with open(features_path, 'r', encoding='utf-8') as f:
                features_data = json.load(f)
            
            return jsonify({
                'features_id': features_id,
                'dream_features': features_data
            })
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error reading dream features: {e}")
            return jsonify({'error': 'Error reading dream features'}), 500
    
    #
    # Content Generation Endpoints
    #
    
    @app.route('/api/content/generate', methods=['POST'])
    def generate_content():
        """
        Generate content from dream features.
        
        Expected request format:
        - JSON with features_id or dream_features object
        - Optional content_types list
        - Optional generation parameters
        
        Returns:
            JSON: Generated content information
        """
        data = request.json
        if not data:
            return jsonify({'error': 'Request data is required'}), 400
        
        # Get dream features
        dream_features = None
        
        if 'features_id' in data:
            features_id = data['features_id']
            features_path = os.path.join(upload_dir, f"{features_id}_dream_features.json")
            
            if not os.path.exists(features_path):
                return jsonify({'error': 'Dream features not found'}), 404
            
            try:
                with open(features_path, 'r', encoding='utf-8') as f:
                    features_data = json.load(f)
                dream_features = DreamFeatures.from_dict(features_data)
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error reading dream features: {e}")
                return jsonify({'error': 'Error reading dream features'}), 500
        
        elif 'dream_features' in data:
            try:
                features_data = data['dream_features']
                dream_features = DreamFeatures.from_dict(features_data)
            except Exception as e:
                logger.error(f"Error parsing dream features: {e}")
                return jsonify({'error': 'Invalid dream features format'}), 400
        
        else:
            return jsonify({'error': 'features_id or dream_features is required'}), 400
        
        # Get content types
        content_types = data.get('content_types', ['image', 'narrative'])
        
        # Get generation parameters
        params = data.get('params', {})
        
        # Generate content
        try:
            content = content_generator.generate_content(
                dream_features,
                content_types=content_types,
                params=params
            )
            
            # Prepare response
            response_data = {
                'status': 'success',
                'content_id': content.content_id,
                'content_type': content.content_type,
                'metadata': content.metadata
            }
            
            # Add content URLs
            if content.image_path:
                response_data['image_url'] = f"/api/content/{content.content_id}/image"
            
            if content.narrative_text:
                response_data['narrative_url'] = f"/api/content/{content.content_id}/narrative"
            
            if content.music_path:
                response_data['music_url'] = f"/api/content/{content.content_id}/music"
            
            return jsonify(response_data)
        
        except Exception as e:
            logger.error(f"Error generating content: {e}")
            return jsonify({'error': 'Error generating content'}), 500
    
    @app.route('/api/content/<content_id>', methods=['GET'])
    def get_content(content_id):
        """
        Get content metadata by ID.
        
        Args:
            content_id: Content ID
            
        Returns:
            JSON: Content metadata
        """
        content_path = os.path.join(
            app.config.get('CONTENT_OUTPUT_DIR', 'output'),
            f"dream_content_{content_id[:8]}.json"
        )
        
        if not os.path.exists(content_path):
            return jsonify({'error': 'Content not found'}), 404
        
        try:
            with open(content_path, 'r', encoding='utf-8') as f:
                content_data = json.load(f)
            
            # Prepare response
            response_data = {
                'content_id': content_data.get('content_id'),
                'content_type': content_data.get('content_type'),
                'metadata': content_data.get('metadata'),
                'creation_params': content_data.get('creation_params')
            }
            
            # Add content URLs
            if content_data.get('image_path'):
                response_data['image_url'] = f"/api/content/{content_id}/image"
            
            if content_data.get('narrative_text'):
                response_data['narrative_url'] = f"/api/content/{content_id}/narrative"
            
            if content_data.get('music_path'):
                response_data['music_url'] = f"/api/content/{content_id}/music"
            
            return jsonify(response_data)
        
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error reading content metadata: {e}")
            return jsonify({'error': 'Error reading content metadata'}), 500
    
    @app.route('/api/content/<content_id>/image', methods=['GET'])
    def get_content_image(content_id):
        """
        Get content image.
        
        Args:
            content_id: Content ID
            
        Returns:
            File: Image file
        """
        content_path = os.path.join(
            app.config.get('CONTENT_OUTPUT_DIR', 'output'),
            f"dream_content_{content_id[:8]}.json"
        )
        
        if not os.path.exists(content_path):
            return jsonify({'error': 'Content not found'}), 404
        
        try:
            with open(content_path, 'r', encoding='utf-8') as f:
                content_data = json.load(f)
            
            image_path = content_data.get('image_path')
            
            if not image_path or not os.path.exists(image_path):
                return jsonify({'error': 'Image not found'}), 404
            
            return send_file(image_path)
        
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error reading content metadata: {e}")
            return jsonify({'error': 'Error reading content metadata'}), 500
    
    @app.route('/api/content/<content_id>/narrative', methods=['GET'])
    def get_content_narrative(content_id):
        """
        Get content narrative text.
        
        Args:
            content_id: Content ID
            
        Returns:
            JSON: Narrative text
        """
        content_path = os.path.join(
            app.config.get('CONTENT_OUTPUT_DIR', 'output'),
            f"dream_content_{content_id[:8]}.json"
        )
        
        if not os.path.exists(content_path):
            return jsonify({'error': 'Content not found'}), 404
        
        try:
            with open(content_path, 'r', encoding='utf-8') as f:
                content_data = json.load(f)
            
            narrative_text = content_data.get('narrative_text')
            
            if not narrative_text:
                return jsonify({'error': 'Narrative not found'}), 404
            
            return jsonify({
                'content_id': content_id,
                'narrative_text': narrative_text
            })
        
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error reading content metadata: {e}")
            return jsonify({'error': 'Error reading content metadata'}), 500
    
    @app.route('/api/content/<content_id>/music', methods=['GET'])
    def get_content_music(content_id):
        """
        Get content music file.
        
        Args:
            content_id: Content ID
            
        Returns:
            File: Music file
        """
        content_path = os.path.join(
            app.config.get('CONTENT_OUTPUT_DIR', 'output'),
            f"dream_content_{content_id[:8]}.json"
        )
        
        if not os.path.exists(content_path):
            return jsonify({'error': 'Content not found'}), 404
        
        try:
            with open(content_path, 'r', encoding='utf-8') as f:
                content_data = json.load(f)
            
            music_path = content_data.get('music_path')
            
            if not music_path or not os.path.exists(music_path):
                return jsonify({'error': 'Music not found'}), 404
            
            return send_file(music_path)
        
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error reading content metadata: {e}")
            return jsonify({'error': 'Error reading content metadata'}), 500
    
    @app.route('/api/content/<content_id>/refine', methods=['PUT'])
    def refine_content(content_id):
        """
        Refine generated content with user feedback.
        
        Expected request format:
        - JSON with refinement parameters
        
        Returns:
            JSON: Refined content information
        """
        data = request.json
        if not data:
            return jsonify({'error': 'Request data is required'}), 400
        
        content_path = os.path.join(
            app.config.get('CONTENT_OUTPUT_DIR', 'output'),
            f"dream_content_{content_id[:8]}.json"
        )
        
        if not os.path.exists(content_path):
            return jsonify({'error': 'Content not found'}), 404
        
        try:
            with open(content_path, 'r', encoding='utf-8') as f:
                content_data = json.load(f)
            
            # In a real implementation, we would modify the content
            # For this demo, we'll pretend the content was refined
            
            # Update metadata
            content_data['refined_at'] = datetime.now().isoformat()
            content_data['refinement_params'] = data
            
            with open(content_path, 'w', encoding='utf-8') as f:
                json.dump(content_data, f, indent=2)
            
            # Prepare response
            response_data = {
                'status': 'success',
                'content_id': content_id,
                'message': 'Content refined successfully',
                'refinement_params': data
            }
            
            # Add content URLs
            if content_data.get('image_path'):
                response_data['image_url'] = f"/api/content/{content_id}/image"
            
            if content_data.get('narrative_text'):
                response_data['narrative_url'] = f"/api/content/{content_id}/narrative"
            
            if content_data.get('music_path'):
                response_data['music_url'] = f"/api/content/{content_id}/music"
            
            return jsonify(response_data)
        
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error refining content: {e}")
            return jsonify({'error': 'Error refining content'}), 500
    
    #
    # Marketplace Endpoints
    #
    
    @app.route('/api/marketplace/list', methods=['GET'])
    def list_marketplace_items():
        """
        List available items in the marketplace.
        
        Returns:
            JSON: List of marketplace items
        """
        # Get query parameters
        page = request.args.get('page', default=1, type=int)
        limit = request.args.get('limit', default=20, type=int)
        sort_by = request.args.get('sort_by', default='created_at', type=str)
        sort_dir = request.args.get('sort_dir', default='desc', type=str)
        content_type = request.args.get('content_type', type=str)
        
        try:
            items = marketplace_manager.list_items(
                page=page,
                limit=limit,
                sort_by=sort_by,
                sort_dir=sort_dir,
                content_type=content_type
            )
            
            return jsonify({
                'items': items,
                'page': page,
                'limit': limit,
                'total': marketplace_manager.count_items(content_type=content_type)
            })
        
        except Exception as e:
            logger.error(f"Error listing marketplace items: {e}")
            return jsonify({'error': 'Error listing marketplace items'}), 500
    
    @app.route('/api/marketplace/item/<item_id>', methods=['GET'])
    def get_marketplace_item(item_id):
        """
        Get detailed information about a marketplace item.
        
        Args:
            item_id: Item ID
            
        Returns:
            JSON: Item details
        """
        try:
            item = marketplace_manager.get_item(item_id)
            
            if not item:
                return jsonify({'error': 'Item not found'}), 404
            
            return jsonify(item)
        
        except Exception as e:
            logger.error(f"Error getting marketplace item: {e}")
            return jsonify({'error': 'Error getting marketplace item'}), 500
    
    @app.route('/api/marketplace/list', methods=['POST'])
    def create_marketplace_listing():
        """
        Create a new marketplace listing.
        
        Expected request format:
        - JSON with listing information
        
        Returns:
            JSON: Created listing details
        """
        data = request.json
        if not data:
            return jsonify({'error': 'Request data is required'}), 400
        
        required_fields = ['content_id', 'price', 'seller_id']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Field {field} is required'}), 400
        
        try:
            # Verify content exists
            content_id = data['content_id']
            content_path = os.path.join(
                app.config.get('CONTENT_OUTPUT_DIR', 'output'),
                f"dream_content_{content_id[:8]}.json"
            )
            
            if not os.path.exists(content_path):
                return jsonify({'error': 'Content not found'}), 404
            
            # Create listing
            listing = marketplace_manager.create_listing(
                content_id=content_id,
                price=data['price'],
                seller_id=data['seller_id'],
                title=data.get('title'),
                description=data.get('description'),
                tags=data.get('tags'),
                royalty_percentage=data.get('royalty_percentage', 10)
            )
            
            return jsonify({
                'status': 'success',
                'message': 'Listing created successfully',
                'listing': listing
            })
        
        except Exception as e:
            logger.error(f"Error creating marketplace listing: {e}")
            return jsonify({'error': 'Error creating marketplace listing'}), 500
    
    @app.route('/api/marketplace/item/<item_id>', methods=['PUT'])
    def update_marketplace_listing(item_id):
        """
        Update a marketplace listing.
        
        Args:
            item_id: Item ID
            
        Returns:
            JSON: Updated listing details
        """
        data = request.json
        if not data:
            return jsonify({'error': 'Request data is required'}), 400
        
        try:
            # Check if listing exists
            existing_item = marketplace_manager.get_item(item_id)
            
            if not existing_item:
                return jsonify({'error': 'Item not found'}), 404
            
            # Update listing
            updated_item = marketplace_manager.update_listing(
                item_id=item_id,
                price=data.get('price'),
                title=data.get('title'),
                description=data.get('description'),
                tags=data.get('tags'),
                status=data.get('status')
            )
            
            return jsonify({
                'status': 'success',
                'message': 'Listing updated successfully',
                'listing': updated_item
            })
        
        except Exception as e:
            logger.error(f"Error updating marketplace listing: {e}")
            return jsonify({'error': 'Error updating marketplace listing'}), 500
    
    @app.route('/api/marketplace/purchase/<item_id>', methods=['POST'])
    def purchase_marketplace_item(item_id):
        """
        Purchase a marketplace item.
        
        Args:
            item_id: Item ID
            
        Returns:
            JSON: Purchase result
        """
        data = request.json
        if not data or 'buyer_id' not in data:
            return jsonify({'error': 'Buyer ID is required'}), 400
        
        try:
            # Check if listing exists and is available
            existing_item = marketplace_manager.get_item(item_id)
            
            if not existing_item:
                return jsonify({'error': 'Item not found'}), 404
            
            if existing_item.get('status') != 'available':
                return jsonify({'error': 'Item is not available for purchase'}), 400
            
            # Process purchase
            purchase_result = marketplace_manager.process_purchase(
                item_id=item_id,
                buyer_id=data['buyer_id'],
                payment_method=data.get('payment_method', 'crypto'),
                payment_details=data.get('payment_details', {})
            )
            
            return jsonify({
                'status': 'success',
                'message': 'Purchase completed successfully',
                'purchase': purchase_result
            })
        
        except Exception as e:
            logger.error(f"Error processing purchase: {e}")
            return jsonify({'error': 'Error processing purchase'}), 500
    
    @app.route('/api/user/<user_id>/listings', methods=['GET'])
    def get_user_listings(user_id):
        """
        Get user's marketplace listings.
        
        Args:
            user_id: User ID
            
        Returns:
            JSON: List of user's listings
        """
        # Get query parameters
        page = request.args.get('page', default=1, type=int)
        limit = request.args.get('limit', default=20, type=int)
        
        try:
            listings = marketplace_manager.get_user_listings(
                user_id=user_id,
                page=page,
                limit=limit
            )
            
            return jsonify({
                'user_id': user_id,
                'listings': listings,
                'page': page,
                'limit': limit,
                'total': marketplace_manager.count_user_listings(user_id=user_id)
            })
        
        except Exception as e:
            logger.error(f"Error getting user listings: {e}")
            return jsonify({'error': 'Error getting user listings'}), 500
    
    @app.route('/api/user/<user_id>/purchases', methods=['GET'])
    def get_user_purchases(user_id):
        """
        Get user's purchases.
        
        Args:
            user_id: User ID
            
        Returns:
            JSON: List of user's purchases
        """
        # Get query parameters
        page = request.args.get('page', default=1, type=int)
        limit = request.args.get('limit', default=20, type=int)
        
        try:
            purchases = marketplace_manager.get_user_purchases(
                user_id=user_id,
                page=page,
                limit=limit
            )
            
            return jsonify({
                'user_id': user_id,
                'purchases': purchases,
                'page': page,
                'limit': limit,
                'total': marketplace_manager.count_user_purchases(user_id=user_id)
            })
        
        except Exception as e:
            logger.error(f"Error getting user purchases: {e}")
            return jsonify({'error': 'Error getting user purchases'}), 500
    
    @app.route('/api/marketplace/recommendations/<user_id>', methods=['GET'])
    def get_marketplace_recommendations(user_id):
        """
        Get personalized marketplace recommendations for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            JSON: List of recommended items
        """
        # Get query parameters
        limit = request.args.get('limit', default=10, type=int)
        
        try:
            recommendations = marketplace_manager.get_recommendations(
                user_id=user_id,
                limit=limit
            )
            
            return jsonify({
                'user_id': user_id,
                'recommendations': recommendations,
                'recommendation_factors': [
                    'purchase_history',
                    'content_similarity',
                    'dream_theme_preference',
                    'browsing_behavior'
                ]
            })
        
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            return jsonify({'error': 'Error getting recommendations'}), 500
    
    @app.route('/api/marketplace/trending', methods=['GET'])
    def get_trending_items():
        """
        Get trending marketplace items.
        
        Returns:
            JSON: List of trending items
        """
        # Get query parameters
        limit = request.args.get('limit', default=10, type=int)
        time_period = request.args.get('time_period', default='week', type=str)
        
        try:
            trending_items = marketplace_manager.get_trending_items(
                limit=limit,
                time_period=time_period
            )
            
            return jsonify({
                'trending_items': trending_items,
                'time_period': time_period
            })
        
        except Exception as e:
            logger.error(f"Error getting trending items: {e}")
            return jsonify({'error': 'Error getting trending items'}), 500
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)
