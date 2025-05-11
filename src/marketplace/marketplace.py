#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dream Content Marketplace

This module provides functionality for listing, discovering, recommending,
and trading dream content as NFTs.
"""

import os
import json
import uuid
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
from web3 import Web3


@dataclass
class ContentListing:
    """Data class for marketplace content listings."""
    
    listing_id: str
    content_id: str
    seller_id: str
    title: str
    description: str
    content_type: str
    price: float
    currency: str
    created_at: str
    expires_at: Optional[str] = None
    status: str = "active"  # "active", "sold", "expired", "cancelled"
    token_id: Optional[str] = None
    contract_address: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    preview_url: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "listing_id": self.listing_id,
            "content_id": self.content_id,
            "seller_id": self.seller_id,
            "title": self.title,
            "description": self.description,
            "content_type": self.content_type,
            "price": self.price,
            "currency": self.currency,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "status": self.status,
            "token_id": self.token_id,
            "contract_address": self.contract_address,
            "metadata": self.metadata,
            "preview_url": self.preview_url
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContentListing':
        """Create from dictionary."""
        return cls(
            listing_id=data.get("listing_id", str(uuid.uuid4())),
            content_id=data.get("content_id", ""),
            seller_id=data.get("seller_id", ""),
            title=data.get("title", ""),
            description=data.get("description", ""),
            content_type=data.get("content_type", ""),
            price=data.get("price", 0.0),
            currency=data.get("currency", "ETH"),
            created_at=data.get("created_at", ""),
            expires_at=data.get("expires_at"),
            status=data.get("status", "active"),
            token_id=data.get("token_id"),
            contract_address=data.get("contract_address"),
            metadata=data.get("metadata"),
            preview_url=data.get("preview_url")
        )


class BlockchainInterface:
    """
    Interface for blockchain interactions.
    Handles NFT creation, trading, and royalty management.
    """
    
    def __init__(
        self,
        provider_url: str = "http://localhost:8545",
        contract_address: Optional[str] = None,
        contract_abi: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Initialize the Blockchain Interface.
        
        Args:
            provider_url: Ethereum provider URL
            contract_address: NFT contract address
            contract_abi: NFT contract ABI
        """
        self.provider_url = provider_url
        self.contract_address = contract_address
        self.contract_abi = contract_abi
        
        # Connect to Web3 provider
        self.web3 = Web3(Web3.HTTPProvider(provider_url))
        
        # Initialize contract if address and ABI are provided
        self.contract = None
        if contract_address and contract_abi:
            self.contract = self.web3.eth.contract(
                address=self.web3.toChecksumAddress(contract_address),
                abi=contract_abi
            )
    
    def is_connected(self) -> bool:
        """Check if connected to Ethereum network."""
        return self.web3.isConnected()
    
    def mint_nft(
        self,
        to_address: str,
        metadata_uri: str,
        content_type: str,
        royalty_basis_points: int,
        from_address: str,
        private_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Mint a new NFT for dream content.
        
        Args:
            to_address: Recipient address
            metadata_uri: URI pointing to content metadata
            content_type: Type of content ("image", "narrative", "music", "mixed")
            royalty_basis_points: Royalty percentage in basis points (100 = 1%)
            from_address: Sender address
            private_key: Private key for transaction signing
            
        Returns:
            Dict: Transaction receipt
        """
        if not self.contract:
            raise ValueError("Contract not initialized")
        
        # Prepare transaction
        tx = self.contract.functions.mintDreamContent(
            self.web3.toChecksumAddress(to_address),
            metadata_uri,
            content_type,
            royalty_basis_points
        ).buildTransaction({
            'from': self.web3.toChecksumAddress(from_address),
            'nonce': self.web3.eth.getTransactionCount(self.web3.toChecksumAddress(from_address)),
            'gas': 3000000,
            'gasPrice': self.web3.eth.gas_price
        })
        
        # Sign and send transaction
        if private_key:
            signed_tx = self.web3.eth.account.sign_transaction(tx, private_key)
            tx_hash = self.web3.eth.send_raw_transaction(signed_tx.rawTransaction)
        else:
            tx_hash = self.web3.eth.send_transaction(tx)
        
        # Wait for transaction receipt
        receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
        
        # Extract token ID from event
        token_id = None
        if receipt['status'] == 1:  # Success
            logs = self.contract.events.DreamContentCreated().processReceipt(receipt)
            if logs:
                token_id = logs[0]['args']['tokenId']
        
        return {
            'status': 'success' if receipt['status'] == 1 else 'failed',
            'transaction_hash': self.web3.toHex(tx_hash),
            'token_id': token_id,
            'receipt': receipt
        }
    
    def list_for_sale(
        self,
        token_id: int,
        price: float,
        from_address: str,
        private_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        List NFT for sale in the marketplace.
        
        Args:
            token_id: NFT token ID
            price: Price in ETH
            from_address: Seller address
            private_key: Private key for transaction signing
            
        Returns:
            Dict: Transaction receipt
        """
        if not self.contract:
            raise ValueError("Contract not initialized")
        
        # Convert price to wei
        price_wei = self.web3.toWei(price, 'ether')
        
        # Prepare transaction
        tx = self.contract.functions.setTokenForSale(
            token_id,
            price_wei
        ).buildTransaction({
            'from': self.web3.toChecksumAddress(from_address),
            'nonce': self.web3.eth.getTransactionCount(self.web3.toChecksumAddress(from_address)),
            'gas': 200000,
            'gasPrice': self.web3.eth.gas_price
        })
        
        # Sign and send transaction
        if private_key:
            signed_tx = self.web3.eth.account.sign_transaction(tx, private_key)
            tx_hash = self.web3.eth.send_raw_transaction(signed_tx.rawTransaction)
        else:
            tx_hash = self.web3.eth.send_transaction(tx)
        
        # Wait for transaction receipt
        receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
        
        return {
            'status': 'success' if receipt['status'] == 1 else 'failed',
            'transaction_hash': self.web3.toHex(tx_hash),
            'receipt': receipt
        }
    
    def buy_token(
        self,
        token_id: int,
        value: float,
        from_address: str,
        private_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Buy a token listed for sale.
        
        Args:
            token_id: NFT token ID
            value: Amount to pay in ETH
            from_address: Buyer address
            private_key: Private key for transaction signing
            
        Returns:
            Dict: Transaction receipt
        """
        if not self.contract:
            raise ValueError("Contract not initialized")
        
        # Convert value to wei
        value_wei = self.web3.toWei(value, 'ether')
        
        # Prepare transaction
        tx = self.contract.functions.buyToken(
            token_id
        ).buildTransaction({
            'from': self.web3.toChecksumAddress(from_address),
            'nonce': self.web3.eth.getTransactionCount(self.web3.toChecksumAddress(from_address)),
            'gas': 300000,
            'gasPrice': self.web3.eth.gas_price,
            'value': value_wei
        })
        
        # Sign and send transaction
        if private_key:
            signed_tx = self.web3.eth.account.sign_transaction(tx, private_key)
            tx_hash = self.web3.eth.send_raw_transaction(signed_tx.rawTransaction)
        else:
            tx_hash = self.web3.eth.send_transaction(tx)
        
        # Wait for transaction receipt
        receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
        
        return {
            'status': 'success' if receipt['status'] == 1 else 'failed',
            'transaction_hash': self.web3.toHex(tx_hash),
            'receipt': receipt
        }
    
    def get_token_details(self, token_id: int) -> Dict[str, Any]:
        """
        Get details of a specific token.
        
        Args:
            token_id: NFT token ID
            
        Returns:
            Dict: Token details
        """
        if not self.contract:
            raise ValueError("Contract not initialized")
        
        # Call contract to get token details
        details = self.contract.functions.getTokenDetails(token_id).call()
        
        return {
            'content_type': details[0],
            'creator': details[1],
            'royalty_basis_points': details[2],
            'created_at': datetime.fromtimestamp(details[3]).isoformat(),
            'price': self.web3.fromWei(details[4], 'ether'),
            'for_sale': details[5]
        }
    
    def get_tokens_for_sale(self) -> List[int]:
        """
        Get list of tokens currently for sale.
        
        Returns:
            List[int]: List of token IDs
        """
        if not self.contract:
            raise ValueError("Contract not initialized")
        
        # Call contract to get tokens for sale
        return self.contract.functions.getTokensForSale().call()


class RecommendationEngine:
    """
    Recommendation engine for dream content marketplace.
    Uses collaborative filtering and content-based approaches.
    """
    
    def __init__(
        self,
        content_data_path: Optional[str] = None,
        user_data_path: Optional[str] = None
    ):
        """
        Initialize the Recommendation Engine.
        
        Args:
            content_data_path: Path to content data JSON
            user_data_path: Path to user interaction data JSON
        """
        # Load content data if provided
        self.content_data = {}
        if content_data_path and os.path.exists(content_data_path):
            with open(content_data_path, 'r', encoding='utf-8') as f:
                self.content_data = json.load(f)
        
        # Load user data if provided
        self.user_data = {}
        if user_data_path and os.path.exists(user_data_path):
            with open(user_data_path, 'r', encoding='utf-8') as f:
                self.user_data = json.load(f)
        
        # Content feature matrix
        self.content_features = {}
        # User-content interaction matrix
        self.user_content_matrix = {}
        
        # Process data
        if self.content_data:
            self._process_content_features()
        if self.user_data:
            self._process_user_interactions()
    
    def _process_content_features(self):
        """Process content data to extract features for content-based filtering."""
        for content_id, content in self.content_data.items():
            # Extract relevant features from content metadata
            features = {}
            
            # Content type
            features['content_type'] = content.get('content_type', '')
            
            # Emotional tone
            emotional_tone = content.get('metadata', {}).get('emotional_tone', {})
            features['valence'] = emotional_tone.get('valence', 0.0)
            features['arousal'] = emotional_tone.get('arousal', 0.0)
            
            # Complexity
            features['complexity'] = content.get('metadata', {}).get('complexity', 0.5)
            
            # Thematic elements
            thematic_elements = content.get('metadata', {}).get('thematic_elements', [])
            features['themes'] = thematic_elements
            
            # Creator
            features['creator'] = content.get('seller_id', '')
            
            # Price range (normalized)
            price = float(content.get('price', 0.0))
            features['price'] = min(1.0, price / 5.0)  # Normalize to 0-1, capping at 5 ETH
            
            # Save features
            self.content_features[content_id] = features
    
    def _process_user_interactions(self):
        """Process user data to build user-content interaction matrix."""
        for user_id, interactions in self.user_data.items():
            self.user_content_matrix[user_id] = {}
            
            for interaction in interactions:
                content_id = interaction.get('content_id', '')
                interaction_type = interaction.get('type', '')
                timestamp = interaction.get('timestamp', '')
                
                # Convert timestamp to recency score (newer = higher score)
                try:
                    interaction_time = datetime.fromisoformat(timestamp)
                    now = datetime.now()
                    days_ago = (now - interaction_time).days
                    recency_score = max(0.0, 1.0 - (days_ago / 30.0))  # 0 to 1, 0 if older than 30 days
                except (ValueError, TypeError):
                    recency_score = 0.0
                
                # Assign interaction score based on type and recency
                score = 0.0
                if interaction_type == 'view':
                    score = 1.0 * recency_score
                elif interaction_type == 'like':
                    score = 3.0 * recency_score
                elif interaction_type == 'purchase':
                    score = 5.0 * recency_score
                elif interaction_type == 'create':
                    score = 2.0 * recency_score
                
                # Update matrix
                self.user_content_matrix[user_id][content_id] = score
    
    def content_based_recommendations(
        self,
        user_id: str,
        n: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Generate content-based recommendations.
        
        Args:
            user_id: User ID
            n: Number of recommendations
            
        Returns:
            List[Tuple[str, float]]: List of (content_id, score) tuples
        """
        # Get user's interaction history
        if user_id not in self.user_content_matrix:
            return []
        
        user_interactions = self.user_content_matrix[user_id]
        
        # If no interactions, return empty list
        if not user_interactions:
            return []
        
        # Find content items the user has interacted with
        user_profile = {}
        for content_id, score in user_interactions.items():
            if content_id in self.content_features:
                features = self.content_features[content_id]
                
                # Update user profile with weighted features
                for key, value in features.items():
                    if key == 'themes':
                        for theme in value:
                            theme_key = f'theme_{theme}'
                            user_profile[theme_key] = user_profile.get(theme_key, 0.0) + score
                    elif isinstance(value, (int, float)):
                        user_profile[key] = user_profile.get(key, 0.0) + value * score
                    elif isinstance(value, str):
                        user_profile[f'{key}_{value}'] = user_profile.get(f'{key}_{value}', 0.0) + score
        
        # Normalize user profile
        total_score = sum(user_interactions.values())
        if total_score > 0:
            user_profile = {k: v / total_score for k, v in user_profile.items()}
        
        # Calculate similarity scores for all content items
        similarity_scores = []
        
        for content_id, features in self.content_features.items():
            # Skip items the user has already interacted with
            if content_id in user_interactions:
                continue
            
            # Calculate similarity
            similarity = 0.0
            
            # Content type match
            content_type = features.get('content_type', '')
            if f'content_type_{content_type}' in user_profile:
                similarity += user_profile[f'content_type_{content_type}'] * 0.2
            
            # Thematic elements match
            themes = features.get('themes', [])
            theme_similarity = 0.0
            for theme in themes:
                theme_key = f'theme_{theme}'
                if theme_key in user_profile:
                    theme_similarity += user_profile[theme_key]
            if themes:
                theme_similarity /= len(themes)
            similarity += theme_similarity * 0.4
            
            # Creator match
            creator = features.get('creator', '')
            if f'creator_{creator}' in user_profile:
                similarity += user_profile[f'creator_{creator}'] * 0.1
            
            # Feature similarity (emotional tone, complexity)
            feature_similarity = 0.0
            for key in ['valence', 'arousal', 'complexity', 'price']:
                feature_value = features.get(key, 0.0)
                profile_value = user_profile.get(key, 0.0)
                feature_similarity += (1.0 - abs(feature_value - profile_value)) * 0.25
            similarity += feature_similarity * 0.3
            
            similarity_scores.append((content_id, similarity))
        
        # Sort by similarity and return top N
        similarity_scores.sort(key=lambda x: x[1], reverse=True)
        return similarity_scores[:n]
    
    def collaborative_filtering_recommendations(
        self,
        user_id: str,
        n: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Generate collaborative filtering recommendations.
        
        Args:
            user_id: User ID
            n: Number of recommendations
            
        Returns:
            List[Tuple[str, float]]: List of (content_id, score) tuples
        """
        # Get user's interaction history
        if user_id not in self.user_content_matrix:
            return []
        
        user_interactions = self.user_content_matrix[user_id]
        
        # If no interactions, return empty list
        if not user_interactions:
            return []
        
        # Calculate user-user similarity
        user_similarity = {}
        for other_user_id, other_interactions in self.user_content_matrix.items():
            if other_user_id == user_id:
                continue
            
            # Find common interactions
            common_items = set(user_interactions.keys()) & set(other_interactions.keys())
            
            if not common_items:
                continue
            
            # Calculate similarity (cosine similarity)
            user_vector = [user_interactions[item] for item in common_items]
            other_vector = [other_interactions[item] for item in common_items]
            
            dot_product = sum(a * b for a, b in zip(user_vector, other_vector))
            user_magnitude = np.sqrt(sum(a * a for a in user_vector))
            other_magnitude = np.sqrt(sum(b * b for b in other_vector))
            
            if user_magnitude == 0 or other_magnitude == 0:
                similarity = 0.0
            else:
                similarity = dot_product / (user_magnitude * other_magnitude)
            
            user_similarity[other_user_id] = similarity
        
        # Generate recommendations based on similar users
        content_scores = {}
        
        for other_user_id, similarity in user_similarity.items():
            if similarity <= 0:
                continue
            
            other_interactions = self.user_content_matrix[other_user_id]
            
            for content_id, score in other_interactions.items():
                # Skip items the user has already interacted with
                if content_id in user_interactions:
                    continue
                
                # Update content score
                content_scores[content_id] = content_scores.get(content_id, 0.0) + score * similarity
        
        # Convert to list and sort
        recommendations = [(content_id, score) for content_id, score in content_scores.items()]
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return recommendations[:n]
    
    def hybrid_recommendations(
        self,
        user_id: str,
        n: int = 10,
        content_weight: float = 0.5
    ) -> List[str]:
        """
        Generate hybrid recommendations combining both approaches.
        
        Args:
            user_id: User ID
            n: Number of recommendations
            content_weight: Weight for content-based recommendations (0-1)
            
        Returns:
            List[str]: List of recommended content IDs
        """
        # Get recommendations from both methods
        content_recs = self.content_based_recommendations(user_id, n=n*2)
        collab_recs = self.collaborative_filtering_recommendations(user_id, n=n*2)
        
        # Combine and normalize scores
        combined_scores = {}
        
        for content_id, score in content_recs:
            combined_scores[content_id] = score * content_weight
        
        for content_id, score in collab_recs:
            combined_scores[content_id] = combined_scores.get(content_id, 0.0) + score * (1.0 - content_weight)
        
        # Convert to list and sort
        recommendations = [(content_id, score) for content_id, score in combined_scores.items()]
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        # Return top N content IDs
        return [content_id for content_id, _ in recommendations[:n]]


class MarketplaceManager:
    """
    Main class for managing the dream content marketplace.
    Handles content listings, recommendations, and transactions.
    """
    
    def __init__(
        self,
        blockchain_interface: Optional[BlockchainInterface] = None,
        recommendation_engine: Optional[RecommendationEngine] = None,
        data_dir: str = "marketplace_data"
    ):
        """
        Initialize the Marketplace Manager.
        
        Args:
            blockchain_interface: Blockchain interface for NFT operations
            recommendation_engine: Engine for content recommendations
            data_dir: Directory for marketplace data storage
        """
        self.blockchain = blockchain_interface
        self.recommender = recommendation_engine
        self.data_dir = data_dir
        
        # Initialize data directory
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(os.path.join(data_dir, "listings"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "users"), exist_ok=True)
        
        # Load listings data
        self.listings = self._load_listings()
    
    def _load_listings(self) -> Dict[str, ContentListing]:
        """Load all content listings from data directory."""
        listings = {}
        
        listings_dir = os.path.join(self.data_dir, "listings")
        if os.path.exists(listings_dir):
            for filename in os.listdir(listings_dir):
                if filename.endswith(".json"):
                    filepath = os.path.join(listings_dir, filename)
                    
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            listing = ContentListing.from_dict(data)
                            listings[listing.listing_id] = listing
                    except (json.JSONDecodeError, IOError) as e:
                        print(f"Error loading listing {filename}: {e}")
        
        return listings
    
    def _save_listing(self, listing: ContentListing) -> None:
        """Save a content listing to file."""
        filepath = os.path.join(self.data_dir, "listings", f"{listing.listing_id}.json")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(listing.to_dict(), f, indent=2)
        
        # Update in-memory listings
        self.listings[listing.listing_id] = listing
    
    def create_listing(
        self,
        content_id: str,
        seller_id: str,
        title: str,
        description: str,
        content_type: str,
        price: float,
        currency: str = "ETH",
        duration_days: int = 30,
        metadata: Optional[Dict[str, Any]] = None,
        preview_url: Optional[str] = None,
        token_id: Optional[str] = None,
        contract_address: Optional[str] = None
    ) -> ContentListing:
        """
        Create a new content listing.
        
        Args:
            content_id: ID of the dream content
            seller_id: ID of the seller
            title: Listing title
            description: Listing description
            content_type: Type of content
            price: Listing price
            currency: Currency (default: "ETH")
            duration_days: Listing duration in days
            metadata: Additional metadata
            preview_url: URL to content preview
            token_id: NFT token ID (if already minted)
            contract_address: NFT contract address
            
        Returns:
            ContentListing: Created listing
        """
        # Generate unique listing ID
        listing_id = str(uuid.uuid4())
        
        # Calculate expiration date
        created_at = datetime.now().isoformat()
        expires_at = (datetime.now() + timedelta(days=duration_days)).isoformat() if duration_days > 0 else None
        
        # Create listing
        listing = ContentListing(
            listing_id=listing_id,
            content_id=content_id,
            seller_id=seller_id,
            title=title,
            description=description,
            content_type=content_type,
            price=price,
            currency=currency,
            created_at=created_at,
            expires_at=expires_at,
            status="active",
            token_id=token_id,
            contract_address=contract_address,
            metadata=metadata,
            preview_url=preview_url
        )
        
        # Save listing
        self._save_listing(listing)
        
        return listing
    
    def get_listing(self, listing_id: str) -> Optional[ContentListing]:
        """
        Get a specific listing by ID.
        
        Args:
            listing_id: Listing ID
            
        Returns:
            Optional[ContentListing]: Listing or None if not found
        """
        return self.listings.get(listing_id)
    
    def update_listing(
        self,
        listing_id: str,
        updates: Dict[str, Any]
    ) -> Optional[ContentListing]:
        """
        Update an existing listing.
        
        Args:
            listing_id: Listing ID
            updates: Dictionary of fields to update
            
        Returns:
            Optional[ContentListing]: Updated listing or None if not found
        """
        listing = self.get_listing(listing_id)
        if not listing:
            return None
        
        # Update fields
        for key, value in updates.items():
            if hasattr(listing, key):
                setattr(listing, key, value)
        
        # Save updated listing
        self._save_listing(listing)
        
        return listing
    
    def get_active_listings(
        self,
        content_type: Optional[str] = None,
        seller_id: Optional[str] = None,
        max_price: Optional[float] = None,
        sort_by: str = "created_at",
        sort_desc: bool = True,
        limit: int = 50,
        offset: int = 0
    ) -> List[ContentListing]:
        """
        Get active listings with optional filtering.
        
        Args:
            content_type: Filter by content type
            seller_id: Filter by seller
            max_price: Maximum price
            sort_by: Field to sort by
            sort_desc: Sort in descending order
            limit: Maximum number of results
            offset: Offset for pagination
            
        Returns:
            List[ContentListing]: List of matching listings
        """
        # Filter listings
        filtered = []
        
        for listing in self.listings.values():
            # Skip inactive listings
            if listing.status != "active":
                continue
            
            # Apply content type filter
            if content_type and listing.content_type != content_type:
                continue
            
            # Apply seller filter
            if seller_id and listing.seller_id != seller_id:
                continue
            
            # Apply price filter
            if max_price is not None and listing.price > max_price:
                continue
            
            filtered.append(listing)
        
        # Sort listings
        if sort_by in ["created_at", "price", "expires_at"]:
            filtered.sort(key=lambda x: getattr(x, sort_by), reverse=sort_desc)
        
        # Paginate
        return filtered[offset:offset+limit]
    
    def get_recommendations(
        self,
        user_id: str,
        n: int = 10
    ) -> List[ContentListing]:
        """
        Get personalized recommendations for a user.
        
        Args:
            user_id: User ID
            n: Number of recommendations
            
        Returns:
            List[ContentListing]: List of recommended listings
        """
        if not self.recommender:
            return []
        
        # Get recommended content IDs
        content_ids = self.recommender.hybrid_recommendations(user_id, n=n)
        
        # Map to active listings
        recommendations = []
        
        for listing in self.listings.values():
            if listing.status == "active" and listing.content_id in content_ids:
                recommendations.append(listing)
        
        return recommendations[:n]
    
    def record_user_interaction(
        self,
        user_id: str,
        content_id: str,
        interaction_type: str
    ) -> None:
        """
        Record a user interaction with content.
        
        Args:
            user_id: User ID
            content_id: Content ID
            interaction_type: Type of interaction (view, like, purchase, create)
        """
        user_file = os.path.join(self.data_dir, "users", f"{user_id}.json")
        
        # Load existing interactions or create new
        interactions = []
        if os.path.exists(user_file):
            try:
                with open(user_file, 'r', encoding='utf-8') as f:
                    interactions = json.load(f)
            except (json.JSONDecodeError, IOError):
                interactions = []
        
        # Add new interaction
        interaction = {
            "content_id": content_id,
            "type": interaction_type,
            "timestamp": datetime.now().isoformat()
        }
        interactions.append(interaction)
        
        # Save interactions
        with open(user_file, 'w', encoding='utf-8') as f:
            json.dump(interactions, f, indent=2)
    
    def mint_and_list(
        self,
        content_id: str,
        seller_id: str,
        title: str,
        description: str,
        content_type: str,
        price: float,
        metadata_uri: str,
        royalty_basis_points: int = 1000,  # 10%
        seller_address: str = None,
        private_key: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        preview_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Mint an NFT and list it for sale in one operation.
        
        Args:
            content_id: ID of the dream content
            seller_id: ID of the seller
            title: Listing title
            description: Listing description
            content_type: Type of content
            price: Listing price
            metadata_uri: URI pointing to content metadata
            royalty_basis_points: Royalty percentage in basis points
            seller_address: Ethereum address of the seller
            private_key: Private key for transaction signing
            metadata: Additional metadata
            preview_url: URL to content preview
            
        Returns:
            Dict: Operation result with listing and transaction details
        """
        if not self.blockchain:
            raise ValueError("Blockchain interface not initialized")
        
        if not seller_address:
            raise ValueError("Seller address is required")
        
        # Mint NFT
        mint_result = self.blockchain.mint_nft(
            to_address=seller_address,
            metadata_uri=metadata_uri,
            content_type=content_type,
            royalty_basis_points=royalty_basis_points,
            from_address=seller_address,
            private_key=private_key
        )
        
        if mint_result['status'] != 'success':
            return {
                'status': 'failed',
                'message': 'Failed to mint NFT',
                'mint_result': mint_result
            }
        
        token_id = mint_result['token_id']
        
        # List NFT for sale
        list_result = self.blockchain.list_for_sale(
            token_id=token_id,
            price=price,
            from_address=seller_address,
            private_key=private_key
        )
        
        if list_result['status'] != 'success':
            return {
                'status': 'failed',
                'message': 'Failed to list NFT for sale',
                'mint_result': mint_result,
                'list_result': list_result
            }
        
        # Create marketplace listing
        listing = self.create_listing(
            content_id=content_id,
            seller_id=seller_id,
            title=title,
            description=description,
            content_type=content_type,
            price=price,
            token_id=str(token_id),
            contract_address=self.blockchain.contract_address,
            metadata=metadata,
            preview_url=preview_url
        )
        
        # Record interaction
        self.record_user_interaction(seller_id, content_id, "create")
        
        return {
            'status': 'success',
            'listing': listing.to_dict(),
            'mint_result': mint_result,
            'list_result': list_result
        }
    
    def purchase_listing(
        self,
        listing_id: str,
        buyer_id: str,
        buyer_address: str,
        private_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Purchase a listed content.
        
        Args:
            listing_id: Listing ID
            buyer_id: Buyer's user ID
            buyer_address: Buyer's Ethereum address
            private_key: Private key for transaction signing
            
        Returns:
            Dict: Purchase result
        """
        if not self.blockchain:
            raise ValueError("Blockchain interface not initialized")
        
        # Get listing
        listing = self.get_listing(listing_id)
        if not listing:
            return {
                'status': 'failed',
                'message': 'Listing not found'
            }
        
        # Check if listing is active
        if listing.status != "active":
            return {
                'status': 'failed',
                'message': f'Listing is not active (status: {listing.status})'
            }
        
        # Check if token ID is available
        if not listing.token_id:
            return {
                'status': 'failed',
                'message': 'Listing does not have a valid token ID'
            }
        
        # Buy token
        buy_result = self.blockchain.buy_token(
            token_id=int(listing.token_id),
            value=listing.price,
            from_address=buyer_address,
            private_key=private_key
        )
        
        if buy_result['status'] != 'success':
            return {
                'status': 'failed',
                'message': 'Failed to buy token',
                'buy_result': buy_result
            }
        
        # Update listing status
        self.update_listing(listing_id, {'status': 'sold'})
        
        # Record interaction
        self.record_user_interaction(buyer_id, listing.content_id, "purchase")
        
        return {
            'status': 'success',
            'listing': listing.to_dict(),
            'buy_result': buy_result
        }


if __name__ == "__main__":
    # Example usage
    # Initialize marketplace
    marketplace = MarketplaceManager(data_dir="marketplace_data")
    
    # Create a sample listing
    listing = marketplace.create_listing(
        content_id="dream_123",
        seller_id="user_456",
        title="Ethereal Journey Through the Clouds",
        description="A vivid dream of flying through colorful clouds and mystical landscapes.",
        content_type="image",
        price=0.25,
        metadata={
            "emotional_tone": {
                "valence": 0.7,
                "arousal": 0.6
            },
            "thematic_elements": ["flying", "nature", "adventure"],
            "complexity": 0.8
        },
        preview_url="https://example.com/previews/dream_123.jpg"
    )
    
    print(f"Created listing: {listing.listing_id}")
    
    # Get active listings
    active_listings = marketplace.get_active_listings(
        content_type="image",
        max_price=1.0,
        limit=10
    )
    
    print(f"Found {len(active_listings)} active listings")
    
    # Record a user interaction
    marketplace.record_user_interaction(
        user_id="user_789",
        content_id="dream_123",
        interaction_type="view"
    )
    
    print("Recorded user interaction")
