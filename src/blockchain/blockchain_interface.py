#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Blockchain Interface for Dream Content Platform

This module provides interfaces to interact with the Ethereum blockchain
for NFT minting, verification, and trading of dream contents.
"""

import json
import time
import os
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime

from web3 import Web3, HTTPProvider
from web3.exceptions import ContractLogicError, InvalidAddress
from eth_account import Account
from eth_account.signers.local import LocalAccount
from eth_typing import ChecksumAddress
import ipfshttpclient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class IPFSClient:
    """Client for interacting with IPFS for storing dream content metadata and media."""

    def __init__(self, ipfs_api_url: str = "/ip4/127.0.0.1/tcp/5001"):
        """
        Initialize IPFS client.
        
        Args:
            ipfs_api_url: URL of the IPFS API
        """
        self.client = None
        try:
            self.client = ipfshttpclient.connect(ipfs_api_url)
            logger.info("Connected to IPFS at %s", ipfs_api_url)
        except Exception as e:
            logger.warning("Could not connect to IPFS: %s", e)
            
    def is_connected(self) -> bool:
        """Check if connected to IPFS."""
        return self.client is not None
    
    def add_file(self, file_path: str) -> Dict[str, Any]:
        """
        Add a file to IPFS.
        
        Args:
            file_path: Path to file
            
        Returns:
            Dict with IPFS hash and other metadata
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to IPFS")
        
        result = self.client.add(file_path)
        logger.info("Added file to IPFS: %s -> %s", file_path, result["Hash"])
        return result
    
    def add_json(self, data: Dict[str, Any]) -> str:
        """
        Add JSON data to IPFS.
        
        Args:
            data: JSON-serializable data
            
        Returns:
            IPFS hash
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to IPFS")
        
        result = self.client.add_json(data)
        logger.info("Added JSON to IPFS: %s", result)
        return result
    
    def get_json(self, ipfs_hash: str) -> Dict[str, Any]:
        """
        Get JSON data from IPFS.
        
        Args:
            ipfs_hash: IPFS hash
            
        Returns:
            Parsed JSON data
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to IPFS")
        
        return self.client.get_json(ipfs_hash)
    
    def cat(self, ipfs_hash: str) -> bytes:
        """
        Get file content from IPFS.
        
        Args:
            ipfs_hash: IPFS hash
            
        Returns:
            File content as bytes
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to IPFS")
        
        return self.client.cat(ipfs_hash)


class BlockchainInterface:
    """Interface for interacting with the Ethereum blockchain."""
    
    def __init__(
        self, 
        provider_url: str, 
        contract_address: str = None, 
        contract_abi: str = None, 
        private_key: str = None,
        ipfs_api_url: str = "/ip4/127.0.0.1/tcp/5001"
    ):
        """
        Initialize blockchain interface.
        
        Args:
            provider_url: URL of the Ethereum node provider
            contract_address: Address of the NFT contract
            contract_abi: ABI of the NFT contract
            private_key: Private key for signing transactions
            ipfs_api_url: URL of the IPFS API
        """
        self.provider_url = provider_url
        self.web3 = None
        self.contract = None
        self.account = None
        
        # Connect to Web3 provider
        try:
            self.web3 = Web3(HTTPProvider(provider_url))
            logger.info("Connected to Ethereum node at %s", provider_url)
        except Exception as e:
            logger.warning("Could not connect to Ethereum node: %s", e)
            return
        
        # Set default account if private key is provided
        if private_key:
            try:
                self.account = Account.from_key(private_key)
                self.web3.eth.default_account = self.account.address
                logger.info("Set default account: %s", self.account.address)
            except Exception as e:
                logger.warning("Could not set account from private key: %s", e)
        
        # Initialize contract
        if contract_address and contract_abi:
            self._initialize_contract(contract_address, contract_abi)
            
        # Initialize IPFS client
        self.ipfs = IPFSClient(ipfs_api_url)
    
    def is_connected(self) -> bool:
        """Check if connected to Ethereum node."""
        return self.web3 is not None and self.web3.is_connected()
    
    def _initialize_contract(self, contract_address: str, contract_abi: Union[str, List[Dict[str, Any]]]) -> None:
        """
        Initialize contract instance.
        
        Args:
            contract_address: Address of the NFT contract
            contract_abi: ABI of the NFT contract (JSON string or parsed object)
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to Ethereum node")
        
        try:
            # Parse ABI if it's a string
            if isinstance(contract_abi, str):
                if os.path.exists(contract_abi):
                    with open(contract_abi, "r", encoding="utf-8") as f:
                        abi = json.load(f)
                else:
                    abi = json.loads(contract_abi)
            else:
                abi = contract_abi
                
            # Create contract instance
            self.contract = self.web3.eth.contract(
                address=Web3.to_checksum_address(contract_address),
                abi=abi
            )
            logger.info("Initialized contract at %s", contract_address)
        except Exception as e:
            logger.error("Could not initialize contract: %s", e)
            self.contract = None
    
    def mint_nft(
        self, 
        content_id: str, 
        content_metadata: Dict[str, Any], 
        creator_address: str
    ) -> Dict[str, Any]:
        """
        Mint a new NFT for dream content.
        
        Args:
            content_id: Unique identifier for the dream content
            content_metadata: Metadata for the dream content
            creator_address: Ethereum address of the content creator
            
        Returns:
            Dict with transaction hash, token ID, and other metadata
        """
        if not self.is_connected() or not self.contract:
            raise ConnectionError("Not properly connected to blockchain")
        
        if not self.account:
            raise ValueError("No account set for signing transactions")
            
        try:
            # Prepare metadata for IPFS
            now = datetime.now().isoformat()
            ipfs_metadata = {
                "content_id": content_id,
                "metadata": content_metadata,
                "creator": creator_address,
                "created_at": now,
                "platform": "Dream Content Platform"
            }
            
            # Upload metadata to IPFS
            ipfs_hash = self.ipfs.add_json(ipfs_metadata)
            metadata_uri = f"ipfs://{ipfs_hash}"
            
            # Prepare transaction
            creator_address = Web3.to_checksum_address(creator_address)
            tx = self.contract.functions.mintDreamContent(
                creator_address,
                metadata_uri,
                content_id
            ).build_transaction({
                "from": self.account.address,
                "nonce": self.web3.eth.get_transaction_count(self.account.address),
                "gasPrice": self.web3.eth.gas_price
            })
            
            # Sign and send transaction
            signed_tx = self.web3.eth.account.sign_transaction(tx, self.account.key)
            tx_hash = self.web3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            # Wait for transaction to be mined
            receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
            
            # Get token ID from event logs
            token_id = None
            for log in receipt["logs"]:
                event = self.contract.events.DreamContentMinted().process_log(log)
                if event:
                    token_id = event["args"]["tokenId"]
                    break
            
            if not token_id:
                raise ValueError("Could not determine token ID from transaction receipt")
                
            logger.info("Minted NFT token ID %s for content %s", token_id, content_id)
            
            return {
                "transaction_hash": tx_hash.hex(),
                "token_id": token_id,
                "metadata_uri": metadata_uri,
                "creator": creator_address,
                "status": "success" if receipt["status"] == 1 else "failed",
                "block_number": receipt["blockNumber"],
                "gas_used": receipt["gasUsed"],
                "timestamp": now
            }
            
        except Exception as e:
            logger.error("Error minting NFT: %s", e)
            raise
    
    def verify_ownership(self, token_id: int, address: str) -> bool:
        """
        Verify ownership of a token.
        
        Args:
            token_id: Token ID
            address: Ethereum address to check
            
        Returns:
            True if the address owns the token, False otherwise
        """
        if not self.is_connected() or not self.contract:
            raise ConnectionError("Not properly connected to blockchain")
            
        try:
            owner = self.contract.functions.ownerOf(token_id).call()
            return owner.lower() == Web3.to_checksum_address(address).lower()
        except Exception as e:
            logger.error("Error verifying ownership: %s", e)
            return False
    
    def get_token_metadata(self, token_id: int) -> Dict[str, Any]:
        """
        Get metadata for a token.
        
        Args:
            token_id: Token ID
            
        Returns:
            Token metadata
        """
        if not self.is_connected() or not self.contract:
            raise ConnectionError("Not properly connected to blockchain")
            
        try:
            # Get token URI
            token_uri = self.contract.functions.tokenURI(token_id).call()
            
            # If IPFS URI, fetch metadata
            if token_uri.startswith("ipfs://"):
                ipfs_hash = token_uri.replace("ipfs://", "")
                metadata = self.ipfs.get_json(ipfs_hash)
                return {
                    "token_id": token_id,
                    "token_uri": token_uri,
                    "metadata": metadata
                }
            else:
                return {
                    "token_id": token_id,
                    "token_uri": token_uri,
                    "metadata": {"error": "Non-IPFS URI not supported"}
                }
        except Exception as e:
            logger.error("Error getting token metadata: %s", e)
            raise
    
    def transfer_token(self, token_id: int, from_address: str, to_address: str) -> Dict[str, Any]:
        """
        Transfer a token to a new owner.
        
        Args:
            token_id: Token ID
            from_address: Current owner address
            to_address: New owner address
            
        Returns:
            Transaction details
        """
        if not self.is_connected() or not self.contract:
            raise ConnectionError("Not properly connected to blockchain")
            
        if not self.account:
            raise ValueError("No account set for signing transactions")
            
        try:
            # Check ownership
            if not self.verify_ownership(token_id, from_address):
                raise ValueError(f"Address {from_address} does not own token {token_id}")
            
            # Check if the signing account is the current owner or is approved
            if self.account.address.lower() != from_address.lower():
                is_approved = self.contract.functions.isApprovedForAll(
                    Web3.to_checksum_address(from_address),
                    self.account.address
                ).call()
                
                if not is_approved:
                    raise ValueError("Signing account is not approved to transfer this token")
            
            # Prepare transaction
            from_address = Web3.to_checksum_address(from_address)
            to_address = Web3.to_checksum_address(to_address)
            
            tx = self.contract.functions.safeTransferFrom(
                from_address,
                to_address,
                token_id
            ).build_transaction({
                "from": self.account.address,
                "nonce": self.web3.eth.get_transaction_count(self.account.address),
                "gasPrice": self.web3.eth.gas_price
            })
            
            # Sign and send transaction
            signed_tx = self.web3.eth.account.sign_transaction(tx, self.account.key)
            tx_hash = self.web3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            # Wait for transaction to be mined
            receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
            
            logger.info("Transferred token %s from %s to %s", token_id, from_address, to_address)
            
            return {
                "transaction_hash": tx_hash.hex(),
                "token_id": token_id,
                "from": from_address,
                "to": to_address,
                "status": "success" if receipt["status"] == 1 else "failed",
                "block_number": receipt["blockNumber"],
                "gas_used": receipt["gasUsed"],
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error("Error transferring token: %s", e)
            raise
    
    def get_royalty_info(self, token_id: int, sale_price: int) -> Tuple[str, int]:
        """
        Get royalty information for a token.
        
        Args:
            token_id: Token ID
            sale_price: Sale price in wei
            
        Returns:
            Tuple of (receiver_address, royalty_amount)
        """
        if not self.is_connected() or not self.contract:
            raise ConnectionError("Not properly connected to blockchain")
            
        try:
            result = self.contract.functions.royaltyInfo(token_id, sale_price).call()
            return result
        except Exception as e:
            logger.error("Error getting royalty info: %s", e)
            raise
    
    def get_creator(self, token_id: int) -> str:
        """
        Get the creator of a token.
        
        Args:
            token_id: Token ID
            
        Returns:
            Creator address
        """
        if not self.is_connected() or not self.contract:
            raise ConnectionError("Not properly connected to blockchain")
            
        try:
            result = self.contract.functions.getCreator(token_id).call()
            return result
        except Exception as e:
            logger.error("Error getting creator: %s", e)
            raise
    
    def get_token_by_content_id(self, content_id: str) -> int:
        """
        Get token ID by content ID.
        
        Args:
            content_id: Content ID
            
        Returns:
            Token ID
        """
        if not self.is_connected() or not self.contract:
            raise ConnectionError("Not properly connected to blockchain")
            
        try:
            result = self.contract.functions.getTokenByContentId(content_id).call()
            return result
        except Exception as e:
            logger.error("Error getting token by content ID: %s", e)
            raise
    
    def get_token_count(self) -> int:
        """
        Get total number of tokens.
        
        Returns:
            Total number of tokens
        """
        if not self.is_connected() or not self.contract:
            raise ConnectionError("Not properly connected to blockchain")
            
        try:
            result = self.contract.functions.totalSupply().call()
            return result
        except Exception as e:
            logger.error("Error getting token count: %s", e)
            raise
    
    def get_tokens_for_owner(self, owner_address: str) -> List[int]:
        """
        Get all tokens owned by an address.
        
        Args:
            owner_address: Owner address
            
        Returns:
            List of token IDs
        """
        if not self.is_connected() or not self.contract:
            raise ConnectionError("Not properly connected to blockchain")
            
        try:
            owner_address = Web3.to_checksum_address(owner_address)
            token_count = self.contract.functions.balanceOf(owner_address).call()
            
            tokens = []
            for i in range(token_count):
                token_id = self.contract.functions.tokenOfOwnerByIndex(owner_address, i).call()
                tokens.append(token_id)
                
            return tokens
        except Exception as e:
            logger.error("Error getting tokens for owner: %s", e)
            raise
