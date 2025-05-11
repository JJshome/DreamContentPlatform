// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC721/extensions/ERC721URIStorage.sol";
import "@openzeppelin/contracts/token/ERC721/extensions/ERC721Enumerable.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/Counters.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";

/**
 * @title DreamContentNFT
 * @dev Smart contract for minting and trading dream-based content as NFTs
 */
contract DreamContentNFT is ERC721URIStorage, ERC721Enumerable, Ownable, ReentrancyGuard {
    using Counters for Counters.Counter;
    Counters.Counter private _tokenIds;
    
    // Events
    event DreamContentCreated(uint256 indexed tokenId, address indexed creator, string contentType);
    event DreamContentSold(uint256 indexed tokenId, address indexed seller, address indexed buyer, uint256 price);
    event RoyaltyPaid(uint256 indexed tokenId, address indexed creator, uint256 amount);

    // Structs
    struct DreamContent {
        string contentType;         // "image", "narrative", "music", "mixed"
        address creator;            // Original creator address
        uint256 royaltyBasisPoints; // Royalty percentage in basis points (100 = 1%)
        uint256 createdAt;          // Timestamp of creation
    }
    
    // Mappings
    mapping(uint256 => DreamContent) public dreamContents;
    mapping(uint256 => uint256) public tokenPrices;  // For marketplace functionality
    mapping(uint256 => bool) public tokenForSale;    // Whether a token is currently for sale

    // Platform fee (in basis points, e.g., 250 = 2.5%)
    uint256 public platformFeeBasisPoints = 250;
    address public platformFeeRecipient;
    
    // Constructor
    constructor(address _platformFeeRecipient) ERC721("DreamContent", "DREAM") {
        platformFeeRecipient = _platformFeeRecipient;
    }
    
    /**
     * @dev Override _beforeTokenTransfer to handle both ERC721Enumerable and ERC721URIStorage
     */
    function _beforeTokenTransfer(
        address from,
        address to,
        uint256 tokenId, 
        uint256 batchSize
    ) internal override(ERC721, ERC721Enumerable) {
        super._beforeTokenTransfer(from, to, tokenId, batchSize);
    }
    
    /**
     * @dev Override burn to handle both ERC721Enumerable and ERC721URIStorage
     */
    function _burn(uint256 tokenId) internal override(ERC721, ERC721URIStorage) {
        super._burn(tokenId);
    }
    
    /**
     * @dev Override tokenURI to handle both ERC721Enumerable and ERC721URIStorage
     */
    function tokenURI(uint256 tokenId) public view override(ERC721, ERC721URIStorage) returns (string memory) {
        return super.tokenURI(tokenId);
    }
    
    /**
     * @dev Override supportsInterface to handle both ERC721Enumerable and ERC721URIStorage
     */
    function supportsInterface(bytes4 interfaceId) public view override(ERC721, ERC721Enumerable) returns (bool) {
        return super.supportsInterface(interfaceId);
    }
    
    /**
     * @dev Mint a new Dream Content NFT
     */
    function mintDreamContent(
        address to,
        string memory _tokenURI,
        string memory contentType,
        uint256 royaltyBasisPoints
    ) public returns (uint256) {
        require(royaltyBasisPoints <= 1000, "Royalty cannot exceed 10%");
        
        _tokenIds.increment();
        uint256 newTokenId = _tokenIds.current();
        
        _safeMint(to, newTokenId);
        _setTokenURI(newTokenId, _tokenURI);
        
        dreamContents[newTokenId] = DreamContent({
            contentType: contentType,
            creator: msg.sender,
            royaltyBasisPoints: royaltyBasisPoints,
            createdAt: block.timestamp
        });
        
        emit DreamContentCreated(newTokenId, msg.sender, contentType);
        
        return newTokenId;
    }
    
    /**
     * @dev Set token for sale in the marketplace
     */
    function setTokenForSale(uint256 tokenId, uint256 price) public {
        require(ownerOf(tokenId) == msg.sender, "Not the token owner");
        require(price > 0, "Price must be greater than 0");
        
        tokenPrices[tokenId] = price;
        tokenForSale[tokenId] = true;
    }
    
    /**
     * @dev Remove token from sale
     */
    function removeTokenFromSale(uint256 tokenId) public {
        require(ownerOf(tokenId) == msg.sender, "Not the token owner");
        tokenForSale[tokenId] = false;
    }
    
    /**
     * @dev Buy a token that is for sale
     */
    function buyToken(uint256 tokenId) public payable nonReentrant {
        require(tokenForSale[tokenId], "Token not for sale");
        require(msg.value >= tokenPrices[tokenId], "Insufficient payment");
        
        address seller = ownerOf(tokenId);
        require(seller != msg.sender, "Cannot buy your own token");
        
        // Calculate royalty for original creator
        DreamContent memory content = dreamContents[tokenId];
        uint256 royaltyAmount = (msg.value * content.royaltyBasisPoints) / 10000;
        
        // Calculate platform fee
        uint256 platformFee = (msg.value * platformFeeBasisPoints) / 10000;
        
        // Calculate amount for seller
        uint256 sellerAmount = msg.value - royaltyAmount - platformFee;
        
        // Transfer ownership
        _transfer(seller, msg.sender, tokenId);
        
        // Mark as not for sale
        tokenForSale[tokenId] = false;
        
        // Send royalty to creator
        if (royaltyAmount > 0 && content.creator != seller) {
            (bool royaltySuccess, ) = payable(content.creator).call{value: royaltyAmount}("");
            require(royaltySuccess, "Failed to send royalty");
            emit RoyaltyPaid(tokenId, content.creator, royaltyAmount);
        } else {
            // If seller is creator, add royalty to seller amount
            sellerAmount += royaltyAmount;
        }
        
        // Send platform fee
        (bool platformSuccess, ) = payable(platformFeeRecipient).call{value: platformFee}("");
        require(platformSuccess, "Failed to send platform fee");
        
        // Send payment to seller
        (bool sellerSuccess, ) = payable(seller).call{value: sellerAmount}("");
        require(sellerSuccess, "Failed to send payment to seller");
        
        emit DreamContentSold(tokenId, seller, msg.sender, msg.value);
    }
    
    /**
     * @dev Set platform fee percentage (only owner)
     */
    function setPlatformFeeBasisPoints(uint256 newFeeBasisPoints) public onlyOwner {
        require(newFeeBasisPoints <= 1000, "Fee too high");
        platformFeeBasisPoints = newFeeBasisPoints;
    }
    
    /**
     * @dev Set platform fee recipient (only owner)
     */
    function setPlatformFeeRecipient(address newRecipient) public onlyOwner {
        require(newRecipient != address(0), "Invalid address");
        platformFeeRecipient = newRecipient;
    }
    
    /**
     * @dev Get token details including price and sale status
     */
    function getTokenDetails(uint256 tokenId) public view returns (
        string memory contentType,
        address creator,
        uint256 royaltyBasisPoints,
        uint256 createdAt,
        uint256 price,
        bool forSale
    ) {
        require(_exists(tokenId), "Token does not exist");
        
        DreamContent memory content = dreamContents[tokenId];
        return (
            content.contentType,
            content.creator,
            content.royaltyBasisPoints,
            content.createdAt,
            tokenPrices[tokenId],
            tokenForSale[tokenId]
        );
    }
    
    /**
     * @dev Get royalty information for a token
     */
    function getRoyaltyInfo(uint256 tokenId, uint256 salePrice) external view returns (address, uint256) {
        require(_exists(tokenId), "Token does not exist");
        
        DreamContent memory content = dreamContents[tokenId];
        uint256 royaltyAmount = (salePrice * content.royaltyBasisPoints) / 10000;
        
        return (content.creator, royaltyAmount);
    }
    
    /**
     * @dev Get all tokens owned by an address
     */
    function getTokensOwnedBy(address owner) public view returns (uint256[] memory) {
        uint256 tokenCount = balanceOf(owner);
        if (tokenCount == 0) {
            return new uint256[](0);
        }
        
        uint256[] memory tokenIds = new uint256[](tokenCount);
        for (uint256 i = 0; i < tokenCount; i++) {
            tokenIds[i] = tokenOfOwnerByIndex(owner, i);
        }
        
        return tokenIds;
    }
    
    /**
     * @dev Get all tokens created by an address
     */
    function getTokensCreatedBy(address creator) public view returns (uint256[] memory) {
        uint256 totalTokens = _tokenIds.current();
        uint256 createdCount = 0;
        
        // First pass: count tokens created by the address
        for (uint256 i = 1; i <= totalTokens; i++) {
            if (dreamContents[i].creator == creator) {
                createdCount++;
            }
        }
        
        // Second pass: collect tokens created by the address
        uint256[] memory createdTokens = new uint256[](createdCount);
        uint256 index = 0;
        
        for (uint256 i = 1; i <= totalTokens; i++) {
            if (dreamContents[i].creator == creator) {
                createdTokens[index] = i;
                index++;
            }
        }
        
        return createdTokens;
    }
    
    /**
     * @dev Get all tokens for sale
     */
    function getTokensForSale() public view returns (uint256[] memory) {
        uint256 totalTokens = _tokenIds.current();
        uint256 forSaleCount = 0;
        
        // First pass: count tokens for sale
        for (uint256 i = 1; i <= totalTokens; i++) {
            if (tokenForSale[i] && _exists(i)) {
                forSaleCount++;
            }
        }
        
        // Second pass: collect tokens for sale
        uint256[] memory forSaleTokens = new uint256[](forSaleCount);
        uint256 index = 0;
        
        for (uint256 i = 1; i <= totalTokens; i++) {
            if (tokenForSale[i] && _exists(i)) {
                forSaleTokens[index] = i;
                index++;
            }
        }
        
        return forSaleTokens;
    }
}
