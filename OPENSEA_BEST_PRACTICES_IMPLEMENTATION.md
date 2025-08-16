# OpenSea Best Practices Implementation

This document outlines how the LabuBank agent has been updated to follow OpenSea's recommended best practices for optimal API usage and data retrieval.

## 🎯 Implemented Best Practices

### 1. **Use Natural Language**

- **Implementation**: Enhanced LLM-based tool selection that understands natural language queries
- **Example**: "Show me trending collections on Polygon this week" automatically selects appropriate tools
- **Benefits**: More intuitive user experience, better query understanding

### 2. **Combine Tools**

- **Implementation**: Intelligent tool selection that combines multiple OpenSea MCP tools for comprehensive insights
- **Example**: Portfolio queries automatically use `get_profile`, `get_token_balances`, and `get_nft_balances` together
- **Benefits**: Richer data, more complete analysis, better user insights

### 3. **Specify Chains**

- **Implementation**: Automatic chain detection and specification in all relevant tool calls
- **Supported Chains**: Ethereum, Polygon, Solana, Arbitrum, Optimism, Base, Zora, Avalanche, Binance
- **Example**: "Show me top collections on Polygon" automatically adds `chain: "polygon"`
- **Benefits**: Targeted blockchain analysis, cross-chain comparison capabilities

### 4. **Check Balances First**

- **Implementation**: New `check_token_balance_before_swap()` function that verifies wallet balances before swap quotes
- **Example**: Before getting a swap quote, the agent checks if the wallet has sufficient tokens
- **Benefits**: Prevents failed swaps, better user experience, follows security best practices

### 5. **Use Collection Slugs**

- **Implementation**: Comprehensive collection slug mapping and automatic conversion
- **Examples**:
  - "Azuki" → "azuki"
  - "BAYC" → "boredapeyachtclub"
  - "CryptoPunks" → "cryptopunks"
- **Benefits**: Accurate collection identification, consistent with OpenSea's slug system

### 6. **Leverage Includes Parameters**

- **Implementation**: Automatic inclusion of relevant data types based on query context
- **Common Includes**:
  - Portfolio queries: `["items", "collections", "activity", "listings", "offers", "balances", "favorites"]`
  - Collection analysis: `["stats", "floor_price", "volume"]`
  - Token analysis: `["prices", "volumes", "market_cap"]`
  - NFT analysis: `["traits", "listings", "offers"]`
- **Benefits**: Richer data, comprehensive insights, better user experience

### 7. **Specify Amounts Correctly**

- **Implementation**: Native unit handling (ETH not wei) in swap quotes and balance checks
- **Example**: Uses "1.5" for 1.5 ETH instead of wei values
- **Benefits**: User-friendly amounts, prevents conversion errors, follows OpenSea specifications

## 🔧 Technical Implementation Details

### Enhanced Tool Selection System

The agent now uses a sophisticated two-tier tool selection system:

1. **LLM-Based Selection**: Uses ASI:One to intelligently select tools based on natural language queries
2. **Rule-Based Fallback**: Comprehensive fallback system with enhanced pattern matching

### Parameter Validation

Enhanced validation now includes:

- Chain parameter validation with supported blockchain list
- Includes parameter validation with allowed data types
- Collection slug format validation
- Wallet address format validation
- Timeframe and sortBy validation

### Portfolio Data Enhancement

The portfolio endpoint now:

- Fetches data from multiple tools in parallel
- Uses comprehensive includes parameters
- Provides rich, actionable insights
- Follows OpenSea's data structure recommendations

## 📊 Example Queries and Tool Selection

### Market Research

```
Query: "What are the top gaming NFT collections by volume on Polygon?"
Tools: get_top_collections, search_collections
Parameters:
  - sortBy: "ONE_DAY_VOLUME"
  - chain: "polygon"
  - includes: ["stats", "floor_prices"]
```

### Portfolio Analysis

```
Query: "Show me all NFTs and tokens owned by wallet 0x123..."
Tools: get_profile, get_token_balances, get_nft_balances
Parameters:
  - address: "0x123..."
  - includes: ["items", "collections", "balances", "activity"]
```

### Collection Analysis

```
Query: "What's the Azuki collection floor price on Ethereum?"
Tools: get_collection
Parameters:
  - collection: "azuki"
  - chain: "ethereum"
  - includes: ["stats", "floor_price", "volume"]
```

### Trending Analysis

```
Query: "Show me trending collections on Solana this week"
Tools: get_trending_collections
Parameters:
  - timeframe: "SEVEN_DAYS"
  - chain: "solana"
  - includes: ["stats", "floor_prices"]
```

## 🧪 Testing

Use the `test_opensea_best_practices.py` script to verify all implementations:

```bash
python test_opensea_best_practices.py
```

This script tests:

- Natural language processing
- Chain specification
- Collection slug usage
- Includes parameters
- Multi-tool combination
- Portfolio analysis
- Market research
- Token discovery

## 🚀 Benefits of Implementation

1. **Better User Experience**: Natural language queries work seamlessly
2. **Comprehensive Data**: Multiple tools combined for rich insights
3. **Accurate Results**: Proper chain specification and collection slugs
4. **Efficient Queries**: Includes parameters provide relevant data upfront
5. **Security**: Balance checking prevents failed operations
6. **Consistency**: Follows OpenSea's recommended patterns
7. **Scalability**: Supports multiple blockchains and data types

## 📝 Usage Examples

### For Developers

```python
# The agent automatically handles:
# - Chain detection from natural language
# - Collection slug conversion
# - Appropriate includes parameters
# - Tool combination for comprehensive data
```

### For Users

```
Natural queries like:
- "Show me trending collections on Polygon this week"
- "What's the Azuki floor price on Ethereum?"
- "Check my portfolio on Arbitrum"
- "Compare top gaming NFTs across chains"
```

## 🔮 Future Enhancements

Planned improvements include:

- Enhanced balance checking for complex swap scenarios
- More sophisticated chain detection patterns
- Additional includes parameter combinations
- Performance optimization for parallel tool execution
- Extended collection slug database

---

The LabuBank agent now fully implements OpenSea's best practices, providing users with a powerful, intuitive, and comprehensive crypto portfolio analysis tool that follows industry standards and delivers exceptional user experiences.
