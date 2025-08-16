# LLM-Based Tool Selection System

## Overview

The new LabuBank agent uses **ASI:One AI** to intelligently select the most appropriate OpenSea MCP tools for user queries, with a rule-based system as a fallback. This approach provides much more flexible and context-aware tool selection.

## How It Works

### 1. **LLM-Based Selection (Primary)**

- **Input**: User query (e.g., "Show me vitalik.eth's NFT collection")
- **Process**: ASI:One analyzes the query and selects appropriate tools
- **Output**: Structured tool selection with arguments and reasoning

### 2. **Rule-Based Fallback (Secondary)**

- **Trigger**: When LLM selection fails or returns invalid tools
- **Process**: Pattern matching and scoring system
- **Output**: Reliable but less flexible tool selection

### 3. **Validation & Execution**

- **Parameter Validation**: Ensures tool arguments are correct
- **Parallel Execution**: Runs selected tools concurrently
- **Result Aggregation**: Combines data from multiple tools

## Key Benefits

### 🧠 **Intelligence**

- **Complex Queries**: Handles multi-intent requests naturally
- **Context Extraction**: Automatically identifies wallet addresses, timeframes, etc.
- **Natural Language**: Understands variations in how users phrase requests

### 🔄 **Flexibility**

- **New Tools**: Adapts to new OpenSea tools without code changes
- **Query Variations**: Handles different ways of asking the same question
- **Edge Cases**: Better handling of ambiguous or complex requests

### 🚀 **Performance**

- **Caching**: 5-minute TTL cache for repeated queries
- **Parallel Execution**: Multiple tools run simultaneously
- **Fallback Safety**: Rule-based system ensures reliability

## Example Queries

### **Simple Queries**

```
Input: "What are trending NFTs?"
LLM Selection: get_trending_collections (timeframe: ONE_DAY)
Rule Fallback: get_trending_collections (timeframe: ONE_DAY)
```

### **Complex Queries**

```
Input: "Show me vitalik.eth's portfolio and trending collections this week"
LLM Selection:
  - get_profile (address: vitalik.eth)
  - get_trending_collections (timeframe: ONE_WEEK)
Rule Fallback: get_profile (address: vitalik.eth)
```

### **Ambiguous Queries**

```
Input: "What's hot in crypto right now?"
LLM Selection: get_trending_tokens + get_trending_collections
Rule Fallback: search (query: "What's hot in crypto right now?")
```

## Tool Selection Process

### **Step 1: LLM Analysis**

```python
selection_prompt = f"""
You are an expert at selecting the most appropriate OpenSea MCP tools for user queries.

Available tools:
{tools_description}

User query: "{user_query}"

Based on the user's query, select 1-3 most appropriate tools and provide the arguments for each tool.

Rules:
1. For wallet addresses (0x... or .eth), use get_profile, get_token_balances, or get_nft_balances
2. For specific collections (like "Azuki"), use get_collection
3. For trending/popular queries, use get_trending_collections or get_top_collections
4. For token swaps, use get_token_swap_quote
5. For general searches, use search, search_collections, or search_tokens
6. Always extract wallet addresses, collection names, timeframes, etc. from the query

Respond in this exact JSON format:
{{
    "tools": [
        {{
            "name": "tool_name",
            "arguments": {{"param": "value"}},
            "reasoning": "Why this tool is appropriate"
        }}
    ]
}}
"""
```

### **Step 2: Response Parsing**

```python
# Extract JSON from response (handle markdown formatting)
json_start = response.find('{')
json_end = response.rfind('}') + 1
if json_start != -1 and json_end != -1:
    json_str = response[json_start:json_end]
    parsed = json.loads(json_str)

    # Convert to our expected format
    selected_tools = []
    for tool_info in parsed.get("tools", []):
        tool_name = tool_info.get("name")
        arguments = tool_info.get("arguments", {})

        if tool_name in AVAILABLE_OPENSEA_TOOLS:
            # Validate and clean arguments
            cleaned_args = {}
            for key, value in arguments.items():
                if isinstance(value, (str, int, float, bool)):
                    cleaned_args[key] = value

            selected_tools.append((tool_name, cleaned_args))
```

### **Step 3: Validation & Fallback**

```python
if llm_selected_tools:
    # Validate and clean the LLM-selected tools
    validated_tools = []
    for tool_name, tool_args in llm_selected_tools:
        is_valid, error_msg = validate_tool_parameters(tool_name, tool_args)
        if is_valid:
            validated_tools.append((tool_name, tool_args))
            logger.info(f"LLM selected tool {tool_name} with score 95 (LLM selection)")
        else:
            logger.warning(f"LLM tool {tool_name} validation failed: {error_msg}")

    if validated_tools:
        # Cache and return LLM selection
        cache_tool_selection(user_query, validated_tools)
        return validated_tools
    else:
        logger.warning("All LLM-selected tools failed validation, falling back to rule-based selection")
else:
    logger.info("LLM tool selection failed or returned no tools, falling back to rule-based selection")
```

## Testing

### **Run the Demo**

```bash
python demo_llm_selection.py
```

### **Run the Test Suite**

```bash
python simple_test.py
```

### **Expected Log Output**

```
INFO:labubank:Attempting LLM-based tool selection...
INFO:labubank:LLM selected tool get_profile with args {'address': 'vitalik.eth'}
INFO:labubank:LLM selected tool get_trending_collections with args {'timeframe': 'ONE_WEEK'}
INFO:labubank:LLM tool selection successful: ['get_profile', 'get_trending_collections']
```

## Configuration

### **Environment Variables**

```bash
ASI_ONE_API_KEY=your_api_key_here
ASI_ONE_MODEL=asi1-mini  # or asi1-pro for better reasoning
```

### **Cache Settings**

```python
_cache_ttl = 300  # 5 minutes cache TTL
max_tools = 3     # Maximum tools per query
```

## Fallback Scenarios

### **When LLM Selection Fails**

1. **API Errors**: Network issues, rate limits, etc.
2. **Invalid Responses**: Malformed JSON, unknown tools
3. **Validation Failures**: Missing required parameters
4. **Timeout Issues**: LLM response too slow

### **Rule-Based Fallback**

- **Pattern Matching**: Regex-based keyword detection
- **Scoring System**: Weighted tool selection
- **Parameter Extraction**: Basic context parsing
- **Reliability**: Always provides a fallback option

## Future Enhancements

### **Planned Improvements**

- **Tool Learning**: Learn from successful tool selections
- **Query Classification**: Categorize queries for better tool matching
- **Performance Metrics**: Track tool selection accuracy
- **Dynamic Prompts**: Adapt prompts based on query patterns

### **Advanced Features**

- **Multi-Modal Queries**: Handle image + text queries
- **Conversation Context**: Remember previous queries in a session
- **Tool Chaining**: Automatically chain related tools
- **Personalization**: Learn user preferences over time

## Conclusion

The LLM-based tool selection system represents a significant improvement over the rule-based approach:

- **More Intelligent**: Understands complex, natural language queries
- **More Flexible**: Adapts to new tools and query patterns
- **More Reliable**: Fallback system ensures robustness
- **Better UX**: Users can ask questions naturally without learning specific syntax

This hybrid approach combines the intelligence of AI with the reliability of rule-based systems, providing the best of both worlds for OpenSea tool selection.
