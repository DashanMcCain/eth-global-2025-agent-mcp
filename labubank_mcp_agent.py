import os
import json
import uuid
import asyncio
import logging
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from uagents import Agent, Context, Model
import httpx
from fastapi import FastAPI
import uvicorn
from dotenv import load_dotenv
import re

# Load environment variables from .env file
load_dotenv()

# ----------------------------
# Configuration & Environment
# ----------------------------

ASI_ONE_URL_DEFAULT = "https://api.asi1.ai/v1/chat/completions"
MCP_PROTOCOL_VERSION = "2025-06-18"  # per current MCP spec
DEFAULT_MCP_ORIGIN = "https://labubank.local"  # helps some servers with Origin checks

def env(name: str, required: bool = True, default: Optional[str] = None) -> str:
    val = os.getenv(name, default)
    if required and not val:
        raise RuntimeError(f"Missing required env var: {name}")
    return val or ""

@dataclass(frozen=True)
class Settings:
    # HTTP
    request_timeout: float = 25.0
    max_retries: int = 3
    retry_backoff_base: float = 0.5  # seconds

    # ASI:One
    asi_url: str = env("ASI_ONE_URL", required=False, default=ASI_ONE_URL_DEFAULT)
    asi_api_key: str = env("ASI_ONE_API_KEY", required=True)
    asi_model: str = env("ASI_ONE_MODEL", required=False, default="asi1-mini")
    asi_max_tokens: int = int(os.getenv("ASI_ONE_MAX_TOKENS", "512"))
    asi_temperature: float = float(os.getenv("ASI_ONE_TEMPERATURE", "0.2"))

    # MCP (OpenSea)
    mcp_api_key: Optional[str] = os.getenv("OPENSEA_API_KEY")
    mcp_url: str = env("OPENSEA_MCP_URL", required=False, default="https://mcp.opensea.io/mcp")
    mcp_origin: str = os.getenv("MCP_ORIGIN", DEFAULT_MCP_ORIGIN)

    # Agent
    agent_name: str = os.getenv("AGENT_NAME", "LabuBank")
    agent_seed: str = os.getenv("AGENT_SEED", "labubank_mcp_agent")
    agent_port: int = int(os.getenv("AGENT_PORT", "8000"))
    agent_endpoint: str = os.getenv("AGENT_ENDPOINT", "http://localhost:8000/query")

SETTINGS = Settings()

# ----------------------------
# Logging (prod-friendly)
# ----------------------------

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("labubank")

# ----------------------------
# HTTP helpers with backoff
# ----------------------------

class HttpClient:
    def __init__(self, timeout: float, max_retries: int, backoff_base: float):
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self._client = httpx.AsyncClient(timeout=timeout)

    async def post_json(
        self,
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        expect_stream: bool = False,
    ) -> httpx.Response:
        # Simple exponential backoff on 5xx & connection errors
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = await self._client.post(url, headers=headers, json=payload)
                if resp.status_code >= 500:
                    raise httpx.HTTPError(f"Server error {resp.status_code}")
                return resp
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.HTTPError) as e:
                if attempt == self.max_retries:
                    raise
                sleep_for = self.backoff_base * (2 ** (attempt - 1))
                logger.warning(f"POST {url} failed ({e}); retrying in {sleep_for:.1f}s...")
                await asyncio.sleep(sleep_for)

    async def aclose(self):
        await self._client.aclose()

HTTP = HttpClient(SETTINGS.request_timeout, SETTINGS.max_retries, SETTINGS.retry_backoff_base)

# ----------------------------
# ASI:One client
# ----------------------------

class ASIOneClient:
    def __init__(self, url: str, api_key: str, model: str, max_tokens: int, temperature: float):
        self.url = url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    async def chat(self, messages: List[Dict[str, str]]) -> str:
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        resp = await HTTP.post_json(self.url, headers=self.headers, payload=body)
        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"ASI response parse error: {e}; body={data}")
            raise RuntimeError("Failed to parse ASI:One response")

ASI = ASIOneClient(
    SETTINGS.asi_url,
    SETTINGS.asi_api_key,
    SETTINGS.asi_model,
    SETTINGS.asi_max_tokens,
    SETTINGS.asi_temperature,
)

# ----------------------------
# MCP (OpenSea) client
# ----------------------------

class MCPClient:
    """
    Minimal-complete MCP HTTP client:
    - initialize() handshake
    - list_tools()
    - call_tool(name, arguments)
    Supports JSON responses and server-sent events (SSE) per spec.
    """

    def __init__(self, endpoint: str, api_key: Optional[str], origin: str):
        self.endpoint = endpoint
        self.api_key = api_key
        self.origin = origin
        self.session_id: Optional[str] = None
        self.protocol_version: str = MCP_PROTOCOL_VERSION

    def _base_headers(self) -> Dict[str, str]:
        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            "Origin": self.origin,
        }
        # Only add session ID after initialization
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        # Recommended by spec post-init:
        headers["MCP-Protocol-Version"] = self.protocol_version
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def initialize(self) -> Dict[str, Any]:
        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "initialize",
            "params": {
                "protocolVersion": self.protocol_version,
                "capabilities": {
                    # request server features you can handle
                    "tools": {"listChanged": True},
                },
                "clientInfo": {"name": "LabuBank-uAgent", "version": "1.0.0"},
            },
        }
        resp = await HTTP.post_json(self.endpoint, headers=self._base_headers(), payload=payload)
        # capture session header if present
        sid = resp.headers.get("Mcp-Session-Id")
        if sid:
            self.session_id = sid
        return sid

    async def list_tools(self) -> List[Dict[str, Any]]:
        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "tools/list",
            "params": {},
        }
        resp = await HTTP.post_json(self.endpoint, headers=self._base_headers(), payload=payload)
        if resp.headers.get("Content-Type", "").startswith("text/event-stream"):
            # If server streams, collect response message
            return await self._read_sse_for_result(resp, result_path=("tools",))
        data = resp.json()
        if "error" in data:
            raise RuntimeError(f"MCP tools/list error: {data['error']}")
        return data.get("result", {}).get("tools", [])

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "jsonrpc": "2.0",
            "id": (req_id := str(uuid.uuid4())),
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        }
        resp = await HTTP.post_json(self.endpoint, headers=self._base_headers(), payload=payload)
        if resp.headers.get("Content-Type", "").startswith("text/event-stream"):
            return await self._read_sse_for_result(resp)
        data = resp.json()
        if "error" in data:
            raise RuntimeError(f"MCP tools/call protocol error: {data['error']}")
        return data.get("result", {})

    async def _read_sse_for_result(
        self,
        response: httpx.Response,
        result_path: Optional[Tuple[str, ...]] = None,
    ) -> Any:
        """
        Read SSE stream until a JSON-RPC response with 'result' arrives.
        """
        # httpx requires streaming iter_lines via .aiter_lines()
        async for line in response.aiter_lines():
            if not line or not line.startswith("data:"):
                continue
            try:
                chunk = json.loads(line.split("data:", 1)[1].strip())
                if "result" in chunk or "error" in chunk:
                    if "error" in chunk:
                        raise RuntimeError(f"MCP SSE protocol error: {chunk['error']}")
                    res = chunk["result"]
                    if result_path:
                        for key in result_path:
                            res = res.get(key, {})
                    return res
            except json.JSONDecodeError:
                continue
        raise RuntimeError("MCP SSE ended without a result.")

MCP = MCPClient(SETTINGS.mcp_url, SETTINGS.mcp_api_key, SETTINGS.mcp_origin)
_mcp_initialized = False  # Track MCP initialization status

# ----------------------------
# uAgents message schemas
# ----------------------------

class UserQuery(Model):
    prompt: str

class AgentReply(Model):
    answer: str
    used_tools: Optional[List[str]] = None
    tool_results: Optional[Dict[str, Any]] = None

class RestRequest(Model):
    prompt: str

class PortfolioRequest(Model):
    wallet_address: str
    include_items: bool = True
    include_collections: bool = True
    include_activity: bool = False
    include_listings: bool = False
    include_offers: bool = False
    include_balances: bool = True
    include_favorites: bool = False

class RestResponse(Model):
    timestamp: int
    answer: str
    used_tools: Optional[List[str]] = None
    tool_results: Optional[Dict[str, Any]] = None
    agent_address: str

class PortfolioResponse(Model):
    timestamp: int
    wallet_address: str
    portfolio_data: Dict[str, Any]
    summary: str
    agent_address: str

class HealthResponse(Model):
    status: str
    timestamp: int
    agent_address: str
    agent_name: str
    version: str
    services: Dict[str, str]
    uptime: str
    error: Optional[str] = None

# ----------------------------
# Orchestration
# ----------------------------

SYSTEM_PROMPT = (
    "You are LabuBank, a helpful crypto portfolio companion. "
    "If context about NFTs or collections is provided, use it. "
    "Be concise, accurate, and avoid speculation."
)

# Cache for tool selection results (TTL: 5 minutes)
_tool_selection_cache = {}
_cache_ttl = 300  # 5 minutes

# Available OpenSea MCP tools for LLM selection
AVAILABLE_OPENSEA_TOOLS = {
    "search": "AI-powered search across OpenSea marketplace data",
    "fetch": "Retrieve full details of a specific OpenSea entity by its unique identifier",
    "search_collections": "Search for NFT collections by name, description, or metadata",
    "get_collection": "Get detailed information about a specific NFT collection",
    "search_items": "Search for individual NFT items/tokens across OpenSea",
    "get_item": "Get detailed information about a specific NFT",
    "search_tokens": "Search for cryptocurrencies and tokens by name or symbol",
    "get_token": "Get information about a specific cryptocurrency",
    "get_token_swap_quote": "Get a swap quote and blockchain actions needed to perform a token swap",
    "get_token_balances": "Retrieve all token balances and points for a wallet address",
    "get_token_balance": "Get the balance of a specific token for a wallet address",
    "get_nft_balances": "Retrieve all NFTs owned by a wallet address with metadata and current listings",
    "get_top_collections": "Get top NFT collections by volume, floor price, sales count, and other metrics",
    "get_trending_collections": "Get trending NFT collections based on recent trading activity",
    "get_top_tokens": "Get top cryptocurrencies and tokens sorted by daily volume",
    "get_trending_tokens": "Get trending cryptocurrencies sorted by daily price change",
    "get_profile": "Get comprehensive profile information for a wallet address"
}

def get_cached_tool_selection(query: str) -> Optional[List[Tuple[str, Dict[str, Any]]]]:
    """Get cached tool selection result if still valid"""
    if query in _tool_selection_cache:
        timestamp, result = _tool_selection_cache[query]
        if time.time() - timestamp < _cache_ttl:
            return result
        else:
            del _tool_selection_cache[query]
    return None

def cache_tool_selection(query: str, result: List[Tuple[str, Dict[str, Any]]]):
    """Cache tool selection result with timestamp"""
    _tool_selection_cache[query] = (time.time(), result)

async def select_tools_with_llm(user_query: str) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Use ASI:One to intelligently select the best OpenSea tools for a user query.
    Returns a list of (tool_name, tool_args) tuples.
    """
    try:
        # Create a prompt for tool selection with exact parameter requirements
        tools_description = "\n".join([f"- {name}: {desc}" for name, desc in AVAILABLE_OPENSEA_TOOLS.items()])
        
        selection_prompt = f"""You are an expert at selecting the most appropriate OpenSea MCP tools for user queries.

Available tools:
{tools_description}

User query: "{user_query}"

Based on the user's query, select 1-3 most appropriate tools and provide the EXACT arguments for each tool.

CRITICAL PARAMETER REQUIREMENTS:
1. get_trending_collections: REQUIRES timeframe parameter with values: "ONE_HOUR", "ONE_DAY", "SEVEN_DAYS", or "THIRTY_DAYS"
2. get_top_collections: REQUIRES sortBy parameter with values: "FLOOR_PRICE", "ONE_DAY_VOLUME", "ONE_DAY_SALES", "ONE_DAY_FLOOR_PRICE_CHANGE", "VOLUME", or "SALES"
3. get_token_swap_quote: REQUIRES fromContractAddress, fromChain, toContractAddress, toChain, fromQuantity, and address parameters
   - For ETH swaps, use: fromContractAddress="0x0000000000000000000000000000000000000000", fromChain="ethereum"
   - For USDC swaps, use: toContractAddress="0xA0b86a33E6441b8c4C8C0C0C0C0C0C0C0C0C0C0C0", toChain="ethereum"
   - For common tokens, provide realistic contract addresses
4. For wallet addresses, use the exact format: 0x... (42 chars) or .eth format
5. For time references, convert to exact values:
   - "this week" → "SEVEN_DAYS"
   - "this month" → "THIRTY_DAYS"
   - "today" → "ONE_DAY"
   - "this hour" → "ONE_HOUR"

Rules:
1. For wallet addresses (0x... or .eth), use get_profile, get_token_balances, or get_nft_balances
2. For specific collections (like "Azuki"), use get_collection
3. For trending/popular queries, use get_trending_collections or get_top_collections
4. For token swaps, use get_token_swap_quote (but only if you have all required parameters)
5. For general searches, use search, search_collections, or search_tokens
6. Always extract wallet addresses, collection names, timeframes, etc. from the query

IMPORTANT: If you cannot provide ALL required parameters for a tool, DO NOT select that tool. Use a simpler alternative instead.

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

Only include tools that are directly relevant to the query. If no specific tools are needed, use "search" with the full query."""
        
        # Get LLM response
        response = await ASI.chat([
            {"role": "system", "content": "You are a tool selection expert. Always respond with valid JSON and provide ALL required parameters for each tool. If you cannot provide all required parameters, choose a simpler tool."},
            {"role": "user", "content": selection_prompt}
        ])
        
        # Parse the response
        try:
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
                        # Clean and validate arguments
                        cleaned_args = {}
                        for key, value in arguments.items():
                            if isinstance(value, (str, int, float, bool)):
                                # Convert time references to exact values
                                if key == "timeframe" and isinstance(value, str):
                                    time_mapping = {
                                        "weekly": "SEVEN_DAYS",
                                        "monthly": "THIRTY_DAYS",
                                        "daily": "ONE_DAY",
                                        "hourly": "ONE_HOUR",
                                        "week": "SEVEN_DAYS",
                                        "month": "THIRTY_DAYS",
                                        "day": "ONE_DAY",
                                        "hour": "ONE_HOUR"
                                    }
                                    cleaned_args[key] = time_mapping.get(value.lower(), value)
                                else:
                                    cleaned_args[key] = value
                        
                        # Special handling for swap quote tool - if missing parameters, skip it
                        if tool_name == "get_token_swap_quote":
                            required_params = ["fromContractAddress", "fromChain", "toContractAddress", "toChain", "fromQuantity", "address"]
                            if not all(param in cleaned_args for param in required_params):
                                logger.warning(f"Skipping {tool_name} due to missing required parameters")
                                continue
                        
                        selected_tools.append((tool_name, cleaned_args))
                        logger.info(f"LLM selected tool {tool_name} with args {cleaned_args}")
                    else:
                        logger.warning(f"LLM suggested unknown tool: {tool_name}")
                
                if selected_tools:
                    logger.info(f"LLM successfully selected {len(selected_tools)} tools")
                    return selected_tools
                else:
                    logger.warning("LLM didn't select any valid tools")
                    return []
                    
            else:
                logger.warning("No JSON found in LLM response")
                return []
                
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM JSON response: {e}")
            logger.debug(f"Raw response: {response}")
            return []
            
    except Exception as e:
        logger.error(f"LLM tool selection failed: {e}")
        return []

def validate_tool_parameters(tool_name: str, tool_args: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate tool parameters before making MCP calls.
    Returns (is_valid, error_message).
    """
    required_params = {
        "get_collection": ["collection"],
        "get_token": ["token"],
        "get_token_balance": ["address", "token"],
        "get_token_balances": ["address"],
        "get_nft_balances": ["address"],
        "get_profile": ["address"],
        "search_collections": ["query"],
        "search_items": ["query"],
        "search_tokens": ["query"],
        "search": ["query"],
        "get_trending_collections": ["timeframe"],
        "get_top_collections": ["sortBy"],
        "get_token_swap_quote": ["fromContractAddress", "fromChain", "toContractAddress", "toChain", "fromQuantity", "address"]
    }
    
    if tool_name in required_params:
        missing_params = [param for param in required_params[tool_name] if param not in tool_args]
        if missing_params:
            return False, f"Missing required parameters for {tool_name}: {missing_params}"
    
    # Validate wallet address format if present
    if "address" in tool_args:
        address = tool_args["address"]
        if not (address.startswith("0x") and len(address) == 42) and not address.endswith(".eth"):
            return False, f"Invalid wallet address format: {address}"
    
    # Validate timeframes for trending tools
    if tool_name in ["get_trending_collections"] and "timeframe" in tool_args:
        valid_timeframes = ["ONE_HOUR", "ONE_DAY", "SEVEN_DAYS", "THIRTY_DAYS"]
        if tool_args["timeframe"] not in valid_timeframes:
            return False, f"Invalid timeframe for {tool_name}: {tool_args['timeframe']}. Valid: {valid_timeframes}"
    
    # Validate sortBy for top collections
    if tool_name in ["get_top_collections"] and "sortBy" in tool_args:
        valid_sort_by = ["FLOOR_PRICE", "ONE_DAY_VOLUME", "ONE_DAY_SALES", "ONE_DAY_FLOOR_PRICE_CHANGE", "VOLUME", "SALES"]
        if tool_args["sortBy"] not in valid_sort_by:
            return False, f"Invalid sortBy for {tool_name}: {tool_args['sortBy']}. Valid: {valid_sort_by}"
    
    return True, ""

async def select_appropriate_tools(user_query: str) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Intelligently select appropriate OpenSea MCP tools based on user query.
    Uses LLM-based selection first, with rule-based fallback for better intelligence.
    Returns a list of (tool_name, tool_args) tuples.
    """
    # Check cache first
    cached_result = get_cached_tool_selection(user_query)
    if cached_result:
        logger.info(f"Using cached tool selection for query: {user_query[:50]}...")
        return cached_result
    
    # Try LLM-based tool selection first
    logger.info("Attempting LLM-based tool selection...")
    llm_selected_tools = await select_tools_with_llm(user_query)
    
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
            logger.info(f"LLM tool selection successful: {[tool[0] for tool in validated_tools]}")
            return validated_tools
        else:
            logger.warning("All LLM-selected tools failed validation, falling back to rule-based selection")
    else:
        logger.info("LLM tool selection failed or returned no tools, falling back to rule-based selection")
    
    # Fallback to rule-based selection
    logger.info("Using rule-based tool selection fallback...")
    user_query_lower = user_query.lower()
    selected_tools = []
    
    # Enhanced patterns with better coverage
    wallet_pattern = re.compile(r'0x[a-fA-F0-9]{40}|[a-zA-Z0-9]+\.eth')
    wallet_matches = wallet_pattern.findall(user_query)
    
    # Expanded token/collection patterns
    token_collection_pattern = re.compile(
        r'\b(?:azuki|bayc|boredapeyachtclub|cryptopunks|doodles|pudgy|penguins|'
        r'bonk|pepe|shib|usdc|usdt|weth|dai|matic|link|uni|aave|comp|mkr|'
        r'ens|unstoppable|opensea|blur|looksrare|rarible|manifold|zora)\b', 
        re.IGNORECASE
    )
    token_collection_matches = token_collection_pattern.findall(user_query)
    
    # Time-related patterns for trending queries
    time_patterns = {
        'ONE_HOUR': r'\b(?:hour|1h|1 hour|last hour)\b',
        'ONE_DAY': r'\b(?:day|24h|24 hour|today|yesterday|daily)\b',
        'SEVEN_DAYS': r'\b(?:week|7d|7 day|weekly|this week|last week)\b',
        'THIRTY_DAYS': r'\b(?:month|30d|30 day|monthly|this month|last month)\b'
    }
    
    # Extract time context
    time_context = None
    for timeframe, pattern in time_patterns.items():
        if re.search(pattern, user_query_lower):
            time_context = timeframe
            break
    
    # Scoring system for tool selection
    tool_scores = {}
    
    # Wallet-related queries (highest priority)
    if wallet_matches:
        wallet_address = wallet_matches[0]
        
        # Portfolio queries
        portfolio_keywords = ['portfolio', 'holdings', 'owned', 'balance', 'wallet', 'assets']
        if any(word in user_query_lower for word in portfolio_keywords):
            if any(word in user_query_lower for word in ['nft', 'collection', 'art']):
                tool_scores[("get_nft_balances", {"address": wallet_address})] = 85
            if any(word in user_query_lower for word in ['token', 'coin', 'crypto']):
                tool_scores[("get_token_balances", {"address": wallet_address})] = 85
            if any(word in user_query_lower for word in ['profile', 'activity', 'trading']):
                tool_scores[("get_profile", {"address": wallet_address})] = 80
        
        # Default comprehensive profile for wallet queries
        if not any(score > 0 for score in tool_scores.values()):
            tool_scores[("get_profile", {"address": wallet_address})] = 75
    
    # Collection-specific queries
    if token_collection_matches:
        collection_name = token_collection_matches[0].lower()
        
        # Direct collection lookup
        if any(word in user_query_lower for word in ['collection', 'floor', 'price', 'volume', 'stats', 'details']):
            tool_scores[("get_collection", {"collection": collection_name})] = 80
        
        # Collection search if not found
        if not any(score > 0 for score in tool_scores.values()):
            tool_scores[("search_collections", {"query": collection_name})] = 70
    
    # Trending and popular queries
    trending_keywords = ['trending', 'hot', 'popular', 'top', 'best', 'most']
    if any(word in user_query_lower for word in trending_keywords):
        if any(word in user_query_lower for word in ['collection', 'nft', 'art']):
            if 'trending' in user_query_lower:
                # Use correct timeframe values
                if time_context == "SEVEN_DAYS":
                    tool_scores[("get_trending_collections", {"timeframe": "SEVEN_DAYS"})] = 75
                elif time_context == "THIRTY_DAYS":
                    tool_scores[("get_trending_collections", {"timeframe": "THIRTY_DAYS"})] = 75
                else:
                    tool_scores[("get_trending_collections", {"timeframe": "ONE_DAY"})] = 75
            else:
                tool_scores[("get_top_collections", {"sortBy": "ONE_DAY_VOLUME"})] = 75
        elif any(word in user_query_lower for word in ['token', 'coin', 'crypto']):
            if 'trending' in user_query_lower:
                tool_scores[("get_trending_tokens", {})] = 75
            else:
                tool_scores[("get_top_tokens", {})] = 75
    
    # NFT item queries
    nft_keywords = ['nft', 'item', 'token', 'rare', 'traits', 'priced', 'under', 'above']
    if any(word in user_query_lower for word in nft_keywords):
        if '#' in user_query or any(word in user_query_lower for word in ['specific', 'details', 'exact']):
            tool_scores[("get_item", {"query": user_query})] = 70
        else:
            tool_scores[("search_items", {"query": user_query})] = 65
    
    # Token queries
    token_keywords = ['token', 'coin', 'cryptocurrency', 'address', 'contract', 'price']
    if any(word in user_query_lower for word in token_keywords):
        if token_collection_matches:
            token_name = token_collection_matches[0].lower()
            tool_scores[("get_token", {"token": token_name})] = 70
        else:
            tool_scores[("search_tokens", {"query": user_query})] = 65
    
    # Swap/quote queries
    swap_keywords = ['swap', 'quote', 'exchange', 'trade', 'convert', 'how much']
    if any(word in user_query_lower for word in swap_keywords):
        tool_scores[("get_token_swap_quote", {"query": user_query})] = 80
    
    # Market analysis queries
    market_keywords = ['market', 'trend', 'analysis', 'compare', 'vs', 'versus']
    if any(word in user_query_lower for word in market_keywords):
        if not tool_scores:  # Only if no other tools selected
            tool_scores[("search", {"query": user_query})] = 60
    
    # General search as fallback (lowest priority)
    if not tool_scores:
        tool_scores[("search", {"query": user_query})] = 50
    
    # Sort tools by score and select top ones
    sorted_tools = sorted(tool_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Select tools with scores above threshold and limit to prevent overwhelming
    threshold = 50  # Lower threshold for fallback
    max_tools = 3
    
    for (tool_name, tool_args), score in sorted_tools:
        if score >= threshold and len(selected_tools) < max_tools:
            # Validate parameters before adding
            is_valid, error_msg = validate_tool_parameters(tool_name, tool_args)
            if is_valid:
                selected_tools.append((tool_name, tool_args))
                logger.info(f"Rule-based selected tool {tool_name} with score {score}")
            else:
                logger.warning(f"Tool {tool_name} validation failed: {error_msg}")
    
    # Calculate confidence score for the selection
    if selected_tools:
        avg_score = sum(score for _, score in sorted_tools[:len(selected_tools)]) / len(selected_tools)
        confidence = "high" if avg_score >= 80 else "medium" if avg_score >= 65 else "low"
        logger.info(f"Rule-based tool selection confidence: {confidence} (avg score: {avg_score:.1f})")
    
    # Cache the result
    cache_tool_selection(user_query, selected_tools)
    
    return selected_tools

async def maybe_fetch_opensea_context(user_query: str) -> Optional[Dict[str, Any]]:
    """
    Intelligently determine and fetch OpenSea context based on user query.
    Returns combined context from multiple tools if applicable.
    """
    # Only initialize MCP if not already done
    global _mcp_initialized
    if not _mcp_initialized:
        try:
            logger.info("Initializing MCP connection...")
            await MCP.initialize()
            _mcp_initialized = True
            logger.info("MCP initialization successful")
        except Exception as e:
            logger.error(f"MCP initialize failed: {e}")
            return None

    try:
        # Select appropriate tools based on query
        selected_tools = await select_appropriate_tools(user_query)
        logger.info(f"Selected tools for query: {[tool[0] for tool in selected_tools]}")
        
        # Execute all selected tools in parallel
        tool_results = {}
        for tool_name, tool_args in selected_tools:
            try:
                result = await MCP.call_tool(tool_name, tool_args)
                tool_results[tool_name] = result
                logger.info(f"Tool {tool_name} executed successfully")
            except Exception as e:
                logger.warning(f"Tool {tool_name} failed: {e}")
                tool_results[tool_name] = {"error": str(e)}
        
        return {
            "tools_used": [tool[0] for tool in selected_tools],
            "results": tool_results,
            "query": user_query
        }
        
    except Exception as e:
        logger.error(f"Failed to fetch OpenSea context: {e}")
        return None

async def answer_with_asi(prompt: str, context_blob: Optional[Dict[str, Any]]) -> str:
    # Build the system message content
    system_content = SYSTEM_PROMPT
    
    if context_blob:
        # Add context to the system message (not as a separate message)
        ctx_str = json.dumps(context_blob)[:4000]
        system_content += f"\n\nOpenSea context:\n{ctx_str}"
    
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": prompt}
    ]
    
    return await ASI.chat(messages)

async def fetch_comprehensive_portfolio(wallet_address: str) -> Dict[str, Any]:
    """Fetch portfolio data from multiple OpenSea tools in parallel"""
    
    # Define all the tool calls to execute in parallel
    tasks = [
        # Tool 1: Get token balances (ERC-20 tokens, ETH, etc.)
        maybe_fetch_opensea_context(f"Check token balances for wallet {wallet_address}"),
        
        # Tool 2: Get NFT balances (NFTs owned)
        maybe_fetch_opensea_context(f"Show NFTs owned by wallet {wallet_address}")
    ]
    
    # Execute all tasks in parallel for better performance
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Combine results, handling any failures gracefully
    portfolio_data = {}
    tool_names = ["token_balances", "nft_balances"]
    
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.warning(f"Tool {tool_names[i]} failed: {result}")
            portfolio_data[tool_names[i]] = None
        else:
            portfolio_data[tool_names[i]] = result
            logger.info(f"{tool_names[i]} data fetched successfully")
    
    return portfolio_data

# ----------------------------
# FastAPI app for health checks
# ----------------------------

app = FastAPI()

@app.get("/ping")
async def ping():
    return {"status": "agent is running"}

# ----------------------------
# uAgent definition & handlers
# ----------------------------

labubank_agent = Agent(
    name=SETTINGS.agent_name,
    seed=SETTINGS.agent_seed,
    port=SETTINGS.agent_port,
    endpoint=[SETTINGS.agent_endpoint],
)

@labubank_agent.on_event("startup")
async def on_startup(ctx: Context):
    ctx.logger.info(
        "\n\n👋 Hey there! I'm LabuBank, your AI sidekick for all things crypto.\n"
        "I'll keep an eye on your portfolio, break down the market, and ping you when something important happens.\n\n"
        "Try sending a UserQuery with a prompt, e.g., 'Top NFT collections by 24h volume?'\n"
        "I'll intelligently select the best OpenSea tools to answer your questions.\n"
    )
    # Warm up MCP, log available tools (non-fatal)
    try:
        global _mcp_initialized
        await MCP.initialize()
        _mcp_initialized = True
    except Exception as e:
        ctx.logger.warning(f"MCP not ready yet or unreachable: {e}")
        _mcp_initialized = False

@labubank_agent.on_message(model=UserQuery, replies=AgentReply)
async def on_user_query(ctx: Context, msg: UserQuery):
    try:
        ctx.logger.info(f"Received query: {msg.prompt}")
        context = await maybe_fetch_opensea_context(msg.prompt)
        answer = await answer_with_asi(msg.prompt, context)
        reply = AgentReply(answer=answer, used_tools=context.get("tools_used") if context else None, tool_results=context.get("results") if context else None)
        await ctx.send(ctx.sender, reply)
    except Exception as e:
        logger.exception("Failed to handle query")
        reply = AgentReply(answer=f"Sorry—something went wrong: {e}")
        await ctx.send(ctx.sender, reply)

@labubank_agent.on_rest_post("/query", RestRequest, RestResponse)
async def handle_rest_query(ctx: Context, req: RestRequest) -> RestResponse:
    """Handle HTTP POST requests to /query endpoint"""
    try:
        ctx.logger.info(f"Received REST query: {req.prompt}")
        context = await maybe_fetch_opensea_context(req.prompt)
        answer = await answer_with_asi(req.prompt, context)
        
        # Return the response directly
        return RestResponse(
            timestamp=int(time.time()),
            answer=answer,
            used_tools=context.get("tools_used") if context else None,
            tool_results=context.get("results") if context else None,
            agent_address=ctx.agent.address
        )
    except Exception as e:
        logger.exception("Failed to handle REST query")
        return RestResponse(
            timestamp=int(time.time()),
            answer=f"Sorry—something went wrong: {e}",
            used_tools=None,
            tool_results=None,
            agent_address=ctx.agent.address
        )

@labubank_agent.on_rest_post("/portfolio", PortfolioRequest, PortfolioResponse)
async def handle_portfolio_query(ctx: Context, req: PortfolioRequest) -> PortfolioResponse:
    """Handle portfolio queries using OpenSea's get_profile tool"""
    try:
        ctx.logger.info(f"Received portfolio query for wallet: {req.wallet_address}")
        
        # Fetch comprehensive portfolio data from multiple OpenSea tools
        portfolio_data = await fetch_comprehensive_portfolio(req.wallet_address)
        if portfolio_data:
            # Generate a summary using ASI:One
            summary_prompt = f"Analyze this portfolio data for wallet {req.wallet_address} and provide a concise summary highlighting: 1) Token balances and values, 2) NFT holdings and collections, 3) Total portfolio value, and 4) Key insights about the wallet's crypto strategy."
            summary = await answer_with_asi(summary_prompt, portfolio_data)
        else:
            summary = f"No portfolio data found for wallet {req.wallet_address}"
            portfolio_data = {}
        
        return PortfolioResponse(
            timestamp=int(time.time()),
            wallet_address=req.wallet_address,
            portfolio_data=portfolio_data,
            summary=summary,
            agent_address=ctx.agent.address
        )
        
    except Exception as e:
        logger.exception("Failed to handle portfolio query")
        return PortfolioResponse(
            timestamp=int(time.time()),
            wallet_address=req.wallet_address,
            portfolio_data={},
            summary=f"Sorry—something went wrong: {e}",
            agent_address=ctx.agent.address
        )

@labubank_agent.on_rest_get("/health", HealthResponse)
async def handle_health_check(ctx: Context) -> HealthResponse:
    """Handle health check requests - fast and lightweight"""
    ctx.logger.info("Health check requested")
    
    try:
        # Check MCP connection status
        mcp_status = "connected" if _mcp_initialized else "disconnected"
        ctx.logger.info(f"MCP status: {mcp_status}")
        
        # Check ASI:One connection (simple ping) - don't let this fail the health check
        asi_status = "unknown"
        try:
            # Simple test message to check ASI:One connectivity
            # Use a very simple prompt to avoid any complex processing
            ctx.logger.info("Testing ASI:One connection...")
            test_response = await ASI.chat([{"role": "user", "content": "Hi"}])
            asi_status = "connected" if test_response and len(test_response) > 0 else "no_response"
            ctx.logger.info(f"ASI:One test response: {test_response[:100] if test_response else 'None'}")
        except Exception as e:
            logger.warning(f"ASI:One health check failed: {e}")
            asi_status = "disconnected"
        
        # Determine overall health status
        # Consider healthy if at least MCP is connected (ASI:One is optional for basic functionality)
        overall_status = "healthy" if mcp_status == "connected" else "degraded"
        ctx.logger.info(f"Overall health status: {overall_status}")
        
        health_response = HealthResponse(
            status=overall_status,
            timestamp=int(time.time()),
            agent_address=ctx.agent.address,
            agent_name=SETTINGS.agent_name,
            version="1.0.0",
            services={
                "mcp_opensea": mcp_status,
                "asi_one": asi_status
            },
            uptime="running"
        )
        
        ctx.logger.info(f"Health check completed successfully: {health_response}")
        return health_response
        
    except Exception as e:
        logger.exception("Health check failed")
        error_response = HealthResponse(
            status="unhealthy",
            timestamp=int(time.time()),
            agent_address=ctx.agent.address,
            agent_name=SETTINGS.agent_name,
            version="1.0.0",
            services={
                "mcp_opensea": "unknown",
                "asi_one": "unknown"
            },
            uptime="unknown",
            error=str(e)
        )
        ctx.logger.error(f"Health check error response: {error_response}")
        return error_response

if __name__ == "__main__":
    # Run agent in a separate thread
    labubank_agent.run()