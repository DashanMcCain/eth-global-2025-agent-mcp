import os
import json
import uuid
import asyncio
import logging
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
# from pydantic import Field  # Commented out due to v1/v2 compatibility issues
from uagents import Agent, Context, Model
import httpx
from fastapi import FastAPI
import uvicorn
from dotenv import load_dotenv


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
    agent_port: int = int(os.getenv("AGENT_PORT", "8011"))
    agent_endpoint: str = os.getenv("AGENT_ENDPOINT", "http://0.0.0.0:8011/query")

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
    opensea_tool: Optional[str] = None
    tool_args: Dict[str, Any] = {}

class AgentReply(Model):
    answer: str
    used_tool: Optional[str] = None
    tool_result: Optional[Dict[str, Any]] = None


class RestRequest(Model):
    prompt: str
    opensea_tool: Optional[str] = None
    tool_args: Dict[str, Any] = {}

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
    used_tool: Optional[str] = None
    tool_result: Optional[Dict[str, Any]] = None
    agent_address: str

class PortfolioResponse(Model):
    timestamp: int
    wallet_address: str
    portfolio_data: Dict[str, Any]
    summary: str
    agent_address: str

# ----------------------------
# Orchestration
# ----------------------------

SYSTEM_PROMPT = (
    "You are LabuBank, a helpful crypto portfolio companion. "
    "If context about NFTs or collections is provided, use it. "
    "Be concise, accurate, and avoid speculation."
)

async def maybe_fetch_opensea_context(tool_name: Optional[str], tool_args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not tool_name:
        return None
    
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
            # Don't reset the flag on failure - let it retry next time
            raise

    # Optional: validate tool exists
    tools = await MCP.list_tools()
    tool_names = {t.get("name") for t in tools}
    if tool_name not in tool_names:
        raise RuntimeError(f"Requested MCP tool '{tool_name}' not found. Available: {sorted(tool_names)}")

    result = await MCP.call_tool(tool_name, tool_args or {})
    return result

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
        maybe_fetch_opensea_context("get_token_balances", {
            "address": wallet_address
        }),
        
        # Tool 2: Get NFT balances (NFTs owned)
        maybe_fetch_opensea_context("get_nft_balances", {
            "address": wallet_address
        })
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

agent = Agent(
    name=SETTINGS.agent_name,
    seed=SETTINGS.agent_seed,
    port=SETTINGS.agent_port,
    endpoint=[SETTINGS.agent_endpoint],
)

@agent.on_event("startup")
async def on_startup(ctx: Context):
    ctx.logger.info(
        "\n\n👋 Hey there! I'm LabuBank, your AI sidekick for all things crypto.\n"
        "I'll keep an eye on your portfolio, break down the market, and ping you when something important happens.\n\n"
        "Try sending a UserQuery with a prompt, e.g., 'Top NFT collections by 24h volume?'\n"
        "Optionally include opensea_tool + tool_args to fetch context before answering.\n"
    )
    # Warm up MCP, log available tools (non-fatal)
    try:
        global _mcp_initialized
        await MCP.initialize()
        _mcp_initialized = True
    except Exception as e:
        ctx.logger.warning(f"MCP not ready yet or unreachable: {e}")
        _mcp_initialized = False

@agent.on_message(model=UserQuery, replies=AgentReply)
async def on_user_query(ctx: Context, msg: UserQuery):
    try:
        ctx.logger.info(f"Received query: {msg.prompt}")
        context = await maybe_fetch_opensea_context(msg.opensea_tool, msg.tool_args or {})
        answer = await answer_with_asi(msg.prompt, context)
        reply = AgentReply(answer=answer, used_tool=msg.opensea_tool, tool_result=context)
        await ctx.send(ctx.sender, reply)
    except Exception as e:
        logger.exception("Failed to handle query")
        reply = AgentReply(answer=f"Sorry—something went wrong: {e}")
        await ctx.send(ctx.sender, reply)

@agent.on_rest_post("/query", RestRequest, RestResponse)
async def handle_rest_query(ctx: Context, req: RestRequest) -> RestResponse:
    """Handle HTTP POST requests to /query endpoint"""
    try:
        ctx.logger.info(f"Received REST query: {req.prompt}")
        context = await maybe_fetch_opensea_context(req.opensea_tool, req.tool_args or {})
        answer = await answer_with_asi(req.prompt, context)
        
        # Return the response directly
        return RestResponse(
            timestamp=int(time.time()),
            answer=answer,
            used_tool=req.opensea_tool,
            tool_result=context,
            agent_address=ctx.agent.address
        )
    except Exception as e:
        logger.exception("Failed to handle REST query")
        return RestResponse(
            timestamp=int(time.time()),
            answer=f"Sorry—something went wrong: {e}",
            used_tool=None,
            tool_result=None,
            agent_address=ctx.agent.address
        )

@agent.on_rest_post("/portfolio", PortfolioRequest, PortfolioResponse)
async def handle_portfolio_query(ctx: Context, req: PortfolioRequest) -> PortfolioResponse:
    """Handle portfolio queries using OpenSea's get_profile tool"""
    try:
        ctx.logger.info(f"Received portfolio query for wallet: {req.wallet_address}")
        
        # Build tool arguments for get_profile
        tool_args = {
            "address": req.wallet_address,
            "include_items": True,
            "include_collections": True,
            "include_activity": True,
            "include_listings": True,
            "include_offers": True,
            "include_balances": True,
            "include_favorites": True
        }
        
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

# --- MAIN (run servers on different ports) ---
def run_agent():
    try:
        agent.run()  # uAgents inbox on 8011
    finally:
        asyncio.run(HTTP.aclose())

if __name__ == "__main__":
    # Run agent in a separate thread
    threading.Thread(target=run_agent, daemon=True).start()
    # Run FastAPI server on 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)