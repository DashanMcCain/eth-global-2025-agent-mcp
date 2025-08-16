import os
import json
import uuid
import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from pydantic import Field
import httpx

from uagents import Agent, Context, Model

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
    mcp_url: str = env("OPENSEA_MCP_URL", required=False, default="https://mcp.opensea.io/mcp")
    mcp_api_key: Optional[str] = os.getenv("OPENSEA_API_KEY")
    mcp_origin: str = os.getenv("MCP_ORIGIN", DEFAULT_MCP_ORIGIN)

    # Agent
    agent_name: str = os.getenv("AGENT_NAME", "LabuBank")
    agent_seed: str = os.getenv("AGENT_SEED", "labubank_mcp_agent")
    agent_port: int = int(os.getenv("AGENT_PORT", "8000"))
    agent_endpoint: str = os.getenv("AGENT_ENDPOINT", f"http://localhost:{os.getenv('AGENT_PORT','8000')}/submit")

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
        data = resp.json()
        if "error" in data:
            raise RuntimeError(f"MCP initialize error: {data['error']}")
        return data.get("result", {})

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

# ----------------------------
# uAgents message schemas
# ----------------------------

class UserQuery(Model):
    prompt: str = Field(..., description="User's request in plain English")
    # If you know the tool you want, pass it; otherwise agent can skip tools
    opensea_tool: Optional[str] = Field(None, description="MCP tool name to call (e.g., 'collections.search')")
    tool_args: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Arguments to MCP tool")

class AgentReply(Model):
    answer: str
    used_tool: Optional[str] = None
    tool_result: Optional[Dict[str, Any]] = None

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
    # Ensure MCP init happened
    try:
        await MCP.initialize()
    except Exception as e:
        logger.error(f"MCP initialize failed: {e}")
        raise

    # Optional: validate tool exists
    tools = await MCP.list_tools()
    tool_names = {t.get("name") for t in tools}
    if tool_name not in tool_names:
        raise RuntimeError(f"Requested MCP tool '{tool_name}' not found. Available: {sorted(tool_names)}")

    result = await MCP.call_tool(tool_name, tool_args or {})
    return result

async def answer_with_asi(prompt: str, context_blob: Optional[Dict[str, Any]]) -> str:
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    if context_blob:
        # Keep context compact
        ctx_str = json.dumps(context_blob)[:4000]
        messages.append({"role": "system", "content": f"OpenSea context:\n{ctx_str}"})
    messages.append({"role": "user", "content": prompt})
    return await ASI.chat(messages)

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
        await MCP.initialize()
        tools = await MCP.list_tools()
        ctx.logger.info(f"MCP ready. Tools discovered: {[t.get('name') for t in tools]}")
    except Exception as e:
        ctx.logger.warning(f"MCP not ready yet or unreachable: {e}")

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

if __name__ == "__main__":
    try:
        agent.run()
    finally:
        asyncio.run(HTTP.aclose())