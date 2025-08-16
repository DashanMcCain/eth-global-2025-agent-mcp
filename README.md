# LabuBank MCP Agent

![tag:labubank](https://img.shields.io/badge/labubank-3D8BD3)
![tag:innovationlab](https://img.shields.io/badge/innovationlab-3D8BD3)
![tag:mcp](https://img.shields.io/badge/mcp-3D8BD3)

LabuBank is an AI-powered crypto portfolio companion that integrates with OpenSea via MCP (Model Context Protocol) to provide real-time NFT and collection insights. The agent uses ASI:One for intelligent responses and can fetch contextual data from OpenSea's MCP server.

## Features

- **Crypto Portfolio Companion**: AI-powered insights and market analysis
- **MCP Integration**: Seamless connection to OpenSea's MCP server
- **ASI:One AI**: Advanced language model for intelligent responses
- **Real-time Data**: Live NFT collection and market data
- **Health Monitoring**: Built-in health checks for deployment

## Input Data Model

````python
class UserQuery(Model):
    prompt: str
    opensea_tool: Optional[str] = None
    tool_args: Dict[str, Any] = {}

## Output Data Model

```python
class AgentReply(Model):
    answer: str
    used_tool: Optional[str] = None
    tool_result: Optional[Dict[str, Any]] = None
````

## Usage

- Send a `UserQuery` with a prompt about crypto, NFTs, or collections
- Use the `/query` endpoint for HTTP requests with `RestRequest` format
- Optionally specify an OpenSea MCP tool and arguments for context
- Receive intelligent responses with relevant market data and insights

## REST API

### POST `/query`

Send queries to the agent via HTTP POST:

**Request Body:**

```json
{
  "prompt": "What are the current trends in the NFT market?",
  "opensea_tool": null,
  "tool_args": {}
}
```

**Response:**

```json
{
  "timestamp": 1709312457,
  "answer": "Based on current market data...",
  "used_tool": null,
  "tool_result": null,
  "agent_address": "agent1qv3h4tkmvqz8jn8hs7q7y9rg8yh6jzfz7yf3xm2x2z7y8q9w2j5q9n8h6j"
}
```

### POST `/portfolio`

Get comprehensive portfolio information for a wallet address:

**Request Body:**

```json
{
  "wallet_address": "0x1234567890abcdef...",
  "include_items": true,
  "include_collections": true,
  "include_activity": false,
  "include_listings": false,
  "include_offers": false,
  "include_balances": true,
  "include_favorites": false
}
```

**Response:**

```json
{
  "timestamp": 1709312457,
  "wallet_address": "0x1234567890abcdef...",
  "portfolio_data": {
    /* OpenSea profile data */
  },
  "summary": "AI-generated portfolio analysis...",
  "agent_address": "agent1qv3h4tkmvqz8jn8hs7q7y9rg8yh6jzfz7yf3xm2x2z7y8q9w2j5q9n8h6j"
}
```

## Environment Variables

- `ASI_ONE_API_KEY`: Required - Your ASI:One API key
- `ASI_ONE_URL`: Optional - ASI:One API endpoint (defaults to production)
- `ASI_ONE_MODEL`: Optional - AI model to use (defaults to "asi1-mini")
- `OPENSEA_MCP_URL`: Optional - OpenSea MCP server URL
- `OPENSEA_API_KEY`: Optional - OpenSea API key for enhanced access
- `AGENT_NAME`: Optional - Custom agent name (defaults to "LabuBank")
- `AGENT_SEED`: Optional - Agent seed for deterministic addresses

## Deployment

Deployed on Render with a public endpoint for Agentverse integration.

## Health Check

The agent provides a `/ping` endpoint to verify it's running and ready to receive requests.
