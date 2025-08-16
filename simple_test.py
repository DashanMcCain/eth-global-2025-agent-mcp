#!/usr/bin/env python3
"""
Simple HTTP test for LabuBank Agent
"""
import asyncio
import httpx
import json

async def test_agent():
    print("🧪 Testing LabuBank Agent via HTTP...")
    
    # Use a single client for all tests
    async with httpx.AsyncClient() as client:
        # Test 1: Health endpoint
        print("\n1️⃣ Testing health endpoint...")
        try:
            response = await client.get("http://localhost:8000/ping")
            if response.status_code == 200:
                print(f"✅ Health check passed: {response.json()}")
            else:
                print(f"❌ Health check failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Health check error: {e}")
        
        # Test 2: Simple query
        print("\n2️⃣ Testing simple query...")
        query1 = {
            "prompt": "What are the current trends in the NFT market?",
            "opensea_tool": None,
            "tool_args": {}
        }
        
        try:
            print("📤 Sending query to agent...")
            response = await client.post(
                "http://localhost:8011/query",
                json=query1,
                timeout=30.0
            )
            print(f"✅ Query 1 response: {response.status_code}")
            if response.status_code == 200:
                print(f"Response: {response.text[:200]}...")
            else:
                print(f"Response body: {response.text}")
        except Exception as e:
            print(f"❌ Query 1 failed: {e}")
        
        # Test 3: Query with MCP tool
        print("\n3️⃣ Testing query with MCP tool...")
        query2 = {
            "prompt": "Show me trending NFT collections",
            "opensea_tool": "search_collections",
            "tool_args": {"query": "trending", "limit": 5}
        }
        
        try:
            print("📤 Sending MCP query to agent...")
            response = await client.post(
                "http://localhost:8011/query",
                json=query2,
                timeout=30.0
            )
            print(f"✅ Query 2 response: {response.status_code}")
            if response.status_code == 200:
                print(f"Response: {response.text[:200]}...")
            else:
                print(f"Response body: {response.text}")
        except Exception as e:
            print(f"❌ Query 2 failed: {e}")
        
        # Test 4: Portfolio endpoint
        print("\n4️⃣ Testing portfolio endpoint...")
        portfolio_request = {
            "wallet_address": "0x87E3c23b4F9EF78041F49F74A18F2DbA8B9fd8a8",
        }
        
        try:
            print("📤 Sending portfolio query...")
            response = await client.post(
                "http://localhost:8011/portfolio",
                json=portfolio_request,
                timeout=60.0  # Longer timeout for portfolio queries
            )
            print(f"✅ Portfolio response: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"Wallet: {data.get('wallet_address')}")
                print(f"Summary: {data.get('summary', '')[:200]}...")
                portfolio_data = data.get('portfolio_data', {})
                print(f"Portfolio data keys: {list(portfolio_data.keys())}")
            else:
                print(f"Response body: {response.text}")
        except Exception as e:
            print(f"❌ Portfolio query failed: {e}")
        
        # Test 5: Check if agent is processing messages
        print("\n5️⃣ Checking agent message processing...")
        print("📋 Look at your agent terminal for these log messages:")
        print("   - 'Received query: [your prompt]'")
        print("   - 'Received portfolio query for wallet: [address]'")
        print("   - ASI:One API calls")
        print("   - MCP tool calls (if using OpenSea)")
        print("   - Response generation")
        
        print("\n🎯 Testing complete! Check your agent logs for processing details.")

if __name__ == "__main__":
    asyncio.run(test_agent())
