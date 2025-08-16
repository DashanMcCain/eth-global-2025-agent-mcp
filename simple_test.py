#!/usr/bin/env python3
"""
Test script for LabuBank Agent with intelligent tool selection
Tests various query types to verify the new weighted scoring system works correctly
"""

import asyncio
import httpx
import json

async def test_agent():
    print("🧪 Testing LabuBank Agent with Intelligent Tool Selection...")
    
    # Use a single client for all tests
    async with httpx.AsyncClient() as client:
        # Test 1: Health endpoint
        print("\n1️⃣ Testing health endpoint...")
        try:
            response = await client.get("http://localhost:8000/health")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Health check passed: {data['status']}")
                print(f"   MCP Status: {data['services']['mcp_opensea']}")
                print(f"   ASI:One Status: {data['services']['asi_one']}")
            else:
                print(f"❌ Health check failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Health check error: {e}")
        
        # Test 2: Simple query (should use general search)
        print("\n2️⃣ Testing simple query (general search)...")
        query1 = {
            "prompt": "What are the current trends in the NFT market?"
        }
        
        try:
            print("📤 Sending general query to agent...")
            response = await client.post(
                "http://localhost:8000/query",
                json=query1,
                timeout=30.0
            )
            print(f"✅ Query 1 response: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"Response: {data['answer'][:200]}...")
                print(f"Tools used: {data.get('used_tools', 'None')}")
                print(f"Tool results: {list(data.get('tool_results', {}).keys()) if data.get('tool_results') else 'None'}")
            else:
                print(f"Response body: {response.text}")
        except Exception as e:
            print(f"❌ Query 1 failed: {e}")
        
        # Test 4: Collection-specific query (should use get_collection)
        print("\n4️⃣ Testing collection-specific query...")
        query3 = {
            "prompt": "Show me details for the Azuki collection including floor price and volume"
        }
        
        try:
            print("📤 Sending collection query...")
            response = await client.post(
                "http://localhost:8000/query",
                json=query3,
                timeout=30.0
            )
            print(f"✅ Query 3 response: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"Response: {data['answer'][:200]}...")
                print(f"Tools used: {data.get('used_tools', 'None')}")
                print(f"Tool results: {list(data.get('tool_results', {}).keys()) if data.get('tool_results') else 'None'}")
            else:
                print(f"Response body: {response.text}")
        except Exception as e:
            print(f"❌ Query 3 failed: {e}")
        
        # Test 5: Trending query (should use get_trending_collections with timeframe)
        print("\n5️⃣ Testing trending query...")
        query4 = {
            "prompt": "What are the trending NFT collections this week?"
        }
        
        try:
            print("📤 Sending trending query...")
            response = await client.post(
                "http://localhost:8000/query",
                json=query4,
                timeout=30.0
            )
            print(f"✅ Query 4 response: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"Response: {data['answer'][:200]}...")
                print(f"Tools used: {data.get('used_tools', 'None')}")
                print(f"Tool results: {list(data.get('tool_results', {}).keys()) if data.get('tool_results') else 'None'}")
            else:
                print(f"Response body: {response.text}")
        except Exception as e:
            print(f"❌ Query 4 failed: {e}")
        
        # Test 6: Token swap query (should use get_token_swap_quote)
        print("\n6️⃣ Testing token swap query...")
        query5 = {
            "prompt": "How much USDC can I get for 1 ETH?"
        }
        
        try:
            print("📤 Sending swap query...")
            response = await client.post(
                "http://localhost:8000/query",
                json=query5,
                timeout=30.0
            )
            print(f"✅ Query 5 response: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"Response: {data['answer'][:200]}...")
                print(f"Tools used: {data.get('used_tools', 'None')}")
                print(f"Tool results: {list(data.get('tool_results', {}).keys()) if data.get('tool_results') else 'None'}")
            else:
                print(f"Response body: {response.text}")
        except Exception as e:
            print(f"❌ Query 5 failed: {e}")
        
        # Test 7: Complex query (should test multiple tool selection)
        print("\n7️⃣ Testing complex query...")
        query6 = {
            "prompt": "Compare the floor prices of BAYC and Azuki collections and show me trending tokens"
        }
        
        try:
            print("📤 Sending complex query...")
            response = await client.post(
                "http://localhost:8000/query",
                json=query6,
                timeout=45.0
            )
            print(f"✅ Query 6 response: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"Response: {data['answer'][:200]}...")
                print(f"Tools used: {data.get('used_tools', 'None')}")
                print(f"Tool results: {list(data.get('tool_results', {}).keys()) if data.get('tool_results') else 'None'}")
            else:
                print(f"Response body: {response.text}")
        except Exception as e:
            print(f"❌ Query 6 failed: {e}")
        
        # Test 8: LLM-specific complex query (should trigger LLM tool selection)
        print("\n8️⃣ Testing LLM-specific complex query...")
        query7 = {
            "prompt": "I want to see the trading activity for vitalik.eth's wallet and also check what's trending in the NFT space this month"
        }
        
        try:
            print("📤 Sending LLM-specific query...")
            response = await client.post(
                "http://localhost:8000/query",
                json=query7,
                timeout=45.0
            )
            print(f"✅ Query 7 response: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"Response: {data['answer'][:200]}...")
                print(f"Tools used: {data.get('used_tools', 'None')}")
                print(f"Tool results: {list(data.get('tool_results', {}).keys()) if data.get('tool_results') else 'None'}")
            else:
                print(f"Response body: {response.text}")
        except Exception as e:
            print(f"❌ Query 7 failed: {e}")
        
        # Test 9: Ambiguous query (should test LLM's reasoning)
        print("\n🔟 Testing ambiguous query...")
        query8 = {
            "prompt": "What's hot right now in crypto and NFTs?"
        }
        
        try:
            print("📤 Sending ambiguous query...")
            response = await client.post(
                "http://localhost:8000/query",
                json=query8,
                timeout=45.0
            )
            print(f"✅ Query 8 response: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"Response: {data['answer'][:200]}...")
                print(f"Tools used: {data.get('used_tools', 'None')}")
                print(f"Tool results: {list(data.get('tool_results', {}).keys()) if data.get('tool_results') else 'None'}")
            else:
                print(f"Response body: {response.text}")
        except Exception as e:
            print(f"❌ Query 8 failed: {e}")
        
        # Test 10: Portfolio endpoint (should use parallel fetching)
        print("\n1️⃣1️⃣ Testing portfolio endpoint...")
        portfolio_request = {
            "wallet_address": "0x87E3c23b4F9EF78041F49F74A18F2DbA8B9fd8a8",
        }
        
        try:
            print("📤 Sending portfolio query...")
            response = await client.post(
                "http://localhost:8000/portfolio",
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
        
        # Test 11: Check agent message processing
        print("\n1️⃣2️⃣ Checking agent message processing...")
        print("📋 Look at your agent terminal for these log messages:")
        print("   - 'Attempting LLM-based tool selection...'")
        print("   - 'LLM selected tool [name] with args [args]'")
        print("   - 'LLM tool selection successful: [tool_names]'")
        print("   - 'Using rule-based tool selection fallback...' (if LLM fails)")
        print("   - 'Rule-based selected tool [name] with score [score]'")
        print("   - 'Tool selection confidence: [high/medium/low]'")
        print("   - 'Using cached tool selection for query: [query]'")
        print("   - 'Tool [name] executed successfully'")
        print("   - ASI:One API calls")
        print("   - MCP tool calls")
        
        print("\n🎯 Testing complete! Check your agent logs for:")
        print("   - LLM vs. rule-based tool selection")
        print("   - Tool selection scores and confidence levels")
        print("   - Caching behavior for repeated queries")
        print("   - Parameter validation results")
        print("   - Parallel tool execution")
        print("   - Error handling and fallbacks")
        print("   - LLM reasoning for tool selection")

if __name__ == "__main__":
    asyncio.run(test_agent())
