#!/usr/bin/env python3
"""
Test script for enhanced RAG integration
Tests both the enhanced RAG service directly and the API endpoints
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.api.services.enhanced_rag_service import EnhancedRagService
import time
import json

async def test_enhanced_rag_service():
    """Test the enhanced RAG service directly"""
    print("🔍 Testing Enhanced RAG Service directly...\n")
    
    # Test queries
    test_queries = [
        "Какую обувь носить летом в офис в 2025?",
        "Как стилизовать платье в дождливую погоду?",
        "Что носить в стиле минимализм?",
        "Какие аксессуары подойдут для романтического образа?",
        "Обувь для холодной погоды"
    ]
    
    try:
        rag_service = EnhancedRagService()
        print("✅ RAG service initialized successfully\n")
        
        for i, query in enumerate(test_queries, 1):
            print(f"📝 Test {i}: {query}")
            print("-" * 50)
            
            start_time = time.time()
            result = await rag_service.enhance_prompt_async(query)
            total_time = time.time() - start_time
            
            if result and result.get("status") == "success":
                print(f"✅ Status: {result['status']}")
                print(f"🎯 Strategy: {result['strategy_used']}")
                print(f"📊 Enhanced: {result['enhanced']}")
                
                # Item detection
                if result.get("item_type"):
                    print(f"👕 Item type: {result['item_type']}")
                    if result.get("item_subtype"):
                        print(f"   Subtype: {result['item_subtype']}")
                
                if result.get("is_styling"):
                    print(f"✨ Styling context: {result['styling_item']}")
                
                # Key nodes
                key_nodes = result.get("key_nodes_found", [])
                if key_nodes:
                    print(f"🔑 Key nodes found: {len(key_nodes)}")
                    for node in key_nodes:
                        print(f"   • {node.get('type', 'Unknown')}: {node.get('name', 'Unknown')}")
                
                # Concepts
                concepts = result.get("concepts_used", [])
                if concepts:
                    print(f"💡 Concepts found: {len(concepts)}")
                    for concept in concepts[:3]:  # Show top 3
                        name = concept.get("name", "").split(':', 1)[1] if ':' in concept.get("name", "") else concept.get("name", "")
                        score = concept.get("intersection_score", 0)
                        print(f"   • {name} (score: {score})")
                
                print(f"⏱ Processing time: {result.get('processing_time', 0):.3f}s")
                print(f"⏱ Total time: {total_time:.3f}s")
                
                if result.get("enhanced"):
                    print(f"📝 Enhanced prompt length: {len(result['enhanced_prompt'])} chars")
                    print(f"📝 Original prompt length: {len(result['original_prompt'])} chars")
                
            else:
                print(f"❌ Error: {result.get('error', 'Unknown error')}")
            
            print("\n" + "="*60 + "\n")
    
    except Exception as e:
        print(f"❌ Error testing RAG service: {e}")
        import traceback
        traceback.print_exc()

async def test_api_endpoints():
    """Test the API endpoints (requires running API server)"""
    print("🌐 Testing API endpoints...\n")
    
    try:
        import aiohttp
        
        test_query = "Какую обувь носить летом в офис в 2025?"
        
        async with aiohttp.ClientSession() as session:
            # Test debug endpoint
            print("🔍 Testing /rag/debug endpoint...")
            async with session.post(
                "http://localhost:8000/rag/debug",
                json={"prompt": test_query},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"✅ Debug endpoint successful")
                    print(f"📊 Strategy: {result.get('strategy_used')}")
                    print(f"⏱ Processing time: {result.get('processing_time', 0):.3f}s")
                    print(f"🔑 Key nodes: {len(result.get('key_nodes_analysis', {}))}")
                    print(f"💡 Concepts: {len(result.get('concepts_analysis', []))}")
                else:
                    print(f"❌ Debug endpoint failed: {response.status}")
            
            print()
            
            # Test enhanced inference endpoint
            print("🧠 Testing /rag/enhanced_inference endpoint...")
            async with session.post(
                "http://localhost:8000/rag/enhanced_inference",
                json={"prompt": test_query},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"✅ Enhanced inference endpoint successful")
                    print(f"📊 Strategy: {result.get('strategy_used')}")
                    print(f"⏱ Processing time: {result.get('processing_time', 0):.3f}s")
                    print(f"🔑 Total keynodes: {result.get('total_keynodes', 0)}")
                    print(f"💡 Concepts count: {result.get('concepts_count', 0)}")
                    
                    # Check for weather debugging
                    if hasattr(result, 'weather_debug'):
                        print(f"🌤️ Weather debug info available")
                else:
                    print(f"❌ Enhanced inference endpoint failed: {response.status}")
    
    except ImportError:
        print("⚠️ aiohttp not available, skipping API tests")
    except Exception as e:
        print(f"❌ Error testing API endpoints: {e}")
        print("💡 Make sure the API server is running on localhost:8000")

def print_bot_usage_instructions():
    """Print instructions for testing the Telegram bot"""
    print("🤖 Telegram Bot Testing Instructions")
    print("="*50)
    print()
    print("The enhanced RAG integration has been added to the Telegram bot.")
    print("You can now test it using these commands:")
    print()
    print("1. 📝 /ask - Interactive mode with enhanced RAG")
    print("   • Choose 'Умный' mode to use enhanced RAG")
    print("   • The bot will show detailed debugging information")
    print()
    print("2. 🔍 /debug_rag <question> - Direct RAG debugging")
    print("   • Shows detailed analysis without generating response")
    print("   • Example: /debug_rag Какую обувь носить летом в офис?")
    print()
    print("3. ⚙️ /ask_with_params <question> use_rag=true - Advanced mode")
    print("   • Example: /ask_with_params Как носить джинсы? use_rag=true temperature=0.8")
    print()
    print("🧪 Test Queries for Weather Debugging:")
    print("• Какую обувь носить в дождливую погоду?")
    print("• Что надеть в солнечную погоду летом?")
    print("• Как одеться в холодную погоду?")
    print("• Обувь для жаркой погоды")
    print()
    print("🎯 Test Queries for Item Type Detection:")
    print("• Какие платья носить летом?")
    print("• Как стилизовать джинсы?")
    print("• Какие аксессуары подойдут для офиса?")
    print()

async def main():
    """Main test function"""
    print("🚀 Enhanced RAG Integration Test Suite")
    print("="*50)
    print()
    
    # Test the service directly
    await test_enhanced_rag_service()
    
    # Test API endpoints
    await test_api_endpoints()
    
    # Print bot usage instructions
    print_bot_usage_instructions()

if __name__ == "__main__":
    asyncio.run(main()) 