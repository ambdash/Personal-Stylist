#!/usr/bin/env python3

import asyncio
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.bot.services.api_client import api_client

async def test_api_connection():
    """Test the API connection from bot's perspective"""
    print("Testing API connection from bot's perspective...")
    print(f"API Base URL: {api_client.base_url}")
    
    # Test health check
    print("\n1. Testing health check...")
    health_result = await api_client.health_check()
    print(f"Health check result: {health_result}")
    
    # Test search for "лето"
    print("\n2. Testing search for 'лето'...")
    search_result = await api_client.search_nodes_by_word("лето")
    print(f"Search result: found={search_result.get('found', False)}, nodes_count={len(search_result.get('nodes', []))}")
    
    if search_result.get('found'):
        print("First few results:")
        for i, node in enumerate(search_result.get('nodes', [])[:3], 1):
            print(f"  {i}. {node.get('name', 'Unknown')} (connections: {node.get('total_connections', 0)})")
    
    # Test search for "кроссовки"
    print("\n3. Testing search for 'кроссовки'...")
    search_result2 = await api_client.search_nodes_by_word("кроссовки")
    print(f"Search result: found={search_result2.get('found', False)}, nodes_count={len(search_result2.get('nodes', []))}")
    
    if search_result2.get('found'):
        print("First few results:")
        for i, node in enumerate(search_result2.get('nodes', [])[:3], 1):
            print(f"  {i}. {node.get('name', 'Unknown')} (connections: {node.get('total_connections', 0)})")

if __name__ == "__main__":
    asyncio.run(test_api_connection()) 