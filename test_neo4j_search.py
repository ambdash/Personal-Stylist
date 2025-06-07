#!/usr/bin/env python3

import requests
import json
from neo4j import GraphDatabase

def test_direct_neo4j_search():
    """Test direct Neo4j search"""
    print("=== Testing Direct Neo4j Search ===")
    
    driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'password123'))
    
    test_queries = ['лето', 'зима', 'кроссовки', 'платье']
    
    for query in test_queries:
        with driver.session() as session:
            result = session.run("""
                MATCH (n) 
                WHERE toLower(n.name) CONTAINS toLower($search) 
                RETURN n.name as name, labels(n)[0] as label 
                LIMIT 5
            """, {'search': query})
            
            results = list(result)
            print(f"\nSearch for '{query}': {len(results)} results")
            for record in results:
                print(f"  - {record['name']} ({record['label']})")
    
    driver.close()

def test_api_search():
    """Test API search endpoint"""
    print("\n=== Testing API Search Endpoint ===")
    
    # Try different ports
    ports = [8000, 8001, 8002]
    api_url = None
    
    for port in ports:
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=2)
            if response.status_code == 200:
                api_url = f"http://localhost:{port}"
                print(f"Found API server at {api_url}")
                break
        except:
            continue
    
    if not api_url:
        print("❌ API server not found on any port")
        return
    
    test_queries = ['лето', 'кроссовки']
    
    for query in test_queries:
        try:
            response = requests.post(
                f"{api_url}/telegram/db/search",
                json={"word": query, "limit": 10},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"\nAPI search for '{query}': Found {len(data.get('nodes', []))} results")
                if data.get('found') and data.get('nodes'):
                    for node in data.get('nodes', [])[:3]:  # Show first 3
                        print(f"  - {node.get('name', 'Unknown')} (connections: {node.get('total_connections', 0)})")
                else:
                    print(f"  No results found for '{query}'")
            else:
                print(f"❌ API search for '{query}' failed: {response.status_code}")
                print(f"Response: {response.text}")
        except Exception as e:
            print(f"❌ API search for '{query}' error: {e}")

def main():
    print("Testing Neo4j Search Functionality")
    print("==================================")
    
    try:
        test_direct_neo4j_search()
        test_api_search()
        print("\n✅ Testing completed!")
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    main() 