#!/usr/bin/env python3.11

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.api.services.enhanced_rag_service import EnhancedRagService

async def test_rag_pipeline():
    """Test the RAG pipeline with the example prompt"""
    
    # Initialize the service
    rag_service = EnhancedRagService()
    
    # Test prompt from the example
    test_prompt = "Какую летнюю обувь лучше выбрать для офиса в 2025 году?"
    
    print(f"🔍 Testing RAG pipeline with prompt: {test_prompt}")
    print("=" * 80)
    
    try:
        # Process the prompt
        result = await rag_service.process_rag_query(test_prompt)
        
        if result["status"] == "success":
            print(f"✅ RAG processing successful!")
            print(f"⏱️  Processing time: {result['processing_time']:.2f}s")
            print(f"🔧 Enhanced: {result['enhanced']}")
            
            if result['enhanced']:
                print(f"\n📝 Original prompt:")
                print(f"   {result['original_prompt']}")
                
                print(f"\n🚀 Enhanced prompt:")
                print(f"   {result['enhanced_prompt']}")
                
                print(f"\n🔑 Key nodes found:")
                for node_type, nodes in result['key_nodes'].items():
                    if nodes:
                        print(f"   {node_type}: {', '.join(n['name'] for n in nodes)}")
                
                print(f"\n💡 Concepts found ({len(result['concepts'])}):")
                for i, concept in enumerate(result['concepts'], 1):
                    print(f"   {i}. {concept['name']} (score: {concept.get('intersection_score', 0)})")
                
                if result.get('item_type'):
                    print(f"\n👕 Item type detected: {result['item_type']}")
                    if result.get('item_subtype'):
                        print(f"   Subtype: {result['item_subtype']}")
                
                if result.get('is_styling'):
                    print(f"\n✨ Styling context detected for: {result.get('styling_item', 'N/A')}")
            else:
                print(f"\n❌ No enhancement applied")
                print(f"   Reason: No relevant concepts found")
        else:
            print(f"❌ RAG processing failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"💥 Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        if hasattr(rag_service, 'neo4j'):
            rag_service.neo4j.close()

if __name__ == "__main__":
    asyncio.run(test_rag_pipeline()) 