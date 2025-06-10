#!/usr/bin/env python3.11
"""
Test script to verify the updated inference engine works with the bot architecture.
"""
import os
import sys
from pathlib import Path
import asyncio
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_inference_engine_direct():
    """Test the inference engine directly"""
    print("=" * 60)
    print("TESTING INFERENCE ENGINE DIRECTLY")
    print("=" * 60)
    
    try:
        from src.ml.inference.engine import InferenceEngine
        
        # Initialize engine
        engine = InferenceEngine()
        
        # Test prompt
        test_prompt = "Как правильно сочетать кожаную куртку в стиле гранж?"
        
        print(f"🧪 Testing prompt: {test_prompt}")
        
        # Test without RAG
        print("\n📝 Testing without RAG...")
        result = engine.generate(
            prompt=test_prompt,
            model_name="t-tech/T-lite-it-1.0",
            adapter_path="src/ml/models/finetuned/t_lite",
            use_rag=False,
            do_sample=False,  # Use greedy decoding
            max_new_tokens=200
        )
        
        print(f"✅ Generation successful!")
        print(f"   Processing time: {result['processing_time']:.2f}s")
        print(f"   Model used: {result['model_used']}")
        print(f"   Response length: {result['response_length']} chars")
        print(f"   Response: {result['generated_text'][:200]}...")
        
        # Test with RAG (if available)
        print("\n📚 Testing with RAG...")
        try:
            result_rag = engine.generate(
                prompt=test_prompt,
                model_name="t-tech/T-lite-it-1.0",
                adapter_path="src/ml/models/finetuned/t_lite",
                use_rag=True,
                do_sample=False,
                max_new_tokens=200
            )
            
            print(f"✅ RAG generation successful!")
            print(f"   Processing time: {result_rag['processing_time']:.2f}s")
            print(f"   RAG enhanced: {result_rag['rag_info']['enhanced'] if result_rag['rag_info'] else False}")
            print(f"   Response: {result_rag['generated_text'][:200]}...")
            
        except Exception as e:
            print(f"⚠️  RAG test failed (this is OK if Neo4j is not available): {e}")
        
        # Test health check
        print("\n🏥 Testing health check...")
        health = engine.health_check()
        print(f"   Status: {health['status']}")
        print(f"   Models loaded: {health['models_loaded']}")
        print(f"   GPU available: {health['gpu_available']}")
        print(f"   GPU count: {health['gpu_count']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Direct engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_celery_worker():
    """Test the Celery worker"""
    print("\n" + "=" * 60)
    print("TESTING CELERY WORKER")
    print("=" * 60)
    
    try:
        from src.celery_app import app as celery_app
        
        # Test if Celery is available
        print("📡 Testing Celery connection...")
        
        # Submit a simple task
        test_prompt = "Какие аксессуары подойдут к черному платью?"
        
        print(f"🧪 Submitting task: {test_prompt}")
        
        task = celery_app.send_task(
            'inference_worker.generate_text',
            kwargs={
                'prompt': test_prompt,
                'use_rag': False,
                'model_name': 't-tech/T-lite-it-1.0',
                'adapter_path': 'src/ml/models/finetuned/t_lite',
                'parameters': {
                    'do_sample': False,
                    'max_new_tokens': 150,
                    'temperature': 0.7
                }
            },
            queue='inference'
        )
        
        print(f"📋 Task submitted with ID: {task.id}")
        print("⏳ Waiting for result (timeout: 60s)...")
        
        # Wait for result with timeout
        try:
            result = task.get(timeout=60)
            
            if result.get('success'):
                task_result = result['result']
                print(f"✅ Task completed successfully!")
                print(f"   Processing time: {task_result['processing_time']:.2f}s")
                print(f"   Model used: {task_result['model_used']}")
                print(f"   Response: {task_result['generated_text'][:200]}...")
                return True
            else:
                print(f"❌ Task failed: {result.get('error', 'Unknown error')}")
                return False
                
        except Exception as e:
            print(f"⏰ Task timeout or error: {e}")
            print("   This might mean the Celery worker is not running")
            return False
            
    except Exception as e:
        print(f"❌ Celery test failed: {e}")
        print("   Make sure Redis and Celery workers are running")
        return False

async def test_fastapi_endpoint():
    """Test the FastAPI endpoint"""
    print("\n" + "=" * 60)
    print("TESTING FASTAPI ENDPOINT")
    print("=" * 60)
    
    try:
        import httpx
        
        # Test endpoint
        url = "http://localhost:8000/v1/unified_inference/generate"
        
        payload = {
            "prompt": "Как носить джинсы с блейзером?",
            "use_rag": False,
            "model_name": "t-tech/T-lite-it-1.0",
            "parameters": {
                "do_sample": False,
                "max_new_tokens": 150,
                "temperature": 0.7
            }
        }
        
        print(f"🌐 Testing endpoint: {url}")
        print(f"📤 Payload: {payload['prompt']}")
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(url, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ API call successful!")
                print(f"   Processing time: {result['processing_time']:.2f}s")
                print(f"   Model used: {result['model_used']}")
                print(f"   Task ID: {result['task_id']}")
                print(f"   Response: {result['generated_text'][:200]}...")
                return True
            else:
                print(f"❌ API call failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
                
    except ImportError:
        print("⚠️  httpx not available, skipping API test")
        print("   Install with: pip install httpx")
        return None
    except Exception as e:
        print(f"❌ API test failed: {e}")
        print("   Make sure FastAPI server is running on localhost:8000")
        return False

async def main():
    """Run all tests"""
    print("🚀 TESTING PERSONAL STYLIST INFERENCE INTEGRATION")
    print("=" * 60)
    
    results = {}
    
    # Test 1: Direct inference engine
    results['direct'] = test_inference_engine_direct()
    
    # Test 2: Celery worker
    results['celery'] = test_celery_worker()
    
    # Test 3: FastAPI endpoint
    results['api'] = await test_fastapi_endpoint()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, result in results.items():
        if result is True:
            status = "✅ PASSED"
        elif result is False:
            status = "❌ FAILED"
        else:
            status = "⚠️  SKIPPED"
        
        print(f"{test_name.upper():12} {status}")
    
    # Recommendations
    print("\n📋 RECOMMENDATIONS:")
    
    if results['direct']:
        print("✅ Inference engine is working correctly")
    else:
        print("❌ Fix inference engine issues first")
    
    if results['celery']:
        print("✅ Celery worker is operational")
    else:
        print("⚠️  Start Celery worker: celery -A src.celery_app worker --loglevel=info --queues=inference")
    
    if results['api']:
        print("✅ FastAPI endpoint is working")
    elif results['api'] is False:
        print("⚠️  Start FastAPI server: python -m src.api.main")
    
    print("\n🎯 If all tests pass, your bot should work correctly!")

if __name__ == "__main__":
    asyncio.run(main()) 