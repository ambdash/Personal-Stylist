#!/usr/bin/env python3.11

import sys
import time
import requests
import redis
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_redis():
    """Test Redis connection"""
    try:
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        result = r.ping()
        print(f"✅ Redis: {result}")
        return True
    except Exception as e:
        print(f"❌ Redis failed: {e}")
        return False

def test_api():
    """Test API server"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ API server: Running")
            return True
        else:
            print(f"❌ API server: Status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API server failed: {e}")
        return False

def test_inference():
    """Test inference endpoint"""
    try:
        payload = {
            "text": "Что надеть на свидание?",
            "max_length": 100
        }
        response = requests.post("http://localhost:8000/inference/generate", 
                               json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Inference: {result.get('generated_text', 'Success')[:50]}...")
            return True
        else:
            print(f"❌ Inference: Status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        return False

def main():
    print("🧪 Testing Personal Stylist Services")
    print("=" * 40)
    
    # Test Redis
    redis_ok = test_redis()
    
    # Wait a bit for API to start
    print("⏳ Waiting for API server...")
    time.sleep(5)
    
    # Test API
    api_ok = test_api()
    
    if api_ok:
        # Test inference
        print("⏳ Testing inference (this may take a moment)...")
        inference_ok = test_inference()
    else:
        inference_ok = False
    
    print("\n" + "=" * 40)
    print("📊 Test Results:")
    print(f"Redis: {'✅' if redis_ok else '❌'}")
    print(f"API: {'✅' if api_ok else '❌'}")
    print(f"Inference: {'✅' if inference_ok else '❌'}")
    
    if all([redis_ok, api_ok, inference_ok]):
        print("\n🎉 All services are working!")
        return 0
    else:
        print("\n⚠️ Some services have issues")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 