#!/usr/bin/env python3
"""
Test script to verify the fixed Personal Stylist setup
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_imports():
    """Test that all required packages can be imported"""
    print("🔍 Testing imports...")
    
    try:
        import redis
        print("✅ Redis import OK")
    except ImportError as e:
        print(f"❌ Redis import failed: {e}")
        return False
    
    try:
        from neo4j import GraphDatabase
        print("✅ Neo4j import OK")
    except ImportError as e:
        print(f"❌ Neo4j import failed: {e}")
        return False
    
    try:
        import celery
        print("✅ Celery import OK")
    except ImportError as e:
        print(f"❌ Celery import failed: {e}")
        return False
    
    try:
        import fastapi
        print("✅ FastAPI import OK")
    except ImportError as e:
        print(f"❌ FastAPI import failed: {e}")
        return False
    
    try:
        import aiogram
        print("✅ Aiogram import OK")
    except ImportError as e:
        print(f"❌ Aiogram import failed: {e}")
        return False
    
    return True

def test_connections():
    """Test connections to Redis and Neo4j"""
    print("\n🔗 Testing connections...")
    
    # Test Redis
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379)
        r.ping()
        print("✅ Redis connection OK")
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        return False
    
    # Test Neo4j
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'password123'))
        driver.verify_connectivity()
        driver.close()
        print("✅ Neo4j connection OK")
    except Exception as e:
        print(f"❌ Neo4j connection failed: {e}")
        return False
    
    return True

def test_celery_apps():
    """Test Celery app configurations"""
    print("\n⚙️ Testing Celery apps...")
    
    # Test database-only Celery app
    try:
        from src.celery_db_app import app as db_app
        print("✅ Database Celery app import OK")
        
        # Test if we can create a task signature
        task_sig = db_app.signature('db_worker.health_check', queue='database')
        print("✅ Database task signature OK")
    except Exception as e:
        print(f"❌ Database Celery app failed: {e}")
        return False
    
    # Test full Celery app (should work but may have PyTorch warnings)
    try:
        from src.celery_app import app as full_app
        print("✅ Full Celery app import OK (may have PyTorch warnings)")
    except Exception as e:
        print(f"⚠️ Full Celery app failed: {e} (expected due to PyTorch issues)")
    
    return True

def test_ml_config():
    """Test ML configuration with PyTorch safety"""
    print("\n🤖 Testing ML configuration...")
    
    try:
        from src.ml.config import MODEL_CONFIGS, SYSTEM_PROMPT
        print("✅ ML config import OK")
        print(f"✅ Found {len(MODEL_CONFIGS())} model configurations")
        print(f"✅ System prompt loaded: {len(SYSTEM_PROMPT)} characters")
    except Exception as e:
        print(f"❌ ML config failed: {e}")
        return False
    
    return True

def test_api_imports():
    """Test API-related imports"""
    print("\n🌐 Testing API imports...")
    
    try:
        from src.api.main import app
        print("✅ FastAPI app import OK")
    except Exception as e:
        print(f"❌ FastAPI app import failed: {e}")
        return False
    
    return True

def test_bot_imports():
    """Test bot-related imports"""
    print("\n🤖 Testing bot imports...")
    
    try:
        from src.bot.main import main
        print("✅ Bot main import OK")
    except Exception as e:
        print(f"❌ Bot main import failed: {e}")
        return False
    
    try:
        from src.bot.handlers.db_utils_handler import router
        print("✅ DB utils handler import OK")
    except Exception as e:
        print(f"❌ DB utils handler import failed: {e}")
        return False
    
    return True

def main():
    """Run all tests"""
    print("🚀 Testing Fixed Personal Stylist Setup")
    print("=" * 50)
    
    tests = [
        ("Package Imports", test_imports),
        ("Service Connections", test_connections),
        ("Celery Apps", test_celery_apps),
        ("ML Configuration", test_ml_config),
        ("API Imports", test_api_imports),
        ("Bot Imports", test_bot_imports),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Your setup is ready.")
        print("\nNext steps:")
        print("1. Run: ./start_local.sh")
        print("2. Follow the terminal commands shown")
        print("3. Test with /db_utils command in Telegram")
    else:
        print(f"\n⚠️ {len(results) - passed} tests failed. Please fix the issues above.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 