#!/usr/bin/env python3

import sys
import os
from pathlib import Path
import asyncio
import time

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_celery_workers():
    """Test if Celery workers are running and responding"""
    print("Testing Celery Workers")
    print("=====================")
    
    try:
        # Test database worker
        print("\n1. Testing Database Worker...")
        from src.celery_db_app import app as db_app
        
        # Check if worker is active
        inspect = db_app.control.inspect()
        active_queues = inspect.active_queues()
        
        if active_queues:
            print("✅ Database worker is active")
            for worker, queues in active_queues.items():
                print(f"   Worker: {worker}")
                for queue in queues:
                    print(f"   Queue: {queue['name']}")
        else:
            print("❌ No active database workers found")
        
        # Test a simple task
        print("\n   Testing database task...")
        try:
            from src.workers.db_worker_safe import search_nodes
            result = search_nodes.delay("лето")
            
            # Wait for result with timeout
            task_result = result.get(timeout=10)
            if task_result and task_result.get('success') and task_result.get('nodes'):
                print(f"   ✅ Database task successful: found {len(task_result.get('nodes', []))} nodes")
            else:
                print("   ⚠️ Database task completed but no results found")
                print(f"   Task result: {task_result}")
        except Exception as e:
            print(f"   ❌ Database task failed: {e}")
            
    except Exception as e:
        print(f"❌ Error testing database worker: {e}")
    
    try:
        # Test inference worker
        print("\n2. Testing Inference Worker...")
        from src.celery_app import app as inference_app
        
        # Check if worker is active
        inspect = inference_app.control.inspect()
        active_queues = inspect.active_queues()
        
        if active_queues:
            print("✅ Inference worker is active")
            for worker, queues in active_queues.items():
                print(f"   Worker: {worker}")
                for queue in queues:
                    print(f"   Queue: {queue['name']}")
        else:
            print("❌ No active inference workers found")
            
        # Test health check
        print("\n   Testing inference health check...")
        try:
            from src.workers.inference_worker import health_check
            result = health_check.delay()
            
            # Wait for result with timeout
            task_result = result.get(timeout=10)
            if task_result and task_result.get('status') == 'healthy':
                print("   ✅ Inference worker health check passed")
            else:
                print(f"   ⚠️ Inference worker health check: {task_result}")
        except Exception as e:
            print(f"   ❌ Inference health check failed: {e}")
            
    except Exception as e:
        print(f"❌ Error testing inference worker: {e}")

def test_redis_connection():
    """Test Redis connection"""
    print("\n3. Testing Redis Connection...")
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, db=0)
        r.ping()
        print("✅ Redis connection successful")
        
        # Check Redis info
        info = r.info()
        print(f"   Redis version: {info.get('redis_version', 'unknown')}")
        print(f"   Connected clients: {info.get('connected_clients', 'unknown')}")
        
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")

def test_neo4j_connection():
    """Test Neo4j connection"""
    print("\n4. Testing Neo4j Connection...")
    try:
        from neo4j import GraphDatabase
        
        driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'password123'))
        with driver.session() as session:
            result = session.run("RETURN 1 as test")
            record = result.single()
            if record and record['test'] == 1:
                print("✅ Neo4j connection successful")
                
                # Check database stats
                stats_result = session.run("MATCH (n) RETURN count(n) as total_nodes")
                stats_record = stats_result.single()
                if stats_record:
                    print(f"   Total nodes in database: {stats_record['total_nodes']}")
            else:
                print("❌ Neo4j connection test failed")
        
        driver.close()
        
    except Exception as e:
        print(f"❌ Neo4j connection failed: {e}")

def main():
    print("Celery Workers and Infrastructure Test")
    print("=====================================")
    
    test_redis_connection()
    test_neo4j_connection()
    test_celery_workers()
    
    print("\n" + "="*50)
    print("Test completed!")
    print("\nTo monitor Celery workers in real-time, use:")
    print("  celery -A src.celery_db_app events")
    print("  celery -A src.celery_app events")
    print("\nTo see worker status:")
    print("  celery -A src.celery_db_app status")
    print("  celery -A src.celery_app status")

if __name__ == "__main__":
    main() 