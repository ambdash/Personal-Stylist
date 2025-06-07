#!/usr/bin/env python3

from src.celery_db_app import app as celery_app
import time

def test_celery_search():
    """Test the Celery search task directly"""
    print("Testing Celery search task...")
    
    # Test the search task directly - use the safe worker task name
    task = celery_app.send_task(
        'db_worker_safe.search_nodes', 
        args=['лето'], 
        kwargs={'limit': 20}, 
        queue='database'
    )
    print(f'Task ID: {task.id}')
    
    # Wait for result
    for i in range(30):
        if task.state == 'SUCCESS':
            print(f'Result: {task.result}')
            break
        elif task.state == 'FAILURE':
            print(f'Failed: {task.info}')
            break
        else:
            print(f'State: {task.state}')
            time.sleep(1)
    
    print("\nTesting with 'кроссовки'...")
    task2 = celery_app.send_task(
        'db_worker_safe.search_nodes', 
        args=['кроссовки'], 
        kwargs={'limit': 20}, 
        queue='database'
    )
    print(f'Task ID: {task2.id}')
    
    # Wait for result
    for i in range(30):
        if task2.state == 'SUCCESS':
            print(f'Result: {task2.result}')
            break
        elif task2.state == 'FAILURE':
            print(f'Failed: {task2.info}')
            break
        else:
            print(f'State: {task2.state}')
            time.sleep(1)

if __name__ == "__main__":
    test_celery_search() 