#!/bin/bash

# Local Development Startup Script
echo "🚀 Starting Personal Stylist Local Development Environment"

# Set Python path
export PYTHONPATH=/Users/dprudnikova/Personal-Stylist

# Check if Redis and Neo4j are running
echo "📋 Checking services..."

# Check Redis
if ! docker ps | grep -q redis-simple; then
    echo "🔴 Redis not running. Starting Redis..."
    docker run -d --name redis-simple -p 6379:6379 redis:7-alpine
    sleep 5
else
    echo "✅ Redis is running"
fi

# Check Neo4j
if ! docker ps | grep -q neo4j-simple; then
    echo "🔴 Neo4j not running. Starting Neo4j..."
    docker run -d --name neo4j-simple -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password123 neo4j:5.15
    echo "⏳ Waiting for Neo4j to start (30 seconds)..."
    sleep 30
else
    echo "✅ Neo4j is running"
fi

# Test connections
echo "🔍 Testing connections..."

# Test Redis
if python -c "import redis; r = redis.Redis(host='localhost', port=6379); r.ping()" 2>/dev/null; then
    echo "✅ Redis connection OK"
else
    echo "❌ Redis connection failed"
    exit 1
fi

# Test Neo4j
if python -c "from neo4j import GraphDatabase; driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'password123')); driver.verify_connectivity()" 2>/dev/null; then
    echo "✅ Neo4j connection OK"
else
    echo "❌ Neo4j connection failed"
    exit 1
fi

echo ""
echo "🎉 All services are ready!"
echo ""
echo "Now open 4 separate terminals and run:"
echo ""
echo "Terminal 1 - Database Worker:"
echo "cd /Users/dprudnikova/Personal-Stylist"
echo "export PYTHONPATH=/Users/dprudnikova/Personal-Stylist"
echo "celery -A src.celery_db_app worker --loglevel=info -Q database --concurrency=4"
echo ""
echo "Terminal 2 - Skip inference worker for now (PyTorch issues)"
echo ""
echo "Terminal 3 - FastAPI:"
echo "cd /Users/dprudnikova/Personal-Stylist"
echo "export PYTHONPATH=/Users/dprudnikova/Personal-Stylist"
echo "uvicorn src.api.main:app --host 0.0.0.0 --port 8002 --reload"
echo ""
echo "Terminal 4 - Telegram Bot:"
echo "cd /Users/dprudnikova/Personal-Stylist"
echo "export PYTHONPATH=/Users/dprudnikova/Personal-Stylist"
echo "python -m src.bot.main"
echo ""
echo "🌐 Monitoring URLs:"
echo "- Neo4j Browser: http://localhost:7474 (neo4j/password123)"
echo "- FastAPI Docs: http://localhost:8002/docs"
echo ""
echo "To stop everything:"
echo "./stop_local.sh" 