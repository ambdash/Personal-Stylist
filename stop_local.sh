#!/bin/bash

# Local Development Stop Script
echo "🛑 Stopping Personal Stylist Local Development Environment"

# Stop Python processes
echo "🔄 Stopping Python processes..."
pkill -f "celery.*worker" 2>/dev/null && echo "✅ Stopped Celery workers"
pkill -f "uvicorn.*src.api.main" 2>/dev/null && echo "✅ Stopped FastAPI"
pkill -f "python.*src.bot.main" 2>/dev/null && echo "✅ Stopped Telegram bot"

# Stop Docker containers
echo "🔄 Stopping Docker containers..."
docker stop redis-simple neo4j-simple 2>/dev/null && echo "✅ Stopped Redis and Neo4j"
docker rm redis-simple neo4j-simple 2>/dev/null && echo "✅ Removed containers"

echo "🎉 All services stopped!" 