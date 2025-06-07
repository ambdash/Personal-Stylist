#!/bin/bash

# Personal Stylist Docker Setup Script
echo "🚀 Starting Personal Stylist Docker Setup..."

# Check if .env file exists
if [ ! -f .env ]; then
    echo "❌ .env file not found!"
    echo "Please create a .env file with the following variables:"
    echo ""
    echo "TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here"
    echo "NEO4J_URI=bolt://neo4j:7687"
    echo "NEO4J_USER=neo4j"
    echo "NEO4J_PASSWORD=password123"
    echo "REDIS_URL=redis://redis:6379/0"
    echo "MODEL_NAME=t-tech/T-lite-it-1.0"
    echo "ADAPTER_PATH=src/ml/models/finetuned/t_lite"
    echo "CUDA_VISIBLE_DEVICES=0"
    echo ""
    exit 1
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose not found. Please install docker-compose."
    exit 1
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p src/ml/models/finetuned/t_lite
mkdir -p data/models/cache

# Build and start services
echo "🔨 Building Docker images..."
docker-compose build

echo "🚀 Starting services..."
docker-compose up -d

# Wait for services to be healthy
echo "⏳ Waiting for services to be ready..."
sleep 30

# Check service status
echo "📊 Service Status:"
docker-compose ps

echo ""
echo "✅ Setup complete!"
echo ""
echo "🌐 Services available at:"
echo "  - API: http://localhost:8002"
echo "  - Neo4j Browser: http://localhost:7474"
echo "  - Flower (Celery Monitor): http://localhost:5555"
echo ""
echo "📱 Your Telegram bot should now be running!"
echo ""
echo "🔧 Useful commands:"
echo "  - View logs: docker-compose logs -f [service_name]"
echo "  - Stop services: docker-compose down"
echo "  - Restart services: docker-compose restart"
echo "  - View service status: docker-compose ps" 