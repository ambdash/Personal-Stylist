#!/bin/bash

# Function to check if a port is in use
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null ; then
        return 0
    else
        return 1
    fi
}

# Function to kill process using a port
kill_port_process() {
    local port=$1
    local pid=$(lsof -ti :$port)
    if [ ! -z "$pid" ]; then
        echo "Killing process using port $port..."
        kill -9 $pid
    fi
}

# Function to kill processes by name pattern
kill_process_by_pattern() {
    local pattern=$1
    pkill -f "$pattern" || true
}

# Function to cleanup all processes
cleanup() {
    echo "Cleaning up processes..."
    # Stop Redis
    brew services stop redis 2>/dev/null || true
    redis-cli shutdown 2>/dev/null || true
    
    # Kill processes
    kill_process_by_pattern "celery worker"
    kill_process_by_pattern "celery -A src.celery_app flower"
    kill_process_by_pattern "uvicorn src.api.main:app"
    kill_process_by_pattern "python -m src.bot.main"
    
    # Kill processes on specific ports
    kill_port_process 8000  # FastAPI
    kill_port_process 5555  # Flower
    kill_port_process 6379  # Redis
    
    echo "Cleanup complete"
}

# Register cleanup function to run on script exit
trap cleanup EXIT

# Function to wait for service
wait_for_service() {
    local port=$1
    local service=$2
    local max_attempts=30
    local attempt=1
    
    echo "Waiting for $service to be ready..."
    while ! check_port $port; do
        if [ $attempt -ge $max_attempts ]; then
            echo "$service failed to start"
            exit 1
        fi
        echo "Waiting for $service... attempt $attempt/$max_attempts"
        sleep 1
        attempt=$((attempt + 1))
    done
    echo "$service is ready!"
}

# Clean up any existing processes before starting
cleanup

# Print startup banner
echo "==================================================="
echo "Personal Stylist Development Environment"
echo "==================================================="
echo "The following services will be started:"
echo "1. Redis (port 6379)"
echo "2. Neo4j (ports 7474, 7687)"
echo "3. FastAPI (port 8000)"
echo "4. Celery Workers (db and llm queues)"
echo "5. Celery Flower monitoring (port 5555)"
echo ""
echo "Credentials for services:"
echo "- Celery Flower UI: http://localhost:5555"
echo "  Username: admin"
echo "  Password: admin"
echo "- FastAPI Swagger UI: http://localhost:8000/docs"
echo "==================================================="
echo ""

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "Homebrew is not installed. Please install it first."
    exit 1
fi

# Install Redis if not installed
if ! command -v redis-server &> /dev/null; then
    echo "Installing Redis..."
    brew install redis
fi

# Install Neo4j if not installed
if ! command -v neo4j &> /dev/null; then
    echo "Installing Neo4j..."
    brew install neo4j
fi

# Stop any running Redis instance
echo "Stopping Redis..."
brew services stop redis 2>/dev/null || true
redis-cli shutdown 2>/dev/null || true
sleep 2

# Create Redis config file
REDIS_CONF="/tmp/redis.conf"
cat > $REDIS_CONF << EOL
requirepass redis_password
port 6379
bind 127.0.0.1
daemonize yes
maxmemory 512mb
maxmemory-policy allkeys-lru
appendonly yes
EOL

# Start Redis with config file
echo "Starting Redis..."
redis-server $REDIS_CONF
wait_for_service 6379 "Redis"

# Test Redis connection
echo "Testing Redis connection..."
if ! redis-cli -a redis_password ping | grep -q "PONG"; then
    echo "Redis connection test failed"
    exit 1
fi
echo "Redis connection successful!"

# Start Neo4j if not running
if ! check_port 7687; then
    echo "Starting Neo4j..."
    neo4j start
    wait_for_service 7687 "Neo4j"
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
else
    source .venv/bin/activate
fi

# Set environment variables
export REDIS_HOST=localhost
export REDIS_PORT=6379
export REDIS_PASSWORD=redis_password
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=password123

# Kill any existing Celery processes
pkill -f 'celery worker' || true
sleep 2

# Start Celery workers
echo "Starting Celery workers..."
# DB worker
celery -A src.celery_app worker --loglevel=info -Q db -n db_worker@%h &
# LLM worker
celery -A src.celery_app worker --loglevel=info -Q llm -n llm_worker@%h &

# Start Celery Flower for monitoring
echo "Starting Celery Flower..."
# Kill any process using port 5555
kill_port_process 5555
celery -A src.celery_app flower --port=5555 --basic_auth=admin:admin &

# Wait for Celery workers to initialize
sleep 5

# Start FastAPI
echo "Starting FastAPI..."
# Kill any process using port 8000
kill_port_process 8000
uvicorn src.api.main:app --host localhost --port 8000 --reload &

# Wait for FastAPI to initialize
wait_for_service 8000 "FastAPI"

echo ""
echo "==================================================="
echo "All services are running!"
echo "- FastAPI is available at: http://localhost:8000"
echo "- Swagger UI is available at: http://localhost:8000/docs"
echo "- Celery Flower UI is available at: http://localhost:5555"
echo "  (use admin:admin to login)"
echo "==================================================="

# Start Telegram Bot
echo "Starting Telegram Bot..."
# Kill any existing bot processes first
kill_process_by_pattern "python -m src.bot.main"
sleep 2
python -m src.bot.main &
BOT_PID=$!

# Wait for interrupt
echo "Services are running. Press Ctrl+C to stop."
wait $BOT_PID 