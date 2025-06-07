#!/bin/bash

# Personal Stylist Workers Startup Script
# Usage: ./start_workers.sh [service_name]
# Services: db, inference, api, bot, all

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠️${NC} $1"
}

print_error() {
    echo -e "${RED}❌${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ️${NC} $1"
}

echo "Personal Stylist Workers Startup Script"
echo "========================================"

# Load environment variables from .env file if it exists
if [ -f .env ]; then
    print_info "Loading environment variables from .env file..."
    export $(grep -v '^#' .env | xargs)
else
    print_warning "No .env file found, using default environment variables"
fi

# Set up environment
print_info "Setting up environment..."
export PYTHONPATH="/app"
export PYTHONPATH="$(pwd)"
cd "$(dirname "$0")"
print_info "PYTHONPATH: $PYTHONPATH"
print_info "Working directory: $(pwd)"

# Function to check if a service is running
check_service() {
    local service=$1
    local port=$2
    if [ -n "$port" ]; then
        if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
            return 0
        else
            return 1
        fi
    else
        return 1
    fi
}

# Function to check prerequisites
check_prerequisites() {
    print_info "Checking prerequisites..."
    
    # Check Redis
    if check_service "redis" 6379; then
        print_status "Redis is running"
    else
        print_error "Redis is not running. Please start Redis first:"
        print_info "  brew services start redis"
        print_info "  or"
        print_info "  redis-server"
        exit 1
    fi
    
    # Check Neo4j
    if check_service "neo4j" 7687; then
        print_status "Neo4j is running"
    else
        print_error "Neo4j is not running. Please start Neo4j first:"
        print_info "  brew services start neo4j"
        print_info "  or"
        print_info "  neo4j start"
        exit 1
    fi
    
    print_status "All prerequisites met!"
}

# Function to initialize Neo4j database
init_neo4j_database() {
    print_info "Initializing Neo4j database with existing data..."
    
    # Check if Neo4j is accessible
    if ! python3 -c "
from neo4j import GraphDatabase
import sys
try:
    driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'password123'))
    with driver.session() as session:
        session.run('RETURN 1')
    driver.close()
    print('Neo4j connection successful')
except Exception as e:
    print(f'Neo4j connection failed: {e}')
    sys.exit(1)
" 2>/dev/null; then
        print_error "Cannot connect to Neo4j. Please check your Neo4j installation and password."
        print_info "Default password should be 'password123'"
        print_info "You can reset it with: neo4j-admin set-initial-password password123"
        exit 1
    fi
    
    # Run the initialization script
    if python3 scripts/init_local_neo4j.py; then
        print_status "Neo4j database initialized successfully with existing data"
    else
        print_error "Failed to initialize Neo4j database"
        print_info "You can try running manually: python3 scripts/init_local_neo4j.py"
        exit 1
    fi
}

# Function to find available port
find_available_port() {
    local start_port=$1
    local port=$start_port
    while lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; do
        port=$((port + 1))
    done
    echo $port
}

# Function to start database worker
start_db_worker() {
    print_info "Starting Database Worker..."
    pkill -f "celery.*database" 2>/dev/null || true
    sleep 2
    
    celery -A src.celery_db_app worker --loglevel=info -Q database --concurrency=2 &
    DB_WORKER_PID=$!
    sleep 3
    
    if kill -0 $DB_WORKER_PID 2>/dev/null; then
        print_status "Database worker started with PID: $DB_WORKER_PID"
        echo $DB_WORKER_PID > .db_worker.pid
    else
        print_error "Failed to start database worker"
        exit 1
    fi
}

# Function to start inference worker
start_inference_worker() {
    print_info "Starting Inference Worker..."
    pkill -f "celery.*inference" 2>/dev/null || true
    sleep 2
    
    celery -A src.celery_app worker --loglevel=info -Q inference --concurrency=1 &
    INFERENCE_WORKER_PID=$!
    sleep 3
    
    if kill -0 $INFERENCE_WORKER_PID 2>/dev/null; then
        print_status "Inference worker started with PID: $INFERENCE_WORKER_PID"
        echo $INFERENCE_WORKER_PID > .inference_worker.pid
    else
        print_error "Failed to start inference worker"
        exit 1
    fi
}

# Function to start API server
start_api_server() {
    print_info "Starting API Server..."
    pkill -f "uvicorn.*src.api.main" 2>/dev/null || true
    sleep 2
    
    # Find available port starting from 8000
    API_PORT=$(find_available_port 8000)
    if [ "$API_PORT" != "8000" ]; then
        print_warning "Port 8000 is busy, using port $API_PORT..."
    fi
    
    python -m uvicorn src.api.main:app --host 0.0.0.0 --port $API_PORT --reload &
    API_SERVER_PID=$!
    sleep 3
    
    if kill -0 $API_SERVER_PID 2>/dev/null; then
        print_status "API server started with PID: $API_SERVER_PID on port $API_PORT"
        echo $API_SERVER_PID > .api_server.pid
        echo $API_PORT > .api_port
    else
        print_error "Failed to start API server"
        exit 1
    fi
}

# Function to start Telegram bot
start_telegram_bot() {
    # Check if bot token is set
    if [ -z "$TELEGRAM_BOT_TOKEN" ]; then
        print_error "TELEGRAM_BOT_TOKEN is not set"
        print_info "Please set your Telegram bot token:"
        print_info "  export TELEGRAM_BOT_TOKEN='your_bot_token_here'"
        exit 1
    fi
    print_status "Telegram bot token is set"
    
    print_info "Starting Telegram Bot..."
    pkill -f "python.*src.bot.main" 2>/dev/null || true
    sleep 2
    
    # Get API port
    if [ -f .api_port ]; then
        API_PORT=$(cat .api_port)
    else
        API_PORT=8000
    fi
    
    # Set API URL for the bot
    export API_BASE_URL="http://localhost:$API_PORT"
    
    print_status "Telegram bot token is set"
    python src/bot/main.py &
    TELEGRAM_BOT_PID=$!
    sleep 3
    
    if kill -0 $TELEGRAM_BOT_PID 2>/dev/null; then
        print_status "Telegram bot started with PID: $TELEGRAM_BOT_PID"
        echo $TELEGRAM_BOT_PID > .telegram_bot.pid
    else
        print_error "Failed to start Telegram bot"
        exit 1
    fi
}

# Function to stop all services
stop_all_services() {
    print_info "Shutting down services..."
    
    # Stop processes using PID files
    for service in db_worker inference_worker api_server telegram_bot; do
        if [ -f ".$service.pid" ]; then
            PID=$(cat ".$service.pid")
            if kill -0 $PID 2>/dev/null; then
                kill $PID
                print_info "$(echo $service | tr '_' ' ' | sed 's/.*/\L&/; s/[a-z]*/\u&/g') stopped"
            fi
            rm -f ".$service.pid"
        fi
    done
    
    # Clean up port file
    rm -f .api_port
    
    # Kill any remaining celery processes
    pkill -f "celery.*worker" 2>/dev/null || true
    pkill -f "uvicorn.*src.api.main" 2>/dev/null || true
    pkill -f "python.*src.bot.main" 2>/dev/null || true
    
    print_status "All services stopped"
}

# Function to show status
show_status() {
    echo "Service Status:"
    echo "==============="
    
    # Check each service
    services=("db_worker:Database Worker" "inference_worker:Inference Worker" "api_server:API Server" "telegram_bot:Telegram Bot")
    
    for service_info in "${services[@]}"; do
        IFS=':' read -r service_file service_name <<< "$service_info"
        if [ -f ".$service_file.pid" ]; then
            PID=$(cat ".$service_file.pid")
            if kill -0 $PID 2>/dev/null; then
                print_status "$service_name: Running (PID: $PID)"
            else
                print_error "$service_name: Not running (stale PID file)"
                rm -f ".$service_file.pid"
            fi
        else
            print_error "$service_name: Not running"
        fi
    done
    
    if [ -f .api_port ]; then
        API_PORT=$(cat .api_port)
        echo ""
        print_info "API available at: http://localhost:$API_PORT"
        print_info "API docs at: http://localhost:$API_PORT/docs"
    fi
}

# Trap to handle Ctrl+C
trap 'stop_all_services; exit 0' INT TERM

# Main logic
SERVICE=${1:-all}

case $SERVICE in
    "db")
        check_prerequisites
        init_neo4j_database
        start_db_worker
        print_info "Database worker is running. Press Ctrl+C to stop."
        wait
        ;;
    "inference")
        check_prerequisites
        start_inference_worker
        print_info "Inference worker is running. Press Ctrl+C to stop."
        wait
        ;;
    "api")
        check_prerequisites
        start_api_server
        print_info "API server is running. Press Ctrl+C to stop."
        wait
        ;;
    "bot")
        check_prerequisites
        start_telegram_bot
        print_info "Telegram bot is running. Press Ctrl+C to stop."
        wait
        ;;
    "all")
        check_prerequisites
        init_neo4j_database
        start_db_worker
        start_inference_worker
        start_api_server
        start_telegram_bot
        
        # Get the API port for display
        if [ -f .api_port ]; then
            API_PORT=$(cat .api_port)
        else
            API_PORT=8000
        fi
        
        echo ""
        print_status "All services are running!"
        echo "- Database Worker: PID $DB_WORKER_PID"
        echo "- Inference Worker: PID $INFERENCE_WORKER_PID"
        echo "- API Server: PID $API_SERVER_PID on port $API_PORT"
        echo "- Telegram Bot: PID $TELEGRAM_BOT_PID"
        echo ""
        print_info "API available at: http://localhost:$API_PORT"
        print_info "API docs at: http://localhost:$API_PORT/docs"
        print_info "Telegram bot is running and ready to receive messages"
        print_info "Press Ctrl+C to stop all services"
        
        # Wait for all background processes
        wait
        ;;
    "stop")
        stop_all_services
        ;;
    "status")
        show_status
        ;;
    *)
        echo "Usage: $0 [db|inference|api|bot|all|stop|status]"
        echo ""
        echo "Services:"
        echo "  db        - Start only database worker"
        echo "  inference - Start only inference worker"
        echo "  api       - Start only API server"
        echo "  bot       - Start only Telegram bot"
        echo "  all       - Start all services (default)"
        echo "  stop      - Stop all services"
        echo "  status    - Show service status"
        exit 1
        ;;
esac 