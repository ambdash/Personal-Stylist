#!/bin/bash

# Personal Stylist Workers Startup Script - LOCAL SETUP
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

echo "Personal Stylist Workers Startup Script - LOCAL SETUP"
echo "====================================================="

# Load environment variables from .env file if it exists
if [ -f .env ]; then
    print_info "Loading environment variables from .env file..."
    export $(grep -v '^#' .env | xargs)
else
    print_warning "No .env file found, using default environment variables"
    # Set default environment variables
    export REDIS_URL="redis://localhost:6379/0"
    export CELERY_BROKER_URL="redis://localhost:6379/0"
    export CELERY_RESULT_BACKEND="redis://localhost:6379/0"
    export NEO4J_URI="bolt://localhost:7687"
    export NEO4J_USER="neo4j"
    export NEO4J_PASSWORD="password123"
    export API_BASE_URL="http://localhost:8000"
    export PYTHONPATH="/home/admin/Personal-Stylist"
fi

# Set up environment
print_info "Setting up environment..."
export PYTHONPATH="$(pwd)"
cd "$(dirname "$0")"
print_info "PYTHONPATH: $PYTHONPATH"
print_info "Working directory: $(pwd)"

# Function to check if a service is running
check_service() {
    local service=$1
    local port=$2
    
    if [ -n "$port" ]; then
        # Try multiple methods to check if port is in use
        
        # Method 1: Try lsof
        if command -v lsof >/dev/null 2>&1; then
            if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
                return 0
            fi
        fi
        
        # Method 2: Try netstat
        if command -v netstat >/dev/null 2>&1; then
            if netstat -tlnp 2>/dev/null | grep ":$port " >/dev/null; then
                return 0
            fi
        fi
        
        # Method 3: Try ss
        if command -v ss >/dev/null 2>&1; then
            if ss -tlnp 2>/dev/null | grep ":$port " >/dev/null; then
                return 0
            fi
        fi
        
        # Method 4: For Redis specifically, try to connect with Python
        if [ "$service" = "redis" ] && [ "$port" = "6379" ]; then
            if python3.11 -c "
import socket
try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)
    result = sock.connect_ex(('127.0.0.1', 6379))
    sock.close()
    exit(0 if result == 0 else 1)
except:
    exit(1)
" 2>/dev/null; then
                return 0
            fi
        fi
        
        # Method 5: For Neo4j specifically, try to connect with Python
        if [ "$service" = "neo4j" ] && [ "$port" = "7687" ]; then
            if python3.11 -c "
import socket
try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)
    result = sock.connect_ex(('127.0.0.1', 7687))
    sock.close()
    exit(0 if result == 0 else 1)
except:
    exit(1)
" 2>/dev/null; then
                return 0
            fi
        fi
        
        return 1
    else
        return 1
    fi
}

# Function to install Redis on Ubuntu
install_redis_ubuntu() {
    print_info "Installing Redis on Ubuntu..."
    
    # Update package list and install Redis
    sudo apt-get update
    sudo apt-get install -y redis-server
    
    # Configure Redis without authentication
    print_info "Configuring Redis without authentication..."
    sudo sed -i 's/^# requirepass .*/# requirepass disabled/' /etc/redis/redis.conf 2>/dev/null || true
    sudo sed -i 's/^requirepass .*/# requirepass disabled/' /etc/redis/redis.conf 2>/dev/null || true
    
    # Ensure Redis binds to localhost
    sudo sed -i 's/^bind .*/bind 127.0.0.1/' /etc/redis/redis.conf 2>/dev/null || true
    
    # Enable and start Redis service
    sudo systemctl enable redis-server
    sudo systemctl restart redis-server
    
    # Wait for Redis to start
    sleep 3
    
    print_status "Redis installed and started"
}

# Function to install Neo4j natively on Ubuntu
install_neo4j_native() {
    print_info "Installing Neo4j natively on Ubuntu..."
    
    # Install Java if not present
    if ! command -v java &> /dev/null; then
        print_info "Installing Java..."
        sudo apt-get update
        sudo apt-get install -y openjdk-11-jdk
    fi
    
    # Add Neo4j repository
    wget -O - https://debian.neo4j.com/neotechnology.gpg.key | sudo apt-key add -
    echo 'deb https://debian.neo4j.com stable latest' | sudo tee -a /etc/apt/sources.list.d/neo4j.list
    
    # Update package list and install Neo4j
    sudo apt-get update
    sudo apt-get install -y neo4j
    
    # Set initial password using the correct command for newer Neo4j versions
    print_info "Setting Neo4j initial password..."
    if sudo neo4j-admin dbms set-initial-password password123 2>/dev/null; then
        print_status "Neo4j password set successfully"
    elif sudo neo4j-admin set-initial-password password123 2>/dev/null; then
        print_status "Neo4j password set successfully (legacy command)"
    else
        print_warning "Could not set Neo4j password automatically. You may need to set it manually."
        print_info "Try: sudo neo4j-admin dbms set-initial-password password123"
    fi
    
    # Enable and start Neo4j service
    sudo systemctl enable neo4j
    sudo systemctl start neo4j
    
    print_status "Neo4j installed and started"
}

# Function to test and fix Redis connection
test_redis_connection() {
    print_info "Testing Redis connection..."
    
    # Try to connect with Python
    if python3.11 -c "
import redis
try:
    r = redis.Redis(host='localhost', port=6379, decode_responses=True)
    r.ping()
    print('Redis connection successful')
    exit(0)
except redis.exceptions.AuthenticationError:
    print('Redis authentication error')
    exit(1)
except Exception as e:
    print(f'Redis connection failed: {e}')
    exit(2)
" 2>/dev/null; then
        print_status "Redis connection test passed"
        return 0
    else
        redis_exit_code=$?
        if [ $redis_exit_code -eq 1 ]; then
            print_warning "Redis requires authentication, fixing configuration..."
            
            # Fix Redis configuration
            if [ -f /etc/redis/redis.conf ]; then
                sudo sed -i 's/^requirepass .*/# requirepass disabled/' /etc/redis/redis.conf 2>/dev/null || true
                sudo sed -i 's/^# requirepass .*/# requirepass disabled/' /etc/redis/redis.conf 2>/dev/null || true
                sudo systemctl restart redis-server
                sleep 3
                
                # Test again
                if python3.11 -c "import redis; r = redis.Redis(host='localhost', port=6379); r.ping()" 2>/dev/null; then
                    print_status "Redis connection fixed"
                    return 0
                else
                    print_error "Failed to fix Redis authentication"
                    return 1
                fi
            else
                print_error "Redis config file not found"
                return 1
            fi
        else
            print_error "Redis connection failed with unknown error"
            return 1
        fi
    fi
}

# Function to check prerequisites
check_prerequisites() {
    print_info "Checking prerequisites..."
    
    # Check Redis
    REDIS_RUNNING=false
    if check_service "redis" 6379; then
        REDIS_RUNNING=true
    elif systemctl is-active --quiet redis-server 2>/dev/null; then
        print_info "Redis service is active according to systemctl"
        REDIS_RUNNING=true
    fi
    
    if [ "$REDIS_RUNNING" = "true" ]; then
        print_status "Redis is running"
        # Test Redis connection
        if ! test_redis_connection; then
            print_error "Redis connection test failed"
            exit 1
        fi
    else
        print_warning "Redis is not running."
        
        # Try to start Redis service if installed
        if command -v redis-server &> /dev/null || systemctl list-unit-files | grep -q redis-server; then
            print_info "Attempting to start Redis service..."
            
            # First, try to fix Redis configuration if it exists
            if [ -f /etc/redis/redis.conf ]; then
                print_info "Fixing Redis configuration..."
                sudo sed -i 's/^requirepass .*/# requirepass disabled/' /etc/redis/redis.conf 2>/dev/null || true
                sudo sed -i 's/^# requirepass .*/# requirepass disabled/' /etc/redis/redis.conf 2>/dev/null || true
                sudo sed -i 's/^bind .*/bind 127.0.0.1/' /etc/redis/redis.conf 2>/dev/null || true
            fi
            
            if sudo systemctl restart redis-server 2>/dev/null; then
                sleep 3
                if check_service "redis" 6379 || systemctl is-active --quiet redis-server; then
                    print_status "Redis started successfully"
                else
                    print_error "Failed to start Redis service"
                    exit 1
                fi
            else
                print_error "Failed to start Redis service"
                exit 1
            fi
        else
            # Redis not installed, offer installation
            print_error "Redis is not installed."
            read -p "Install Redis now? (y/N): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                install_redis_ubuntu
            else
                print_error "Redis is required. Please install it manually:"
                print_info "  sudo apt-get install redis-server"
                exit 1
            fi
        fi
    fi
    
    # Check Neo4j
    NEO4J_RUNNING=false
    if check_service "neo4j" 7687; then
        NEO4J_RUNNING=true
    elif systemctl is-active --quiet neo4j 2>/dev/null; then
        print_info "Neo4j service is active according to systemctl"
        NEO4J_RUNNING=true
    fi
    
    if [ "$NEO4J_RUNNING" = "true" ]; then
        print_status "Neo4j is running"
    else
        print_warning "Neo4j is not running."
        
        # Check if Neo4j is installed
        if command -v neo4j &> /dev/null || systemctl list-unit-files | grep -q neo4j; then
            print_info "Attempting to start Neo4j service..."
            if sudo systemctl start neo4j 2>/dev/null; then
                sleep 5
                if check_service "neo4j" 7687 || systemctl is-active --quiet neo4j; then
                    print_status "Neo4j started successfully"
                else
                    print_error "Failed to start Neo4j service"
                    exit 1
                fi
            else
                print_error "Failed to start Neo4j service"
                exit 1
            fi
        else
            # Neo4j not installed, offer installation
            print_error "Neo4j is not installed."
            read -p "Install Neo4j now? (y/N): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                install_neo4j_native
            else
                print_error "Neo4j is required. Please install it manually or use Docker setup."
                exit 1
            fi
        fi
    fi
    
    print_status "All prerequisites met!"
}

# Function to initialize database concepts
init_database_concepts() {
    print_info "Initializing database concepts..."
    
    # List of concept scripts to run
    concept_scripts=(
        "scripts/add_shoe_concepts.py"
        "scripts/add_dress_concepts.py" 
        "scripts/add_jeans_concepts.py"
    )
    
    # Check if concept scripts exist and run them
    for script in "${concept_scripts[@]}"; do
        if [ -f "$script" ]; then
            script_name=$(basename "$script" .py)
            print_info "Running $script_name..."
            
            if python3.11 "$script" 2>/dev/null; then
                print_status "$script_name completed successfully"
            else
                print_warning "$script_name failed or had warnings (this may be normal if concepts already exist)"
            fi
        else
            print_warning "Concept script not found: $script"
        fi
    done
    
    print_status "Database concept initialization completed"
}

# Function to check if required Python packages are available
check_python_packages() {
    print_info "Checking required Python packages..."
    
    # Check if neo4j package is available
    if ! python3.11 -c "import neo4j" 2>/dev/null; then
        print_error "Python neo4j package is not installed"
        print_info "Install with: pip3.11 install neo4j"
        return 1
    fi
    
    # Check if python-dotenv package is available
    if ! python3.11 -c "import dotenv" 2>/dev/null; then
        print_warning "Python python-dotenv package is not installed"
        print_info "Installing python-dotenv..."
        python3.11 -m pip install python-dotenv
    fi
    
    # Check if csv package is available (should be built-in)
    if ! python3.11 -c "import csv" 2>/dev/null; then
        print_error "Python csv package is not available"
        return 1
    fi
    
    print_status "Required Python packages are available"
    return 0
}

# Function to initialize Neo4j database
init_neo4j_database() {
    print_info "Checking Neo4j database status..."
    
    # Check if required Python packages are available
    if ! check_python_packages; then
        print_error "Cannot initialize Neo4j without required Python packages"
        return 1
    fi
    
    # Load environment variables
    NEO4J_USER_ENV=$(grep "^NEO4J_USER=" .env 2>/dev/null | cut -d'=' -f2 || echo "neo4j")
    NEO4J_PASSWORD_ENV=$(grep "^NEO4J_PASSWORD=" .env 2>/dev/null | cut -d'=' -f2 || echo "password123")
    NEO4J_URI_ENV=$(grep "^NEO4J_URI=" .env 2>/dev/null | cut -d'=' -f2 || echo "bolt://localhost:7687")
    
    print_info "Using Neo4j credentials from .env file: user=$NEO4J_USER_ENV"
    
    # Check if Neo4j is accessible
    if ! python3.11 -c "
from neo4j import GraphDatabase
import sys
try:
    driver = GraphDatabase.driver('$NEO4J_URI_ENV', auth=('$NEO4J_USER_ENV', '$NEO4J_PASSWORD_ENV'))
    with driver.session() as session:
        session.run('RETURN 1')
    driver.close()
    print('Neo4j connection successful')
except Exception as e:
    print(f'Neo4j connection failed: {e}')
    sys.exit(1)
" 2>/dev/null; then
        print_error "Cannot connect to Neo4j. Please check your Neo4j installation and password."
        print_info "Current credentials: user=$NEO4J_USER_ENV, password=$NEO4J_PASSWORD_ENV"
        print_info "You can reset it with: sudo neo4j-admin dbms set-initial-password $NEO4J_PASSWORD_ENV"
        return 1
    fi
    
    # Check if database already has data
    NODE_COUNT=$(python3.11 -c "
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv
load_dotenv()
driver = GraphDatabase.driver('$NEO4J_URI_ENV', auth=('$NEO4J_USER_ENV', '$NEO4J_PASSWORD_ENV'))
with driver.session() as session:
    result = session.run('MATCH (n) RETURN count(n) as count')
    count = result.single()['count']
    print(count)
driver.close()
" 2>/dev/null)
    
    if [ "$NODE_COUNT" -gt 100 ]; then
        print_status "Neo4j database already contains $NODE_COUNT nodes - skipping initialization"
        print_info "To force reinitialize, run: python3.11 scripts/init_local_neo4j.py"
        return 0
    fi
    
    print_info "Neo4j database is empty or has few nodes ($NODE_COUNT) - initializing with data..."
    
    # Check if data files exist - specifically look for new_nodes.csv and new_edges.csv
    NEO4J_DATA_DIR="src/api/db/neo4j"
    if [ ! -f "$NEO4J_DATA_DIR/new_nodes.csv" ] || [ ! -f "$NEO4J_DATA_DIR/new_edges.csv" ]; then
        print_warning "Neo4j data files not found in $NEO4J_DATA_DIR"
        print_info "Expected files: new_nodes.csv, new_edges.csv"
        
        # Check if we can find the source files and copy them
        if [ -f "src/data/neo4j_data/filtered_nodes.csv" ] && [ -f "src/data/neo4j_data/filtered_edges.csv" ]; then
            print_info "Found source data files, copying to expected location..."
            mkdir -p "$NEO4J_DATA_DIR"
            cp "src/data/neo4j_data/filtered_nodes.csv" "$NEO4J_DATA_DIR/new_nodes.csv"
            cp "src/data/neo4j_data/filtered_edges.csv" "$NEO4J_DATA_DIR/new_edges.csv"
            print_status "Data files copied successfully"
        else
            print_error "Cannot find Neo4j source data files."
            print_info "Expected source files:"
            print_info "  - src/data/neo4j_data/filtered_nodes.csv"
            print_info "  - src/data/neo4j_data/filtered_edges.csv"
            print_info "Please ensure the data files exist or run the data generation script first."
            return 1
        fi
        
        # Create basic Cypher script if it doesn't exist
        if [ ! -f "$NEO4J_DATA_DIR/clean_and_transform.cypher" ]; then
            print_info "clean_and_transform.cypher not found, but this is expected to exist"
            print_info "Please ensure the clean_and_transform.cypher file exists in $NEO4J_DATA_DIR"
        fi
    else
        print_status "Found required data files: new_nodes.csv and new_edges.csv"
    fi
    
    # Try using the separate initialization script first (more reliable)
    if [ -f "scripts/init_local_neo4j.py" ]; then
        print_info "Using separate initialization script with environment variables..."
        if python3.11 scripts/init_local_neo4j.py; then
            print_status "Neo4j database initialized successfully using separate script"
            
            # Initialize database concepts after successful database setup
            init_database_concepts
            return 0
        else
            print_warning "Separate initialization script failed, trying inline approach..."
        fi
    fi
    
    # Fallback to inline Python approach
    print_info "Using inline initialization approach..."
    
    # Run the initialization using Python
    if python3.11 -c "
import sys
import os
sys.path.append('$(pwd)')

from neo4j import GraphDatabase
import logging
from pathlib import Path
import csv
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Local Neo4j connection using environment variables
NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
NEO4J_USER = os.getenv('NEO4J_USER', 'neo4j')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', 'password123')

def run_cypher_statements(session, cypher_file):
    '''Run multiple Cypher statements from a file'''
    with open(cypher_file, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Split the content into individual statements
    statements = []
    current_statement = []
    
    for line in content.split('\n'):
        line = line.strip()
        if not line or line.startswith('//'):  # Skip empty lines and comments
            continue
        
        current_statement.append(line)
        if line.endswith(';'):  # Statement end
            statements.append(' '.join(current_statement))
            current_statement = []
    
    # Add any remaining statement
    if current_statement:
        statements.append(' '.join(current_statement))
    
    # Run each statement
    for statement in statements:
        if statement.strip():  # Skip empty statements
            try:
                session.run(statement)
                logger.info(f'Executed statement: {statement[:100]}...')
            except Exception as e:
                logger.error(f'Error executing statement: {statement[:100]}... Error: {e}')

def create_node_with_label(session, label, node_id, name):
    '''Create a node with the given label'''
    query = f'''
    CREATE (n:{label} {{id: \$id, name: \$name}})
    RETURN n
    '''
    session.run(query, {'id': node_id, 'name': name})

def create_relationship(session, start_id, end_id, rel_type):
    '''Create a relationship between nodes'''
    query = f'''
    MATCH (a {{id: \$start_id}})
    MATCH (b {{id: \$end_id}})
    CREATE (a)-[r:{rel_type}]->(b)
    RETURN r
    '''
    session.run(query, {'start_id': start_id, 'end_id': end_id})

def import_csv_data(driver, data_dir):
    '''Import CSV data into Neo4j'''
    try:
        with driver.session() as session:
            # Clear existing data
            session.run('MATCH (n) DETACH DELETE n')
            logger.info('Cleared existing data')
            
            # Get paths - use new_nodes.csv and new_edges.csv
            nodes_path = data_dir / 'new_nodes.csv'
            edges_path = data_dir / 'new_edges.csv'
            clean_cypher_path = data_dir / 'clean_and_transform.cypher'
            
            if not nodes_path.exists():
                raise FileNotFoundError(f'Nodes file not found: {nodes_path}')
            if not edges_path.exists():
                raise FileNotFoundError(f'Edges file not found: {edges_path}')
            
            # Import nodes
            node_count = 0
            with open(nodes_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    create_node_with_label(session, row['label'], row['id'], row['name'])
                    node_count += 1
                    if node_count % 1000 == 0:  # Progress indicator every 1000 nodes
                        logger.info(f'Imported {node_count} nodes...')
            logger.info(f'Imported {node_count} nodes from new_nodes.csv')
            
            # Import relationships
            edge_count = 0
            with open(edges_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    create_relationship(session, row['start'], row['end'], row['relation'])
                    edge_count += 1
                    if edge_count % 1000 == 0:  # Progress indicator every 1000 relationships
                        logger.info(f'Imported {edge_count} relationships...')
            logger.info(f'Imported {edge_count} relationships from new_edges.csv')
            
            # Run cleaning and transformation
            if clean_cypher_path.exists():
                logger.info('Running clean_and_transform.cypher script...')
                run_cypher_statements(session, clean_cypher_path)
                logger.info('Executed cleaning and transformation commands')
            else:
                logger.warning(f'Clean cypher file not found: {clean_cypher_path}')
                # Create basic indexes anyway
                basic_indexes = [
                    'CREATE INDEX IF NOT EXISTS FOR (n:Одежда) ON (n.id);',
                    'CREATE INDEX IF NOT EXISTS FOR (n:Аксессуар) ON (n.id);',
                    'CREATE INDEX IF NOT EXISTS FOR (n:Обувь) ON (n.id);',
                    'CREATE INDEX IF NOT EXISTS FOR (n:Цвет) ON (n.id);',
                    'CREATE INDEX IF NOT EXISTS FOR (n:Материал) ON (n.id);',
                    'CREATE INDEX IF NOT EXISTS FOR (n:Концепт) ON (n.id);',
                    'CREATE INDEX IF NOT EXISTS FOR (n:Эстетика) ON (n.id);'
                ]
                for index_query in basic_indexes:
                    session.run(index_query)
                logger.info('Created basic indexes')
            
            logger.info(f'Successfully imported and cleaned all data in Neo4j: {node_count} nodes, {edge_count} relationships')
    except Exception as e:
        logger.error(f'Error importing CSV data: {e}')
        raise

def init_neo4j():
    '''Initialize Neo4j database with data'''
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    try:
        # Use the Neo4j directory
        data_dir = Path('$NEO4J_DATA_DIR')
        
        logger.info(f'Using Neo4j directory: {data_dir}')
        
        # Verify directory exists
        if not data_dir.exists():
            raise FileNotFoundError(f'Neo4j directory not found: {data_dir}')
        
        # Import data into Neo4j
        import_csv_data(driver, data_dir)
        
        logger.info('Neo4j database initialized successfully')
    except Exception as e:
        logger.error(f'Error initializing Neo4j: {e}')
        raise
    finally:
        driver.close()

if __name__ == '__main__':
    init_neo4j()
"; then
        print_status "Neo4j database initialized successfully with existing data"
        
        # Initialize database concepts after successful database setup
        init_database_concepts
    else
        print_error "Failed to initialize Neo4j database"
        print_info "You can try running manually: python3.11 scripts/init_local_neo4j.py"
        return 1
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
    
    celery -A src.celery_inference_app worker --loglevel=info -Q inference --concurrency=1 --pool=solo &
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
    
    python3.11 -m uvicorn src.api.main:app --host 0.0.0.0 --port $API_PORT --reload &
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
    
    print_info "Starting Telegram Bot..."
    pkill -f "python3.11.*src.bot.main" 2>/dev/null || true
    sleep 2
    
    # Get API port
    if [ -f .api_port ]; then
        API_PORT=$(cat .api_port)
    else
        API_PORT=8000
    fi
    
    # Set API URL for the bot
    export API_BASE_URL="http://localhost:$API_PORT"
    
    python3.11 src/bot/main.py &
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
    pkill -f "python3.11.*src.bot.main" 2>/dev/null || true
    
    print_status "All services stopped"
}

# Function to show status
show_status() {
    echo "Service Status:"
    echo "==============="
    
    # Check Redis status
    if check_service "redis" 6379; then
        print_status "Redis: Running"
    else
        print_error "Redis: Not running"
    fi
    
    # Check Neo4j status
    if check_service "neo4j" 7687; then
        print_status "Neo4j: Running"
    else
        print_error "Neo4j: Not running"
    fi
    
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
    
    echo ""
    print_info "Neo4j Web Interface: http://localhost:7474"
    print_info "Neo4j Bolt Port: 7687"
}

# Trap to handle Ctrl+C
trap 'stop_all_services; exit 0' INT TERM

# Main logic
SERVICE=${1:-all}

case $SERVICE in
    "redis")
        if check_service "redis" 6379; then
            print_status "Redis is already running"
        else
            if command -v redis-server &> /dev/null; then
                sudo systemctl start redis-server
                print_status "Redis started"
            else
                install_redis_ubuntu
            fi
        fi
        ;;
    "neo4j")
        if check_service "neo4j" 7687; then
            print_status "Neo4j is already running"
        else
            if command -v neo4j &> /dev/null; then
                sudo systemctl start neo4j
                print_status "Neo4j started"
            else
                install_neo4j_native
            fi
        fi
        ;;
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
        echo "Usage: $0 [redis|neo4j|db|inference|api|bot|all|stop|status]"
        echo ""
        echo "Services:"
        echo "  redis     - Start Redis database"
        echo "  neo4j     - Start Neo4j database"
        echo "  db        - Start only database worker"
        echo "  inference - Start only inference worker"
        echo "  api       - Start only API server"
        echo "  bot       - Start only Telegram bot"
        echo "  all       - Start all services (default)"
        echo "  stop      - Stop all services"
        echo "  status    - Show service status"
        echo ""
        echo "Prerequisites:"
        echo "  - Redis server (will be installed if missing)"
        echo "  - Neo4j database (will be installed if missing)"
        echo "  - Python dependencies (install with: pip install -r requirements.txt)"
        echo ""
        echo "Environment Variables:"
        echo "  TELEGRAM_BOT_TOKEN - Required for bot service"
        exit 1
        ;;
esac 