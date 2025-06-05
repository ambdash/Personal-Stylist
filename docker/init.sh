#!/bin/bash

set -e

# Function to check if Neo4j is running
check_neo4j() {
    local port=$1
    local max_attempts=30
    local attempt=1

    echo "Checking Neo4j on port $port..."
    while [ $attempt -le $max_attempts ]; do
        if curl -s "http://localhost:$port" > /dev/null; then
            echo "Neo4j is up on port $port"
            return 0
        fi
        echo "Attempt $attempt/$max_attempts: Neo4j is not ready on port $port"
        sleep 2
        attempt=$((attempt + 1))
    done
    echo "Failed to start Neo4j on port $port after $max_attempts attempts"
    return 1
}

echo "Starting Neo4j..."
neo4j start

# Wait for Neo4j to be ready
if ! check_neo4j 7474; then
    echo "Neo4j failed to start on HTTP port"
    exit 1
fi

echo "Neo4j is up - waiting additional time for full initialization..."
sleep 15

echo "Running initialization script..."
cd /var/lib/neo4j/import
if ! python init_docker.py; then
    echo "Initialization script failed"
    exit 1
fi

echo "Initialization complete"

# Stop the background Neo4j
neo4j stop

# Start Neo4j in the foreground
exec neo4j console 