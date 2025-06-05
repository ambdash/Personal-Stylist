#!/bin/bash

# Stop any running containers
docker-compose down

# Remove existing Neo4j volume if it exists
docker volume rm personal-stylist_neo4j_data || true

# Start Neo4j container
docker-compose up -d neo4j

# Wait for Neo4j to be ready
echo "Waiting for Neo4j to be ready..."
sleep 30

# Run initialization script
docker-compose run --rm api python -m src.api.db.neo4j_init

# Stop containers
docker-compose down

echo "Neo4j initialization complete! The data is now stored in the Docker volume." 