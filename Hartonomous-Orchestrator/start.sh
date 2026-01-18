#!/bin/bash

echo "Building Docker image..."
docker-compose build

echo "Starting OpenAI gateway..."
docker-compose up -d

echo ""
echo "Gateway running at http://localhost:8700"
echo ""
echo "Test it with:"
echo "curl http://localhost:8700/health"
echo ""
echo "View logs:"
echo "docker-compose logs -f"
