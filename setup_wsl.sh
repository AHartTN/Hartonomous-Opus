#!/bin/bash

# Add PostgreSQL official repository for latest version
echo "Adding PostgreSQL official repository..."
sudo apt install -y curl ca-certificates
sudo rm -f /usr/share/keyrings/postgresql-keyring.gpg
curl -fsSL https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo gpg --dearmor -o /usr/share/keyrings/postgresql-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/postgresql-keyring.gpg] https://apt.postgresql.org/pub/repos/apt noble-pgdg main" | sudo tee /etc/apt/sources.list.d/postgresql.list

# Install ninja, PostgreSQL 18, PostGIS, and dotnet
echo "Installing ninja-build, PostgreSQL 18, PostGIS, and dotnet-sdk-9.0..."
sudo apt update
sudo apt install -y ninja-build postgresql-18 postgresql-contrib-18 postgresql-18-postgis-3 libpq-dev dotnet-sdk-10.0

# Ensure PostgreSQL is running
echo "Starting PostgreSQL service..."
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Create PostgreSQL user and database
echo "Setting up PostgreSQL user and database..."
sudo -u postgres dropuser hartonomous || true
sudo -u postgres dropdb hypercube || true
sudo -u postgres createuser --createdb --login hartonomous
sudo -u postgres psql -c "ALTER USER hartonomous PASSWORD 'hartonomous';"
sudo -u postgres createdb hypercube -O hartonomous

# Build and install PostgreSQL extensions
echo "Building and installing PostgreSQL extensions..."
cd cpp
rm -rf build
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
ninja
sudo ninja install

# Load schema
echo "Loading database schema..."
PGPASSWORD=hartonomous psql -h localhost -U hartonomous -d hypercube -f sql/hypercube_schema.sql

echo "Setup complete. Please source your .bashrc or restart terminal for PIP_CACHE_DIR."