#!/bin/bash
#
# DTOcean Project Setup Script for Docker Sandbox
# This script installs the DTOcean monorepo project with all dependencies
#

set -e  # Exit on error

echo "=========================================="
echo "DTOcean Project Setup"
echo "=========================================="

# ============================================================================
# Step 1: Install Python 3.13 using uv
# ============================================================================
echo ""
echo "Step 1: Installing Python 3.13..."
uv python install 3.13

# ============================================================================
# Step 2: Install Poetry
# ============================================================================
echo ""
echo "Step 2: Installing Poetry..."
uv tool install poetry

# ============================================================================
# Step 3: Configure Poetry environment
# ============================================================================
echo ""
echo "Step 3: Configuring Poetry environment..."
poetry env use python3.13

# ============================================================================
# Step 4: Install Poetry plugins
# ============================================================================
echo ""
echo "Step 4: Installing Poetry plugins..."
poetry self add poetry-monoranger-plugin
poetry self add poetry-dynamic-versioning

# ============================================================================
# Step 5: Install project dependencies and packages
# ============================================================================
echo ""
echo "Step 5: Installing DTOcean packages (this may take a while)..."
poetry install --with test --with audit --with docs --with release --with tox

# ============================================================================
# Step 6: Pull Git LFS files
# ============================================================================
echo ""
echo "Step 6: Pulling Git LFS files..."
git lfs install
git lfs pull

# ============================================================================
# Step 7: Initialize DTOcean data files
# ============================================================================
echo ""
echo "Step 7: Initializing DTOcean data files..."
poetry run dtocean init

# ============================================================================
# Step 8: Verify installation
# ============================================================================
echo ""
echo "Step 8: Verifying installation..."
poetry run dtocean -h

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
