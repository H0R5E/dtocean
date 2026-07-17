#!/bin/bash
#
# System Setup Script for Docker Sandbox
# This script installs the system dependencies for DTOcean
#

set -e  # Exit on error

echo "=========================================="
echo "System Setup"
echo "=========================================="

# ============================================================================
# Step 1: Update package lists
# ============================================================================
echo ""
echo "Step 1: Updating package lists..."
sudo apt-get update

# ============================================================================
# Step 2: Install system dependencies
# ============================================================================
echo ""
echo "Step 2: Installing system dependencies..."
sudo apt-get install -y \
    build-essential \
    gfortran \
    meson \
    ninja-build \
    python3-dev \
    libglib2.0-0 \
    libxkbcommon0 \
    libfontconfig1 \
    libgl1 \
    libdbus-1-3 \
    libx11-xcb1 \
    libxcb-keysyms1 \
    libxcb-image0 \
    libxcb-shm0 \
    libxcb-icccm4 \
    libxcb-sync1 \
    libxcb-xfixes0 \
    libxcb-shape0 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-xinerama0 \
    libxcb-cursor0 \
    libxcb-xkb1 \
    libxkbcommon-x11-0 \
    libnss3 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    libxtst6 \
    libxi6 \
    libxss1 \
    libasound2t64 \
    libegl1 \
    libxfixes3 \
    libpulse0 \
    libxkbfile1 \
    libxcursor1 \
    libwoff1 \
    libevent-2.1-7t64 \
    libxslt1.1 \
    git-lfs

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
