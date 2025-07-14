#!/bin/bash

echo "[qBraid] Starting postStart.sh..."

# 1. Upgrade pip
echo "[qBraid] Upgrading pip..."
pip install --upgrade pip

# 2. Clone QOkit repo
echo "[qBraid] Cloning QOkit repository..."
git clone https://github.com/qbraid/QOkit.git

# 3. Install QOkit with GPU-CUDA12 support
echo "[qBraid] ⚙️ Installing QOkit in editable mode with GPU-CUDA12 support..."
cd QOkit
pip install -e .[GPU-CUDA12]
cd ..

# 4. Install project requirements
echo "[qBraid] Installing your project dependencies..."
pip install -r requirements.txt

# 5. Clean pip cache
echo "[qBraid] Purging pip cache..."
pip cache purge

# 6. Clean temporary files
echo "[qBraid] Removing temporary files..."
rm -rf /tmp/* ~/.cache/pip ~/.cache/matplotlib

echo "[qBraid] All done! Environment is ready to use."
