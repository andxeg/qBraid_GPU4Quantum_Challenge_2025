#!/bin/bash

echo "[qBraid] Starting postStart.sh..."

# 1. Upgrade pip
echo "[qBraid] Upgrading pip..."
pip install --upgrade pip

# 2. Clone QOkit repo
echo "\n[qBraid] Cloning QOkit repository...\n"
git clone https://github.com/jpmorganchase/QOKit.git

# 3. Install QOkit with GPU-CUDA12 support
echo "\n[qBraid] Installing QOkit in editable mode with GPU-CUDA12 support...\n"
cd QOKit
pip install -e .[GPU-CUDA12]
cd ..

# 4. Install project requirements
echo "\n[qBraid] Installing your project dependencies...\n"
pip install -r requirements.txt

# 5. Clean pip cache
echo "\n[qBraid] Purging pip cache...\n"
pip cache purge

# 6. Clean temporary files
echo "\n[qBraid] Removing temporary files...\n"
rm -rf /tmp/* ~/.cache/pip ~/.cache/matplotlib

echo "\n[qBraid] All done! Environment is ready to use."
