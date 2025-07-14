#!/bin/bash

echo "[qBraid] Starting postStart.sh..."

# 1. Upgrade pip
echo "[qBraid] Upgrading pip..."
pip install --upgrade pip

# 2. Clone QOkit repo
echo "\n[qBraid] Cloning QOkit repository...\n"
git clone https://github.com/jpmorganchase/QOKit.git

# 3. Replace QOkit/qokit/portfolio_optimization.py with your custom version
echo -e "\n[qBraid] Replacing QOkit/qokit/portfolio_optimization.py with your custom version...\n"
cp benchmarks/portfolio_optimization.py QOkit/qokit/portfolio_optimization.py

# 4. Install QOkit with GPU-CUDA12 support
echo "\n[qBraid] Installing QOkit in editable mode with GPU-CUDA12 support...\n"
cd QOKit
pip install -e .[GPU-CUDA12]
cd ..

# 5. Install project requirements
echo "\n[qBraid] Installing your project dependencies...\n"
pip install -r requirements.txt

# 6. Install Git LFT
echo -e "\n[qBraid] Installing Git LFS...\n"
sudo apt update && sudo apt install -y git-lfs
git lfs install

# 7. Clean pip cache
echo "\n[qBraid] Purging pip cache...\n"
pip cache purge

# 8. Clean temporary files
echo "\n[qBraid] Removing temporary files...\n"
rm -rf /tmp/* ~/.cache/pip ~/.cache/matplotlib

echo "\n[qBraid] All done! Environment is ready to use."
