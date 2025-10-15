#!/bin/bash

printf "\n[qBraid] Starting postStart.sh...\n"

# 1. Upgrade pip
printf "\n[qBraid] Upgrading pip...\n"
pip install --upgrade pip

# 2. Clone QOkit repo
printf "\n[qBraid] Cloning QOkit repository...\n"
git clone https://github.com/jpmorganchase/QOKit.git

# 3. Replace QOkit/qokit/portfolio_optimization.py with your custom version
printf "\n[qBraid] Replacing QOkit/qokit/portfolio_optimization.py with your custom version...\n"
cp benchmarks/portfolio_optimization.py QOKit/qokit/portfolio_optimization.py

# 4. Install QOkit with GPU-CUDA12 support
printf "\n[qBraid] Installing QOkit in editable mode with GPU-CUDA12 support...\n"
cd QOKit
pip install -e .[GPU-CUDA12]
cd ..

# 5. Install project requirements
printf "\n[qBraid] Installing your project dependencies...\n"
pip install -r requirements.txt

# # 6. Install Git LFT
# printf "\n[qBraid] Installing Git LFS...\n"
# sudo apt update && sudo apt install -y git-lfs
# git lfs install

# 7. Clean pip cache
printf "\n[qBraid] Purging pip cache...\n"
pip cache purge

# 8. Clean temporary files
printf "\n[qBraid] Removing temporary files...\n"
rm -rf /tmp/* ~/.cache/pip ~/.cache/matplotlib

printf "\n[qBraid] All done! Environment is ready to use.\n"
