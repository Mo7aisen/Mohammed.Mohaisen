#!/bin/bash
# Master setup script for XAI Lung Segmentation project

set -e

echo "========================================================"
echo "    XAI LUNG SEGMENTATION PROJECT SETUP                 "
echo "========================================================"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Python
if ! command_exists python3; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

# Check CUDA availability
if command_exists nvidia-smi; then
    echo "✓ CUDA GPU detected"
    nvidia-smi --query-gpu=name,memory.total --format=csv
else
    echo "⚠ No CUDA GPU detected - will use CPU (slower)"
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Make all scripts executable
echo "Making scripts executable..."
chmod +x run_training.sh
chmod +x run_evaluation.sh
chmod +x run_extended_training.sh
chmod +x run_extended_evaluation.sh

# Create output directory structure
echo "Creating directory structure..."
mkdir -p outputs/organized_models
mkdir -p outputs/extended_training_logs

# Generate data split manifests
echo "Generating data split manifests..."
python3 generate_split_manifest.py

# Check dataset paths
echo ""
echo "Checking dataset configuration..."
python3 -c "
import yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

print('Configured datasets:')
for name, info in config['datasets'].items():
    path = info['path']
    import os
    exists = '✓' if os.path.exists(path) else '✗'
    print(f'  {exists} {name}: {path}')
"

echo ""
echo "========================================================"
echo "    SETUP COMPLETE!                                     "
echo "========================================================"
echo ""
echo "Next steps:"
echo "1. Ensure dataset paths in config.yaml are correct"
echo "2. Run initial training: ./run_training.sh"
echo "3. Run extended training: ./run_extended_training.sh"
echo "4. Run evaluation: ./run_evaluation.sh && ./run_extended_evaluation.sh"
echo "5. Organize checkpoints: python3 organize_model_checkpoints.py"
echo "6. Launch app: streamlit run app.py"
echo ""
echo "For detailed instructions, see README.md"