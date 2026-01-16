#!/usr/bin/env bash
set -e

eval "$(mise activate bash)"

echo "Installing Python dependencies..."

python -m pip install --upgrade pip

if [ -f requirements.txt ]; then
  python -m pip install -r requirements.txt
fi

if command -v nvidia-smi &>/dev/null; then
  echo "GPU detected → installing CUDA PyTorch (cu121)"
  python -m pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121
else
  echo "No GPU detected → installing CPU PyTorch"
  python -m pip install torch torchvision torchaudio
fi

echo "Verifying torch install..."
python - <<'EOF'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
EOF
