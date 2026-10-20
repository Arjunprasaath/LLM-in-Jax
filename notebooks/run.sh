#!/usr/bin/env bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

JAX_PYTHON="$HOME/miniconda3/envs/llm_jax/bin/python"
TORCH_PYTHON="$HOME/miniconda3/envs/llm_torch/bin/python"

echo "=========================================================="
echo " 1. Running JAX (env: llm_jax)..."
echo "=========================================================="
"$JAX_PYTHON" run_jax.py

echo ""
echo "=========================================================="
echo " 2. Running PyTorch (env: llm_torch)..."
echo "=========================================================="
"$TORCH_PYTHON" run_pytorch.py

echo ""
echo "=========================================================="
echo " 3. Comparing Results..."
echo "=========================================================="
"$TORCH_PYTHON" compare.py
