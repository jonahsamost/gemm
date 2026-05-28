sudo apt update
sudo apt install -y zip vim pv tmux rsync
sudo apt install -y python3.12-dev nsight-compute-2026.1.1
sudo ln -s /opt/nvidia/nsight-compute/2026.1.1/ncu /usr/local/bin/ncu

curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

uv venv
source .venv/bin/activate
uv pip install -U git+https://github.com/NTT123/cute-viz.git
uv pip install apache-tvm-ffi nvidia-cutlass-dsl torch-c-dlpack-ext ipython

uv pip uninstall torch
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
