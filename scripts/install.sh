apt update
apt install -y zip vim pv tmux rsync

curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

uv venv
source .venv/bin/activate
uv pip install -U git+https://github.com/NTT123/cute-viz.git
uv pip install apache-tvm-ffi nvidia-cutlass-dsl torch-c-dlpack-ext ipython
uv pip uninstall torch
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
