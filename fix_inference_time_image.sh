# Installs the NVIDIA CUDA compiler without touching PyTorch
conda install -y -c nvidia cuda-nvcc

# Installs the build system flashinfer uses to compile the C++ code
pip install ninja