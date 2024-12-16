# Setup_Instructions.md

## Prerequisites
1. **Hardware**: A GPU-enabled machine with NVIDIA CUDA support.
2. **Software**:
   - NVIDIA CUDA Toolkit (latest version).
   - Python 3.8+.
   - Libraries:
     - `cudf`, `cupy`, `numba`, `cuml`, `nvtx`, `scikit-learn`, `kagglehub`.
   - C++ compiler with CUDA support (for CUDA C implementations).
   - OpenACC-enabled compiler (e.g., PGI Compiler).

## Installation Steps

### 1. Clone the Repository
```bash
git clone https://github.com/your-repo/gpu_sentiment_analysis.git
cd gpu_sentiment_analysis
```

### 2. Install Dependencies
#### Python Libraries
```bash
pip install cudf cupy numba cuml nvtx scikit-learn kagglehub
```
#### CUDA Toolkit
- Download and install the latest [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads).

### 3. Dataset Installation
```python
import kagglehub
path = kagglehub.dataset_download("kazanova/sentiment140")
print("Path to dataset files:", path)
```

### 4. Set Up Environment
- Ensure the `CUDA_HOME` environment variable is set.
```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### 5. Nsight Systems Installation (Optional)
If running on Google Colab, use the following commands to download the dataset:
To run and profile using NVIDIA Nsight Systems (nys) on Colab, use the following commands:
```bash
!cat /etc/os-release | grep "VERSION_ID"
!echo "Machine's architecture: `uname -i`"
!wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/nsight-systems-2023.2.3_2023.2.3.1001-1_amd64.deb
!apt update
!apt install ./nsight-systems-2023.2.3_2023.2.3.1001-1_amd64.deb
!apt --fix-broken install
```

### 6. Compile and Run CUDA C Code
Navigate to the `src/CUDA_C/` directory and compile the code:
```bash
!nvcc -o sentimentanalysis sentimentanalysis.cu -run
!nsys profile --stats=true ./sentimentanalysis
```

### 7. Run Python Code with Profiling
For the CUDA Python implementation:
```bash
python sentimentanalysis1.py
!nsys profile --stats=true -o sentimentanalysis-report python sentimentanalysis1.py
```

---

