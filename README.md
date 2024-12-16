# README.md

## GPU-Accelerated Sentiment Analysis

This repository implements a GPU-accelerated pipeline for sentiment analysis using CUDA. By leveraging GPU programming techniques, this project significantly improves the speed and efficiency of large-scale Natural Language Processing (NLP) tasks.

### Features
- GPU-accelerated tokenization and data preprocessing.
- Binary sentiment classification using Logistic Regression.
- Implementations in CUDA Python, CUDA C/C++, and OpenACC.
- Profiling and performance analysis with NVIDIA Nsight Systems.

### Dataset
This project uses the [Sentiment140 Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140). The dataset contains 1.6 million tweets labeled for sentiment (positive or negative). It is suitable for large-scale binary sentiment classification tasks.

### Repository Structure
```
GPU_Sentiment_Analysis/
├── src/
│   ├── OpenACC/
│   │── sentimentanalysis2.py
│   ├── CUDA_C/
├── docs/
│   ├── Implementation.md
│   ├── Setup_Instructions.md
│   └── Usage_Guide.md
├── results/
│   ├── graphs/
│   ├── profiling/
│   
├── LICENSE
└── README.md
```

### Prerequisites
1. A GPU-enabled machine with NVIDIA CUDA support.
2. Software:
   - NVIDIA CUDA Toolkit
   - Python 3.8+
   - Libraries: `cudf`, `cupy`, `numba`, `cuml`, `nvtx`, `scikit-learn`, `kagglehub`.

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/gpu_sentiment_analysis.git
   cd gpu_sentiment_analysis
   ```
2. Install dependencies:
   ```bash
   pip install cudf cupy numba cuml nvtx scikit-learn kagglehub
   ```
3. Download the dataset:
   ```python
   import kagglehub
   path = kagglehub.dataset_download("kazanova/sentiment140")
   print("Path to dataset files:", path)
   ```
4. Compile CUDA C code (if using the CUDA C implementation):
   ```bash
   nvcc -o sentimentanalysis sentimentanalysis.cu
   ```

### Usage
Run the Python implementation:
```bash
python src/CUDA_Python/sentimentanalysis2.py
```
To profile the performance using NVIDIA Nsight Systems:
```bash
nsys profile --stats=true python src/CUDA_Python/sentimentanalysis2.py
```

### Results
Performance metrics, graphs, and profiling reports are stored in the `results/` directory. These include execution times, memory usage, and kernel performance statistics.

### License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

# LICENSE

MIT License

Copyright (c) 2024 AMANY ALATTAS & AYSHA ALMAHMOUD

You are free to use, modify, distribute, and share this software for any purpose. This software is provided "as is" without warranty of any kind. Use it at your own risk. The authors are not liable for any damages or issues arising from its use.

