# Usage_Guide.md

## Running the Pipeline
1. **Data Preparation**
   - Place the input dataset (`sampled_dataset.csv`) in the root directory.
2. **Execution**
   - Run the Python script for the CUDA Python implementation:
     ```bash
     python src/sentimentanalysis.py
     ```
   - To run CUDA C or OpenACC implementations, navigate to their directories and execute the compiled binaries.

## Understanding Outputs
- **Console Logs**:
  - Provides step-by-step updates, including data preprocessing, tokenization, model training, and evaluation.
- **Results Folder**:
  - Contains performance metrics, graphs, and profiling reports.

## Troubleshooting
- **Common Errors**:
  - "CUDA out of memory": Reduce batch size or truncate input data.
  - "Library not found": Ensure all dependencies are installed and paths are configured correctly.
- **Debugging Tips**:
  - Use NVTX annotations for profiling.
  - Run with `--debug` flag to enable verbose logging.

## Extending the Project
- Add support for more complex models like BERT or LSTMs.
- Incorporate additional datasets to improve model generalization.
- Explore multi-GPU training for larger datasets.

