# Self-Compressing Depthwise Attention – Compression Module

This folder contains the code, datasets, model architectures, and training scripts related to the compression experiments for self-compressing vision models.

## Folder Structure

```

compression/
├── **pycache**/                  # Python bytecode cache
├── after\_pe.png                  # Example visualization or positional encoding result
├── assets/                       # Model checkpoints from various experiments
├── data/                         # Dataset loaders and preprocessing scripts
│   ├── country\_data.py           # Loader for Country211 dataset
│   └── mnist\_data.py             # Loader for MNIST dataset
├── qmodules/                     # Quantized modules and model architectures
│   ├── QConv.py                  # Quantized convolution layer
│   ├── QLinear.py                # Quantized linear layer
│   ├── QEagerLinAttn.py          # Quantized linear attention module
│   ├── QPE.py                    # Quantized positional encoding
│   ├── quant\_func.py             # Quantization helper functions
│   ├── qutils.py                 # Utility functions for quantization
│   ├── Models/                   # Model architectures
│   │   ├── ConvModel.py
│   │   ├── PureConvModel.py
│   │   └── eagerDWLin.py
│   ├── fused\_kernels/            # Optimized CUDA kernels for attention
│   └── spec/                     # Submodule implementations
├── QTrainer.py                   # Training utilities for quantized models
├── train.py                       # Main training script
└── utils.py                        # General utility functions

```

## Usage

1. **Preparing datasets**:  
   - `data/country_data.py` and `data/mnist_data.py` handle loading and preprocessing.  
   - Ensure required datasets are downloaded and accessible.

2. **Training models**:  
   - Use `train.py` for running experiments.  
   - `QTrainer.py` provides helper functions for training quantized networks.

3. **Model architectures**:  
   - Available under `qmodules/Models/`.  
   - Includes ConvModel, PureConvModel, and eagerDWLin.  
   - Quantized layers (QConv, QLinear, QEagerLinAttn) are modular and can be swapped in different architectures.

4. **Checkpoints**:  
   - Stored in `assets/`.  
   - Named according to experiment type and date.  
   - Can be loaded for evaluation or fine-tuning.

5. **Utilities**:  
   - `qutils.py` and `quant_func.py` contain helper functions for quantization and mixed-precision operations.  
   - `utils.py` contains general-purpose helper functions for training and evaluation.

## Notes

- The project uses PyTorch and custom CUDA kernels for efficient training and inference of compressed models.  
- `__pycache__/` folders contain compiled bytecode and can be safely ignored.  
- Ensure CUDA-enabled hardware is available for training with fused kernels and attention modules.
