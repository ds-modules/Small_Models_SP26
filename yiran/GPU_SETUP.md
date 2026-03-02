# GPU Setup Guide for RAG Notebooks

## Overview

These notebooks use two components that benefit from GPU acceleration:

| Component | What it does | GPU tech |
|-----------|-------------|----------|
| **llama-cpp-python** | LLM inference (Qwen2 1.5B) | CUDA (Linux) / Metal (Mac) |
| **onnxruntime-gpu** | Embedding model (all-MiniLM-L6-v2) | CUDA (Linux) / CPU (Mac) |

## JupyterHub Setup (Linux + CUDA)

### llama-cpp-python with CUDA

The notebooks auto-detect your platform and install with the right flags. If you need to install manually:

```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

### onnxruntime-gpu with CUDA

```bash
pip install onnxruntime-gpu
```

The notebooks automatically select `CUDAExecutionProvider` when a CUDA GPU is detected.

## Mac (Apple Silicon / Metal) Setup

### llama-cpp-python with Metal

On Apple Silicon Macs, Metal support is built in automatically:

```bash
pip install llama-cpp-python
```

No special `CMAKE_ARGS` needed — `llama-cpp-python` detects Apple Silicon and enables Metal by default.

### onnxruntime on Mac

ONNX Runtime does not support Metal/MPS acceleration. The notebooks fall back to `CPUExecutionProvider` on Mac. This is fine — the embedding model (all-MiniLM-L6-v2) is small and runs fast on CPU.

## How GPU Acceleration Works

### llama-cpp-python: `n_gpu_layers`

The key parameter is `n_gpu_layers`:

| Value | Meaning |
|-------|---------|
| `0` | All layers on CPU (slow) |
| `-1` | All layers on GPU (fastest) |
| `N` | First N layers on GPU, rest on CPU |

All notebooks now set `n_gpu_layers=N_GPU_LAYERS` where `N_GPU_LAYERS` is auto-detected:
- `-1` (all on GPU) when CUDA or Metal is available
- `0` (CPU only) when no GPU is detected

### onnxruntime: Execution Providers

ONNX Runtime tries providers in order. The notebooks set:
- CUDA available: `["CUDAExecutionProvider", "CPUExecutionProvider"]`
- No CUDA: `["CPUExecutionProvider"]`

## Verifying GPU Is Being Used

### Check llama-cpp-python

When loading the model with `verbose=True`, look for these lines in the output:

```
# CUDA — you should see:
ggml_cuda_init: found 1 CUDA devices
llm_load_tensors: offloading 28 repeating layers to GPU
llm_load_tensors: offloaded 28/29 layers to GPU

# Metal — you should see:
ggml_metal_init: allocating
llm_load_tensors: offloading 28 repeating layers to GPU
```

If you see `llm_load_tensors: offloaded 0/29 layers to GPU`, GPU is NOT being used.

### Check onnxruntime

The notebooks print the active provider after loading:

```python
print(onnx_session.get_providers())
# Should show ['CUDAExecutionProvider', 'CPUExecutionProvider'] on JupyterHub
```

### Check with torch (if available)

The GPU setup cell at the top of each notebook reports:
```
Platform: Linux x86_64
CUDA GPU: NVIDIA A10G
llama.cpp will use n_gpu_layers=-1
```

## Expected Speedups

Rough estimates for the Qwen2 1.5B Q4_0 model:

| Operation | CPU | CUDA GPU | Metal (M1/M2) |
|-----------|-----|----------|----------------|
| Model load | ~3s | ~2s | ~2s |
| Prompt eval (100 tokens) | ~10s | ~1s | ~2s |
| Token generation | ~8 tok/s | ~40 tok/s | ~25 tok/s |

For embeddings (all-MiniLM-L6-v2):

| Operation | CPU | CUDA GPU |
|-----------|-----|----------|
| Single query embed | ~50ms | ~10ms |
| Batch of 100 texts | ~2s | ~0.3s |

Actual performance depends on your specific hardware.

## Troubleshooting

### "No GPU detected" on JupyterHub
- Make sure your JupyterHub instance has a GPU allocated
- Run `nvidia-smi` in a terminal to verify CUDA is available
- You may need to restart the kernel after installing GPU packages

### llama-cpp-python not using GPU
- Reinstall with CUDA flags: `CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir`
- Check that `n_gpu_layers=-1` is set (not `0` or missing)

### onnxruntime falling back to CPU
- Install the GPU version: `pip install onnxruntime-gpu` (not `onnxruntime`)
- Check CUDA toolkit is installed: `nvcc --version`

### Slow on Mac
- Ensure you're on Apple Silicon (M1/M2/M3), not Intel Mac
- llama-cpp-python Metal should be automatic — if not, try: `CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --force-reinstall --no-cache-dir`
