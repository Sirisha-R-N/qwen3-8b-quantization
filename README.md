# Qwen3-8B Quantization Evaluation

This project benchmarks and analyzes the performance of the **Qwen3-8B** Large Language Model in various **quantized forms (INT8 and INT4)** compared to its **full-precision (FP16/FP32)** counterpart.

The primary goal is to understand the trade-offs between **inference latency, memory usage, and task accuracy** when deploying quantized models. This helps assess the feasibility of using Qwen-8B in resource-constrained production environments.

---

## Project Objectives

-  Apply **Post-Training Quantization (PTQ)** and **Quantization-Aware Training (QAT)** on Qwen-8B using popular tools (BitsAndBytes, GPTQ, etc.)
-  Benchmark **inference latency, throughput, and memory usage** on an NVIDIA H100 GPU
-  Evaluate model performance on a downstream task (e.g., sentiment classification using IMDb dataset)
-  Analyze trade-offs between **model size, speed, and accuracy**

---

<!--## Dependencies

We recommend using a Conda environment or virtualenv.

### Quick Install (via pip)

```bash
pip install -r requirements.txt
```
### Or if you're using Conda

``` bash
conda create -n qwen3_quant python=3.10
conda activate qwen3_quant
pip install -r requirements.txt
``` -->

