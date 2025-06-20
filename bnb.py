import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def get_quant_config(precision):
    if precision == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    elif precision == "int4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        return None

def benchmark(model_name, precision):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    quant_config = get_quant_config(precision)
    dtype = None
    if precision == "fp16":
        dtype = torch.float16
    elif precision == "bf16":
        dtype = torch.bfloat16

    # Clear CUDA stats
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=quant_config,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model.eval()

    # Prepare input
    prompt = "Explain quantum computing in simple terms."
    model_device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(model_device)


    # Warmup
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=5)

    # Measure memory before inference
    if device == "cuda":
        start_mem = torch.cuda.memory_allocated()
    else:
        start_mem = model.get_memory_footprint()

    # Timed inference
    num_runs = 5
    total_time = 0
    for _ in range(num_runs):
        torch.cuda.synchronize() if device == "cuda" else None
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50)
        torch.cuda.synchronize() if device == "cuda" else None
        total_time += time.perf_counter() - start

    # Peak memory
    if device == "cuda":
        peak_mem = torch.cuda.max_memory_allocated()
        total_mem_used = (peak_mem - start_mem) / 1024**3
    else:
        total_mem_used = (model.get_memory_footprint() - start_mem) / 1024**3

    # Parameter memory
    param_mem = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3

    # Throughput
    tokens_generated = outputs.shape[1] - inputs.input_ids.shape[1]
    avg_latency = total_time / num_runs
    tokens_per_sec = tokens_generated / avg_latency

    return {
        "precision": precision,
        "param_mem_gb": param_mem,
        "peak_inference_mem_gb": total_mem_used,
        "avg_latency_s": avg_latency,
        "tokens_per_sec": tokens_per_sec,
        "tokens_generated": tokens_generated,
    }

model_name = "Qwen/Qwen3-8B"
results = []
for precision in ["fp16", "bf16", "int8", "int4"]:
    print(f"Benchmarking {precision.upper()}...")
    res = benchmark(model_name, precision)
    results.append(res)

print("\n| Precision | Param Mem (GB) | Peak Inf Mem (GB) | Avg Latency (s) | Tokens/sec |")
print("|-----------|----------------|-------------------|-----------------|------------|")
for r in results:
    print(f"| {r['precision'].upper():<9} | {r['param_mem_gb']:.2f}         | {r['peak_inference_mem_gb']:.2f}            | {r['avg_latency_s']:.2f}           | {r['tokens_per_sec']:.2f}      |")
