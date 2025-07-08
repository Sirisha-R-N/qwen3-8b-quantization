import torch
import time
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset

MODEL_PATH = "qwen3-8b"  # Replace with your model
BATCH_SIZE = 1
MAX_LENGTH = 128

def load_model(model_path, tokenizer):
    # 4-bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    '''
    # 8-bit
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True)
    '''
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    return model

def parameter_memory_usage_gb(model):
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    buffers = sum(b.numel() * b.element_size() for b in model.buffers())
    return (total + buffers) / (1024 ** 3)  # GB

def measure_metrics(model, tokenizer, texts, batch_size=1, max_length=128):
    model.eval()
    inputs = tokenizer(
        texts,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors="pt"
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    n_samples = len(texts)
    n_batches = (n_samples + batch_size - 1) // batch_size
    latencies = []
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for i in range(n_batches):
            batch_slice = slice(i * batch_size, min((i + 1) * batch_size, n_samples))
            batch_inputs = {k: v[batch_slice] for k, v in inputs.items()}
            torch.cuda.synchronize()
            start = time.time()
            _ = model(**batch_inputs)
            torch.cuda.synchronize()
            end = time.time()
            latencies.append(end - start)
    total_time = sum(latencies)
    avg_latency = total_time / n_samples
    throughput = n_samples / total_time
    return avg_latency, throughput

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # Load a subset of the dataset for benchmarking
    dataset = load_dataset("imdb", split="test[:100]")  
    texts = dataset["text"]

    model = load_model(MODEL_PATH, tokenizer)
    param_mem = parameter_memory_usage_gb(model)
    avg_latency, throughput = measure_metrics(
        model, tokenizer, texts, batch_size=BATCH_SIZE, max_length=MAX_LENGTH
    )
    print(f"Parameter Memory: {param_mem:.3f} GB")
    print(f"Avg Latency per sample: {avg_latency:.4f} s")
    print(f"Throughput: {throughput:.2f} samples/s")

if __name__ == "__main__":
    main()
