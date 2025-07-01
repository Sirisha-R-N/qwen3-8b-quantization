import torch
import time
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset

MODEL_PATH = "qwen3-ft"  # replace with your model

def parameter_memory_usage_gb(model):
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    buffers = sum(b.numel() * b.element_size() for b in model.buffers())
    return (total + buffers) / (1024 ** 3)  # GB


def evaluate_accuracy(model, tokenizer, num_samples=200):
    model.eval()
    dataset = load_dataset("imdb", split="test").select(range(num_samples))
    preds, labels = [], []
    for item in dataset:
        inputs = tokenizer(
            item["text"],
            truncation=True,
            padding='max_length',  
            max_length=512,
            return_tensors="pt"
        ).to(model.device)
        with torch.no_grad():
            logits = model(**inputs).logits
        pred = torch.argmax(logits, dim=-1).item()
        preds.append(pred)
        labels.append(item["label"])
    return np.mean(np.array(preds) == np.array(labels))

def measure_latency_throughput(model, tokenizer, num_samples=50):
    model.eval()
    # Load IMDB test dataset with a limited number of samples
    imdb_test_samples = load_dataset("imdb", split="test").select(range(num_samples))
    texts = [item["text"] for item in imdb_test_samples]
    inputs = tokenizer(
        texts,
        truncation=True,
        padding='max_length',  
        max_length=512,
        return_tensors="pt"
    ).to(model.device)
    
    with torch.no_grad():
        # Warmup
        _ = model(**{k: v[:1] for k, v in inputs.items()})
        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        outputs = model(**inputs)  
        end = time.time()
    
    total_time = end - start
    avg_latency = total_time / num_samples
    throughput = num_samples / total_time
    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
    return avg_latency, throughput, peak_mem

def load_model(quant_type, tokenizer):
    if quant_type == "fp16":
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
    elif quant_type == "bf16":
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
    elif quant_type == "int8":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_PATH,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
    elif quant_type == "int4":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_PATH,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        raise ValueError("Unknown quant_type")
    
    # Set pad token in model config
    model.config.pad_token_id = tokenizer.pad_token_id
    return model

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    results = {}
    for quant_type in ["fp16", "bf16", "int8", "int4"]: # edit this if you want only one type
        print(f"\n--- Evaluating {quant_type.upper()} ---")
        model = load_model(quant_type, tokenizer)
        param_mem = parameter_memory_usage_gb(model)
        accuracy = evaluate_accuracy(model, tokenizer)
        avg_latency, throughput, peak_mem = measure_latency_throughput(model, tokenizer)
        
        results[quant_type] = {
            "Parameter Memory (GB)": param_mem,
            "Peak Inference Memory (MB)": peak_mem,
            "Accuracy": accuracy,
            "Avg Latency (s)": avg_latency,
            "Throughput (samples/s)": throughput
        }
        
        print(f"Parameter Memory: {param_mem:.2f} GB")
        print(f"Peak Inference Memory: {peak_mem:.2f} MB")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Avg Latency: {avg_latency:.4f} s")
        print(f"Throughput: {throughput:.2f} samples/s")
        torch.cuda.empty_cache()
    
    # Print summary table
    print("\n=== Summary ===")
    print(f"{'Version':<8} | {'Param Mem (GB)':<15} | {'Peak Mem (MB)':<15} | {'Accuracy':<9} | {'Latency(s)':<10} | {'Throughput':<10}")
    print("-"*75)
    for qt, res in results.items():
        print(f"{qt.upper():<8} | {res['Parameter Memory (GB)']:<15.2f} | {res['Peak Inference Memory (MB)']:<15.2f} | {res['Accuracy']:<9.4f} | {res['Avg Latency (s)']:<10.4f} | {res['Throughput (samples/s)']:<10.2f}")

if __name__ == "__main__":
    main()
