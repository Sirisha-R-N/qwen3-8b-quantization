from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig

model_id = "merged_qwen3-8b"
quant_path = "gptq-4"

calibration_dataset = load_dataset(
    "imdb",
    split="train"
).select(range(1024))["text"]

quant_config = QuantizeConfig(bits=4, group_size=128)

model = GPTQModel.load(model_id, quant_config)

# increase `batch_size` to match gpu/vram specs to speed up quantization
model.quantize(calibration_dataset, batch_size=2)

model.save(quant_path)


# Evaluating latency, memory usage, throughput
import torch
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# CONFIGURATION
MODEL_PATH = "gptq-4"  
BATCH_SIZE = 1               
SEQ_LENGTH = 128
NUM_ITERATIONS = 20          

# LOAD TOKENIZER AND MODEL
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.float16
).eval()

# ENSURE PAD TOKEN IS SET (required for batch processing)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

device = next(model.parameters()).device

# CREATE SINGLE DUMMY INPUT
dummy_texts = ["This is a test sentence."]
inputs = tokenizer(
    dummy_texts,
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=SEQ_LENGTH
)
inputs = {k: v.to(device) for k, v in inputs.items()}

# PARAMETER MEMORY USAGE IN GB
def parameter_memory_usage_gb(model):
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    buffers = sum(b.numel() * b.element_size() for b in model.buffers())
    return (total + buffers) / (1024 ** 3)  # GB

# INFERENCE LATENCY (PER-SAMPLE)
def measure_latency(model, inputs, num_warmup=1, num_runs=10):
    with torch.no_grad():
        # Warmup
        for _ in range(num_warmup):
            _ = model(**inputs)
        torch.cuda.synchronize()

        start = time.time()
        for _ in range(num_runs):
            _ = model(**inputs)
        torch.cuda.synchronize()
        end = time.time()

    avg_latency = (end - start) / num_runs
    return avg_latency


# THROUGHPUT (samples/sec, for reference)
def measure_throughput(model, inputs, num_iter=10, num_warmup=2):
    with torch.no_grad():
        # Warmup
        for _ in range(num_warmup):
            _ = model(**inputs)
        torch.cuda.synchronize()

        start = time.time()
        for _ in range(num_iter):
            _ = model(**inputs)
        torch.cuda.synchronize()
        end = time.time()

    total_samples = num_iter * inputs['input_ids'].shape[0]
    return total_samples / (end - start)


# RUN MEASUREMENTS
latency = measure_latency(model, inputs)
throughput = measure_throughput(model, inputs)
mem_usage_gb = parameter_memory_usage_gb(model)

print("="*50)
print(f"{'Metric':<25} | {'Value':<20} | Details")
print("="*50)
print(f"{'Per-sample Latency':<25} | {latency:.6f} sec ")
print(f"{'Throughput':<25} | {throughput:.2f} samples/sec ")
print(f"{'Parameter Memory':<25} | {mem_usage_gb:.3f} GB ")
print("="*50)
