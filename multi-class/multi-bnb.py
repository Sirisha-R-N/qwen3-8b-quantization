import torch
import time
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from peft import PeftModel
from datasets import load_from_disk  

# === CONFIG ===
BASE_MODEL = "Qwen/Qwen3-8B"
LORA_ADAPTER_PATH = "qwen-product-classifier/checkpoint-6750"
MAX_NEW_TOKENS = 20
BATCH_SIZE = 1
N_SAMPLES = 5  # Limit evaluation to 5 samples

# Load dataset
test_dataset = dataset["test"]
prompts = test_dataset["text"][:N_SAMPLES]  # Use only a few samples

# === Model Loader ===
def load_model(quant_mode):
    if quant_mode == "fp16":
        return AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
    elif quant_mode == "bf16":
        return AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
    elif quant_mode == "int8":
        return AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            load_in_8bit=True,
            device_map="auto",
            trust_remote_code=True
        )
    elif quant_mode == "int4":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        return AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )

# === Benchmark Function ===
def benchmark_quant(quant_mode):
    print(f"\n===== Benchmarking: {quant_mode.upper()} =====")

    torch.cuda.empty_cache()
    gc.collect()

    # Load model + LoRA
    base = load_model(quant_mode)
    model = PeftModel.from_pretrained(base, LORA_ADAPTER_PATH)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.padding_side = "left"
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    # Warm-up
    input_ids = tokenizer(prompts[0], return_tensors="pt").to("cuda")
    _ = model.generate(**input_ids, max_new_tokens=MAX_NEW_TOKENS)
    print(tokenizer.decode(_[0], skip_special_tokens=True))

    # Measure latency, throughput, memory on all N_SAMPLES
    torch.cuda.reset_peak_memory_stats()
    start = time.time()

    pipe(prompts, batch_size=BATCH_SIZE, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)

    total_time = time.time() - start
    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
    latency = total_time / len(prompts)
    throughput = len(prompts) / total_time

    result = {
        "Quant": quant_mode,
        "Latency (s/sample)": round(latency, 3),
        "Throughput (samples/sec)": round(throughput, 2),
        "Peak Memory (MB)": round(peak_mem, 2)
    }

    print(result)
    return result

# === Run Benchmarks ===
all_results = []
for mode in ["fp16", "bf16", "int8", "int4"]:
    result = benchmark_quant(mode)
    all_results.append(result)
