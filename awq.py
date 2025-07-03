from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
from datasets import load_dataset
import torch

# Paths
model_path = "merged_qwen3-8b"    # Your fine-tuned base model path
quantized_path = "awq-4"          # Output directory for quantized model

# Quantization configuration
quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM"
}

# Load model and tokenizer
model = AutoAWQForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Prepare task-specific calibration data (IMDB in this example)
def load_calibration_data():
    data = load_dataset("imdb", split="train")
    # Use only non-empty, sufficiently long texts for calibration
    return [text[:512] for text in data["text"] if text.strip()][:256]

# Quantize the model
model.quantize(
    tokenizer,
    quant_config=quant_config,
    calib_data=load_calibration_data(),
    n_parallel_calib_samples=4,
    max_calib_samples=256
)

# Save quantized model and tokenizer
model.save_quantized(quantized_path)
tokenizer.save_pretrained(quantized_path)
print(f"Quantized model saved at {quantized_path}")
