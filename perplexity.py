import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen3-8B"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16 #remove this line for 4 bit and 8 bit model
    
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load a language modeling benchmark dataset 
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test[:100]")  

def compute_perplexity(model, tokenizer, dataset):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for example in dataset:
        inputs = tokenizer(example["text"], return_tensors="pt", truncation=True, max_length=512)

        # Skip if input is empty after tokenization
        if inputs["input_ids"].shape[1] < 2:
            continue

        input_ids = inputs["input_ids"].to(model.device).long()

        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss

        total_loss += loss.item() * input_ids.size(1)
        total_tokens += input_ids.size(1)

    if total_tokens == 0:
        raise ValueError("No valid tokens in dataset. Check input content.")

    perplexity = torch.exp(torch.tensor(total_loss / total_tokens))
    return perplexity.item()


ppl = compute_perplexity(model, tokenizer, dataset)
print(f"Perplexity: {ppl:.2f}")
