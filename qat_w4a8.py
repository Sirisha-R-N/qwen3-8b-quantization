import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torchao.quantization.prototype.qat import Int8DynActInt4WeightQATQuantizer

# Load your model
model = AutoModelForSequenceClassification.from_pretrained(
    "merged_qwen3-8b", # Replace with your model
    trust_remote_code=True
).cuda()

tokenizer = AutoTokenizer.from_pretrained("merged_qwen3-8b") # Replace with your model

# Set pad token
tokenizer.pad_token = tokenizer.eos_token  
model.config.pad_token_id = tokenizer.eos_token_id  

# Initialize QAT quantizer
qat_quantizer = Int8DynActInt4WeightQATQuantizer()

# Prepare model for QAT 
model = qat_quantizer.prepare(model)

# Training setup
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# Classification training loop
model.train()
for step, batch in enumerate(dataloader):
    batch = {k: v.cuda() for k, v in batch.items()}
    current_loss = None  
    
    if step < 1000:
        with torch.no_grad():
            outputs = model(**batch)
            current_loss = outputs.loss.item()  # Capture loss for monitoring
    else:
        outputs = model(**batch)
        current_loss = outputs.loss.item()  # Capture loss before backward
        outputs.loss.backward()  # Backpropagate the tensor, not the scalar
        optimizer.step()
        optimizer.zero_grad()
    
    # Print every 50 steps
    if step % 50 == 0 and current_loss is not None:
        print(f"Step {step}: Loss = {current_loss:.4f}")


# Convert to real quantized model
model = qat_quantizer.convert(model)

# Save quantized model
model.save_pretrained("qat-qwen3-8b") # Replace with your destination path
