import pandas as pd
from datasets import Dataset

csv_file = "styles.csv"  # Replace with your file path

# Load CSV and skip bad lines
df = pd.read_csv(csv_file, on_bad_lines='skip')

# Keep only relevant columns and drop missing values
df = df[['productDisplayName', 'masterCategory']].dropna()

# Encode labels
labels = sorted(df['masterCategory'].unique())
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}
df['label'] = df['masterCategory'].map(label2id)

# Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df)
def preprocess(example):
    return tokenizer(
        example['productDisplayName'],
        truncation=True,
        padding='max_length',
        max_length=64  # Adjust as needed
    )
dataset = dataset.map(preprocess, batched=False)
# Rename 'label' column for compatibility with Hugging Face Trainer
dataset = dataset.rename_column("label", "labels")
dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Split into train/test (optional, adjust as needed)
splits = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = splits['train']
eval_dataset = splits['test']


from torch.utils.data import DataLoader

train_dataloader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True
)


import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torchao.quantization.prototype.qat import Int8DynActInt4WeightQATQuantizer

# 1. Load your fine-tuned model
num_labels=7
model = AutoModelForSequenceClassification.from_pretrained(
    "fashion-merged",
    trust_remote_code=True,
    num_labels=num_labels
).cuda()

tokenizer = AutoTokenizer.from_pretrained("fashion-merged")

tokenizer.pad_token = tokenizer.eos_token  # Set pad token to EOS token
model.config.pad_token_id = tokenizer.eos_token_id  # Update model config


# 2. Initialize QAT quantizer
qat_quantizer = Int8DynActInt4WeightQATQuantizer()

# 3. Prepare model for QAT (insert fake quantization ops)
model = qat_quantizer.prepare(model)

# 4. Training setup
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# 5. Classification training loop
import gc

model.train()
for step, batch in enumerate(train_dataloader):
    batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
    torch.cuda.empty_cache()
    gc.collect()

    current_loss = None

    try:
        if step < 1000:
            with torch.no_grad():
                outputs = model(**batch)
                current_loss = outputs.loss.item()
        else:
            outputs = model(**batch)
            current_loss = outputs.loss.item()
            outputs.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"[OOM] at step {step}")
            torch.cuda.empty_cache()
            gc.collect()
            continue
        else:
            raise e


    # Free up GPU memory manually
    del outputs
    del batch
    torch.cuda.empty_cache()
    gc.collect()



# Free any remaining memory
torch.cuda.empty_cache()
gc.collect()

# Move model to CPU to save memory
model.cpu()

# Step 6: Convert to quantized model
model = qat_quantizer.convert(model)

# Step 7: Save model
model.save_pretrained("qat-classifier")
tokenizer.save_pretrained("qat-classifier")


from torch.utils.data import DataLoader

eval_dataloader = DataLoader(
    eval_dataset,
    batch_size=8,
    shuffle=True
)


from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

model.eval()
correct, total = 0, 0

all_preds = []
all_labels = []

import gc
torch.cuda.empty_cache()
gc.collect()

model.cuda()  # Ensure model is on GPU

for batch in eval_dataloader:
    batch = {k: v.cuda() for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
        # ... continue with accuracy, etc.

        preds = torch.argmax(outputs['logits'], dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(batch['labels'].cpu().tolist())

accuracy = accuracy_score(all_labels, all_preds)
print(f"Accuracy: {accuracy:.4f}")


import time

model.eval()
batch_size = eval_dataloader.batch_size

start_time = time.time()
n_batches = 0

with torch.no_grad():
    for batch in eval_dataloader:
        batch = {k: v.cuda() for k, v in batch.items()}
        _ = model(**batch)
        n_batches += 1

total_time = time.time() - start_time
num_samples = n_batches * batch_size
latency_per_sample = total_time / num_samples
throughput = num_samples / total_time

print(f"Inference Latency: {latency_per_sample:.4f} sec/sample")
print(f"Inference Throughput: {throughput:.2f} samples/sec")


def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    return param_size / (1024 ** 3)  # in MB

model_size_mb = get_model_size(model)
print(f"Parameter memory: {model_size_mb:.2f} GB")
