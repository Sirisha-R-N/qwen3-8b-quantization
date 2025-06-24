from transformers import AutoTokenizer, Qwen3ForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import numpy as np

model_name = "Qwen/Qwen3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Set pad token if not present
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

lora_config = LoraConfig(
    r=8,  
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_CLS,
)

model = Qwen3ForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    torch_dtype="auto",
    trust_remote_code=True,
)
model.config.pad_token_id = tokenizer.pad_token_id
model = get_peft_model(model, lora_config)

dataset = load_dataset("imdb")

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",  # Ensures all sequences are padded to max_length
        max_length=512,
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir="./results_imdb_qwen3_8b",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    bf16=True, 
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = np.mean(predictions == labels)
    return {"accuracy": accuracy}

trainer.compute_metrics = compute_metrics

results = trainer.evaluate()
print(f"Accuracy: {results['eval_accuracy']:.4f}")

model.save_pretrained("./qwen3-ft")
tokenizer.save_pretrained("./qwen3-ft")
