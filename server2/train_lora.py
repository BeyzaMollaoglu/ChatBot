import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

# === MODEL ===
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

print("Tokenizer ve model yükleniyor...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype="auto"
)

# === LoRA Config ===
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05
)
model = get_peft_model(model, lora_config)

# === Dataset ===
print("Dataset yükleniyor...")
dataset = load_dataset("json", data_files="train.json")["train"]

def merge_columns(example):
    # input + output birleştirilip "text" oluşturuluyor
    return {
        "text": f"User: {example['input']}\nAssistant: {example['output']}"
    }

dataset = dataset.map(merge_columns)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )

dataset = dataset.map(
    tokenize,
    batched=True,
    remove_columns=dataset.column_names
)

# Trainer için veri collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# === Training Args ===
args = TrainingArguments(
    output_dir="out-lora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=3,
    fp16=True,       # GTX 1650 destekliyor
    bf16=False,
    save_strategy="epoch",
    logging_steps=10,
    optim="adamw_torch"
)

# === Trainer ===
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
    data_collator=data_collator
)

# === Training ===
print("Eğitim başlıyor...")
trainer.train()

# === Save ===
model.save_pretrained("out-lora")
print("LoRA modeli kaydedildi: out-lora")
