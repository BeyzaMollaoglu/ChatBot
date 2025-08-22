# train_lora_incremental.py
# Qwen/Qwen2.5-1.5B-Instruct tabanı ile LoRA eğitimine "kaldığın yerden devam" script'i.
# - Mevcut adapter (out-lora) varsa yükler ve ÜZERİNE eğitim yapar.
# - Yoksa yeni LoRA başlatır.
# - "The model did not return a loss" hatasını önlemek için labels=input_ids kullanır.

import os
import json
from typing import Dict, Any

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    default_data_collator,
)

from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    PeftConfig,
    TaskType,
)

# ================== Ayarlar ==================
BASE_MODEL = os.getenv("HF_BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
OUT_DIR    = os.getenv("LORA_DIR", "out-lora")          # LoRA çıktı/adapter klasörü
MAX_LEN    = int(os.getenv("MAX_LEN", "512"))
EPOCHS     = float(os.getenv("EPOCHS", "3"))
LR         = float(os.getenv("LR", "2e-4"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))
GR_ACC     = int(os.getenv("GR_ACC", "8"))

HF_TOKEN   = os.getenv("HUGGINGFACE_HUB_TOKEN", None)

# Dataset dosyası: varsa .jsonl, yoksa .json, yoksa train_fixed.json
CANDIDATE_DATA = ["train.jsonl", "train1.json"]

# Qwen 2 için tipik LoRA hedef modülleri:
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# ================== Yardımcılar ==================
def pick_dataset_path() -> str:
    for p in CANDIDATE_DATA:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"Dataset bulunamadı. Şunlardan biri lazım: {', '.join(CANDIDATE_DATA)}"
    )

def build_text(example: Dict[str, Any]) -> Dict[str, str]:
    """
    Beklenen giriş formatı:
      {"input": "...", "output": "..."}  (JSONL veya JSON)
    Farklı alan adları varsa burada uyarlayabilirsin.
    """
    user = example.get("input", "")
    assistant = example.get("output", "")
    # Birleştir
    txt = f"User: {user}\nAssistant: {assistant}"
    return {"text": txt}

# ================== Tokenizer & Model ==================
print(f"Tokenizer ve Base Model yükleniyor: {BASE_MODEL}")
tok_kwargs = {"trust_remote_code": True}
if HF_TOKEN:
    tok_kwargs["token"] = HF_TOKEN

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, **tok_kwargs)

# pad token yoksa eos'u kullan
if tokenizer.pad_token is None and tokenizer.eos_token is not None:
    tokenizer.pad_token = tokenizer.eos_token

model_kwargs = {
    "trust_remote_code": True,
    "device_map": "auto",
    "torch_dtype": "auto",
}
if HF_TOKEN:
    model_kwargs["token"] = HF_TOKEN

base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **model_kwargs)

# Eğitim için öneri: use_cache kapat (gc + gradient_checkpointing ile uyum)
if hasattr(base_model, "config"):
    base_model.config.use_cache = False
try:
    base_model.gradient_checkpointing_enable()
except Exception:
    pass  # bazı modellerde olmayabilir

# ================== LoRA kurulum / mevcut adaptörü devam ettirme ==================
have_adapter = os.path.exists(os.path.join(OUT_DIR, "adapter_config.json"))

if have_adapter:
    print("✅ Mevcut LoRA bulundu, üzerine eğitim yapılacak.")
    # out-lora içindeki adapter'ı base_model'e tak ve trainable yap
    model = PeftModel.from_pretrained(
        base_model,
        OUT_DIR,
        is_trainable=True
    )
else:
    print("ℹ️  Mevcut LoRA yok; yeni LoRA başlatılıyor.")
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=TARGET_MODULES,
        bias="none"
    )
    model = get_peft_model(base_model, lora_cfg)

# Eğitimde hangi parametrelerin trainable olduğunu yazalım (debug faydalı)
try:
    model.print_trainable_parameters()
except Exception:
    pass

# ================== Dataset ==================
data_path = pick_dataset_path()
print(f"Dataset yükleniyor: {data_path}")
dataset = load_dataset("json", data_files=data_path)["train"]

# input+output -> text
dataset = dataset.map(build_text, remove_columns=[c for c in dataset.column_names if c != "text"])

# tokenizasyon + labels
def tokenize_fn(batch):
    tokens = tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_LEN,
        padding="max_length",
    )
    # Causal LM için labels = input_ids
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

dataset = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)

# ================== Training Arguments ==================
args = TrainingArguments(
    output_dir=OUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GR_ACC,
    learning_rate=LR,
    num_train_epochs=EPOCHS,
    fp16=True,             # GTX 1650 vb. için uygun
    bf16=False,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    report_to="none",
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    optim="adamw_torch",
    label_names=["labels"],  # labels alanını açıkça bildir
)

# ================== Trainer ==================
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
    data_collator=default_data_collator,  # labels zaten dataset'te
)

# ================== Train ==================
print("Eğitim başlıyor...")
trainer.train()

# ================== Save ==================
print(f"Adapter kaydediliyor → {OUT_DIR}")
model.save_pretrained(OUT_DIR)
print("✅ Bitti.")


# === Train.jsonl sıfırlama ===
open("train.jsonl", "w", encoding="utf-8").close()
print("train.jsonl temizlendi ✅")