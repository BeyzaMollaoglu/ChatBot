from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "Qwen/Qwen-1_8B"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    trust_remote_code=True
)
