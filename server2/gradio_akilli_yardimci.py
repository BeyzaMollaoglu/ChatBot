import os
import re
import json
import uuid
import time
import traceback
import pyodbc
import gradio as gr

# === Transformers & PEFT ===
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel

OFFLOAD_DIR = os.getenv("OFFLOAD_DIR", "./offload")  # CPU offload klasörü
os.makedirs(OFFLOAD_DIR, exist_ok=True)

# ========= Config =========
DB_DRIVER = os.getenv("DB_DRIVER", "{ODBC Driver 18 for SQL Server}")
DB_SERVER = os.getenv("DB_SERVER", "localhost")
DB_DATABASE = os.getenv("DB_DATABASE", "testdb")
DB_USER = os.getenv("DB_USER", "")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_ENCRYPT = os.getenv("DB_ENCRYPT", "yes")

PROMPT_PATH = os.getenv("PROMPT_PATH", "prompt.txt")

# === LoRA Config ===
HF_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_DIR = "out-lora"

# ========= Prompt Handling =========
_DEFAULT_SYSTEM_PROMPT = r"""Siz 'AkıllıYardımcı' adlı şirket AI asistanısınız.
Türkçe, doğal ve net yanıtlar verin ve SADECE şu JSON formatında döndürün:

{
  "intent": "greeting|general|database|website|action|clarification|not_found|security",
  "reply": "Türkçe yanıt (kısa ve net)",
  "options": [{"type": "database|open_url|clarification", "label": "string", "value": "string"}],
  "needsClarification": false
}

Kurallar:
- Minimalist olun; tek cümle idealdir.
- Hassas verileri paylaşmayın (maaş vb. -> intent: security).
- İsim belirsizse intent: clarification + seçenekler verin.
- Sohbet tetikleyicilerinde intent='general' kullanın.
- Bilgi yoksa "Bilgi bulunamadı" deyin, uydurmayın.
- Yanıt kesinlikle JSON olmalı.
- Şu veritabanı şeması sadece bağlam içindir: {SCHEMA_INFO}
"""

_prompt_cache = {"text": None, "mtime": None}

def load_system_prompt(schema_info_text: str) -> str:
    try:
        st = os.stat(PROMPT_PATH)
        mtime = st.st_mtime
        if _prompt_cache["mtime"] != mtime:
            with open(PROMPT_PATH, "r", encoding="utf-8") as f:
                txt = f.read()
            _prompt_cache["text"] = txt
            _prompt_cache["mtime"] = mtime
    except FileNotFoundError:
        _prompt_cache["text"] = None
        _prompt_cache["mtime"] = None
    base = _prompt_cache["text"] if _prompt_cache["text"] else _DEFAULT_SYSTEM_PROMPT
    return base.replace("{SCHEMA_INFO}", schema_info_text or "")

# ========= DB Utils =========
_conn_cache = {"conn": None, "last_ok": 0}

def _build_conn_str() -> str:
    if DB_USER and DB_PASSWORD:
        return f"DRIVER={DB_DRIVER};SERVER={DB_SERVER};DATABASE={DB_DATABASE};UID={DB_USER};PWD={DB_PASSWORD};Encrypt={'yes' if DB_ENCRYPT.lower()=='yes' else 'no'};TrustServerCertificate=yes;"
    else:
        return f"DRIVER={DB_DRIVER};SERVER={DB_SERVER};DATABASE={DB_DATABASE};Trusted_Connection=yes;Encrypt={'yes' if DB_ENCRYPT.lower()=='yes' else 'no'};TrustServerCertificate=yes;"

def get_conn():
    if _conn_cache["conn"] is not None:
        try:
            _conn_cache["conn"].cursor().execute("SELECT 1")
            return _conn_cache["conn"]
        except:
            _conn_cache["conn"] = None
    conn = pyodbc.connect(_build_conn_str(), timeout=5, autocommit=True)
    _conn_cache["conn"] = conn
    return conn

def ensure_chat_table():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
    IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'ChatHistory')
    CREATE TABLE ChatHistory (
        Id INT IDENTITY(1,1) PRIMARY KEY,
        ChatId NVARCHAR(64) NOT NULL,
        Role NVARCHAR(20) NOT NULL,
        Content NVARCHAR(MAX) NOT NULL,
        Timestamp DATETIME DEFAULT GETDATE()
    );
    """)

def save_chat(chat_id: str, role: str, content: str):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("INSERT INTO ChatHistory (ChatId, Role, Content) VALUES (?, ?, ?)", (chat_id, role, content))

def fetch_schema_info():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
    SELECT s.name AS [schema], t.name AS [table], c.name AS [column]
    FROM sys.tables t
    JOIN sys.schemas s ON s.schema_id = t.schema_id
    JOIN sys.columns c ON c.object_id = t.object_id
    WHERE t.is_ms_shipped = 0
    ORDER BY s.name, t.name, c.column_id
    """)
    rows = cur.fetchall()
    mp = {}
    for schema, table, column in rows:
        key = f"{schema}.{table}"
        mp.setdefault(key, {"table": key, "columns": []})
        mp[key]["columns"].append(column)
    return json.dumps(list(mp.values()), ensure_ascii=False)

# ========= LoRA Helpers =========
_hf_pipe = None

def init_hf_lora():
    """LoRA adaptörünü yükleyip text-generation pipeline hazırlar (low-VRAM uyumlu)."""
    global _hf_pipe
    if _hf_pipe is not None:
        return _hf_pipe
    try:
        tok = AutoTokenizer.from_pretrained(
            HF_BASE_MODEL,
            use_fast=True,
            trust_remote_code=True
        )
        # pad_token yoksa eos'u kullan
        if tok.pad_token is None and tok.eos_token is not None:
            tok.pad_token = tok.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            HF_BASE_MODEL,
            trust_remote_code=True,
            device_map="auto",           # eldeki GPU + CPU'ya dağıt
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            offload_folder=OFFLOAD_DIR,  # <<< ÖNEMLİ: CPU offload klasörü
            attn_implementation="eager"  # flash-attn yoksa güvenli yol
        )

        model = PeftModel.from_pretrained(
            base,
            LORA_DIR
        )

        _hf_pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tok,
            device_map="auto",
            torch_dtype="auto"
        )
        print(f"HF LoRA yüklendi: base={HF_BASE_MODEL}, adapter={LORA_DIR}, offload_dir={OFFLOAD_DIR}")
    except Exception as e:
        print(f"HF LoRA yüklenemedi: {e}")
        _hf_pipe = None
        raise
    return _hf_pipe

def build_hf_prompt(system_prompt: str, history_messages, user_message: str) -> str:
    lines = ["<s>[SYSTEM]", system_prompt.strip(), "[/SYSTEM]"]
    if history_messages:
        for m in history_messages[-6:]:
            if m["role"] == "user":
                lines += ["[USER]", m["content"], "[/USER]"]
    lines += ["[USER]", user_message.strip(), "[/USER]", "[ASSISTANT]"]
    return "\n".join(lines)

def hf_generate_json(system_prompt: str, history_messages, user_message: str) -> str:
    pipe = init_hf_lora()
    prompt = build_hf_prompt(system_prompt, history_messages, user_message)
    out = pipe(prompt, max_new_tokens=256, do_sample=False, temperature=0.0)[0]["generated_text"]
    resp = out[len(prompt):]
    s, e = resp.find("{"), resp.rfind("}")
    if s != -1 and e != -1:
        return resp[s:e+1].strip()
    return resp.strip()

# ========= Model Call =========
def call_model(user_message: str, history_messages, chat_id: str):
    save_chat(chat_id, "user", user_message)
    try:
        schema_info = fetch_schema_info()
        system_prompt = load_system_prompt(schema_info)
        cleaned = hf_generate_json(system_prompt, history_messages, user_message)
        try:
            parsed = json.loads(cleaned)
        except:
            parsed = {"intent": "general", "reply": cleaned[:200], "options": [], "needsClarification": False}
        save_chat(chat_id, "assistant", parsed.get("reply", ""))
        return parsed
    except Exception as e:
        err = f"⚠️ LoRA çağrı hatası: {e}"
        save_chat(chat_id, "assistant", err)
        return {"intent": "general", "reply": err, "options": [], "needsClarification": True}

# ========= Test Utils =========
def test_db_connection():
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("SELECT 1")
        return "✅ Veritabanı bağlantısı başarılı"
    except Exception as e:
        return f"❌ DB hatası: {e}"

def test_lora_connection():
    try:
        init_hf_lora()
        return f"✅ LoRA yüklendi ({HF_BASE_MODEL} + {LORA_DIR})"
    except Exception as e:
        return f"❌ LoRA hatası: {e}"

# ========= Gradio UI =========
with gr.Blocks(title="AkıllıYardımcı - LoRA + DB", css="footer {visibility: hidden}") as demo:
    gr.Markdown("## AkıllıYardımcı • LoRA + DB")

    with gr.Row():
        db_status = gr.Textbox(label="Veritabanı Durumu", interactive=False)
        lora_status = gr.Textbox(label="LoRA Durumu", interactive=False)

    with gr.Row():
        test_db_btn = gr.Button("DB Test")
        test_lora_btn = gr.Button("LoRA Test")

    chat_id = gr.State()
    chatbot = gr.Chatbot(height=450, type="messages")
    msg = gr.Textbox(placeholder="Mesaj yazın…")
    send = gr.Button("Gönder", variant="primary")

    def on_load():
        ensure_chat_table()
        return str(uuid.uuid4()), [], test_db_connection(), test_lora_connection()

    demo.load(on_load, outputs=[chat_id, chatbot, db_status, lora_status])
    test_db_btn.click(test_db_connection, outputs=[db_status])
    test_lora_btn.click(test_lora_connection, outputs=[lora_status])

    def send_message(user_message, history, chat_id):
        data = call_model(user_message, history, chat_id)
        reply = data.get("reply", "")
        new_history = history + [{"role": "user", "content": user_message}, {"role": "assistant", "content": reply}]
        return new_history

    send.click(send_message, inputs=[msg, chatbot, chat_id], outputs=[chatbot]).then(lambda: "", outputs=[msg])
    msg.submit(send_message, inputs=[msg, chatbot, chat_id], outputs=[chatbot]).then(lambda: "", outputs=[msg])

if __name__ == "__main__":
    ensure_chat_table()
    demo.launch(debug=True)
