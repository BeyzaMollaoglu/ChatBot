import os
import re
import json
import uuid
import time
import traceback
import pyodbc
import requests
import gradio as gr

# ========= Config =========
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-r1:7b")

DB_DRIVER = os.getenv("DB_DRIVER", "{ODBC Driver 18 for SQL Server}")
DB_SERVER = os.getenv("DB_SERVER", "localhost")
DB_DATABASE = os.getenv("DB_DATABASE", "testdb")
DB_USER = os.getenv("DB_USER", "")           # boş ise Windows kimlik doğrulama dener
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_ENCRYPT = os.getenv("DB_ENCRYPT", "yes")  # yes/no (Driver 18 ile zorunlu olabilir)

PROMPT_PATH = os.getenv("PROMPT_PATH", "prompt.txt")

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
- Yanıt kesinlikle JSON olmalı; ek metin, kod bloğu, açıklama olmayacak.
- Şu veritabanı şeması sadece bağlam içindir: {SCHEMA_INFO}
"""

_prompt_cache = {"text": None, "mtime": None}

def load_system_prompt(schema_info_text: str) -> str:
    """
    Load prompt.txt if exists and inject {SCHEMA_INFO}; else use default.
    Caches content by file mtime.
    """
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
        # SQL Auth
        return (
            f"DRIVER={DB_DRIVER};"
            f"SERVER={DB_SERVER};"
            f"DATABASE={DB_DATABASE};"
            f"UID={DB_USER};PWD={DB_PASSWORD};"
            f"Encrypt={'yes' if DB_ENCRYPT.lower()=='yes' else 'no'};TrustServerCertificate=yes;"
        )
    else:
        # Windows Auth
        return (
            f"DRIVER={DB_DRIVER};"
            f"SERVER={DB_SERVER};"
            f"DATABASE={DB_DATABASE};"
            f"Trusted_Connection=yes;"
            f"Encrypt={'yes' if DB_ENCRYPT.lower()=='yes' else 'no'};TrustServerCertificate=yes;"
        )

def get_conn(retry=False):
    # Basic cached connection
    if _conn_cache["conn"] is not None:
        try:
            _conn_cache["conn"].cursor().execute("SELECT 1")
            _conn_cache["last_ok"] = time.time()
            return _conn_cache["conn"]
        except Exception:
            try:
                _conn_cache["conn"].close()
            except Exception:
                pass
            _conn_cache["conn"] = None

    try:
        conn = pyodbc.connect(_build_conn_str(), timeout=5, autocommit=True)
        _conn_cache["conn"] = conn
        _conn_cache["last_ok"] = time.time()
        return conn
    except Exception as e:
        print(f"DB Bağlantı Hatası: {e}")
        if not retry and "ODBC Driver 18" in DB_DRIVER:
            # Fallback to Driver 17 automatically
            os.environ["DB_DRIVER"] = "{ODBC Driver 17 for SQL Server}"
            print("ODBC Driver 17'ye geçiliyor...")
            return get_conn(retry=True)
        raise e

def ensure_chat_table():
    try:
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
        print("ChatHistory tablosu kontrol edildi/oluşturuldu.")
    except Exception as e:
        print("ChatHistory kontrol/oluşturma hatası:", e)

def save_chat(chat_id: str, role: str, content: str):
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO ChatHistory (ChatId, Role, Content) VALUES (?, ?, ?)",
            (chat_id, role, content)
        )
        print(f"Chat kaydedildi: {role} - {content[:50]}...")
    except Exception as e:
        print("ChatHistory kayıt hatası:", e)

def load_chat_history(chat_id: str, limit=100):
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("""
            SELECT TOP (?) Role, Content
            FROM ChatHistory WITH (NOLOCK)
            WHERE ChatId = ?
            ORDER BY Timestamp ASC, Id ASC
        """, (limit, chat_id))
        rows = cur.fetchall()
        
        # Convert to Gradio messages format
        messages = []
        for role, content in rows:
            role = (role or "").lower()
            content = content or ""
            
            if role == "user":
                messages.append({"role": "user", "content": content})
            elif role == "assistant":
                messages.append({"role": "assistant", "content": content})
        
        print(f"Chat geçmişi yüklendi: {len(messages)} mesaj")
        return messages
    except Exception as e:
        print("ChatHistory yükleme hatası:", e)
        return []

def fetch_schema_info():
    """
    Return schema info as JSON string: [{table, columns:[...]}, ...]
    """
    try:
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
        
        # build map
        mp = {}
        for schema, table, column in rows:
            key = f"{schema}.{table}"
            mp.setdefault(key, {"table": key, "columns": []})
            mp[key]["columns"].append(column)
        
        arr = list(mp.values())
        schema_json = json.dumps(arr, ensure_ascii=False)
        print(f"Şema bilgisi alındı: {len(arr)} tablo")
        return schema_json
    except Exception as e:
        print("Şema alma hatası:", e)
        return "[]"

SAFE_SQL_RE = re.compile(r"^\s*select\b", re.IGNORECASE)
BLOCKLIST = re.compile(r"\b(insert|update|delete|drop|alter|truncate|exec|merge|grant|revoke|xp_)\b", re.IGNORECASE)

def _inject_top_clause(sql: str, top_n: int = 50) -> str:
    # naive: insert TOP N after first SELECT if not already present
    m = re.match(r"^\s*select\s+", sql, re.IGNORECASE)
    if not m:
        return sql
    head_end = m.end()
    # if already SELECT TOP, leave
    if re.match(r"top\s+\d+", sql[head_end:], re.IGNORECASE):
        return sql
    return sql[:head_end] + f"TOP {top_n} " + sql[head_end:]

def run_readonly_select(sql: str):
    if not sql or not SAFE_SQL_RE.search(sql) or BLOCKLIST.search(sql):
        return False, "Yalnızca SELECT sorgularına izin verilir."
    try:
        conn = get_conn()
        cur = conn.cursor()
        q = _inject_top_clause(sql.strip(), 50)
        print(f"SQL sorgusu çalıştırılıyor: {q}")
        cur.execute(q)
        cols = [d[0] for d in cur.description] if cur.description else []
        rows = cur.fetchall()
        
        # render as simple markdown table
        if not cols:
            return True, "Sorgu çalıştı (satır/sütun döndürmedi)."
        
        # limit rows to 50
        out_lines = []
        out_lines.append("| " + " | ".join(cols) + " |")
        out_lines.append("|" + "|".join(["---"]*len(cols)) + "|")
        
        for r in rows[:50]:
            vals = [str(x) if x is not None else "" for x in r]
            out_lines.append("| " + " | ".join(vals) + " |")
        
        result = "\n".join(out_lines)
        print(f"Sorgu sonucu: {len(rows)} satır, {len(cols)} sütun")
        return True, result
    except Exception as e:
        error_msg = f"Sorgu hatası: {e}"
        print(error_msg)
        return False, error_msg

# ========= Model Call =========
def call_model(user_message: str, history_messages, chat_id: str):
    """
    Builds messages (using prompt.txt + schema info) and calls Ollama with format=json.
    Saves user + assistant messages to DB.
    """
    print(f"Model çağrılıyor - Kullanıcı mesajı: {user_message}")
    
    # Save user message immediately
    save_chat(chat_id, "user", user_message)

    try:
        schema_info = fetch_schema_info()
        system_prompt = load_system_prompt(schema_info)

        # Build messages - sadece system ve mevcut user mesajını ekle
        msgs = [{"role": "system", "content": system_prompt}]
        
        # Önceki mesajları ekle
        if history_messages:
            for msg in history_messages:
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    msgs.append({"role": msg["role"], "content": str(msg["content"])})
        
        # Mevcut kullanıcı mesajını ekle
        msgs.append({"role": "user", "content": user_message})

        body = {
            "model": MODEL_NAME,
            "format": "json",
            "messages": msgs,
            "temperature": 0.3,
            "max_tokens": 1024,
            "stream": False
        }
        
        print(f"Ollama'ya istek gönderiliyor: {OLLAMA_URL}/api/chat")
        r = requests.post(f"{OLLAMA_URL}/api/chat", json=body, timeout=120)
        r.raise_for_status()
        
        data = r.json()
        raw = data.get("message", {}).get("content", "") or ""
        
        # JSON temizleme
        cleaned = (
            raw.replace("\ufeff","")
               .replace("```json","")
               .replace("```","")
               .strip()
        )
        
        s = cleaned.find("{")
        e = cleaned.rfind("}")
        if s != -1 and e != -1 and e > s:
            cleaned = cleaned[s:e+1]
        
        try:
            parsed = json.loads(cleaned)
            print(f"JSON başarıyla parse edildi: {parsed.get('intent', 'unknown')}")
        except Exception as parse_error:
            print(f"JSON parse hatası: {parse_error}")
            print(f"Ham yanıt: {raw}")
            parsed = {
                "intent": "general",
                "reply": "Yanıt üretilemedi. Lütfen yeniden dener misiniz?",
                "options": [],
                "needsClarification": True
            }
        
        if not isinstance(parsed.get("options"), list):
            parsed["options"] = []
            
        # Save assistant message
        assistant_reply = parsed.get("reply", "")
        save_chat(chat_id, "assistant", assistant_reply)
        
        return parsed
        
    except Exception as e:
        error_msg = f"Ollama çağrı hatası: {e}"
        print(f"Model çağrı hatası: {e}")
        print(traceback.format_exc())
        save_chat(chat_id, "assistant", error_msg)
        return {
            "intent": "general",
            "reply": f"⚠️ {error_msg}",
            "options": [],
            "needsClarification": True
        }

# ========= Gradio App =========
def send_message(user_message, history, chat_id, opt_payloads):
    if not user_message.strip():
        return history, gr.update(choices=[], value=None, interactive=True), [], "", "general"
    
    try:
        print(f"Mesaj gönderiliyor: {user_message}")
        
        # History'den message listesi oluştur
        history_messages = []
        if history:
            for msg in history:
                if isinstance(msg, dict):
                    history_messages.append(msg)
                elif isinstance(msg, list) and len(msg) == 2:
                    # Eski format desteği
                    user_msg, bot_msg = msg
                    if user_msg:
                        history_messages.append({"role": "user", "content": str(user_msg)})
                    if bot_msg:
                        history_messages.append({"role": "assistant", "content": str(bot_msg)})
        
        data = call_model(user_message, history_messages, chat_id)
        bot_text = data.get("reply", "")
        intent = data.get("intent", "general")
        options = data.get("options", []) or []
        
        labels = [o.get("label", "Seçenek") for o in options]
        payloads = [{"type": o.get("type"), "value": o.get("value"), "label": o.get("label", "Seçenek")} for o in options]
        
        # Yeni mesajları history'ye ekle
        new_history = (history or []) + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": bot_text}
        ]
        
        raw_json = json.dumps(data, ensure_ascii=False, indent=2)
        
        print(f"Yanıt oluşturuldu: {intent} - {bot_text[:50]}...")
        
        return (
            new_history,
            gr.update(choices=labels, value=None, interactive=True),
            payloads,
            raw_json,
            intent
        )
    except Exception as e:
        tb = traceback.format_exc()
        print(f"send_message hatası: {e}")
        print(tb)
        err = f"⚠️ Hata: {e}"
        new_history = (history or []) + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": err}
        ]
        return new_history, gr.update(choices=[], value=None), [], json.dumps({"error": str(e)}, ensure_ascii=False, indent=2), "error"

def run_option(selected_label, history, chat_id, opt_payloads):
    if not selected_label:
        return history
    
    payload = next((p for p in (opt_payloads or []) if p.get("label") == selected_label), None)
    if not payload:
        return (history or []) + [{"role": "assistant", "content": "⚠️ Geçersiz seçenek."}]
    
    t = (payload.get("type") or "").lower()
    v = payload.get("value") or ""

    if t == "open_url":
        response_content = f"🔗 Bağlantı: {v}"
    elif t == "clarification":
        # Save as a small assistant message for traceability
        save_chat(chat_id, "assistant", f"[Clarification seçildi] {v}")
        response_content = f"📝 Seçildi: {v}"
    elif t == "database":
        # If 'v' looks like a table name, run "SELECT TOP 10 * FROM [v]"
        sql = v.strip()
        if not sql:
            response_content = "⚠️ Geçersiz DB isteği."
        else:
            if not SAFE_SQL_RE.search(sql) and re.fullmatch(r"[A-Za-z0-9_\.\[\]]{1,128}", sql):
                sql = f"SELECT TOP 10 * FROM {sql}"
            ok, msg = run_readonly_select(sql)
            response_content = f"🗄️ Veritabanı Sonucu:\n{msg}"
    else:
        response_content = f"Seçenek: {t} {v}"

    new_history = (history or []) + [
        {"role": "user", "content": f"[Seçenek: {selected_label}]"},
        {"role": "assistant", "content": response_content}
    ]
    return new_history

def run_sql(sql_text, history):
    if not sql_text.strip():
        return history
    
    ok, msg = run_readonly_select(sql_text or "")
    prefix = "✅ Sorgu Başarılı" if ok else "❌ Sorgu Hatası"
    
    new_history = (history or []) + [
        {"role": "user", "content": f"🧮 SQL: {sql_text}"},
        {"role": "assistant", "content": f"{prefix}\n{msg}"}
    ]
    return new_history

def test_db_connection():
    """Test database connection and return status"""
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("SELECT 1 as test")
        result = cur.fetchone()
        if result and result[0] == 1:
            return "✅ Veritabanı bağlantısı başarılı"
        else:
            return "❌ Veritabanı bağlantı testi başarısız"
    except Exception as e:
        return f"❌ Veritabanı bağlantı hatası: {e}"

def test_ollama_connection():
    """Test Ollama connection and return status"""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [m.get('name', '') for m in models]
            if MODEL_NAME in model_names:
                return f"✅ Ollama bağlantısı başarılı - Model: {MODEL_NAME}"
            else:
                return f"⚠️ Ollama bağlı ama model '{MODEL_NAME}' bulunamadı. Mevcut modeller: {', '.join(model_names)}"
        else:
            return f"❌ Ollama bağlantı hatası: HTTP {response.status_code}"
    except Exception as e:
        return f"❌ Ollama bağlantı hatası: {e}"

with gr.Blocks(title="AkıllıYardımcı - Python + Gradio + DB", css="footer {visibility: hidden}") as demo:
    gr.Markdown("## AkıllıYardımcı • Standalone (Python + Gradio + MSSQL)")
    
    # Connection status
    with gr.Row():
        db_status = gr.Textbox(label="Veritabanı Durumu", interactive=False, value="Kontrol ediliyor...")
        ollama_status = gr.Textbox(label="Ollama Durumu", interactive=False, value="Kontrol ediliyor...")
    
    with gr.Row():
        test_db_btn = gr.Button("DB Bağlantısını Test Et")
        test_ollama_btn = gr.Button("Ollama Bağlantısını Test Et")

    chat_id = gr.State()
    chatbot = gr.Chatbot(height=450, type="messages", show_copy_button=True)
    opt_payloads = gr.State([])
    last_raw = gr.State("")
    last_intent = gr.State("")

    with gr.Row():
        msg = gr.Textbox(placeholder="Mesaj yazın…", scale=8)
        send = gr.Button("Gönder", variant="primary", scale=1)

    with gr.Row():
        options = gr.Radio(label="Seçenekler", choices=[], interactive=True, scale=6)
        apply_btn = gr.Button("Seçeneği Uygula", scale=1)

    with gr.Accordion("Veritabanı • Yalnızca SELECT", open=False):
        sql_box = gr.Textbox(label="SQL (SELECT)", placeholder="SELECT TOP 10 * FROM dbo.Personel", lines=3)
        run_sql_btn = gr.Button("Sorguyu Çalıştır")

    with gr.Accordion("Ham JSON Yanıt", open=False):
        raw_view = gr.Code(language="json", interactive=False)

    # Initialize on load
    def on_load():
        try:
            ensure_chat_table()
            cid = str(uuid.uuid4())
            print(f"Yeni chat session: {cid}")
            
            # Test connections
            db_stat = test_db_connection()
            ollama_stat = test_ollama_connection()
            
            return cid, [], db_stat, ollama_stat
        except Exception as e:
            print(f"Başlatma hatası: {e}")
            return str(uuid.uuid4()), [], f"❌ Başlatma hatası: {e}", "❌ Test edilemedi"

    demo.load(
        on_load,
        outputs=[chat_id, chatbot, db_status, ollama_status]
    )

    # Test button handlers
    test_db_btn.click(test_db_connection, outputs=[db_status])
    test_ollama_btn.click(test_ollama_connection, outputs=[ollama_status])

    # Send handlers
    send_click = send.click(
        send_message,
        inputs=[msg, chatbot, chat_id, opt_payloads],
        outputs=[chatbot, options, opt_payloads, raw_view, last_intent]
    )
    send_click.then(lambda: "", outputs=[msg])
    
    msg.submit(
        send_message,
        inputs=[msg, chatbot, chat_id, opt_payloads],
        outputs=[chatbot, options, opt_payloads, raw_view, last_intent]
    ).then(lambda: "", outputs=[msg])

    apply_btn.click(
        run_option,
        inputs=[options, chatbot, chat_id, opt_payloads],
        outputs=[chatbot]
    )

    run_sql_btn.click(
        run_sql,
        inputs=[sql_box, chatbot],
        outputs=[chatbot]
    )

if __name__ == "__main__":
    print("AkıllıYardımcı başlatılıyor...")
    try:
        ensure_chat_table()
        print("Veritabanı tabloları kontrol edildi.")
        print(f"Ollama URL: {OLLAMA_URL}")
        print(f"Model: {MODEL_NAME}")
        print(f"DB Server: {DB_SERVER}/{DB_DATABASE}")
        demo.launch(debug=True)
    except Exception as e:
        print(f"Uygulama başlatma hatası: {e}")
        print(traceback.format_exc())