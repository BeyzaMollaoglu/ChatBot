# export_chat_to_jsonl.py  (incremental)
import os, json, time, pyodbc, hashlib, pathlib

DB_DRIVER = os.getenv("DB_DRIVER", "{ODBC Driver 18 for SQL Server}")
DB_SERVER = os.getenv("DB_SERVER", "localhost")
DB_DATABASE = os.getenv("DB_DATABASE", "testdb")
DB_USER = os.getenv("DB_USER", "")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_ENCRYPT = os.getenv("DB_ENCRYPT", "yes")

OUT_PATH   = os.getenv("EXPORT_OUT", "train.jsonl")
STATE_PATH = os.getenv("EXPORT_STATE", "export_state.json")
SEEN_PATH  = os.getenv("EXPORT_SEEN",  "export_seen.txt")  # opsiyonel dedup

def _build_conn_str() -> str:
    if DB_USER and DB_PASSWORD:
        return (f"DRIVER={DB_DRIVER};SERVER={DB_SERVER};DATABASE={DB_DATABASE};"
                f"UID={DB_USER};PWD={DB_PASSWORD};Encrypt={'yes' if DB_ENCRYPT.lower()=='yes' else 'no'};TrustServerCertificate=yes;")
    else:
        return (f"DRIVER={DB_DRIVER};SERVER={DB_SERVER};DATABASE={DB_DATABASE};"
                f"Trusted_Connection=yes;Encrypt={'yes' if DB_ENCRYPT.lower()=='yes' else 'no'};TrustServerCertificate=yes;")

def _load_state():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"last_user_id": 0}

def _save_state(state):
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

def _load_seen():
    seen = set()
    if os.path.exists(SEEN_PATH):
        with open(SEEN_PATH, "r", encoding="utf-8") as f:
            for line in f:
                h = line.strip()
                if h:
                    seen.add(h)
    return seen

def _append_seen(new_hashes):
    if not new_hashes:
        return
    with open(SEEN_PATH, "a", encoding="utf-8") as f:
        for h in new_hashes:
            f.write(h + "\n")

def export_incremental(out_path=OUT_PATH, limit=20000, use_dedup=True):
    state = _load_state()
    last_user_id = int(state.get("last_user_id", 0))

    conn = pyodbc.connect(_build_conn_str(), timeout=10, autocommit=True)
    cur = conn.cursor()

    # last_user_id’dan büyük olan yeni user→assistant çiftleri
    cur.execute(f"""
        SELECT TOP ({limit})
            u.Id       AS UserId,
            u.Content  AS UserMsg,
            a.Content  AS AssistantMsg
        FROM ChatHistory u
        JOIN ChatHistory a
             ON u.ChatId = a.ChatId AND a.Id = u.Id + 1
        WHERE u.Role='user' AND a.Role='assistant'
          AND LEN(a.Content) > 5
          AND a.Content NOT LIKE '%hata%' AND a.Content NOT LIKE '%⚠️%'
          AND u.Id > ?
        ORDER BY u.Id ASC;
    """, (last_user_id,))

    rows = cur.fetchall()
    if not rows:
        print("✓ Yeni kayıt yok. (last_user_id =", last_user_id, ")")
        return

    # out_path yoksa oluştur; varsa append modunda aç
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    f = open(out_path, "a", encoding="utf-8")

    seen = _load_seen() if use_dedup else set()
    added_hashes = []
    written = 0
    max_user_id_seen = last_user_id

    for uid, utext, atext in rows:
        utext = (utext or "").strip()
        atext = (atext or "").strip()
        if not utext or not atext:
            continue

        # Eğitim şeması: (input, output(JSON string))
        output_obj = {
            "intent": "general",
            "reply": atext,
            "options": [],
            "needsClarification": False
        }
        rec = {
            "input": utext,
            "output": json.dumps(output_obj, ensure_ascii=False)
        }

        # opsiyonel dedup (input+output hash)
        if use_dedup:
            h = hashlib.sha1((rec["input"] + "||" + rec["output"]).encode("utf-8")).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            added_hashes.append(h)

        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        written += 1
        if uid > max_user_id_seen:
            max_user_id_seen = uid

    f.close()

    # state & seen güncelle
    state["last_user_id"] = max_user_id_seen
    _save_state(state)
    if use_dedup:
        _append_seen(added_hashes)

    print(f"✓ {written} yeni örnek eklendi → {out_path}")
    print(f"→ last_user_id={max_user_id_seen} kaydedildi ({STATE_PATH})")
    if use_dedup:
        print(f"→ {len(added_hashes)} hash eklendi ({SEEN_PATH})")

if __name__ == "__main__":
    export_incremental()
