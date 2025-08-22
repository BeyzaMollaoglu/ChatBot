# export_chat_to_jsonl.py
import os, json, time, pyodbc

DB_DRIVER = os.getenv("DB_DRIVER", "{ODBC Driver 18 for SQL Server}")
DB_SERVER = os.getenv("DB_SERVER", "localhost")
DB_DATABASE = os.getenv("DB_DATABASE", "testdb")
DB_USER = os.getenv("DB_USER", "")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_ENCRYPT = os.getenv("DB_ENCRYPT", "yes")

def _build_conn_str() -> str:
    if DB_USER and DB_PASSWORD:
        return (f"DRIVER={DB_DRIVER};SERVER={DB_SERVER};DATABASE={DB_DATABASE};"
                f"UID={DB_USER};PWD={DB_PASSWORD};Encrypt={'yes' if DB_ENCRYPT.lower()=='yes' else 'no'};TrustServerCertificate=yes;")
    else:
        return (f"DRIVER={DB_DRIVER};SERVER={DB_SERVER};DATABASE={DB_DATABASE};"
                f"Trusted_Connection=yes;Encrypt={'yes' if DB_ENCRYPT.lower()=='yes' else 'no'};TrustServerCertificate=yes;")

def export_jsonl(out_path="train.jsonl", limit=20000):
    conn = pyodbc.connect(_build_conn_str(), timeout=10, autocommit=True)
    cur = conn.cursor()
    # User → Assistant ardışık çiftler (son 30 gün, hatalı yanıtları ele)
    cur.execute(f"""
        SELECT TOP ({limit})
            u.Content AS UserMsg,
            a.Content AS AssistantMsg
        FROM ChatHistory u
        JOIN ChatHistory a
          ON u.ChatId = a.ChatId AND a.Id = u.Id + 1
        WHERE u.Role='user' AND a.Role='assistant'
          AND LEN(a.Content) > 5
          AND a.Content NOT LIKE '%hata%' AND a.Content NOT LIKE '%⚠️%'
          AND u.Timestamp > DATEADD(day, -30, GETDATE())
        ORDER BY u.Timestamp ASC;
    """)
    rows = cur.fetchall()

    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for utext, atext in rows:
            utext = (utext or "").strip()
            atext = (atext or "").strip()
            if not utext or not atext:
                continue

            # Not: mevcut uygulama sadece "reply" metnini kaydediyor.
            # Bu yüzden JSON şemanızı burada biz sarıyoruz.
            output_obj = {
                "intent": "general",
                "reply": atext,           # cevabı olduğu gibi koy
                "options": [],            # elimizde yok
                "needsClarification": False
            }
            rec = {
                "input": utext,
                "output": json.dumps(output_obj, ensure_ascii=False)
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1

    print(f"✓ {n} örnek {out_path} dosyasına yazıldı.")

if __name__ == "__main__":
    export_jsonl()
