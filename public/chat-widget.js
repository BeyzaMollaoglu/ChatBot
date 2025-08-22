(() => {
  "use strict";
  if (window.__ai_widget_initialized) return;
  window.__ai_widget_initialized = true;

  const SERVER_URL = "http://localhost:3001/api/chat"; // index.js ile uyumlu endpoint
  const chatId = "ep_" + Date.now() + "_" + Math.random().toString(36).slice(2, 8);

  // ---------- Shadow DOM ----------
  const host = document.createElement("div");
  host.id = "aiw-host";
  document.body.appendChild(host);
  const shadow = host.attachShadow({ mode: "open" });

  const style = document.createElement("style");
  style.textContent = `
    :host, :host * { box-sizing: border-box; }
    .hidden { display: none !important; }

    #root {
      position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%);
      width: 600px; height: 80vh;
      background: #0f1a1a; color: #e9f1f1;
      border-radius: 12px; box-shadow: 0 10px 30px rgba(0,0,0,.25);
      display: flex; flex-direction: column; overflow: hidden;
      z-index: 999999;
      font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
    }

    #header {
      background: #122425; border-bottom: 1px solid #1c3435;
      padding: 10px 12px; font-weight: 700;
      display: flex; align-items: center; justify-content: space-between;
    }
    #title { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    #close {
      background: transparent; border: 0; color: #e9f1f1;
      font-size: 20px; line-height: 1; width: auto; height: auto;
      padding: 0; margin: 0; cursor: pointer;
      display: inline-flex; align-items: center; justify-content: center;
    }

    #msgs {
      flex: 1; overflow: auto; padding: 12px;
      display: flex; flex-direction: column; gap: 8px;
    }

    .row { display: flex; }
    .row.you { justify-content: flex-end; }
    .bubble {
      max-width: 90%; white-space: pre-wrap; word-break: break-word;
      padding: 10px 12px; border-radius: 10px; border: 1px solid #20393a;
    }
    .you .bubble { background: #dbeafe; color: #0b2a4a; }
    .bot .bubble { background: #1b2d2e; color: #e9f1f1; }

    #composer { display: flex; gap: 8px; padding: 10px; border-top: 1px solid #1c3435; }
    #inp {
      flex: 1; padding: 10px 12px; border-radius: 10px;
      background: #0c1616; color: #e9f1f1; border: 1px solid #2a4a4c;
      outline: none;
    }
    #send {
      padding: 10px 14px; border-radius: 10px; border: 0;
      background: #1aa3a8; color: #fff; font-weight: 700; cursor: pointer;
      white-space: nowrap;
    }
  `;
  shadow.appendChild(style);

  // ---------- Panel ----------
  const root = document.createElement("div");
  root.id = "root";
  shadow.appendChild(root);

  const header = document.createElement("div");
  header.id = "header";
  const title = document.createElement("div");
  title.id = "title";
  title.textContent = "AkıllıYardımcı";
  const closeBtn = document.createElement("button");
  closeBtn.id = "close";
  closeBtn.textContent = "×";
  header.appendChild(title);
  header.appendChild(closeBtn);

  const msgs = document.createElement("div");
  msgs.id = "msgs";

  const composer = document.createElement("div");
  composer.id = "composer";
  const inp = document.createElement("input");
  inp.id = "inp";
  inp.type = "text";
  inp.placeholder = "Mesaj yazın… (ör. Beyza izni)";
  const sendBtn = document.createElement("button");
  sendBtn.id = "send";
  sendBtn.textContent = "Gönder";
  composer.appendChild(inp);
  composer.appendChild(sendBtn);

  root.appendChild(header);
  root.appendChild(msgs);
  root.appendChild(composer);

  // ---------- Balon Oluştur ----------
  function bubble(role, text) {
    const row = document.createElement("div");
    row.className = "row " + (role === "you" ? "you" : "bot");
    const b = document.createElement("div");
    b.className = "bubble";
    b.textContent = text;
    row.appendChild(b);
    msgs.appendChild(row);
    msgs.scrollTop = msgs.scrollHeight;
    return b;
  }

  // ---------- Mesaj Gönder ----------
  async function send(text) {
    if (!text.trim()) return;
    bubble("you", text);
    const loading = bubble("bot", "Yükleniyor…");

    try {
      const response = await fetch(SERVER_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: text.trim(),
          chatId: chatId
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      loading.textContent = data.reply || "Bir şey ters gitti.";

      // Opsiyonları göster
      if (data.options && data.options.length) {
        data.options.forEach(opt => {
          bubble("bot", `${opt.label}: ${opt.value}`);
        });
      }

      // Chat geçmişini güncelle (görselde değil, istemci tarafında tut)
    } catch (err) {
      console.error("Mesaj gönderme hatası:", err);
      loading.textContent = "Sunucuya ulaşılamadı.";
    }
  }

  // ---------- History Yükleme ----------
  async function loadHistory() {
    try {
      const response = await fetch(SERVER_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: "", chatId: chatId })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      const chatHistory = data.chatHistory || [];
      msgs.innerHTML = ""; // Mesajları temizle

      chatHistory.forEach(msg => {
        let content = msg.content;
        try {
          const parsed = JSON.parse(content); // JSON ise parse et
          content = parsed.reply || content; // Sadece reply'i al
          if (parsed.options && parsed.options.length) {
            parsed.options.forEach(opt => {
              bubble("bot", `${opt.label}: ${opt.value}`);
            });
          }
        } catch (e) {
          // JSON değilse olduğu gibi kullan
        }
        bubble(msg.role, content);
      });

      if (chatHistory.length === 0) {
        bubble("bot", "Merhaba! Nasıl bir arama yapmak istersiniz?");
      }
    } catch (error) {
      console.error("History yükleme hatası:", error);
      bubble("bot", "Geçmiş yüklenemedi, yeni bir sohbet başlıyor.");
    }
  }

  // ---------- Eventler ----------
  sendBtn.onclick = () => {
    const v = inp.value;
    inp.value = "";
    send(v);
  };
  inp.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      sendBtn.click();
    }
  });

  closeBtn.onclick = () => {
    host.remove(); // Widget'ı kapat
  };

  // ---------- Başlat ----------
  bubble("bot", "Merhaba! Nasıl bir arama yapmak istersiniz?");
  inp.focus();
  loadHistory(); // History'i yükle
})();