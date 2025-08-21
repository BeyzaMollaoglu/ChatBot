import "dotenv/config";
import express from "express";
import cors from "cors";
import path from "path";
import { fileURLToPath } from "url";
import { readFileSync, readdirSync } from "fs";
import { load as loadHTML } from "cheerio";
import sql from "mssql";
import fetch from 'node-fetch';

//const OLLAMA_BASE = process.env.OLLAMA_BASE_URL || "http://localhost:11434";
//const OLLAMA_MODEL = process.env.OLLAMA_MODEL || "llama3.1:8b";

// LangChain (RAG)
import { Document } from "@langchain/core/documents";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/hf_transformers";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
app.use(cors());
app.use(express.json());

const PUBLIC_DIR = path.join(__dirname, "..", "public");
app.use(express.static(PUBLIC_DIR));

// Global state
let DB_SCHEMA = new Map();
let siteRetriever = null;
let siteStore = null;
let siteStats = { files: 0, sections: 0, ts: null };
let dbRetriever = null;
let dbStore = null;
let dbStats = { tables: 0, rows: 0, ts: null };

const MAX_OPTIONS = parseInt(process.env.SEARCH_MAX_OPTIONS || "6");

// Helper functions
function normTr(s = "") {
  return s.toLowerCase()
    .replaceAll("ı", "i").replaceAll("İ", "i")
    .replaceAll("ğ", "g").replaceAll("Ğ", "g")
    .replaceAll("ü", "u").replaceAll("Ü", "u")
    .replaceAll("ş", "s").replaceAll("Ş", "s")
    .replaceAll("ö", "o").replaceAll("Ö", "o")
    .replaceAll("ç", "c").replaceAll("Ç", "c")
    .normalize("NFD").replace(/[\u0300-\u036f]/g, "");
}

function cleanHeadingTitle(t = "") { return t.replace(/\s*\(h[1-6]\)\s*$/i, "").trim(); }
const SUPPORTED_EXTENSIONS = new Set([".html", ".htm", ".txt"]);
const EXCLUDE_DIRS = new Set(["node_modules", ".git", "assets", "images", "img", "css", "js"]);

function listSupportedFiles(dir, acc = []) {
  try {
    for (const ent of readdirSync(dir, { withFileTypes: true })) {
      const name = ent.name;
      const full = path.join(dir, name);
      if (ent.isDirectory()) {
        if (EXCLUDE_DIRS.has(name) || name.startsWith(".")) continue;
        listSupportedFiles(full, acc);
      } else {
        const ext = path.extname(name).toLowerCase();
        if (SUPPORTED_EXTENSIONS.has(ext)) acc.push(full);
      }
    }
  } catch (error) {
    console.warn(`Warning: Could not read directory ${dir}:`, error.message);
  }
  return acc;
}

function slugify(s = "") {
  return s.toLowerCase()
    .normalize("NFD").replace(/[\u0300-\u036f]/g, "")
    .replace(/[^\w\s-]/g, "")
    .trim().replace(/\s+/g, "-")
    .slice(0, 64);
}

function extractContentFromFile(absPath) {
  const relPath = path.relative(PUBLIC_DIR, absPath).replace(/\\/g, "/");
  const ext = path.extname(absPath).toLowerCase();
  const fileName = path.basename(absPath, ext);
  try {
    if (ext === ".txt") return extractSectionsFromTxt(absPath, relPath, fileName);
    if (ext === ".html" || ext === ".htm") return extractSectionsFromHtml(absPath, relPath, fileName);
  } catch (err) {
    console.warn(`Warning: Could not process ${relPath}:`, err.message);
  }
  return [];
}

function extractSectionsFromTxt(absPath, relPath, fileName) {
  const content = readFileSync(absPath, "utf-8");
  if (!content.trim()) return [];
  const lines = content.split("\n").filter(l => l.trim());
  const sections = [];
  let currentSection = "", currentTitle = fileName;

  for (const line of lines) {
    const isLikelyHeader = line.length < 100 && (line.match(/^[A-Z][^.!?]*$/) || line.match(/^\d+\.?\s/) || line.match(/^[-=]{3,}/) || line.toUpperCase() === line);
    if (isLikelyHeader && currentSection.length > 100) {
      sections.push({ title: currentTitle, url: `/${relPath}#section-${sections.length+1}`, content: currentSection.trim(), fileType: "txt" });
      currentTitle = line.length < 50 ? line : fileName;
      currentSection = "";
    } else currentSection += line + "\n";
  }
  if (currentSection.trim()) {
    sections.push({ title: currentTitle, url: `/${relPath}#section-${sections.length+1}`, content: currentSection.trim().slice(0,4000), fileType: "txt" });
  }
  if (sections.length === 0) {
    sections.push({ title: fileName, url: `/${relPath}`, content: content.trim().slice(0,4000), fileType: "txt" });
  }
  return sections;
}

function extractSectionsFromHtml(absPath, relPath, fileName) {
  const html = readFileSync(absPath, "utf-8");
  const $ = loadHTML(html);
  const pageTitle = $("title").text().trim() || fileName;
  const metaDescription = $('meta[name="description"]').attr("content") || "";
  const headings = $("h1, h2, h3, h4, h5, h6").toArray();

  if (headings.length === 0) {
    const bodyText = $("body").text().replace(/\s+/g, " ").trim();
    if (!bodyText) return [];
    return [{ title: pageTitle, url: `/${relPath}`, content: (metaDescription + "\n" + bodyText).slice(0,4000), fileType: "html" }];
  }

  const sections = [];
  const pageOverview = $("body").children().first().text().replace(/\s+/g, " ").trim();
  if (pageOverview && pageOverview.length > 50) {
    sections.push({ title: `${pageTitle} (Overview)`, url: `/${relPath}`, content: (metaDescription + "\n" + pageOverview).slice(0,4000), fileType: "html" });
  }

  headings.forEach((h, idx) => {
    const node = $(h);
    const tag = h.tagName.toLowerCase();
    const rawTitle = node.text().trim() || `Section ${idx + 1}`;
    const title = cleanHeadingTitle(rawTitle);
    let id = node.attr("id"); if (!id) id = slugify(title);

    const currentLevel = parseInt(tag.replace("h", ""));
    const siblings = node.nextAll().toArray();
    let content = node.text();
    for (const sibling of siblings) {
      const siblingTag = sibling.tagName?.toLowerCase();
      if (siblingTag && siblingTag.match(/^h[1-6]$/)) {
        const siblingLevel = parseInt(siblingTag.replace("h", ""));
        if (siblingLevel <= currentLevel) break;
      }
      content += "\n" + $(sibling).text();
    }
    content = content.replace(/\s+/g, " ").trim();
    if (!content || content.length < 20) return;
    sections.push({ title: `${title} (${tag})`, url: `/${relPath}#${id}`, content: content.slice(0,4000), fileType: "html" });
  });

  return sections;
}

async function buildSiteIndex() {
  console.log("🌐 Building site search index...");
  const files = listSupportedFiles(PUBLIC_DIR);
  console.log(`📁 Found ${files.length} supported files`);

  const docs = [];
  let processedFiles = 0;

  for (const absPath of files) {
    try {
      const sections = extractContentFromFile(absPath);
      for (const section of sections) {
        const enhancedContent = `${section.title}\n\n${section.content}`;
        docs.push(new Document({
          pageContent: enhancedContent,
          metadata: { url: section.url, title: cleanHeadingTitle(section.title || ""), fileType: section.fileType, fileName: path.basename(absPath) },
        }));
      }
      processedFiles++;
    } catch (error) {
      console.warn(`⚠️ Could not process ${absPath}:`, error.message);
    }
  }

  if (!docs.length) {
    console.warn("⚠️ No indexable content found in public/ directory.");
    siteRetriever = null; siteStore = null;
    siteStats = { files: processedFiles, sections: 0, ts: new Date().toISOString() };
    return;
  }

  const embeddings = new HuggingFaceTransformersEmbeddings({ modelName: "Xenova/all-MiniLM-L6-v2" });
  siteStore = await MemoryVectorStore.fromDocuments(docs, embeddings);
  siteRetriever = siteStore.asRetriever({ k: 30 });

  siteStats = { files: processedFiles, sections: docs.length, ts: new Date().toISOString() };
  console.log(`✅ Site search index ready: ${siteStats.files} files, ${siteStats.sections} sections`);
}

async function buildDbIndex() {
  console.log("🗄️ Building database search index...");
  if (!DB_SCHEMA.size) {
    console.warn("⚠️ No database schema available.");
    dbRetriever = null; dbStore = null;
    dbStats = { tables: 0, rows: 0, ts: new Date().toISOString() };
    return;
  }

  const docs = [];
  let processedRows = 0;
  let processedTables = 0;
  const pool = await getPool();
  const embeddings = new HuggingFaceTransformersEmbeddings({ modelName: "Xenova/all-MiniLM-L6-v2" });

  for (const [tableKey, tableDef] of DB_SCHEMA) {
    const allCols = tableDef.columns.map(c => `[${c.name}]`);
    if (!allCols.length) continue;

    const idCol = tableDef.columns.find(c => normTr(c.name).includes("id") || normTr(c.name).includes("key"))?.name || tableDef.columns[0].name;
    const query = `SELECT TOP 1000 [${idCol}] AS rowId, ${allCols.join(", ")} FROM [${tableDef.schema}].[${tableDef.table}] WITH (NOLOCK)`;

    try {
      const result = await pool.request().query(query);
      const rows = result.recordset || [];

      for (const row of rows) {
        let content = "";
        for (const col of tableDef.columns) {
          const val = row[col.name] != null ? row[col.name] : "";
          content += `${col.name}: ${val}\n`;
        }
        content = content.trim().slice(0, 4000);
        if (!content) continue;

        const title = tableDef.table + (row[idCol] ? ` (ID: ${row[idCol]})` : "");
        docs.push(new Document({
          pageContent: `${title}\n\n${content}`,
          metadata: { table: tableKey, rowId: row.rowId || "unknown", tableName: tableDef.table },
        }));
        processedRows++;
      }
      processedTables++;
    } catch (e) {
      console.warn(`⚠️ Could not process table ${tableKey}:`, e.message);
    }
  }

  if (!docs.length) {
    console.warn("⚠️ No indexable content found in database.");
    dbRetriever = null; dbStore = null;
    dbStats = { tables: processedTables, rows: 0, ts: new Date().toISOString() };
    return;
  }

  dbStore = await MemoryVectorStore.fromDocuments(docs, embeddings);
  dbRetriever = dbStore.asRetriever({ k: 30 });

  dbStats = { tables: processedTables, rows: docs.length, ts: new Date().toISOString() };
  console.log(`✅ Database search index ready: ${dbStats.tables} tables, ${dbStats.rows} rows`);
}

function sqlConfigFromEnv() {
  return {
    user: process.env.SQL_USER,
    password: process.env.SQL_PASSWORD,
    server: process.env.SQL_SERVER,
    database: process.env.SQL_DATABASE,
    pool: { max: 10, min: 0, idleTimeoutMillis: 30000 },
    options: {
      encrypt: String(process.env.SQL_ENCRYPT || "true") === "true",
      trustServerCertificate: String(process.env.SQL_TRUST_SERVER_CERT || "false") === "true",
      enableArithAbort: true,
    },
  };
}

let _poolPromise = null;
async function getPool() {
  if (_poolPromise) return _poolPromise;
  const cfg = sqlConfigFromEnv();
  _poolPromise = new sql.ConnectionPool(cfg).connect()
    .then(pool => { console.log("✅ MSSQL pool connected:", cfg.server, cfg.database); return pool; })
    .catch(err => { _poolPromise = null; console.error("❌ MSSQL connection failed:", err.message); throw err; });
  return _poolPromise;
}

async function loadDbSchema() {
  const pool = await getPool();
  const q = `SELECT s.name AS [schema], t.name AS [table], c.name AS [column], ty.name AS [type]
             FROM sys.tables t
             JOIN sys.schemas s ON s.schema_id = t.schema_id
             JOIN sys.columns c ON c.object_id = t.object_id
             JOIN sys.types ty ON ty.user_type_id = c.user_type_id
             WHERE t.is_ms_shipped = 0
             ORDER BY s.name, t.name, c.column_id`;
  const r = await pool.request().query(q);
  const map = new Map();
  for (const row of r.recordset) {
    const key = `${row.schema}.${row.table}`;
    if (!map.has(key)) map.set(key, { schema: row.schema, table: row.table, columns: [] });
    map.get(key).columns.push({ name: row.column, type: row.type });
  }
  return map;
}

async function initDb() {
  try {
    await getPool();
    DB_SCHEMA = await loadDbSchema();
    console.log(`📚 DB schema loaded: ${DB_SCHEMA.size} tables`);
    await buildDbIndex();
  } catch (e) {
    console.warn("⚠️ DB init failed:", e.message);
  }
}

function generatePreview(content, query, maxLength = 150) {
  const sentences = content.split(/[.!?]+/).filter(s => s.trim().length > 10);
  let best = sentences[0] || content, maxMatches = 0;
  const queryWords = normTr(query).split(/\s+/).filter(w => w.length > 2);
  for (const s of sentences) {
    const low = normTr(s);
    const m = queryWords.filter(k => low.includes(k)).length;
    if (m > maxMatches) { maxMatches = m; best = s; }
  }
  const preview = best.trim();
  return preview.length > maxLength ? preview.slice(0, maxLength) + "..." : preview;
}

// ---- BUTTON & MINIMAL REPLY HELPERS ----
const VALID_OPTION_TYPES = new Set(["open_url", "database", "clarification"]);

function normalizeOptions(options = []) {
  return options
    .filter(o => o && typeof o === "object")
    .map(o => ({
      type: VALID_OPTION_TYPES.has(o.type) ? o.type : "open_url",
      label: typeof o.label === "string" ? o.label : "Seç",
      value: typeof o.value === "string" ? o.value : ""
    }))
    .filter(o => o.value)
    .slice(0, MAX_OPTIONS);
}

function buttonsFromSiteResults(siteResults = []) {
  return normalizeOptions(
    siteResults.slice(0, 3).map(r => ({
      type: "open_url",
      label: r.title || r.url,
      value: r.url
    }))
  );
}

// "izin" soruları için minimal geri dönüş (model boş dönerse)
function extractNameFromQuery(q = "") {
  const m = q.match(/^(.+?)\s+(?:izni|izin)\b/i);
  if (m) return m[1].trim().replace(/[?]+$/, "");
  return q.trim();
}
function extractKalanIzin(content = "") {
  const m =
    content.match(/kalan[\s_]*iz[ıi]n[:\s]*([0-9]+)/i) ||
    content.match(/\bkalan[:\s]*([0-9]+)/i);
  return m ? Number(m[1]) : null;
}
function makeMinimalReplyFromDb(content = "", userQuery = "") {
  if (!/izin/i.test(userQuery)) return "";
  const name = extractNameFromQuery(userQuery) || "Çalışan";
  const kalan = extractKalanIzin(content);
  if (Number.isFinite(kalan)) return `${name}’nın ${kalan} gün izni kaldı.`;
  return "";
}

const PROMPT_PATH = path.join(__dirname, "prompt.txt");

async function processQueryWithDeepSeek(query, chatHistory = []) {
  try {
    // Şema bilgisini hazırlama
    const schemaInfo = JSON.stringify(
      [...DB_SCHEMA.entries()].map(([k, v]) => ({
        table: k,
        columns: v.columns.map(c => c.name)
      }))
    );

    // Prompt dosyasını yükleme
    let basePrompt;
    try {
      basePrompt = readFileSync(PROMPT_PATH, "utf-8");
    } catch (e) {
      console.error("prompt.txt okunamadı:", e.message);
      basePrompt = "Sen bir şirketin AI asistanısın, adı 'AkıllıYardımcı'. Yanıtın SADECE JSON olsun.";
    }
    const prompt = basePrompt.replace("{SCHEMA_INFO}", schemaInfo);

    // Chat geçmişini DeepSeek formatına çevirme
    const historyMessages = Array.isArray(chatHistory)
      ? chatHistory.map(m => ({
          role: m.role === "assistant" ? "assistant" : "user",
          content: String(m.content || "")
        }))
      : [];

    // DeepSeek API isteği için body
    const body = {
      model: process.env.DEEPSEEK_MODEL || "deepseek-chat", // .env'den model al
      messages: [
        { role: "system", content: prompt },
        ...historyMessages,
        { role: "user", content: query }
      ],
      temperature: 0.7, // Yaratıcılık derecesi, isteğe bağlı
      max_tokens: 2048, // Maksimum token sayısı, isteğe bağlı
      stream: false // Akış modunu kapattık
    };

    // DeepSeek API'ye istek atma
    const resp = await fetch(`${process.env.DEEPSEEK_BASE_URL || "https://api.deepseek.com/v1"}/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${process.env.DEEPSEEK_API_KEY}` // API anahtarı ile kimlik doğrulama
      },
      body: JSON.stringify(body),
    });

    if (!resp.ok) throw new Error(`DeepSeek error ${resp.status}: ${await resp.text()}`);
    const data = await resp.json();

    // Yanıtı işleme (DeepSeek genellikle 'choices' dizisi döndürür)
    const raw = data.choices?.[0]?.message?.content || "";
    if (!raw) throw new Error("Boş yanıt alındı.");

    // JSON temizleme
    const cleaned = String(raw)
      .trim()
      .replace(/^\uFEFF/, "")
      .replace(/^```json\s*/i, "")
      .replace(/^```\s*/i, "")
      .replace(/\s*```$/i, "");

    let parsed;
    try {
      parsed = JSON.parse(cleaned);
    } catch (e) {
      console.error("JSON parse hatası:", e.message, "| Raw:", raw);
      parsed = { intent: "general", reply: "Biraz daha detay verebilir misin?", options: [], needsClarification: true };
    }

    const safe = {
      intent: typeof parsed.intent === "string" ? parsed.intent : "general",
      reply: typeof parsed.reply === "string" ? parsed.reply : "Biraz daha detay verebilir misin?",
      options: Array.isArray(parsed.options)
        ? parsed.options.filter(o => o && typeof o.type === "string" && typeof o.label === "string" && typeof o.value === "string")
        : [],
      needsClarification: Boolean(parsed.needsClarification)
    };
    const ALLOWED_INTENTS = new Set(["greeting", "general", "database", "website", "action", "clarification", "not_found", "security"]);
    if (!ALLOWED_INTENTS.has(safe.intent)) safe.intent = "general";

    console.log(`[AkıllıYardımcı/DeepSeek] Sorgu: ${query} | Yanıt: ${JSON.stringify(safe)}`);
    return safe;
  } catch (e) {
    console.error("DeepSeek error:", e.message);
    return { intent: "general", reply: "DeepSeek ile bir sorun oluştu. Tekrar dener misin?", options: [], needsClarification: false };
  }
}

async function siteSearch(query) {
  if (!siteRetriever) {
    console.warn("⚠️ Site search index not ready.");
    return [];
  }
  const results = await siteRetriever.invoke(query);
  return results.map(doc => ({
    url: doc.metadata.url,
    title: doc.metadata.title,
    content: doc.pageContent,
  })).slice(0, MAX_OPTIONS);
}

async function dbSearch(query) {
  if (!dbRetriever) {
    console.warn("⚠️ Database search index not ready.");
    return [];
  }
  const results = await dbRetriever.invoke(query);
  return results.map(doc => ({
    table: doc.metadata.table,
    rowId: doc.metadata.rowId,
    title: doc.metadata.tableName + (doc.metadata.rowId ? ` (ID: ${doc.metadata.rowId})` : ""),
    content: doc.pageContent,
  })).slice(0, MAX_OPTIONS);
}

app.post("/api/reindex-site", async (_req, res) => {
  try { await buildSiteIndex(); res.json({ ok: true, stats: siteStats }); }
  catch (e) { res.status(500).json({ ok: false, error: e.message }); }
});

app.post("/api/reindex-db", async (_req, res) => {
  try { await buildDbIndex(); res.json({ ok: true, stats: dbStats }); }
  catch (e) { res.status(500).json({ ok: false, error: e.message }); }
});

app.get("/api/search-stats", (_req, res) => { res.json({ siteStats, dbStats }); });

// Chat history DB functions
async function ensureChatTable() {
  try {
    const pool = await getPool();
    const createQuery = `
      IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'ChatHistory')
      CREATE TABLE ChatHistory (
        Id INT IDENTITY(1,1) PRIMARY KEY,
        ChatId NVARCHAR(50) NOT NULL,
        Role NVARCHAR(20) NOT NULL,
        Content NVARCHAR(MAX) NOT NULL,
        Timestamp DATETIME DEFAULT GETDATE()
      );
    `;
    await pool.request().query(createQuery);
    console.log("✅ ChatHistory tablosu oluşturuldu veya zaten var.");
  } catch (e) {
    console.error("⚠️ ChatHistory tablosu oluşturma hatası:", e.message);
  }
}

async function loadChatHistory(chatId) {
  try {
    const pool = await getPool();
    const query = `SELECT Role, Content FROM ChatHistory WHERE ChatId = @chatId ORDER BY Timestamp ASC`;
    const request = pool.request();
    request.input('chatId', sql.NVarChar, chatId);
    const result = await request.query(query);
    console.log(`📥 History yüklendi: ${result.recordset.length} mesaj (chatId: ${chatId})`);
    return result.recordset.map(row => ({ role: row.Role, content: row.Content }));
  } catch (e) {
    console.error("⚠️ History yükleme hatası:", e.message);
    return [];
  }
}

async function saveChatMessage(chatId, role, content) {
  try {
    const pool = await getPool();
    const query = `INSERT INTO ChatHistory (ChatId, Role, Content) VALUES (@chatId, @role, @content)`;
    const request = pool.request();
    request.input('chatId', sql.NVarChar, chatId);
    request.input('role', sql.NVarChar, role);
    request.input('content', sql.NVarChar(sql.MAX), content);
    await request.query(query);
    console.log(`💾 Mesaj kaydedildi: ${role} (chatId: ${chatId}) ${chatId}`);
  } catch (e) {
    console.error("⚠️ Mesaj kaydetme hatası:", e.message);
  }
}

app.post("/api/chat", async (req, res) => {
  try {
    const message = (req.body?.message || "").trim();
    const chatId = req.body?.chatId || "default";
    let chatHistory = await loadChatHistory(chatId);
    if (!message) return res.status(400).json({ error: "Message is required" });

    await saveChatMessage(chatId, "user", message);

    let result = await processQueryWithDeepSeek(message, chatHistory);

    if (!Array.isArray(result.options)) result.options = [];

    if (result.intent === "database") {
      const dbResults = await dbSearch(message);

      if (!dbResults.length) {
        result.intent = "not_found";
        result.reply = "Uygun bir kayıt bulamadım. Ad–soyadı veya departmanı netleştirebilir misin?";
        result.needsClarification = true;
        result.options = normalizeOptions([
          { type: "clarification", label: "Benzer isimleri göster", value: "similar_names" }
        ]);
      } else {
        if (!result.reply || !result.reply.trim()) {
          const minimal = makeMinimalReplyFromDb(dbResults[0].content, message);
          if (minimal) result.reply = minimal;
        }
        if (!result.options.length) {
          result.options = normalizeOptions([
            { type: "database", label: "Detayı Göster", value: message }
          ]);
        } else {
          result.options = normalizeOptions(result.options);
        }
      }
    } else if (result.intent === "action") {
      const siteResults = await siteSearch(message + " login giriş");
      if (siteResults.length) {
        result.options = buttonsFromSiteResults(siteResults);
      }
      if (!result.options.length) {
        if (message.toLowerCase().includes("login") || message.toLowerCase().includes("giriş")) {
          result.options = normalizeOptions([{ type: "open_url", label: "Giriş Sayfası", value: "/login.html" }]);
        } else if (message.toLowerCase().includes("ana sayfa") || message.toLowerCase().includes("home")) {
          result.options = normalizeOptions([{ type: "open_url", label: "Ana Sayfa", value: "/index.html" }]);
        } else if (message.toLowerCase().includes("rapor")) {
          result.options = normalizeOptions([{ type: "open_url", label: "Raporlar", value: "/reports.html" }]);
        }
      }
    } else if (result.intent === "website") {
      const siteResults = await siteSearch(message);
      if (siteResults.length) {
        const preview = generatePreview(siteResults[0].content, message);
        result.reply = result.reply?.trim()
          ? result.reply
          : `Webden buldum: ${siteResults[0].title} - ${preview}.`;
        result.options = buttonsFromSiteResults(siteResults);
      } else {
        result.options = normalizeOptions(result.options);
      }
    } else {
      result.options = normalizeOptions(result.options);
    }

    await saveChatMessage(chatId, "assistant", result.reply);
    chatHistory = await loadChatHistory(chatId);

    return res.json({ ...result, chatHistory });
  } catch (err) {
    console.error("Chat error:", err.message);
    res.status(500).json({ error: "Server error", details: process.env.NODE_ENV === "development" ? err.message : undefined });
  }
});

app.get("/api/db/ping", async (_req, res) => {
  try {
    const r = await getPool().then(p => p.request().query("SELECT @@VERSION AS version"));
    res.json({ ok: true, version: r.recordset?.[0]?.version });
  } catch (e) { res.status(500).json({ ok: false, error: e.message }); }
});

app.post("/api/action", async (req, res) => {
  try {
    const { type, value } = req.body;

    if (type === "open_url") {
      return res.json({
        success: true,
        action: "redirect",
        url: value,
        message: `${value} sayfasına yönlendiriliyor...`
      });
    } else if (type === "database") {
      const dbResults = await dbSearch(value);
      if (dbResults.length) {
        const result = dbResults[0];
        return res.json({
          success: true,
          action: "show_data",
          data: result.content,
          message: `${result.title} bilgileri:`
        });
      } else {
        return res.json({ success: false, message: "Veri bulunamadı." });
      }
    } else if (type === "clarification") {
      const msg =
        value === "similar_names"
          ? "Benzer isimleri göstermek için lütfen tam ad veya departman belirtin."
          : "Biraz daha detay verebilir misin? (Örn: soyadı veya departman)";
      return res.json({ success: true, action: "reply", message: msg });
    } else {
      return res.status(400).json({ error: "Geçersiz aksiyon tipi" });
    }
  } catch (err) {
    console.error("Action error:", err.message);
    res.status(500).json({ error: "Server error" });
  }
});

(async () => {
  await initDb();
  //await ensureChatTable();
  await buildSiteIndex();
  const PORT = process.env.PORT || 3001;
  app.listen(PORT, "localhost", () => {
    console.log(`🚀 AI Search Server running at http://localhost:${PORT} (Time: ${new Date().toLocaleString('tr-TR', { timeZone: 'Europe/Istanbul' })})`);
    console.log(`📊 Index stats:`, { siteStats, dbStats });
  });
})();