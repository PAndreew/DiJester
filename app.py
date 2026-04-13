#!/usr/bin/env python3
"""
Voice-to-data pipeline  —  v3
Multi-agent, configurable schedule, PWA + push notifications
Web UI on http://0.0.0.0:8000
"""

import base64, json, os, queue, re, sqlite3, subprocess, threading, time, wave
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path

import httpx
import numpy as np
from faster_whisper import WhisperModel
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel
import uvicorn

try:
    from pywebpush import webpush, WebPushException
    from cryptography.hazmat.primitives.asymmetric.ec import generate_private_key, SECP256R1
    from cryptography.hazmat.primitives.serialization import (
        Encoding, PublicFormat, PrivateFormat, NoEncryption,
    )
    PUSH_AVAILABLE = True
except ImportError:
    PUSH_AVAILABLE = False
    print("[push] pywebpush/cryptography not available — push disabled")

# ── Config ────────────────────────────────────────────────────────────────────
SAMPLE_RATE   = 16000
CHUNK_SECONDS = 30
WHISPER_MODEL = "large-v3"
LLAMA_URL     = os.getenv("LLAMA_URL", "http://127.0.0.1:8080/v1/chat/completions")
BASE_DIR      = Path(__file__).parent
DATA_DIR      = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH   = DATA_DIR / "data.db"
AUDIO_DIR = DATA_DIR / "audio_chunks"
AUDIO_DIR.mkdir(exist_ok=True)

# ── Global state ──────────────────────────────────────────────────────────────
_audio_queue:  queue.Queue = queue.Queue()
_is_recording  = False
_whisper_ready = False
_audio_level   = 0.0
_running_agents: set = set()
_state_lock    = threading.Lock()
_vapid_private = None
_vapid_public  = None

# ── Default agent prompts ─────────────────────────────────────────────────────
_TODOS_SYSTEM = (
    "You are a helpful assistant extracting actionable items from transcribed "
    "Hungarian household conversations. Respond ONLY with valid JSON, no markdown."
)
_TODOS_USER = (
    'Extract todo items and shopping list entries. Use this exact JSON schema:\n'
    '{"todos": ["feladat 1"], "shopping": [{"item": "termék", "quantity": "mennyiség"}]}\n\n'
    'Transcriptions:\n'
)

_HEALTH_SYSTEM = (
    "You are a health tracking assistant. Extract health-related mentions from "
    "transcribed Hungarian household conversations. Respond ONLY with valid JSON, no markdown."
)
_HEALTH_USER = (
    'Extract health mentions (symptoms, medications, appointments, exercise, mood, food/drink). '
    'Use this exact JSON schema:\n'
    '{"health": [{"text": "fejfájás reggel", "category": "symptom"}]}\n\n'
    'Transcriptions:\n'
)

_HN_SYSTEM = (
    "You are a tech news curator. Summarise Hacker News stories concisely. "
    "Respond ONLY with valid JSON, no markdown."
)
_HN_USER = (
    'Summarise these top Hacker News stories. For each provide a 2-3 sentence summary '
    'and the most insightful comment. Use this exact JSON schema:\n'
    '{"summaries": [{"title": "...", "url": "...", "summary": "...", "top_comment": "..."}]}\n\n'
    'Stories:\n'
)

# ── Database ──────────────────────────────────────────────────────────────────
def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def _init_db():
    with _db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS transcriptions (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT    NOT NULL,
                text      TEXT    NOT NULL,
                processed INTEGER DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS todos (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                text       TEXT NOT NULL,
                done       INTEGER DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS shopping (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                item       TEXT NOT NULL,
                quantity   TEXT DEFAULT '',
                done       INTEGER DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS health (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                text       TEXT NOT NULL,
                category   TEXT DEFAULT '',
                done       INTEGER DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS summaries (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                source     TEXT NOT NULL,
                title      TEXT NOT NULL,
                url        TEXT DEFAULT '',
                content    TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS extractions (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp      TEXT NOT NULL,
                agent_id       INTEGER,
                segments_count INTEGER NOT NULL,
                raw_response   TEXT NOT NULL,
                todos_count    INTEGER NOT NULL DEFAULT 0,
                shopping_count INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS agents (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                name          TEXT    NOT NULL,
                description   TEXT    NOT NULL,
                source        TEXT    NOT NULL,
                output_type   TEXT    NOT NULL,
                system_prompt TEXT    NOT NULL,
                user_prompt   TEXT    NOT NULL,
                interval_min  INTEGER NOT NULL DEFAULT 60,
                enabled       INTEGER NOT NULL DEFAULT 1,
                last_run      TEXT,
                last_result   TEXT
            );
            CREATE TABLE IF NOT EXISTS push_subscriptions (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                endpoint TEXT NOT NULL UNIQUE,
                p256dh   TEXT NOT NULL,
                auth     TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS settings (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
        """)
    _migrate_db()
    _seed_agents()
    _init_vapid()

def _migrate_db():
    """Apply incremental schema changes to existing DBs."""
    with _db() as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(extractions)").fetchall()}
        if "agent_id" not in cols:
            conn.execute("ALTER TABLE extractions ADD COLUMN agent_id INTEGER")

def _seed_agents():
    with _db() as conn:
        if conn.execute("SELECT COUNT(*) FROM agents").fetchone()[0] > 0:
            return
        for row in [
            ("Todos & Shopping",
             "Extracts todo items and shopping list entries from conversations",
             "transcriptions", "todos_shopping",
             _TODOS_SYSTEM, _TODOS_USER, 60),
            ("Health Log",
             "Tracks health mentions: symptoms, medications, appointments, mood",
             "transcriptions", "health",
             _HEALTH_SYSTEM, _HEALTH_USER, 1440),
            ("Hacker News Digest",
             "Summarises the top 3 HN stories and their key comments",
             "hackernews", "summary",
             _HN_SYSTEM, _HN_USER, 180),
        ]:
            conn.execute(
                "INSERT INTO agents (name,description,source,output_type,"
                "system_prompt,user_prompt,interval_min,enabled) VALUES (?,?,?,?,?,?,?,1)", row
            )

def _init_vapid():
    global _vapid_private, _vapid_public
    if not PUSH_AVAILABLE:
        return
    with _db() as conn:
        priv = conn.execute("SELECT value FROM settings WHERE key='vapid_private'").fetchone()
        pub  = conn.execute("SELECT value FROM settings WHERE key='vapid_public'").fetchone()
    if priv and pub:
        _vapid_private, _vapid_public = priv["value"], pub["value"]
        return
    key = generate_private_key(SECP256R1())
    _vapid_private = key.private_bytes(Encoding.PEM, PrivateFormat.TraditionalOpenSSL, NoEncryption()).decode()
    pub_bytes = key.public_key().public_bytes(Encoding.X962, PublicFormat.UncompressedPoint)
    _vapid_public = base64.urlsafe_b64encode(pub_bytes).rstrip(b"=").decode()
    with _db() as conn:
        conn.execute("INSERT OR REPLACE INTO settings VALUES ('vapid_private',?)", (_vapid_private,))
        conn.execute("INSERT OR REPLACE INTO settings VALUES ('vapid_public',?)",  (_vapid_public,))
    print(f"[vapid] new keys generated, public={_vapid_public[:24]}…")

# ── Push notifications ────────────────────────────────────────────────────────
def _send_push(title: str, body: str, tag: str = "agent"):
    if not PUSH_AVAILABLE or not _vapid_private:
        return
    with _db() as conn:
        subs = conn.execute("SELECT * FROM push_subscriptions").fetchall()
    for sub in subs:
        try:
            webpush(
                subscription_info={
                    "endpoint": sub["endpoint"],
                    "keys": {"p256dh": sub["p256dh"], "auth": sub["auth"]},
                },
                data=json.dumps({"title": title, "body": body, "tag": tag}),
                vapid_private_key=_vapid_private,
                vapid_claims={"sub": "mailto:voicepipeline@localhost"},
            )
        except Exception as e:
            print(f"[push] error: {e}")
            try:
                if hasattr(e, "response") and e.response and e.response.status_code in (404, 410):
                    with _db() as conn:
                        conn.execute("DELETE FROM push_subscriptions WHERE endpoint=?", (sub["endpoint"],))
            except Exception:
                pass

# ── Audio recording ───────────────────────────────────────────────────────────
_USB_MIC_SOURCE = "alsa_input.usb-MUSIC-BOOST_USB_Microphone_MB-306-00.mono-fallback"
_SMALL_BYTES    = SAMPLE_RATE * 2 // 10
_WHISPER_BYTES  = SAMPLE_RATE * 2 * CHUNK_SECONDS

def _recorder():
    global _audio_level
    cmd = ["parec", "--device", _USB_MIC_SOURCE, "--format=s16le",
           f"--rate={SAMPLE_RATE}", "--channels=1", "--raw", "--latency-msec=50"]
    print(f"[recorder] starting parec on {_USB_MIC_SOURCE}")
    while True:
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            buf = b""
            while proc.poll() is None:
                data = proc.stdout.read(_SMALL_BYTES)
                if not data:
                    break
                samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                _audio_level = min(1.0, float(np.sqrt(np.mean(samples**2))) / 6000.0)
                if _is_recording:
                    buf += data
                    if len(buf) >= _WHISPER_BYTES:
                        chunk = np.frombuffer(buf[:_WHISPER_BYTES], dtype=np.int16)
                        rms = float(np.sqrt(np.mean(chunk.astype(np.float32)**2)))
                        if rms >= 500:
                            _audio_queue.put((datetime.now().isoformat(), chunk.reshape(-1,1).copy()))
                        else:
                            print(f"[recorder] silent chunk skipped (rms={rms:.0f})")
                        buf = b""
                else:
                    buf = b""
            proc.kill()
        except Exception as e:
            print(f"[recorder] error: {e}")
            _audio_level = 0.0
        time.sleep(2)

# ── Transcription ─────────────────────────────────────────────────────────────
def _transcriber():
    global _whisper_ready
    print(f"[whisper] loading {WHISPER_MODEL} …")
    model = WhisperModel(WHISPER_MODEL, device="cuda", compute_type="int8")
    _whisper_ready = True
    print("[whisper] ready")
    while True:
        try:
            timestamp, audio = _audio_queue.get(timeout=1)
        except queue.Empty:
            continue
        wav = AUDIO_DIR / f"chunk_{int(time.time()*1000)}.wav"
        try:
            with wave.open(str(wav), "w") as wf:
                wf.setnchannels(1); wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE); wf.writeframes(audio.tobytes())
            segs, _ = model.transcribe(str(wav), language="hu", beam_size=5,
                                       vad_filter=True, vad_parameters={"threshold": 0.5})
            text = " ".join(s.text for s in segs).strip()
        except Exception as e:
            print(f"[whisper] error: {e}"); text = ""
        finally:
            wav.unlink(missing_ok=True)
        if text:
            with _db() as conn:
                conn.execute("INSERT INTO transcriptions (timestamp,text) VALUES (?,?)", (timestamp, text))
            print(f"[{timestamp[:19]}] {text}")

# ── HN data source ────────────────────────────────────────────────────────────
def _fetch_hn_top(n: int = 3) -> str:
    try:
        ids = httpx.get("https://hacker-news.firebaseio.com/v0/topstories.json", timeout=15).json()[:n*3]
    except Exception as e:
        return f"[HN fetch error: {e}]"
    parts, fetched = [], 0
    for sid in ids:
        if fetched >= n:
            break
        try:
            story = httpx.get(f"https://hacker-news.firebaseio.com/v0/item/{sid}.json", timeout=10).json()
        except Exception:
            continue
        if not story or story.get("type") != "story" or story.get("dead") or story.get("deleted"):
            continue
        comments = []
        for kid in (story.get("kids") or [])[:10]:
            try:
                c = httpx.get(f"https://hacker-news.firebaseio.com/v0/item/{kid}.json", timeout=10).json()
                if c and c.get("text") and not c.get("deleted") and not c.get("dead"):
                    t = re.sub(r"<[^>]+>", " ", c["text"])
                    for ent, rep in [("&gt;",">"),("&lt;","<"),("&amp;","&"),("&#x27;","'"),("&quot;",'"')]:
                        t = t.replace(ent, rep)
                    t = re.sub(r"\s+", " ", t).strip()
                    if len(t) > 50:
                        comments.append(t[:400])
            except Exception:
                continue
            if len(comments) >= 3:
                break
        block = (f"Title: {story.get('title','')}\n"
                 f"URL: {story.get('url','no url')}\n"
                 f"Score: {story.get('score',0)} | Comments: {story.get('descendants',0)}\n")
        if story.get("text"):
            block += f"Post: {re.sub(r'<[^>]+>', ' ', story['text'])[:400]}\n"
        if comments:
            block += "Top comments:\n" + "\n".join(f"  * {c}" for c in comments)
        parts.append(block)
        fetched += 1
    return "\n\n---\n\n".join(parts) if parts else "[No stories found]"

# ── Agent runner ──────────────────────────────────────────────────────────────
def _run_agent(agent: dict):
    aid = agent["id"]
    with _state_lock:
        if aid in _running_agents:
            print(f"[agent:{agent['name']}] already running, skipping")
            return
        _running_agents.add(aid)
    try:
        _do_run_agent(agent)
    finally:
        with _state_lock:
            _running_agents.discard(aid)

def _do_run_agent(agent: dict):
    aid, name = agent["id"], agent["name"]
    print(f"[agent:{name}] starting")

    # 1. Gather input
    if agent["source"] == "transcriptions":
        since = agent.get("last_run") or (
            datetime.now() - timedelta(minutes=max(agent["interval_min"], 1))
        ).isoformat()
        with _db() as conn:
            rows = conn.execute(
                "SELECT id, timestamp, text FROM transcriptions WHERE timestamp > ? ORDER BY timestamp",
                (since,)
            ).fetchall()
        if not rows:
            print(f"[agent:{name}] nothing new")
            _set_agent_result(aid, "nothing new to process")
            return
        input_block = "\n".join(f"[{r['timestamp'][:19]}] {r['text']}" for r in rows)
        seg_count = len(rows)
    elif agent["source"] == "hackernews":
        print(f"[agent:{name}] fetching HN top stories…")
        input_block = _fetch_hn_top(3)
        if input_block.startswith("["):
            _set_agent_result(aid, input_block)
            return
        seg_count = 3
    else:
        _set_agent_result(aid, f"unknown source: {agent['source']}")
        return

    # 2. Call LLM
    try:
        resp = httpx.post(LLAMA_URL, json={
            "model": "gemma",
            "messages": [
                {"role": "system", "content": agent["system_prompt"]},
                {"role": "user",   "content": agent["user_prompt"] + input_block},
            ],
            "max_tokens": 1024, "temperature": 0.1,
        }, timeout=120)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        msg = f"LLM error: {e}"
        print(f"[agent:{name}] {msg}")
        _set_agent_result(aid, msg)
        return

    # 3. Parse & store, then notify
    result = _parse_and_store(agent, content, seg_count)
    _set_agent_result(aid, result)
    _send_push(name, result, tag=f"agent-{aid}")
    print(f"[agent:{name}] {result}")

def _parse_and_store(agent: dict, content: str, seg_count: int) -> str:
    now = datetime.now().isoformat()
    n_todos = n_shop = 0
    try:
        s, e = content.find("{"), content.rfind("}") + 1
        data = json.loads(content[s:e]) if s >= 0 and e > s else {}
    except Exception:
        data = {}

    if agent["output_type"] == "todos_shopping":
        with _db() as conn:
            for todo in data.get("todos", []):
                if str(todo).strip():
                    conn.execute("INSERT INTO todos (created_at,text) VALUES (?,?)", (now, str(todo).strip()))
                    n_todos += 1
            for entry in data.get("shopping", []):
                item = (entry.get("item","") if isinstance(entry,dict) else str(entry)).strip()
                qty  = entry.get("quantity","") if isinstance(entry,dict) else ""
                if item:
                    conn.execute("INSERT INTO shopping (created_at,item,quantity) VALUES (?,?,?)", (now,item,qty))
                    n_shop += 1
        result = f"{n_todos} todo(s), {n_shop} shopping item(s) from {seg_count} segment(s)"

    elif agent["output_type"] == "health":
        n = 0
        with _db() as conn:
            for item in data.get("health", []):
                text = (item.get("text","") if isinstance(item,dict) else str(item)).strip()
                cat  = item.get("category","") if isinstance(item,dict) else ""
                if text:
                    conn.execute("INSERT INTO health (created_at,text,category) VALUES (?,?,?)", (now,text,cat))
                    n += 1
        result = f"{n} health item(s) from {seg_count} segment(s)"

    elif agent["output_type"] == "summary":
        n = 0
        with _db() as conn:
            for s in data.get("summaries", []):
                title   = s.get("title","No title") if isinstance(s,dict) else "Summary"
                summary = s.get("summary","")       if isinstance(s,dict) else str(s)
                url     = s.get("url","")           if isinstance(s,dict) else ""
                tc      = s.get("top_comment","")   if isinstance(s,dict) else ""
                full    = summary + ("\n\nTop comment: " + tc if tc else "")
                conn.execute(
                    "INSERT INTO summaries (created_at,source,title,url,content) VALUES (?,?,?,?,?)",
                    (now, agent["name"], title, url, full)
                )
                n += 1
        result = f"{n} summary(ies) from {seg_count} story(ies)"
    else:
        result = "done"

    with _db() as conn:
        conn.execute(
            "INSERT INTO extractions (timestamp,agent_id,segments_count,raw_response,todos_count,shopping_count) "
            "VALUES (?,?,?,?,?,?)",
            (now, agent["id"], seg_count, content, n_todos, n_shop)
        )
    return result

def _set_agent_result(agent_id: int, result: str):
    with _db() as conn:
        conn.execute("UPDATE agents SET last_run=?,last_result=? WHERE id=?",
                     (datetime.now().isoformat(), result, agent_id))

# ── Scheduler ─────────────────────────────────────────────────────────────────
def _scheduler():
    while True:
        time.sleep(60)
        now = datetime.now()
        with _db() as conn:
            agents = [dict(r) for r in conn.execute("SELECT * FROM agents WHERE enabled=1").fetchall()]
        for agent in agents:
            last = datetime.fromisoformat(agent["last_run"]) if agent["last_run"] else datetime.min
            if (now - last).total_seconds() >= agent["interval_min"] * 60:
                threading.Thread(target=_run_agent, args=(agent,), daemon=True,
                                 name=f"agent-{agent['id']}").start()

# ── FastAPI ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(_app: FastAPI):
    _init_db()
    threading.Thread(target=_recorder,    daemon=True, name="recorder").start()
    threading.Thread(target=_transcriber, daemon=True, name="transcriber").start()
    threading.Thread(target=_scheduler,   daemon=True, name="scheduler").start()
    yield

app = FastAPI(lifespan=lifespan)

# ── PWA assets ────────────────────────────────────────────────────────────────
@app.get("/manifest.json")
def manifest():
    return JSONResponse({
        "name": "DiJester",
        "short_name": "DiJester",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#000000",
        "theme_color": "#000000",
        "icons": [{"src":"/icon.svg","sizes":"any","type":"image/svg+xml","purpose":"any maskable"}]
    })

@app.get("/icon.svg")
def icon_svg():
    return Response(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
        '<rect width="100" height="100" rx="20" fill="#1a1d27"/>'
        '<rect x="42" y="18" width="16" height="30" rx="8" fill="#667eea"/>'
        '<path d="M28 46 a22 22 0 0 0 44 0" stroke="#667eea" stroke-width="4" fill="none" stroke-linecap="round"/>'
        '<rect x="48" y="68" width="4" height="13" fill="#667eea"/>'
        '<rect x="35" y="79" width="30" height="4" rx="2" fill="#48bb78"/>'
        '</svg>',
        media_type="image/svg+xml"
    )

@app.get("/sw.js")
def service_worker():
    return Response("""
self.addEventListener('push', e => {
  const d = e.data ? e.data.json() : {};
  e.waitUntil(self.registration.showNotification(d.title || 'Voice Pipeline', {
    body: d.body || '', icon: '/icon.svg', badge: '/icon.svg',
    tag: d.tag || 'vp', renotify: true
  }));
});
self.addEventListener('notificationclick', e => {
  e.notification.close();
  e.waitUntil(clients.matchAll({type:'window'}).then(list => {
    for (const c of list) if ('focus' in c) return c.focus();
    if (clients.openWindow) return clients.openWindow('/');
  }));
});
""", media_type="application/javascript")

# ── Recording ─────────────────────────────────────────────────────────────────
@app.post("/recording/start")
def start_recording():
    global _is_recording; _is_recording = True;  return {"recording": True}

@app.post("/recording/stop")
def stop_recording():
    global _is_recording; _is_recording = False; return {"recording": False}

# ── Status ────────────────────────────────────────────────────────────────────
@app.get("/api/status")
def status():
    with _state_lock:
        running = list(_running_agents)
    return {
        "recording":      _is_recording,
        "whisper_ready":  _whisper_ready,
        "queue":          _audio_queue.qsize(),
        "level":          _audio_level,
        "running_agents": running,
    }

# ── Transcriptions ────────────────────────────────────────────────────────────
@app.get("/api/transcriptions")
def transcriptions(limit: int = 200):
    with _db() as conn:
        rows = conn.execute(
            "SELECT id,timestamp,text,processed FROM transcriptions ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]

# ── Todos ─────────────────────────────────────────────────────────────────────
@app.get("/api/todos")
def todos():
    with _db() as c: return [dict(r) for r in c.execute("SELECT * FROM todos ORDER BY done,created_at DESC").fetchall()]

@app.post("/api/todos/{tid}/toggle")
def toggle_todo(tid: int):
    with _db() as c: c.execute("UPDATE todos SET done=1-done WHERE id=?", (tid,)); return {"ok":True}

@app.delete("/api/todos/{tid}")
def delete_todo(tid: int):
    with _db() as c: c.execute("DELETE FROM todos WHERE id=?", (tid,)); return {"ok":True}

# ── Shopping ──────────────────────────────────────────────────────────────────
@app.get("/api/shopping")
def shopping():
    with _db() as c: return [dict(r) for r in c.execute("SELECT * FROM shopping ORDER BY done,created_at DESC").fetchall()]

@app.post("/api/shopping/{sid}/toggle")
def toggle_shopping(sid: int):
    with _db() as c: c.execute("UPDATE shopping SET done=1-done WHERE id=?", (sid,)); return {"ok":True}

@app.delete("/api/shopping/{sid}")
def delete_shopping(sid: int):
    with _db() as c: c.execute("DELETE FROM shopping WHERE id=?", (sid,)); return {"ok":True}

# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/api/health")
def health_items():
    with _db() as c: return [dict(r) for r in c.execute("SELECT * FROM health ORDER BY done,created_at DESC").fetchall()]

@app.post("/api/health/{hid}/toggle")
def toggle_health(hid: int):
    with _db() as c: c.execute("UPDATE health SET done=1-done WHERE id=?", (hid,)); return {"ok":True}

@app.delete("/api/health/{hid}")
def delete_health(hid: int):
    with _db() as c: c.execute("DELETE FROM health WHERE id=?", (hid,)); return {"ok":True}

# ── Summaries ─────────────────────────────────────────────────────────────────
@app.get("/api/summaries")
def summaries():
    with _db() as c: return [dict(r) for r in c.execute("SELECT * FROM summaries ORDER BY created_at DESC LIMIT 50").fetchall()]

@app.delete("/api/summaries/{sid}")
def delete_summary(sid: int):
    with _db() as c: c.execute("DELETE FROM summaries WHERE id=?", (sid,)); return {"ok":True}

# ── Extraction log ────────────────────────────────────────────────────────────
@app.get("/api/extractions")
def extractions(limit: int = 50):
    with _db() as conn:
        rows = conn.execute(
            "SELECT e.*,a.name as agent_name FROM extractions e "
            "LEFT JOIN agents a ON e.agent_id=a.id ORDER BY e.timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]

# ── Agents ────────────────────────────────────────────────────────────────────
@app.get("/api/agents")
def get_agents():
    with _state_lock: running = set(_running_agents)
    with _db() as conn:
        rows = [dict(r) for r in conn.execute("SELECT * FROM agents ORDER BY id").fetchall()]
    for r in rows:
        r["running"] = r["id"] in running
    return rows

class AgentUpdate(BaseModel):
    interval_min:  int  | None = None
    enabled:       bool | None = None
    system_prompt: str  | None = None
    user_prompt:   str  | None = None

@app.put("/api/agents/{aid}")
def update_agent(aid: int, body: AgentUpdate):
    sets, vals = [], []
    if body.interval_min is not None:
        if not (1 <= body.interval_min <= 44640):
            raise HTTPException(400, "interval_min must be 1–44640")
        sets.append("interval_min=?"); vals.append(body.interval_min)
    if body.enabled is not None:
        sets.append("enabled=?"); vals.append(1 if body.enabled else 0)
    if body.system_prompt is not None:
        sets.append("system_prompt=?"); vals.append(body.system_prompt)
    if body.user_prompt is not None:
        sets.append("user_prompt=?"); vals.append(body.user_prompt)
    if not sets:
        raise HTTPException(400, "nothing to update")
    vals.append(aid)
    with _db() as conn:
        conn.execute(f"UPDATE agents SET {','.join(sets)} WHERE id=?", vals)
    return {"ok": True}

@app.post("/api/agents/{aid}/run")
def run_agent_now(aid: int):
    with _db() as conn:
        row = conn.execute("SELECT * FROM agents WHERE id=?", (aid,)).fetchone()
    if not row:
        raise HTTPException(404, "agent not found")
    threading.Thread(target=_run_agent, args=(dict(row),), daemon=True).start()
    return {"ok": True, "message": f"Agent '{row['name']}' started"}

# Backward-compat
@app.post("/api/extract")
def manual_extract():
    with _db() as conn:
        agent = conn.execute("SELECT * FROM agents WHERE output_type='todos_shopping' LIMIT 1").fetchone()
    if agent:
        threading.Thread(target=_run_agent, args=(dict(agent),), daemon=True).start()
    return {"ok": True, "message": "Extraction started"}

# ── Push ──────────────────────────────────────────────────────────────────────
@app.get("/api/push/vapid-public-key")
def vapid_public_key():
    if not PUSH_AVAILABLE or not _vapid_public:
        raise HTTPException(503, "Push not available")
    return Response(_vapid_public, media_type="text/plain")

class PushSub(BaseModel):
    endpoint: str
    keys: dict

@app.post("/api/push/subscribe")
def push_subscribe(sub: PushSub):
    with _db() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO push_subscriptions (endpoint,p256dh,auth) VALUES (?,?,?)",
            (sub.endpoint, sub.keys.get("p256dh",""), sub.keys.get("auth",""))
        )
    return {"ok": True}

@app.delete("/api/push/unsubscribe")
async def push_unsubscribe(req: Request):
    body = await req.json()
    with _db() as conn:
        conn.execute("DELETE FROM push_subscriptions WHERE endpoint=?", (body.get("endpoint",""),))
    return {"ok": True}

@app.post("/api/push/test")
def push_test():
    _send_push("Voice Pipeline", "Push notifications are working!")
    return {"ok": True}

# ── UI ────────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def index():
    return _HTML

_HTML = """<!DOCTYPE html>
<html lang="hu">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="theme-color" content="#000000">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<link rel="manifest" href="/manifest.json">
<link rel="icon" href="/icon.svg">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
<link href="https://api.fontshare.com/v2/css?f[]=satoshi@300,400,500,700&display=swap" rel="stylesheet">
<title>DiJester</title>
<style>
:root{
  --bg:#000;
  --surface:#090909;
  --surface-2:#111;
  --surface-3:#161616;
  --border:rgba(255,255,255,0.06);
  --border-hover:rgba(255,255,255,0.12);
  --text:#ccc;
  --text-bright:#e8e8e8;
  --text-dim:#525252;
  --green:#3dba6e;
  --green-dim:rgba(61,186,110,0.12);
  --green-border:rgba(61,186,110,0.28);
  --red:#f87171;
  --red-dim:rgba(248,113,113,0.1);
  --silver:#909090;
  --silver-light:#c0c0c0;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Inter',system-ui,sans-serif;background:var(--bg);color:var(--text);
     min-height:100vh;-webkit-font-smoothing:antialiased}
a{color:var(--green);text-decoration:none}
a:hover{text-decoration:underline}

/* ── Header ── */
header{background:var(--bg);border-bottom:1px solid var(--border);
       padding:.75rem 1.25rem;display:flex;align-items:center;gap:.75rem;flex-wrap:wrap}
h1{font-family:'Satoshi',sans-serif;font-size:1rem;font-weight:700;
   letter-spacing:.04em;color:var(--text-bright);flex:1}

/* ── Recording dot ── */
.status-dot{width:8px;height:8px;border-radius:50%;background:var(--text-dim);flex-shrink:0;
            transition:background .3s}
.status-dot.recording{background:var(--red);animation:pulse-dot 1.4s infinite}
.status-dot.loading{background:#d4a017}
@keyframes pulse-dot{0%,100%{opacity:1;box-shadow:0 0 0 0 rgba(248,113,113,.4)}
                     50%{opacity:.7;box-shadow:0 0 0 6px rgba(248,113,113,0)}}

/* ── Mic bars ── */
.mic-bars{display:flex;align-items:flex-end;gap:3px;height:18px;flex-shrink:0}
.mic-bars .bar{width:3px;border-radius:2px;background:var(--surface-3);
               transition:height .1s ease,background .1s ease}
.mic-bars.active .bar{background:var(--green)}

/* ── Badge / queue ── */
.badge{font-size:.7rem;color:var(--text-dim);letter-spacing:.04em}

/* ── Buttons ── */
.btn{padding:.38rem .85rem;border-radius:6px;border:1px solid var(--border);
     cursor:pointer;font-size:.8rem;font-weight:500;font-family:'Inter',sans-serif;
     background:var(--surface-2);color:var(--text);transition:border-color .2s,background .2s,color .2s}
.btn:hover:not(:disabled){border-color:var(--border-hover);background:var(--surface-3);color:var(--text-bright)}
.btn:disabled{opacity:.35;cursor:default}
.btn-green{background:var(--green-dim);border-color:var(--green-border);color:var(--green)}
.btn-green:hover:not(:disabled){background:rgba(61,186,110,.2)}
.btn-red{background:var(--red-dim);border-color:rgba(248,113,113,.25);color:var(--red)}
.btn-red:hover:not(:disabled){background:rgba(248,113,113,.16)}
.btn-sm{padding:.28rem .65rem;font-size:.75rem}

/* ── Tabs ── */
.tabs{display:flex;gap:0;background:var(--bg);border-bottom:1px solid var(--border);
      padding:0 1.25rem;overflow-x:auto}
.tab{padding:.6rem 1rem;cursor:pointer;color:var(--text-dim);font-size:.78rem;
     font-weight:500;letter-spacing:.04em;white-space:nowrap;
     border-bottom:2px solid transparent;transition:color .2s;flex-shrink:0}
.tab:hover{color:var(--silver)}
.tab.active{color:var(--text-bright);border-bottom-color:var(--green)}

/* ── Panels ── */
.panel{display:none;padding:1.1rem 1.25rem}
.panel.active{display:block}
.feed{max-height:calc(100vh - 210px);overflow-y:auto}

/* ── Section label ── */
.section-label{font-size:.68rem;font-weight:600;letter-spacing:.18em;
               text-transform:uppercase;color:var(--text-dim);margin-bottom:.7rem}

/* ── Toolbar ── */
.toolbar{display:flex;gap:.5rem;margin-bottom:.85rem;align-items:center;flex-wrap:wrap}

/* ── Entry cards (feed / extractions) ── */
.entry{padding:.55rem .8rem;margin-bottom:.35rem;background:var(--surface);
       border-radius:6px;border:1px solid var(--border);border-left:2px solid var(--border)}
.entry.unprocessed{border-left-color:var(--green)}
.ts{font-size:.68rem;color:var(--text-dim);margin-bottom:.15rem;letter-spacing:.02em}
.txt{font-size:.87rem;line-height:1.55;color:var(--text)}

/* ── Checklist ── */
ul.checklist{list-style:none}
ul.checklist li{display:flex;align-items:center;gap:.55rem;padding:.5rem .75rem;
                margin-bottom:.3rem;background:var(--surface);border-radius:6px;
                border:1px solid var(--border);transition:border-color .2s}
ul.checklist li:hover{border-color:var(--border-hover)}
ul.checklist li.done .label{text-decoration:line-through;color:var(--text-dim)}
.label{flex:1;font-size:.87rem;color:var(--text)}
.qty{font-size:.76rem;color:var(--text-dim);margin-right:.35rem}

/* Circular checkbox */
.cb{appearance:none;-webkit-appearance:none;width:16px;height:16px;border-radius:50%;
    border:1px solid var(--text-dim);background:transparent;cursor:pointer;
    flex-shrink:0;position:relative;transition:border-color .2s,background .2s}
.cb:checked{background:var(--green);border-color:var(--green)}
.cb:checked::after{content:"";position:absolute;left:4px;top:2px;
                   width:5px;height:8px;border:2px solid #000;
                   border-top:none;border-left:none;transform:rotate(45deg)}
.del{background:none;border:none;color:var(--text-dim);cursor:pointer;
     font-size:.85rem;padding:0 .2rem;transition:color .2s;line-height:1}
.del:hover{color:var(--red)}

/* ── Category badge ── */
.cat-badge{font-size:.66rem;padding:.1rem .45rem;border-radius:10px;
           background:var(--surface-2);border:1px solid var(--border);
           color:var(--silver);flex-shrink:0;letter-spacing:.04em}

/* ── Summary cards ── */
.summary-card{background:var(--surface);border:1px solid var(--border);
              border-radius:8px;margin-bottom:.65rem;overflow:hidden;
              transition:border-color .2s}
.summary-card:hover{border-color:var(--border-hover)}
.summary-header{padding:.65rem .85rem;display:flex;align-items:flex-start;gap:.6rem}
.summary-title{flex:1;font-size:.88rem;font-weight:600;color:var(--text-bright);
               line-height:1.4;font-family:'Satoshi',sans-serif}
.summary-meta{font-size:.68rem;color:var(--text-dim);padding:0 .85rem .4rem;letter-spacing:.03em}
.summary-body{font-size:.83rem;line-height:1.6;color:var(--text);
              padding:.5rem .85rem .85rem;border-top:1px solid var(--border);white-space:pre-wrap}

/* ── Extraction log ── */
.ext-entry{background:var(--surface);border:1px solid var(--border);
           border-radius:6px;margin-bottom:.5rem;overflow:hidden}
.ext-header{padding:.42rem .75rem;display:flex;gap:.75rem;align-items:center;
            font-size:.72rem;color:var(--text-dim);border-bottom:1px solid var(--border)}
.ext-counts{margin-left:auto;color:var(--silver)}
.ext-raw{padding:.55rem .75rem;font-family:'Courier New',monospace;font-size:.78rem;
         white-space:pre-wrap;word-break:break-word;color:rgba(61,186,110,.8);
         max-height:240px;overflow-y:auto}

/* ── Settings / agent cards ── */
.agent-card{background:var(--surface);border:1px solid var(--border);border-radius:8px;
            padding:.85rem 1rem;margin-bottom:.6rem;transition:border-color .2s}
.agent-card:hover{border-color:var(--border-hover)}
.agent-header{display:flex;align-items:center;gap:.6rem;margin-bottom:.35rem}
.agent-name{font-family:'Satoshi',sans-serif;font-weight:700;font-size:.9rem;
            color:var(--text-bright);flex:1;letter-spacing:.02em}
.source-badge{font-size:.65rem;padding:.1rem .42rem;border-radius:10px;flex-shrink:0;
              letter-spacing:.05em;font-weight:600;text-transform:uppercase}
.src-transcriptions{background:rgba(61,186,110,.1);border:1px solid var(--green-border);color:var(--green)}
.src-hackernews{background:rgba(251,179,65,.08);border:1px solid rgba(251,179,65,.25);color:#fbb341}
.agent-desc{font-size:.8rem;color:var(--text-dim);margin-bottom:.65rem;line-height:1.5}
.agent-footer{display:flex;align-items:center;gap:.5rem;flex-wrap:wrap;margin-top:.6rem}
.last-run{font-size:.72rem;color:var(--text-dim);flex:1;letter-spacing:.02em}

/* Toggle switch */
.toggle{position:relative;display:inline-block;width:36px;height:20px;flex-shrink:0}
.toggle input{opacity:0;width:0;height:0}
.slider{position:absolute;cursor:pointer;inset:0;background:var(--surface-3);
        border:1px solid var(--border);border-radius:20px;transition:.25s}
.slider::before{content:"";position:absolute;width:14px;height:14px;left:2px;bottom:2px;
                background:var(--text-dim);border-radius:50%;transition:.25s}
input:checked+.slider{background:var(--green-dim);border-color:var(--green-border)}
input:checked+.slider::before{transform:translateX(16px);background:var(--green)}

/* Interval row */
.interval-row{display:flex;align-items:center;gap:.5rem;flex-wrap:wrap}
.interval-row label{font-size:.74rem;color:var(--text-dim);letter-spacing:.04em}
.interval-select,.interval-custom{
  background:var(--surface-2);color:var(--text);
  border:1px solid var(--border);border-radius:5px;
  padding:.28rem .5rem;font-size:.78rem;font-family:'Inter',sans-serif;
  cursor:pointer;transition:border-color .2s}
.interval-select:hover,.interval-custom:focus{border-color:var(--border-hover);outline:none}
.interval-custom{width:64px;display:none}

/* Notification card */
.notif-card{background:var(--surface);border:1px solid var(--border);
            border-radius:8px;padding:.85rem 1rem;margin-bottom:1.1rem}
.notif-status{font-size:.82rem;color:var(--text-dim);margin:.35rem 0 .7rem;line-height:1.5}

/* Empty state */
.empty{color:var(--text-dim);font-size:.83rem;padding:.85rem 0;letter-spacing:.02em}

/* ── Toast ── */
#toast{position:fixed;bottom:1.5rem;left:50%;transform:translateX(-50%);
       background:var(--surface-2);color:var(--text-bright);
       padding:.6rem 1.25rem;border-radius:7px;font-size:.84rem;
       border:1px solid var(--border);
       box-shadow:0 8px 32px rgba(0,0,0,.7);opacity:0;
       transition:opacity .25s;pointer-events:none;z-index:999;white-space:nowrap}
#toast.show{opacity:1}
#toast.ok{border-color:var(--green-border);color:var(--green)}
#toast.warn{border-color:rgba(251,179,65,.35);color:#fbb341}
#toast.err{border-color:rgba(248,113,113,.3);color:var(--red)}

/* Scrollbar */
::-webkit-scrollbar{width:4px;height:4px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--surface-3);border-radius:4px}
::-webkit-scrollbar-thumb:hover{background:var(--text-dim)}
</style>
</head>
<body>
<header>
  <span class="status-dot" id="dot"></span>
  <div class="mic-bars" id="mic-bars">
    <div class="bar" id="b1" style="height:3px"></div>
    <div class="bar" id="b2" style="height:3px"></div>
    <div class="bar" id="b3" style="height:3px"></div>
    <div class="bar" id="b4" style="height:3px"></div>
    <div class="bar" id="b5" style="height:3px"></div>
  </div>
  <h1>DiJester</h1>
  <span class="badge" id="queue-badge"></span>
  <button class="btn btn-green" id="btn-start" onclick="setRecording(true)">&#9654; Start</button>
  <button class="btn btn-red"   id="btn-stop"  onclick="setRecording(false)" disabled>&#9632; Stop</button>
</header>

<nav class="tabs">
  <div class="tab active" onclick="switchTab('feed',this)">Feed</div>
  <div class="tab"        onclick="switchTab('todos',this)">Todos</div>
  <div class="tab"        onclick="switchTab('shopping',this)">Shopping</div>
  <div class="tab"        onclick="switchTab('health',this)">Health</div>
  <div class="tab"        onclick="switchTab('summaries',this)">Summaries</div>
  <div class="tab"        onclick="switchTab('extractions',this)">Log</div>
  <div class="tab"        onclick="switchTab('settings',this)">Settings</div>
</nav>

<div id="feed" class="panel active">
  <div class="feed" id="feed-list"></div>
</div>

<div id="todos" class="panel">
  <ul class="checklist" id="todo-list"></ul>
</div>

<div id="shopping" class="panel">
  <ul class="checklist" id="shop-list"></ul>
</div>

<div id="health" class="panel">
  <ul class="checklist" id="health-list"></ul>
</div>

<div id="summaries" class="panel">
  <div id="summary-list"></div>
</div>

<div id="extractions" class="panel">
  <div class="section-label" style="margin-bottom:.85rem">Raw LLM responses</div>
  <div class="feed" id="extraction-list"></div>
</div>

<div id="settings" class="panel">
  <div class="notif-card">
    <div class="section-label">Push Notifications</div>
    <div class="notif-status" id="notif-status">Checking…</div>
    <div style="display:flex;gap:.5rem;flex-wrap:wrap">
      <button class="btn btn-green btn-sm" id="btn-notif" onclick="setupNotifications()">Enable Notifications</button>
      <button class="btn btn-sm" onclick="testNotification()">Send Test</button>
    </div>
  </div>
  <div class="section-label">Scheduled Agents</div>
  <div id="agents-list"></div>
</div>

<div id="toast"></div>

<script>
let activeTab = 'feed';

// ── Tab switching ─────────────────────────────────────────────────────────────
function switchTab(name, el) {
  activeTab = name;
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  el.classList.add('active');
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.getElementById(name).classList.add('active');
  refreshAll();
}

// ── Toast ─────────────────────────────────────────────────────────────────────
let _toastTimer = null;
function showToast(msg, type='ok') {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.className = 'show ' + type;
  clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => { t.className = ''; }, 4000);
}

// ── Recording ─────────────────────────────────────────────────────────────────
async function setRecording(on) {
  await fetch('/recording/' + (on ? 'start' : 'stop'), {method:'POST'});
  refreshStatus();
}

// ── Status ────────────────────────────────────────────────────────────────────
let _prevRunning = new Set();
async function refreshStatus() {
  const s = await fetch('/api/status').then(r=>r.json()).catch(()=>({}));
  const dot = document.getElementById('dot');
  dot.className = 'status-dot' + (s.recording ? ' recording' : (!s.whisper_ready ? ' loading' : ''));
  document.getElementById('btn-start').disabled = s.recording;
  document.getElementById('btn-stop').disabled  = !s.recording;
  document.getElementById('queue-badge').textContent =
    s.whisper_ready ? (s.queue > 0 ? `queue: ${s.queue}` : '') : 'whisper loading…';
  updateBars(s.level || 0);

  const nowRunning = new Set(s.running_agents || []);
  // detect agents that just finished
  for (const id of _prevRunning) {
    if (!nowRunning.has(id)) {
      // fetch updated result
      const agents = await fetch('/api/agents').then(r=>r.json()).catch(()=>[]);
      const a = agents.find(x => x.id === id);
      if (a) showToast(`${a.name}: ${a.last_result || 'done'}`,
                       (a.last_result||'').startsWith('error') ? 'err' :
                       (a.last_result||'').includes('nothing') ? 'warn' : 'ok');
      refreshAll();
    }
  }
  _prevRunning = nowRunning;
  // update running spinners on settings tab
  if (activeTab === 'settings') _updateAgentRunning(nowRunning);
}

function updateBars(level) {
  const profile = [0.5, 0.75, 1.0, 0.75, 0.5];
  const alive = level > 0.01;
  document.getElementById('mic-bars').className = 'mic-bars' + (alive ? ' active' : '');
  for (let i = 1; i <= 5; i++) {
    const h = alive ? Math.max(3, Math.round(level * profile[i-1] * 20)) : 3;
    document.getElementById('b'+i).style.height = h + 'px';
  }
}

// ── Live feed ─────────────────────────────────────────────────────────────────
async function refreshFeed() {
  const rows = await fetch('/api/transcriptions?limit=150').then(r=>r.json());
  const el = document.getElementById('feed-list');
  if (!rows.length) { el.innerHTML = '<div class="empty">No transcriptions yet.</div>'; return; }
  el.innerHTML = rows.map(r => `
    <div class="entry ${r.processed ? '' : 'unprocessed'}">
      <div class="ts">${r.timestamp.slice(0,19).replace('T',' ')}${r.processed ? '' : ' · pending'}</div>
      <div class="txt">${esc(r.text)}</div>
    </div>`).join('');
}

// ── Todos ─────────────────────────────────────────────────────────────────────
async function refreshTodos() {
  const rows = await fetch('/api/todos').then(r=>r.json());
  const el = document.getElementById('todo-list');
  if (!rows.length) { el.innerHTML = '<li class="empty">No todos yet.</li>'; return; }
  el.innerHTML = rows.map(r => `
    <li class="${r.done?'done':''}" id="todo-${r.id}">
      <input class="cb" type="checkbox" ${r.done?'checked':''} onchange="toggle('todos',${r.id})">
      <span class="label">${esc(r.text)}</span>
      <button class="del" onclick="del('todos',${r.id})">&#10005;</button>
    </li>`).join('');
}

// ── Shopping ──────────────────────────────────────────────────────────────────
async function refreshShopping() {
  const rows = await fetch('/api/shopping').then(r=>r.json());
  const el = document.getElementById('shop-list');
  if (!rows.length) { el.innerHTML = '<li class="empty">No shopping items yet.</li>'; return; }
  el.innerHTML = rows.map(r => `
    <li class="${r.done?'done':''}" id="shop-${r.id}">
      <input class="cb" type="checkbox" ${r.done?'checked':''} onchange="toggle('shopping',${r.id})">
      <span class="label">${esc(r.item)}</span>
      ${r.quantity ? `<span class="qty">${esc(r.quantity)}</span>` : ''}
      <button class="del" onclick="del('shopping',${r.id})">&#10005;</button>
    </li>`).join('');
}

// ── Health ────────────────────────────────────────────────────────────────────
const CAT_COLORS = {
  symptom:'#f87171',medication:'#fbb341',appointment:'#67e8f9',
  exercise:'#3dba6e',mood:'#c084fc',food:'#a3e635'
};
async function refreshHealth() {
  const rows = await fetch('/api/health').then(r=>r.json());
  const el = document.getElementById('health-list');
  if (!rows.length) { el.innerHTML = '<li class="empty">No health data yet.</li>'; return; }
  el.innerHTML = rows.map(r => {
    const color = CAT_COLORS[r.category] || '#a0aec0';
    return `<li class="${r.done?'done':''}" id="health-${r.id}">
      <input class="cb" type="checkbox" ${r.done?'checked':''} onchange="toggle('health',${r.id})">
      <span class="label">${esc(r.text)}</span>
      ${r.category ? `<span class="cat-badge" style="color:${color}">${esc(r.category)}</span>` : ''}
      <button class="del" onclick="del('health',${r.id})">&#10005;</button>
    </li>`;
  }).join('');
}

// ── Summaries ─────────────────────────────────────────────────────────────────
async function refreshSummaries() {
  const rows = await fetch('/api/summaries').then(r=>r.json());
  const el = document.getElementById('summary-list');
  if (!rows.length) { el.innerHTML = '<div class="empty">No summaries yet.</div>'; return; }
  el.innerHTML = rows.map(r => `
    <div class="summary-card">
      <div class="summary-header">
        <div>
          <div class="summary-title">${esc(r.title)}</div>
          ${r.url ? `<a href="${esc(r.url)}" target="_blank" style="font-size:.75rem;color:#667eea">${esc(r.url.replace(/^https?:\/\//,'').slice(0,60))}</a>` : ''}
        </div>
        <button class="del" onclick="delSummary(${r.id})">&#10005;</button>
      </div>
      <div class="summary-meta">${r.created_at.slice(0,19).replace('T',' ')} &middot; ${esc(r.source)}</div>
      <div class="summary-body">${esc(r.content)}</div>
    </div>`).join('');
}

async function delSummary(id) {
  await fetch('/api/summaries/'+id, {method:'DELETE'});
  refreshSummaries();
}

// ── Extraction log ────────────────────────────────────────────────────────────
async function refreshExtractions() {
  const rows = await fetch('/api/extractions?limit=30').then(r=>r.json());
  const el = document.getElementById('extraction-list');
  if (!rows.length) { el.innerHTML = '<div class="empty">No extractions yet.</div>'; return; }
  el.innerHTML = rows.map(r => {
    const pretty = tryJson(r.raw_response);
    const agentLabel = r.agent_name ? ` &middot; ${esc(r.agent_name)}` : '';
    return `<div class="ext-entry">
      <div class="ext-header">
        <span>${r.timestamp.slice(0,19).replace('T',' ')}${agentLabel}</span>
        <span>${r.segments_count} input(s)</span>
        <span class="ext-counts">${r.todos_count}T / ${r.shopping_count}S</span>
      </div>
      <div class="ext-raw">${esc(pretty)}</div>
    </div>`;
  }).join('');
}

function tryJson(s) {
  try { return JSON.stringify(JSON.parse(s), null, 2); }
  catch { return s; }
}

// ── Settings & Agents ─────────────────────────────────────────────────────────
const INTERVALS = [
  [15,'15 min'],[30,'30 min'],[60,'1 hour'],[120,'2 hours'],
  [180,'3 hours'],[360,'6 hours'],[720,'12 hours'],[1440,'Daily'],[0,'Custom…']
];

async function refreshSettings() {
  updateNotifStatus();
  const agents = await fetch('/api/agents').then(r=>r.json()).catch(()=>[]);
  const el = document.getElementById('agents-list');
  if (!agents.length) { el.innerHTML = '<div class="empty">No agents.</div>'; return; }
  el.innerHTML = agents.map(a => {
    const srcClass = 'src-' + a.source;
    const srcLabel = a.source === 'hackernews' ? 'HN' : 'MIC';
    const isCustom = !INTERVALS.slice(0,-1).find(([v])=>v===a.interval_min);
    const selOpts = INTERVALS.map(([v,l]) =>
      `<option value="${v}" ${(isCustom?v===0:v===a.interval_min)?'selected':''}>${l}</option>`
    ).join('');
    const lastInfo = a.last_run
      ? `Last run ${a.last_run.slice(0,16).replace('T',' ')} &middot; ${esc(a.last_result||'')}`
      : 'Never run';
    return `<div class="agent-card" id="agent-card-${a.id}">
      <div class="agent-header">
        <span class="agent-name">${esc(a.name)}</span>
        <span class="source-badge ${srcClass}">${srcLabel}</span>
        <label class="toggle" title="${a.enabled?'Enabled':'Disabled'}">
          <input type="checkbox" ${a.enabled?'checked':''} onchange="toggleAgent(${a.id},this.checked)">
          <span class="slider"></span>
        </label>
      </div>
      <div class="agent-desc">${esc(a.description)}</div>
      <div class="interval-row">
        <label>Run every</label>
        <select class="interval-select" id="sel-${a.id}" onchange="intervalChanged(${a.id},this)">
          ${selOpts}
        </select>
        <input class="interval-custom" id="cust-${a.id}" type="number" min="1" max="44640"
               value="${isCustom?a.interval_min:60}" placeholder="min"
               onchange="setCustomInterval(${a.id},this.value)"
               style="display:${isCustom?'block':'none'}">
        ${isCustom ? `<span style="font-size:.78rem;color:#718096">min</span>` : ''}
      </div>
      <div class="agent-footer">
        <span class="last-run" id="last-${a.id}">${lastInfo}</span>
        <button class="btn btn-green btn-sm" id="run-${a.id}"
                onclick="runAgent(${a.id})" ${a.running?'disabled':''}>
          ${a.running ? '&#9203; Running…' : '&#9654; Run now'}
        </button>
      </div>
    </div>`;
  }).join('');
}

function _updateAgentRunning(runningSet) {
  for (const el of document.querySelectorAll('[id^="run-"]')) {
    const id = parseInt(el.id.replace('run-',''));
    const running = runningSet.has(id);
    el.disabled = running;
    el.innerHTML = running ? '&#9203; Running…' : '&#9654; Run now';
  }
}

async function toggleAgent(id, enabled) {
  await fetch('/api/agents/'+id, {method:'PUT',headers:{'Content-Type':'application/json'},
              body: JSON.stringify({enabled})});
  showToast(enabled ? 'Agent enabled' : 'Agent disabled', 'ok');
}

function intervalChanged(id, sel) {
  const val = parseInt(sel.value);
  const custEl = document.getElementById('cust-'+id);
  if (val === 0) {
    custEl.style.display = 'block';
    return;
  }
  custEl.style.display = 'none';
  setIntervalMin(id, val);
}

async function setCustomInterval(id, val) {
  const v = parseInt(val);
  if (v >= 1 && v <= 44640) setIntervalMin(id, v);
}

async function setIntervalMin(id, min) {
  await fetch('/api/agents/'+id, {method:'PUT',headers:{'Content-Type':'application/json'},
              body: JSON.stringify({interval_min: min})});
  showToast('Schedule saved', 'ok');
}

async function runAgent(id) {
  const r = await fetch('/api/agents/'+id+'/run', {method:'POST'});
  const d = await r.json();
  showToast(d.message, 'ok');
  document.getElementById('run-'+id).disabled = true;
  document.getElementById('run-'+id).innerHTML = '&#9203; Running…';
}

// ── Push notifications ────────────────────────────────────────────────────────
let _swReg = null;

async function registerSW() {
  if (!('serviceWorker' in navigator)) return;
  try {
    _swReg = await navigator.serviceWorker.register('/sw.js');
  } catch(e) { console.warn('SW register failed:', e); }
}

function updateNotifStatus() {
  const el = document.getElementById('notif-status');
  const btn = document.getElementById('btn-notif');
  if (!('Notification' in window)) {
    el.textContent = 'Notifications not supported in this browser.';
    btn.disabled = true; return;
  }
  const p = Notification.permission;
  if (p === 'granted') {
    el.textContent = 'Push notifications are enabled.';
    btn.textContent = 'Unsubscribe';
    btn.onclick = unsubscribeNotifications;
  } else if (p === 'denied') {
    el.textContent = 'Notifications blocked — allow them in browser settings.';
    btn.disabled = true;
  } else {
    el.textContent = 'Notifications are not set up yet.';
    btn.textContent = 'Enable Notifications';
    btn.onclick = setupNotifications;
    btn.disabled = false;
  }
}

function urlBase64ToUint8Array(b64) {
  const pad = '='.repeat((4 - b64.length % 4) % 4);
  const raw = atob((b64 + pad).replace(/-/g,'+').replace(/_/g,'/'));
  return Uint8Array.from([...raw].map(c => c.charCodeAt(0)));
}

async function setupNotifications() {
  if (!_swReg) { showToast('Service worker not ready', 'err'); return; }
  const perm = await Notification.requestPermission();
  if (perm !== 'granted') { showToast('Permission denied', 'err'); return; }
  try {
    const pubKey = await fetch('/api/push/vapid-public-key').then(r => {
      if (!r.ok) throw new Error('Push not available on server');
      return r.text();
    });
    const sub = await _swReg.pushManager.subscribe({
      userVisibleOnly: true,
      applicationServerKey: urlBase64ToUint8Array(pubKey)
    });
    await fetch('/api/push/subscribe', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify(sub.toJSON())
    });
    showToast('Notifications enabled!', 'ok');
    updateNotifStatus();
  } catch(e) { showToast('Setup failed: ' + e.message, 'err'); }
}

async function unsubscribeNotifications() {
  if (!_swReg) return;
  const sub = await _swReg.pushManager.getSubscription();
  if (sub) {
    await fetch('/api/push/unsubscribe', {method:'DELETE',headers:{'Content-Type':'application/json'},
               body: JSON.stringify({endpoint: sub.endpoint})});
    await sub.unsubscribe();
  }
  showToast('Unsubscribed', 'warn');
  updateNotifStatus();
}

async function testNotification() {
  await fetch('/api/push/test', {method:'POST'});
  showToast('Test notification sent', 'ok');
}

// ── Helpers ───────────────────────────────────────────────────────────────────
async function toggle(type, id) {
  await fetch(`/api/${type}/${id}/toggle`, {method:'POST'});
  if (type==='todos') refreshTodos();
  else if (type==='shopping') refreshShopping();
  else if (type==='health') refreshHealth();
}

async function del(type, id) {
  await fetch(`/api/${type}/${id}`, {method:'DELETE'});
  if (type==='todos') refreshTodos();
  else if (type==='shopping') refreshShopping();
  else if (type==='health') refreshHealth();
}

function esc(s) {
  return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function refreshAll() {
  refreshStatus();
  if (activeTab==='feed')        refreshFeed();
  if (activeTab==='todos')       refreshTodos();
  if (activeTab==='shopping')    refreshShopping();
  if (activeTab==='health')      refreshHealth();
  if (activeTab==='summaries')   refreshSummaries();
  if (activeTab==='extractions') refreshExtractions();
  if (activeTab==='settings')    refreshSettings();
}

// ── Boot ──────────────────────────────────────────────────────────────────────
registerSW();
refreshAll();
setInterval(refreshAll, 3000);
setInterval(async () => {
  const s = await fetch('/api/status').then(r=>r.json()).catch(()=>({level:0}));
  updateBars(s.level || 0);
}, 150);
</script>
</body>
</html>
"""

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
