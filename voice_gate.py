#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
語音門禁助理 (voice_gate.py)

按住 GPIO 18 (pull-down) 開始錄音 → Gemini 2.5 Flash 辨識訪客意圖 → TTS 語音回饋
無螢幕操作。使用 gpiozero 驅動按鈕（相容新版 Pi OS kernel）。
"""

import json
import logging
import math
import os
import signal
import sqlite3
import struct
import subprocess
import sys
import tempfile
import threading
import time
import wave

import alsaaudio
from gpiozero import Button
from google import genai
from google.genai import types as genai_types
from gtts import gTTS

# ─────────────────────────────────────────────────────────────────────────────
# 設定
# ─────────────────────────────────────────────────────────────────────────────
BUTTON_PIN     = 18
SAMPLE_RATE    = 16000
CHANNELS       = 1
FORMAT         = alsaaudio.PCM_FORMAT_S16_LE
PERIOD_SIZE    = 1024
CAPTURE_DEVICE = "plughw:UACDemoV10"   # 啟動時自動偵測
PLAYBACK_CARD  = "CD002AUDIO"          # USB 喇叭；3.5mm 改 "Headphones"

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
WAV_PATH       = "/tmp/visitor_voice.wav"
TTS_LANG       = "zh-TW"
TTS_CACHE_DIR  = "/tmp/voice_gate_tts"

DEBOUNCE_S   = 0.05
MIN_RECORD_S = 0.5
MAX_RECORD_S = 12.0

ADMIN_DB_PATH = "/home/rob/intercom_system/data/admin.db"

# 公司資料：{id: {"name": str, "extension": str}}，啟動時從 DB 載入
COMPANIES: dict = {}

SYSTEM_PROMPT = ""  # 啟動時依 COMPANIES 動態產生

def _build_system_prompt() -> str:
    company_list = ", ".join(
        f"ID {cid}: {info['name']}" for cid, info in sorted(COMPANIES.items())
    )
    return (
        "你是一個門禁語音助理。請聽取音檔，辨識訪客說的公司名稱，並回傳對應的公司 ID。"
        f" 公司名單：{company_list}。"
        " 注意：音檔可能有雜訊或輕微回音，請盡力從人聲中辨識公司名稱。"
        " 訪客可能只說公司名稱的一部分，請模糊比對最接近的公司。"
        ' 若能判定，輸出格式：{"id": 數字, "name": "公司名稱"}。'
        ' 若完全無法判定（例如音檔無人聲、嚴重雜訊），才回傳 {"id": 0}。'
    )

PHRASES = {
    "ready":   "系統已就緒，請按住按鈕說話",
    "wait":    "收到，正在幫您辨識，請稍候",
    "fail":    "抱歉，我聽不清楚，請再按一次按鈕說話",
    "network": "網路連線異常，請稍後再試",
    "short":   "按鈕時間太短，請按住按鈕說完再放開",
    "mic_busy":"麥克風暫時忙碌，請稍後再試",
}

# ─────────────────────────────────────────────────────────────────────────────
# 日誌
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("voice-gate")

# ─────────────────────────────────────────────────────────────────────────────
# 全域
# ─────────────────────────────────────────────────────────────────────────────
client: genai.Client = None  # type: ignore

_state      = "IDLE"   # IDLE | RECORDING | PROCESSING
_state_lock = threading.Lock()
_stop_rec   = threading.Event()
_current_rec_thread: threading.Thread = None  # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
# 裝置偵測
# ─────────────────────────────────────────────────────────────────────────────
def _detect_capture_device() -> str:
    preference = [
        "plughw:UACDemoV10",
        "plughw:CARD=UACDemoV10,DEV=0",
    ]
    available = alsaaudio.pcms(alsaaudio.PCM_CAPTURE)
    for p in preference:
        if p in available:
            return p
    for dev in available:
        if "plughw" in dev and "Headphones" not in dev and "vc4" not in dev.lower():
            return dev
    return "default"


# ─────────────────────────────────────────────────────────────────────────────
# 音訊播放
# ─────────────────────────────────────────────────────────────────────────────
_audio_proc: "subprocess.Popen | None" = None
_audio_proc_lock = threading.Lock()


def _aplay(wav_path: str):
    """播放 WAV 檔，同時追蹤執行中的 process（讓 stop_audio() 可以中斷）"""
    global _audio_proc
    cmd = ["aplay", "-q", "-D", f"plughw:{PLAYBACK_CARD}", wav_path]
    proc = subprocess.Popen(cmd)
    with _audio_proc_lock:
        _audio_proc = proc
    proc.wait()
    with _audio_proc_lock:
        if _audio_proc is proc:
            _audio_proc = None


def stop_audio():
    """立刻中斷正在播放的音訊（按鈕按下時呼叫，清除 TTS 佔用）"""
    global _audio_proc
    with _audio_proc_lock:
        proc = _audio_proc
        _audio_proc = None
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=0.3)
        except subprocess.TimeoutExpired:
            proc.kill()


def beep(freq: int = 880, duration_ms: int = 150, volume: float = 0.7):
    n = int(SAMPLE_RATE * duration_ms / 1000)
    buf = bytearray(n * 2)
    for i in range(n):
        fade = 1.0 - i / n
        val = int(32767 * volume * fade * math.sin(2 * math.pi * freq * i / SAMPLE_RATE))
        struct.pack_into("<h", buf, i * 2, max(-32768, min(32767, val)))
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = f.name
    try:
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(bytes(buf))
        _aplay(path)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def speak(text: str, silent: bool = False):
    os.makedirs(TTS_CACHE_DIR, exist_ok=True)
    safe = "".join(c if c.isalnum() else "_" for c in text)[:60]
    cache_wav = os.path.join(TTS_CACHE_DIR, safe + ".wav")

    if not os.path.exists(cache_wav):
        log.info(f"TTS 合成: {text}")
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                mp3 = f.name
            gTTS(text=text, lang=TTS_LANG, slow=False).save(mp3)
            subprocess.run(
                ["ffmpeg", "-y", "-i", mp3, "-ar", str(SAMPLE_RATE), "-ac", "1", cache_wav],
                check=True, capture_output=True,
            )
            os.unlink(mp3)
        except Exception as e:
            log.error(f"TTS 失敗: {e}")
            return

    if not silent:
        log.info(f"TTS 播放: {text}")
        _aplay(cache_wav)


def _precache_tts():
    log.info("預先快取 TTS（靜音合成）...")
    for text in PHRASES.values():
        speak(text, silent=True)
    for info in COMPANIES.values():
        speak(f"好的，正在幫您接通 {info['name']}", silent=True)
    log.info("TTS 快取完成")


# ─────────────────────────────────────────────────────────────────────────────
# 錄音
# ─────────────────────────────────────────────────────────────────────────────
_active_pcm = None          # 供 SIGTERM handler 關閉 ALSA
_active_pcm_lock = threading.Lock()

def record_audio() -> bool:
    global _active_pcm
    log.info("錄音開始...")
    frames = []
    start = time.monotonic()

    # 重試開啟 ALSA（最多 6 次，每次等 0.5s）
    # 原因：前次 voice_gate 被 SIGKILL 時可能留下 zombie PCM handle
    pcm = None
    for attempt in range(6):
        try:
            pcm = alsaaudio.PCM(
                alsaaudio.PCM_CAPTURE,
                alsaaudio.PCM_NORMAL,
                device=CAPTURE_DEVICE,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                format=FORMAT,
                periodsize=PERIOD_SIZE,
            )
            break
        except Exception as e:
            if 'busy' in str(e).lower() or 'resource' in str(e).lower():
                if attempt < 5:
                    log.warning(f"麥克風忙碌，重試 {attempt+1}/6: {e}")
                    time.sleep(0.5)
                    continue
            log.error(f"錄音失敗: {e}")
            return False

    if pcm is None:
        log.error("錄音失敗：麥克風仍忙碌，放棄")
        return False

    with _active_pcm_lock:
        _active_pcm = pcm

    try:
        while not _stop_rec.is_set():
            if time.monotonic() - start >= MAX_RECORD_S:
                log.warning("超過最長錄音時間，自動截斷")
                break
            n, data = pcm.read()
            if n > 0:
                frames.append(data)
    finally:
        with _active_pcm_lock:
            _active_pcm = None
        try:
            pcm.close()
        except Exception:
            pass

    duration = time.monotonic() - start
    log.info(f"錄音結束，時長 {duration:.2f}s")

    if duration < MIN_RECORD_S:
        log.warning("錄音太短")
        return False

    try:
        raw = b"".join(frames)
        # ── 軟體放大：自動將過低音量放大，提升辨識率 ──────────────────
        try:
            import array as _array
            samples = _array.array("h", raw)
            sq_sum = sum(s * s for s in samples)
            rms = (sq_sum / len(samples)) ** 0.5 if samples else 0
            if 30 < rms < 1500:  # 有聲音但太小聲才放大
                gain = min(1500.0 / rms, 8.0)
                samples = _array.array(
                    "h",
                    (max(-32768, min(32767, int(s * gain))) for s in samples),
                )
                raw = samples.tobytes()
                log.info(f"音量自動放大: RMS {rms:.0f} → 目標 1500 (x{gain:.1f})")
            else:
                log.info(f"音量正常: RMS {rms:.0f}")
        except Exception as amp_e:
            log.warning(f"放大略過: {amp_e}")
        # ────────────────────────────────────────────────────────────────
        with wave.open(WAV_PATH, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(raw)
        log.info(f"WAV 儲存: {WAV_PATH}")
        return True
    except Exception as e:
        log.error(f"WAV 儲存失敗: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Gemini 辨識
# ─────────────────────────────────────────────────────────────────────────────
def recognize() -> dict:
    uploaded = None
    try:
        log.info("上傳音檔至 Gemini...")
        with open(WAV_PATH, "rb") as f:
            uploaded = client.files.upload(
                file=f,
                config={"mime_type": "audio/wav"},
            )

        log.info("等待 Gemini 辨識...")
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[uploaded, "請辨識訪客要找哪間公司。"],
            config=genai_types.GenerateContentConfig(
                system_instruction=_build_system_prompt(),
                temperature=0.0,
            ),
        )
        raw = resp.text.strip()
        log.info(f"Gemini 回應: {raw}")

        if "```" in raw:
            parts = raw.split("```")
            raw = parts[1].lstrip("json").strip() if len(parts) > 1 else raw

        return json.loads(raw)

    except TimeoutError:
        log.error("Gemini API 超時")
        return {"id": -1, "error": "timeout"}
    except Exception as e:
        msg = str(e).lower()
        log.error(f"Gemini 失敗: {e}")
        if any(k in msg for k in ("network", "connection", "unreachable", "name or service",
                                   "name resolution", "errno -3", "errno -2", "timed out",
                                   "socket", "ssl")):
            return {"id": -2, "error": "network"}
        return {"id": -1, "error": str(e)}
    finally:
        if uploaded:
            try:
                client.files.delete(name=uploaded.name)
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# 撥號觸發（寫入 voice_call_queue，由 main.py 輪詢後觸發 SIP）
# ─────────────────────────────────────────────────────────────────────────────
CALL_QUEUE_DEDUP_WINDOW_S = 15    # pending 去重短窗（秒）：防止按鈕抖動/重複觸發
CALL_QUEUE_PROCESSING_TTL_S = 600 # processing 視為「通話/撥號 busy」的最大存活秒數（10 分鐘）

def _trigger_call(company_id: int, name: str, extension: str):
    """
    辨識成功後寫入撥號佇列（由 main.py 輪詢後觸發 SIP 撥號）
    去重策略：
      1) pending：只看 CALL_QUEUE_DEDUP_WINDOW_S 短窗（防抖動）
      2) processing：視為 busy（通話進行中），用 claimed_at + CALL_QUEUE_PROCESSING_TTL_S 判斷
    """
    extension = str(extension).strip()
    if not extension:
        log.error("辨識成功但 extension 為空，略過寫入 voice_call_queue")
        return False
    try:
        conn = sqlite3.connect(ADMIN_DB_PATH, timeout=5)
        cur = conn.cursor()
        cur.execute("BEGIN IMMEDIATE")
        # 同分機：若 pending（短窗）或 processing（長 TTL）存在，視為 busy 不重複 enqueue
        cur.execute(
            """
            SELECT id, status, created_at, claimed_at
            FROM voice_call_queue
            WHERE extension = ?
              AND (
                    (status = 'pending'
                     AND (strftime('%s','now') - strftime('%s', created_at, 'utc')) BETWEEN 0 AND ?)
                 OR (status = 'processing'
                     AND (claimed_at IS NULL OR claimed_at = ''
                          OR (strftime('%s','now') - strftime('%s', claimed_at, 'utc')) BETWEEN 0 AND ?))
              )
            ORDER BY id DESC
            LIMIT 1
            """,
            (extension, CALL_QUEUE_DEDUP_WINDOW_S, CALL_QUEUE_PROCESSING_TTL_S),
        )
        row = cur.fetchone()
        if row:
            existing_id, status, created_at, claimed_at = row
            log.warning(
                f"[VoiceGate] 去重：忽略撥號請求 ext={extension} "
                f"id={existing_id} status={status} created_at={created_at} claimed_at={claimed_at}"
            )
            conn.commit()
            conn.close()
            return False
        cur.execute(
            "INSERT INTO voice_call_queue (company_id, company_name, extension, status) "
            "VALUES (?, ?, ?, 'pending')",
            (company_id, name, extension),
        )
        conn.commit()
        conn.close()
        log.info(f"[VoiceGate] 撥號請求已寫入佇列: {name} → 分機 {extension}")
        return True
    except sqlite3.IntegrityError as e:
        # DB 層 unique partial index 觸發（雙重保護）
        log.warning(f"[VoiceGate] DB UNIQUE 去重觸發，忽略 ext={extension}: {e}")
        return False
    except Exception as e:
        log.error(f"[VoiceGate] 寫入撥號佇列失敗: {e}")
        return False


def _load_companies_from_db() -> dict:
    """從 admin.db 載入公司名單，回傳 {id: {'name': str, 'extension': str}}"""
    try:
        conn = sqlite3.connect(ADMIN_DB_PATH, timeout=5)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, name, extension FROM companies ORDER BY id"
        ).fetchall()
        conn.close()
        result = {row["id"]: {"name": row["name"], "extension": row["extension"]} for row in rows}
        log.info(f"載入 {len(result)} 家公司: {[v['name'] for v in result.values()]}")
        return result
    except Exception as e:
        log.error(f"載入公司資料失敗: {e}")
        return {}


def _ensure_call_queue_table():
    """確保 voice_call_queue 資料表存在（含向下相容欄位擴充與索引）"""
    try:
        conn = sqlite3.connect(ADMIN_DB_PATH, timeout=5)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS voice_call_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                company_id INTEGER,
                company_name TEXT,
                extension TEXT,
                status TEXT DEFAULT 'pending',
                created_at DATETIME DEFAULT (datetime('now','localtime')),
                claimed_at DATETIME,
                claimed_by TEXT,
                processed_at DATETIME,
                result TEXT
            )
        """)
        # 舊資料庫向下相容：補欄位（已存在會拋 OperationalError，直接略過）
        for ddl in (
            "ALTER TABLE voice_call_queue ADD COLUMN claimed_at DATETIME",
            "ALTER TABLE voice_call_queue ADD COLUMN claimed_by TEXT",
            "ALTER TABLE voice_call_queue ADD COLUMN result TEXT",
        ):
            try:
                cur.execute(ddl)
            except sqlite3.OperationalError:
                pass
        # 加索引：加速 consumer 查詢與 producer 去重
        cur.execute("CREATE INDEX IF NOT EXISTS idx_vcq_status_created ON voice_call_queue(status, created_at)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_vcq_ext_status_created ON voice_call_queue(extension, status, created_at)")
        conn.commit()
        conn.close()
    except Exception as e:
        log.error(f"建立 voice_call_queue 失敗: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 處理流程（放開按鈕後執行）
# ─────────────────────────────────────────────────────────────────────────────
def session_thread(rec_thread: threading.Thread):
    global _state

    rec_thread.join(timeout=MAX_RECORD_S + 8)  # +8: 含重試等待時間

    if not os.path.exists(WAV_PATH):
        # 若是麥克風忙碌（重試失敗），給不同提示
        phrase = PHRASES["mic_busy"] if _stop_rec.is_set() else PHRASES["short"]
        speak(phrase)
        with _state_lock:
            _state = "IDLE"
        speak(PHRASES["ready"])
        return

    speak(PHRASES["wait"])
    result = recognize()
    cid = result.get("id", 0)

    call_triggered = False
    if cid == -2:
        speak(PHRASES["network"])
    elif cid <= 0:
        speak(PHRASES["fail"])
    else:
        info = COMPANIES.get(cid, {})
        name = info.get("name", result.get("name", f"分機 {cid}"))
        extension = info.get("extension", "")
        speak(f"好的，正在幫您接通 {name}")
        log.info(f"辨識成功 → ID {cid}: {name} (分機 {extension})")
        call_triggered = _trigger_call(cid, name, extension)

    with _state_lock:
        _state = "IDLE"

    # 撥號已觸發時不播放「就緒」提示，避免 TTS 佔用 USB 喇叭導致通話無聲
    if not call_triggered:
        speak(PHRASES["ready"])


# ─────────────────────────────────────────────────────────────────────────────
# 按鈕回呼
# ─────────────────────────────────────────────────────────────────────────────
_last_press = 0.0
_last_release = 0.0


def on_pressed():
    global _state, _last_press, _current_rec_thread

    now = time.monotonic()
    if now - _last_press < DEBOUNCE_S:
        return
    _last_press = now

    with _state_lock:
        current = _state

    if current != "IDLE":
        log.debug(f"忽略按下（狀態={current}）")
        return

    log.info("按鈕按下 → 錄音")
    with _state_lock:
        _state = "RECORDING"

    # 立刻中斷正在播放的 TTS（否則 beep() 要等 TTS 結束才能佔用音訊裝置）
    stop_audio()

    _stop_rec.clear()
    if os.path.exists(WAV_PATH):
        try:
            os.unlink(WAV_PATH)
        except OSError:
            pass

    beep()              # 同步播放嗵聲 ~150ms
    time.sleep(0.30)    # 等殘響消散（縮短到 300ms，beep 已有 fade-out，回音短）

    rec = threading.Thread(target=record_audio, daemon=True)
    rec.start()
    _current_rec_thread = rec


def on_released():
    global _state, _last_release

    now = time.monotonic()
    if now - _last_release < DEBOUNCE_S:
        return
    _last_release = now

    with _state_lock:
        current = _state

    if current != "RECORDING":
        return

    log.info("按鈕放開 → 停止錄音，開始辨識")
    with _state_lock:
        _state = "PROCESSING"

    _stop_rec.set()

    rec = _current_rec_thread
    threading.Thread(target=session_thread, args=(rec,), daemon=True).start()


# ─────────────────────────────────────────────────────────────────────────────
# 主程式
# ─────────────────────────────────────────────────────────────────────────────
def main():
    global client, CAPTURE_DEVICE, COMPANIES

    if not GEMINI_API_KEY:
        log.error("請設定環境變數 GEMINI_API_KEY")
        sys.exit(1)

    client = genai.Client(api_key=GEMINI_API_KEY)

    # 從資料庫載入真實公司名單
    _ensure_call_queue_table()
    COMPANIES = _load_companies_from_db()
    if not COMPANIES:
        log.warning("公司資料庫為空，使用預設空名單")

    CAPTURE_DEVICE = _detect_capture_device()
    log.info(f"錄音裝置: {CAPTURE_DEVICE}")
    log.info(f"播放裝置: plughw:{PLAYBACK_CARD}")

    # SIGTERM handler：systemd 重啟前正確釋放 ALSA，防止 zombie PCM handle
    def _sigterm_handler(signum, frame):
        log.info("收到 SIGTERM，正在釋放音訊資源...")
        stop_audio()                    # 停止正在播放的 TTS
        _stop_rec.set()                 # 停止錄音迴圈
        with _active_pcm_lock:
            if _active_pcm is not None:
                try:
                    _active_pcm.close()
                    log.info("ALSA PCM 已關閉")
                except Exception:
                    pass
        sys.exit(0)

    signal.signal(signal.SIGTERM, _sigterm_handler)

    btn = Button(BUTTON_PIN, pull_up=False, bounce_time=DEBOUNCE_S)
    btn.when_pressed  = on_pressed
    btn.when_released = on_released
    log.info(f"GPIO {BUTTON_PIN} 初始化完成（gpiozero, pull-down）")

    _precache_tts()
    speak(PHRASES["ready"])

    log.info("待機中... (Ctrl+C 結束)")
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        log.info("中斷")
    finally:
        stop_audio()
        btn.close()


if __name__ == "__main__":
    main()
