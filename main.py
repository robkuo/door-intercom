#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
門口對講機系統 - 主程式

功能：
1. 觸控選擇 8 間公司撥打 SIP 電話
2. 通話中遠端開門（DTMF）
3. NFC 卡片開門、密碼開門
"""

import tkinter as tk
import sys
import os
import signal
import time as _time
import traceback as _tb
import urllib.request
import json

# 加入模組路徑
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sqlite3

from config import (
    COMPANIES, SIP_SERVER, SIP_PORT, SIP_USERNAME, SIP_PASSWORD, SIP_DOMAIN,
    GPIO_RELAY_PIN, DOOR_UNLOCK_DURATION,
    DATABASE_PATH,
    DTMF_UNLOCK_CODE, DTMF_UNLOCK_CODE_ALT,
    SCREEN_WIDTH, SCREEN_HEIGHT, FULLSCREEN,
    LOG_FILE, LOG_LEVEL,
    NFC_ENABLED, NFC_DATABASE_PATH, NFC_SCAN_INTERVAL,
    WEB_ADMIN_DB_PATH
)

# 公司資料更新間隔（秒）
COMPANY_UPDATE_INTERVAL = 5000  # 5 秒
# 通話畫面看門狗間隔（ms）：若 call_window 顯示但 SIP 已斷，強制返回主畫面
CALL_WATCHDOG_INTERVAL = 10_000  # 10 秒
# 網頁開門請求輪詢間隔（ms）
DOOR_REQUEST_POLL_INTERVAL = 500
VOICE_CALL_QUEUE_FRESH_S = 15           # 只處理 N 秒內的新請求（語音門禁）
VOICE_CALL_QUEUE_COOLDOWN_S = 10.0      # 通話結束後冷卻（防止結束即回撥）
VOICE_CALL_MIN_INTERVAL_S = 60.0        # 上次撥號開始後至少間隔 N 秒才能再撥（防假陽性斷線後快速重撥）
VOICE_CALL_QUEUE_PROCESSING_TTL_S = 600 # processing TTL（秒）：超時視為卡死可被清理


def load_companies_from_db():
    """從資料庫載入公司資料"""
    try:
        if not os.path.exists(WEB_ADMIN_DB_PATH):
            return COMPANIES  # 資料庫不存在，使用預設值

        conn = sqlite3.connect(WEB_ADMIN_DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('SELECT id, name, extension, floor FROM companies ORDER BY id')
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return COMPANIES  # 沒有資料，使用預設值

        companies = {}
        for row in rows:
            companies[row['id']] = {
                'name': row['name'],
                'extension': row['extension'],
                'floor': row['floor'] or ''
            }
        return companies
    except Exception as e:
        print(f"載入公司資料失敗: {e}")
        return COMPANIES  # 發生錯誤，使用預設值
from utils.logger import setup_logger, get_logger
from door.lock_control import DoorLock
from nfc.nfc_manager import NFCManager, NFCResult
from sip.sip_client import SIPClient, CallState
from gui.main_window import MainWindow
from gui.call_window import CallWindow
from gui.password_window import PasswordWindow
# 管理介面已移至網頁版，不再使用觸控螢幕管理

# 密碼驗證 API URL
PASSWORD_VERIFY_URL = "http://localhost:5000/api/passwords/verify"


class IntercomSystem:
    """
    門口對講機系統主類別

    整合所有模組，控制整體流程
    """

    def __init__(self):
        """初始化系統"""
        # 初始化日誌
        self.logger = setup_logger(LOG_FILE, LOG_LEVEL)
        self.logger.info("系統啟動中...")

        # 建立 Tkinter 根視窗
        self.root = tk.Tk()
        self.root.title("門口對講機")
        self.root.geometry(f"{SCREEN_WIDTH}x{SCREEN_HEIGHT}")
        self.root.configure(bg='#1a1a2e')

        if FULLSCREEN:
            self.root.attributes('-fullscreen', True)

        # 綁定 ESC 鍵退出全螢幕
        self.root.bind('<Escape>', self._toggle_fullscreen)

        self._call_ended_count = 0
        self._last_call_end_time: float = 0.0   # 上次通話結束時間，用於防止短時間重複撥號
        self._last_call_start_time: float = 0.0  # 上次撥號開始時間，用於防止假陽性後快速重撥
        self._last_called: dict = {}             # {extension: timestamp}，防止同一分機短時間重複撥號
        self._current_outbound_extension: str = ""  # 目前撥出的分機（用於 per-ext cooldown）
        self._queue_consumer_id = f"main:{os.getpid()}"  # 唯一 consumer ID（transactional claim 用）
        # Voice call queue active tracking（讓 call_ended 可 finalize 對應 DB row）
        self._active_vcq_id: int = None
        self._active_vcq_extension: str = ""
        self._active_vcq_connected: bool = False

        # 初始化硬體模組
        self._init_hardware()

        # 初始化 GUI
        self._init_gui()

        # 設定回調
        self._setup_callbacks()

        # 啟動 NFC 連續掃描
        if self.nfc_manager:
            self.nfc_manager.start_continuous_scan(self._on_nfc_scan)

        self.logger.info("系統初始化完成")

    def _init_hardware(self):
        """初始化硬體模組"""
        self.logger.info("初始化硬體模組...")

        # 門鎖控制
        self.door_lock = DoorLock(
            relay_pin=GPIO_RELAY_PIN,
            unlock_duration=DOOR_UNLOCK_DURATION
        )

        # NFC 讀卡器
        self.nfc_manager = None
        if NFC_ENABLED:
            self.nfc_manager = NFCManager(
                database_path=NFC_DATABASE_PATH,
                scan_interval=NFC_SCAN_INTERVAL
            )

        # SIP 客戶端
        self.sip_client = SIPClient(
            server=SIP_SERVER,
            port=SIP_PORT,
            username=SIP_USERNAME,
            password=SIP_PASSWORD,
            domain=SIP_DOMAIN
        )

        # 註冊 SIP
        self.sip_client.register()

    def _init_gui(self):
        """初始化 GUI"""
        self.logger.info("初始化 GUI...")

        # 載入公司資料（優先從資料庫，否則使用 config.py）
        self.companies = load_companies_from_db()

        # 主畫面（管理介面已移至網頁版）
        self.main_window = MainWindow(
            self.root,
            self.companies,
            on_company_selected=self._on_company_selected,
            on_password_click=self._on_password_click
        )

        # 通話畫面
        self.call_window = CallWindow(
            self.root,
            on_hangup=self._on_hangup,
            on_answer=self._on_incoming_answer
        )

        # 密碼開門畫面
        self.password_window = PasswordWindow(
            self.root,
            on_password_submit=self._on_password_submit,
            on_cancel=self._on_password_cancel
        )

        # 預設顯示主畫面
        self.main_window.show()

        # 啟動公司資料定時更新（每 5 秒檢查一次）
        self._start_company_update_timer()
        self._start_sip_status_poll()
        # 啟動通話畫面看門狗（每 10 秒同步 UI 與 SIP 狀態）
        self._start_call_watchdog()

        # 啟動網頁開門請求輪詢（NFC 網頁開門）
        self._poll_web_door_requests()

        # 啟動語音門禁撥號佇列輪詢
        self._poll_voice_call_queue()

    def _start_call_watchdog(self):
        """啟動通話畫面看門狗"""
        self.root.after(CALL_WATCHDOG_INTERVAL, self._call_watchdog)

    def _call_watchdog(self):
        """
        定期檢查 call_window 顯示狀態是否與 SIP 狀態一致。
        若 call_window 還在但 SIP 已是 idle/disconnected，強制返回主畫面。
        這能自動修復以下情況：
          - AMI 斷線後監控執行緒卡死，_on_call_ended 沒有被觸發
          - 放置一晚後通話無聲但畫面仍在
        """
        try:
            if self.call_window.is_visible():
                sip_state = self.sip_client.call_state
                from sip.sip_client import CallState
                if sip_state in (CallState.IDLE, CallState.DISCONNECTED):
                    self.logger.warning(
                        f"看門狗：call_window 顯示中但 SIP={sip_state.value}，強制返回主畫面"
                    )
                    # ⚠️ 不可直接呼叫 _on_call_ended()：會重置 _last_call_end_time，
                    # 導致 cooldown 歸零，queue 立刻被撿到觸發幽靈回撥。
                    # 只需把畫面切回主畫面即可（_on_call_ended 已由 sip_client 觸發過）
                    self._show_main()
        except Exception as e:
            self.logger.error(f"call_watchdog 錯誤: {e}")
        self.root.after(CALL_WATCHDOG_INTERVAL, self._call_watchdog)

    def _poll_web_door_requests(self):
        """輪詢 Web 開門請求（NFC 網頁開門 IPC）"""
        try:
            conn = sqlite3.connect(WEB_ADMIN_DB_PATH)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, requester_name FROM door_open_requests
                WHERE status = 'pending'
                  AND (strftime('%s','now') - strftime('%s',created_at)) <= 10
                ORDER BY created_at ASC LIMIT 1
            """)
            row = cursor.fetchone()
            if row:
                req_id, name = row
                cursor.execute(
                    "UPDATE door_open_requests SET status='done' WHERE id=?", (req_id,)
                )
                conn.commit()
                conn.close()
                self.logger.info(f"[Web Door] 開門請求來自 {name} (id={req_id})")
                self._unlock_door()
                self.main_window.show_message(f"歡迎 {name}！(Web)", "success")
            else:
                conn.close()
        except Exception as e:
            self.logger.error(f"[Web Door] 輪詢失敗: {e}")
        finally:
            self.root.after(DOOR_REQUEST_POLL_INTERVAL, self._poll_web_door_requests)

    def _poll_voice_call_queue(self):
        """輪詢語音門禁撥號佇列（voice_gate.py 辨識後觸發 SIP 撥號）"""
        # SIP 非空閒時不消費 queue（避免通話中把 pending 直接丟棄）
        if self.sip_client._call_state not in (CallState.IDLE, CallState.DISCONNECTED):
            self.root.after(DOOR_REQUEST_POLL_INTERVAL, self._poll_voice_call_queue)
            return
        # 通話結束後冷卻（防止結束後立即回撥）
        if _time.time() - self._last_call_end_time < VOICE_CALL_QUEUE_COOLDOWN_S:
            self.root.after(DOOR_REQUEST_POLL_INTERVAL, self._poll_voice_call_queue)
            return
        # 撥號開始後的最小間隔（防止假陽性斷線後在冷卻內重新撥號）
        since_start = _time.time() - self._last_call_start_time
        if since_start < VOICE_CALL_MIN_INTERVAL_S:
            self.logger.debug(f"[VoiceGate] 距上次撥號開始 {since_start:.1f}s < {VOICE_CALL_MIN_INTERVAL_S}s，略過")
            self.root.after(DOOR_REQUEST_POLL_INTERVAL, self._poll_voice_call_queue)
            return
        try:
            # isolation_level=None → autocommit 模式；所有事務用明確 BEGIN/COMMIT 控制
            # 這樣清理 UPDATE 和 BEGIN IMMEDIATE claim 不會互相衝突
            conn = sqlite3.connect(WEB_ADMIN_DB_PATH, timeout=5, isolation_level=None)
            cursor = conn.cursor()
            # 向下相容：補欄位（已存在則略過）
            for ddl in (
                "ALTER TABLE voice_call_queue ADD COLUMN claimed_at DATETIME",
                "ALTER TABLE voice_call_queue ADD COLUMN claimed_by TEXT",
                "ALTER TABLE voice_call_queue ADD COLUMN result TEXT",
            ):
                try:
                    cursor.execute(ddl)
                except sqlite3.OperationalError:
                    pass
            # 清理過期 pending（超過 freshness 視窗的舊請求）—autocommit，不開隱式事務
            cursor.execute(
                """
                UPDATE voice_call_queue
                SET status='failed', processed_at=datetime('now','localtime'), result='expired'
                WHERE status='pending'
                  AND (strftime('%s','now') - strftime('%s', created_at, 'utc')) > ?
                """,
                (VOICE_CALL_QUEUE_FRESH_S,),
            )
            # 清理卡死 processing（main 崩潰後殘留的 processing 超過 TTL）—autocommit
            cursor.execute(
                """
                UPDATE voice_call_queue
                SET status='failed', processed_at=datetime('now','localtime'), result='stale_processing'
                WHERE status='processing'
                  AND claimed_at IS NOT NULL AND claimed_at != ''
                  AND (strftime('%s','now') - strftime('%s', claimed_at, 'utc')) > ?
                """,
                (VOICE_CALL_QUEUE_PROCESSING_TTL_S,),
            )
            # 明確開啟 IMMEDIATE 事務做原子性 claim（isolation_level=None 下必須明確 BEGIN）
            cursor.execute("BEGIN IMMEDIATE")
            # created_at 是 localtime 字串，用 'utc' modifier 轉回 UTC 再與 now(UTC) 算秒差
            cursor.execute(
                """
                SELECT id, company_id, company_name, extension, created_at
                FROM voice_call_queue
                WHERE status = 'pending'
                  AND (claimed_at IS NULL OR claimed_at = '')
                  AND (strftime('%s','now') - strftime('%s', created_at, 'utc')) BETWEEN 0 AND ?
                ORDER BY created_at ASC
                LIMIT 1
                """,
                (VOICE_CALL_QUEUE_FRESH_S,),
            )
            row = cursor.fetchone()
            if not row:
                conn.commit()
                conn.close()
                self.root.after(DOOR_REQUEST_POLL_INTERVAL, self._poll_voice_call_queue)
                return
            req_id, company_id, name, extension, created_at = row
            extension = str(extension).strip()
            if not extension:
                cursor.execute(
                    "UPDATE voice_call_queue SET status='failed', processed_at=datetime('now','localtime'), result='empty_extension' WHERE id=?",
                    (req_id,),
                )
                conn.commit()
                conn.close()
                self.root.after(DOOR_REQUEST_POLL_INTERVAL, self._poll_voice_call_queue)
                return
            # per-extension cooldown：同分機在冷卻期（或最小間隔）內不重撥
            last_ext_end = self._last_called.get(extension, 0.0)
            now = _time.time()
            cooldown_ok = (now - last_ext_end) >= VOICE_CALL_QUEUE_COOLDOWN_S
            interval_ok = (now - self._last_call_start_time) >= VOICE_CALL_MIN_INTERVAL_S
            if not (cooldown_ok and interval_ok):
                conn.commit()
                conn.close()
                self.logger.info(
                    f"[VoiceGate] ext={extension} 冷卻中 "
                    f"(cooldown={now - last_ext_end:.1f}s/{VOICE_CALL_QUEUE_COOLDOWN_S}s, "
                    f"interval={now - self._last_call_start_time:.1f}s/{VOICE_CALL_MIN_INTERVAL_S}s)"
                )
                self.root.after(DOOR_REQUEST_POLL_INTERVAL, self._poll_voice_call_queue)
                return
            # Transactional claim：pending → processing（確保同一筆只被處理一次）
            cursor.execute(
                """
                UPDATE voice_call_queue
                SET status='processing',
                    claimed_at=datetime('now','localtime'),
                    claimed_by=?
                WHERE id=?
                  AND status='pending'
                  AND (claimed_at IS NULL OR claimed_at = '')
                """,
                (self._queue_consumer_id, req_id),
            )
            if cursor.rowcount != 1:
                # 已被其他 consumer claim（競態）
                conn.rollback()
                conn.close()
                self.root.after(DOOR_REQUEST_POLL_INTERVAL, self._poll_voice_call_queue)
                return
            conn.commit()
            conn.close()
            # 撥號
            self.logger.info(f"[VoiceGate] claim id={req_id} → 撥號 {name} (分機 {extension}) created_at={created_at}")
            self._call_ended_count = 0  # 每次新撥號前重置計數（包含撥號失敗的情況）
            self._current_outbound_extension = extension
            self._last_call_start_time = _time.time()  # 記錄撥號開始時間（防假陽性後快速重撥）
            # 記錄 active queue row（由 _on_call_ended finalize，保持 processing 做 busy 訊號）
            self._active_vcq_id = req_id
            self._active_vcq_extension = extension
            self._active_vcq_connected = False
            self._show_call(name)
            success = self.sip_client.call(extension)
            if not success:
                # 撥號沒啟動：立刻標記 failed，並清除 active 記錄
                conn2 = sqlite3.connect(WEB_ADMIN_DB_PATH, timeout=5)
                conn2.execute(
                    "UPDATE voice_call_queue SET status='failed', processed_at=datetime('now','localtime'), result='dial_failed' WHERE id=?",
                    (req_id,),
                )
                conn2.commit()
                conn2.close()
                self._active_vcq_id = None
                self._active_vcq_extension = ""
                self._active_vcq_connected = False
                self.logger.warning(f"[VoiceGate] 撥號失敗，{name}（分機 {extension}）不在線")
                self.call_window.set_status("unavailable")
                self.root.after(2500, self._show_main)
            # 撥號成功時：不改 status，保持 processing 直到 _on_call_ended 做 finalize
            # 這樣 voice_gate._trigger_call() 的 processing busy 訊號在整段通話期間都有效
        except Exception as e:
            self.logger.error(f"[VoiceGate] 輪詢失敗: {e}")
        finally:
            self.root.after(DOOR_REQUEST_POLL_INTERVAL, self._poll_voice_call_queue)

    def _setup_callbacks(self):
        """設定回調函數"""
        # SIP 通話狀態
        self.sip_client.set_on_state_changed(self._on_call_state_changed)
        self.sip_client.set_on_dtmf_received(self._on_dtmf_received)
        self.sip_client.set_on_call_connected(self._on_call_connected)
        self.sip_client.set_on_call_ended(self._on_call_ended)
        self.sip_client.set_on_incoming_call(self._on_incoming_call_detected)

        # 門鎖狀態
        self.door_lock.set_on_unlock(lambda: self.logger.info("門已開啟"))
        self.door_lock.set_on_lock(lambda: self.logger.info("門已上鎖"))

    # =========================================================================
    # GUI 切換
    # =========================================================================
    def _show_main(self):
        """顯示主畫面"""
        self.call_window.hide()
        self.password_window.hide()
        self.main_window.show()
        # 重設 SIP 狀態到 IDLE，確保下次來電偵測正常運作（持鎖，thread-safe）
        self.sip_client.reset_to_idle()
        # 恢復 NFC 掃描
        if self.nfc_manager:
            self.nfc_manager.start_continuous_scan(self._on_nfc_scan)

    def _show_call(self, company_name: str):
        """顯示通話畫面"""
        self.main_window.hide()
        self.call_window.show(company_name)

    # =========================================================================
    # 事件處理
    # =========================================================================
    def _on_company_selected(self, company_id: int, company_info: dict):
        """選擇公司撥號"""
        # 通話中忽略（CONNECTED 狀態）
        if self.sip_client._call_state not in (CallState.IDLE, CallState.DISCONNECTED):
            self.logger.info(f"忽略按鈕（通話中）: {company_info['name']}")
            return
        self.logger.info(f"撥打: {company_info['name']} (分機 {company_info['extension']})")

        # 切換到通話畫面
        self._show_call(company_info['name'])

        # 撥打 SIP 電話
        success = self.sip_client.call(company_info['extension'])
        if not success:
            self.logger.warning(f"撥號失敗，{company_info['name']} 不在線")
            self.call_window.set_status("unavailable")
            self.root.after(2500, self._show_main)

    def _on_hangup(self):
        """掛斷電話"""
        self.logger.info("掛斷通話")
        self.sip_client.hangup()
        self.call_window.hide()
        self._show_main()

    def _on_incoming_call_detected(self, caller_id: str, channel: str):
        """偵測到來電（AMI 事件觸發，需 thread-safe 切回主執行緒）"""
        self.logger.info(f"來電: {caller_id} (channel={channel})")
        # 回到 Tkinter 主執行緒更新 GUI
        self.root.after(0, self._handle_incoming_call, caller_id)

    def _handle_incoming_call(self, caller_id: str):
        """在主執行緒顯示來電畫面"""
        if self.sip_client._call_state not in (CallState.IDLE, CallState.DISCONNECTED):
            self.logger.info(f"通話中，忽略來電: {caller_id}")
            return
        self.logger.info(f"顯示來電畫面: {caller_id}")
        self._show_call(caller_id or "來電")
        # 來電模式：顯示接聽按鈕
        self.call_window.show_incoming(caller_id or "來電")

    def _on_incoming_answer(self):
        """接聽來電按鈕點擊"""
        self.logger.info("接聽來電")
        self.sip_client.answer_incoming_call()

    def _on_call_state_changed(self, state: CallState):
        """通話狀態變更"""
        state_map = {
            CallState.DIALING: 'dialing',
            CallState.RINGING: 'ringing',
            CallState.CONNECTED: 'connected',
            CallState.DISCONNECTED: 'disconnected',
        }
        gui_state = state_map.get(state, 'dialing')
        self.call_window.set_status(gui_state)

    def _on_call_connected(self):
        """通話連接"""
        self._call_ended_count = 0  # 每次新通話重設計數
        self.logger.info("通話已連接")
        if self._active_vcq_id:
            self._active_vcq_connected = True

    def _on_call_ended(self):
        """通話結束"""
        now = _time.time()
        # Idempotent guard：2 秒內重複呼叫直接忽略（防止 watchdog/多路徑重入）
        if self._call_ended_count > 0 and (now - self._last_call_end_time) < 2.0:
            self.logger.warning(
                f"_on_call_ended 重複呼叫已忽略（距上次 {now - self._last_call_end_time:.3f}s）"
            )
            return
        self._last_call_end_time = now  # 記錄結束時間，防止短時間重複撥號
        # per-extension cooldown：同分機也加記錄
        if self._current_outbound_extension:
            self._last_called[self._current_outbound_extension] = self._last_call_end_time
            self._current_outbound_extension = ""
        # 診斷：偵測重複呼叫
        self._call_ended_count = getattr(self, '_call_ended_count', 0) + 1
        caller = _tb.extract_stack()[-2]
        self.logger.info(
            f"通話已結束 (第 {self._call_ended_count} 次) ← "
            f"{caller.filename.split('/')[-1]}:{caller.lineno} {caller.name}()"
        )
        if self._call_ended_count > 1:
            self.logger.warning(f"[重複] _on_call_ended 被呼叫第 {self._call_ended_count} 次！")
        # Finalize voice_call_queue row（只針對 voice_gate 觸發的通話）
        if self._active_vcq_id:
            vcq_id = self._active_vcq_id
            vcq_ext = self._active_vcq_extension
            ended_result = "call_ended_connected" if self._active_vcq_connected else "call_ended"
            self._active_vcq_id = None
            self._active_vcq_extension = ""
            self._active_vcq_connected = False
            try:
                conn_vcq = sqlite3.connect(WEB_ADMIN_DB_PATH, timeout=5)
                conn_vcq.execute(
                    "UPDATE voice_call_queue SET status='done', processed_at=datetime('now','localtime'), result=? "
                    "WHERE id=? AND status='processing'",
                    (ended_result, vcq_id),
                )
                # 額外保險：清除同分機殘留的 pending，防止掛斷後立刻被撥出第二通
                if vcq_ext:
                    conn_vcq.execute(
                        "UPDATE voice_call_queue SET status='failed', processed_at=datetime('now','localtime'), result='suppressed_after_call' "
                        "WHERE status='pending' AND extension=?",
                        (vcq_ext,),
                    )
                conn_vcq.commit()
                conn_vcq.close()
                self.logger.info(f"[VoiceGate] finalize id={vcq_id} result={ended_result}，同分機殘留 pending 已清除")
            except Exception as e:
                self.logger.error(f"[VoiceGate] finalize voice_call_queue 失敗: {e}")
        # 延遲返回主畫面
        self.root.after(1500, self._show_main)

    def _on_dtmf_received(self, digit: str):
        """收到 DTMF 按鍵"""
        self.logger.info(f"收到 DTMF: {digit}")

        # 檢查是否為開門指令
        if digit == DTMF_UNLOCK_CODE or digit == DTMF_UNLOCK_CODE_ALT:
            self.logger.info("收到遠端開門指令")
            self._unlock_door()
            self.call_window.show_door_opened()

    def _unlock_door(self):
        """開門"""
        self.logger.info("執行開門")
        self.door_lock.unlock()

    # =========================================================================
    # 密碼開門功能
    # =========================================================================
    def _on_password_click(self):
        """密碼開門按鈕點擊"""
        self.logger.info("進入密碼開門畫面")
        # 暫停 NFC 掃描
        if self.nfc_manager:
            self.nfc_manager.stop_continuous_scan()
        # 切換畫面
        self.main_window.hide()
        self.password_window.show()

    def _on_password_submit(self, password: str):
        """密碼提交處理"""
        self.logger.info(f"驗證密碼...")
        try:
            # 呼叫 API 驗證密碼
            data = json.dumps({'password': password}).encode('utf-8')
            req = urllib.request.Request(
                PASSWORD_VERIFY_URL,
                data=data,
                headers={'Content-Type': 'application/json'},
                method='POST'
            )
            with urllib.request.urlopen(req, timeout=5) as response:
                result = json.loads(response.read().decode('utf-8'))

            if result.get('success') and result.get('valid'):
                name = result.get('name', '使用者')
                self.logger.info(f"密碼驗證成功: {name}")
                self.password_window.show_success(name)
                self._unlock_door()
                # 2 秒後返回主畫面
                self.root.after(2000, self._show_main)
            elif result.get('locked'):
                # 帳號已鎖定
                error_msg = result.get('error', '請稍後再試')
                self.logger.warning(f"密碼嘗試鎖定: {error_msg}")
                self.password_window.show_error(error_msg)
            else:
                error_msg = result.get('error', '密碼錯誤')
                self.logger.warning(f"密碼驗證失敗: {error_msg}")
                self.password_window.show_error(error_msg)
        except Exception as e:
            self.logger.error(f"密碼驗證發生錯誤: {e}")
            self.password_window.show_error("系統錯誤")

    def _on_password_cancel(self):
        """密碼取消處理"""
        self.logger.info("取消密碼開門")
        self._show_main()

    # =========================================================================
    # NFC 功能
    # =========================================================================
    def _start_sip_status_poll(self):
        """啟動 SIP 分機狀態輪詢（每 30 秒檢查一次）"""
        self._poll_sip_status()

    def _poll_sip_status(self):
        """檢查各公司分機是否已在 Asterisk 登錄，並更新主畫面狀態列"""
        try:
            if self.sip_client._call_state not in (CallState.IDLE, CallState.DISCONNECTED):
                self.root.after(30_000, self._poll_sip_status)
                return
            offline = []
            seen_ext = set()
            for company in self.companies.values():
                ext = str(company.get('extension', '')).strip()
                if not ext or ext in seen_ext:
                    continue
                seen_ext.add(ext)
                if not self.sip_client.check_extension_registered(ext):
                    offline.append(company.get('name', ext) + '(' + ext + ')')
            self.main_window.set_sip_offline_hint(offline)
        except Exception as e:
            self.logger.warning(f'SIP 狀態輪詢失敗: {e}')
        self.root.after(30_000, self._poll_sip_status)

    def _on_nfc_scan(self, result: NFCResult, card):
        """NFC 掃描回調"""
        if result == NFCResult.SUCCESS and card:
            self.logger.info(f"NFC 卡片識別成功: {card.name}")
            self._unlock_door()
            self.main_window.show_message(f"歡迎 {card.name}！(NFC)", "success")

        elif result == NFCResult.UNKNOWN:
            self.logger.warning("未授權的 NFC 卡片")
            # 可選：顯示錯誤訊息
            # self.main_window.show_message("未授權的卡片", "error")

        elif result == NFCResult.DISABLED:
            self.logger.warning(f"卡片已停用: {card.name if card else 'unknown'}")
            self.main_window.show_message("此卡片已停用", "error")

    # =========================================================================
    # 公司資料動態更新
    # =========================================================================
    def _start_company_update_timer(self):
        """啟動公司資料定時更新"""
        self._check_company_update()

    def _check_company_update(self):
        """檢查並更新公司資料"""
        try:
            new_companies = load_companies_from_db()
            if new_companies != self.companies:
                self.logger.info("偵測到公司資料變更，更新 GUI")
                self.companies = new_companies
                self.main_window.update_companies(new_companies)
        except Exception as e:
            self.logger.error(f"更新公司資料失敗: {e}")

        # 安排下次檢查
        self.root.after(COMPANY_UPDATE_INTERVAL, self._check_company_update)

    # =========================================================================
    # 系統控制
    # =========================================================================
    def _toggle_fullscreen(self, event=None):
        """切換全螢幕"""
        current = self.root.attributes('-fullscreen')
        self.root.attributes('-fullscreen', not current)

    def run(self):
        """啟動主迴圈"""
        self.logger.info("系統已啟動")
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.shutdown()

    def shutdown(self):
        """關閉系統"""
        self.logger.info("系統關閉中...")

        # 停止 NFC 掃描
        if self.nfc_manager:
            self.nfc_manager.stop_continuous_scan()

        # 掛斷通話
        if self.sip_client._call_state not in (CallState.IDLE, CallState.DISCONNECTED):
            self.sip_client.hangup()

        # 清理資源
        if self.nfc_manager:
            self.nfc_manager.cleanup()
        self.sip_client.cleanup()
        self.door_lock.cleanup()

        self.logger.info("系統已關閉")

        # 關閉視窗
        self.root.quit()


def main():
    """主函數"""
    # 建立系統實例
    system = IntercomSystem()

    # 設定信號處理
    def signal_handler(sig, frame):
        print("\n收到終止信號，關閉系統...")
        system.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 啟動系統
    system.run()


if __name__ == "__main__":
    main()
