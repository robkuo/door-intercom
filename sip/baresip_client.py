# -*- coding: utf-8 -*-
"""
BaresipClient - 透過 baresip ctrl_tcp 控制 SIP 通話（含 H.264 視訊）
Drop-in replacement for SIPClient (AMI-based).
baresip 負責：攝影機串流、音訊、SIP 撥號/掛斷/DTMF 偵測。
AMI 仍用於：分機在線檢查（check_extension_registered）。
"""

import socket
import threading
import time
import json
from typing import Callable, Optional
from enum import Enum
from dataclasses import dataclass

import sys
sys.path.append('..')
from utils.logger import get_logger


class CallState(Enum):
    """通話狀態"""
    IDLE = "idle"
    DIALING = "dialing"
    RINGING = "ringing"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"


@dataclass
class CallInfo:
    """通話資訊"""
    state: CallState
    remote_uri: str = ""
    duration: float = 0.0
    dtmf_digits: str = ""


class BaresipClient:
    """
    使用 baresip SIP UA 發起帶有 H.264 視訊的通話。
    baresip 需要預先啟動並在 TCP 4444 提供 ctrl_tcp 介面。
    對外介面與 SIPClient 相同（drop-in replacement）。
    """

    def __init__(
        self,
        server: str = "127.0.0.1",
        port: int = 5060,
        username: str = "100",
        password: str = "password100",
        domain: str = None,
        ami_port: int = 5038,
        ami_username: str = "intercom",
        ami_password: str = "intercom123",
        baresip_host: str = "127.0.0.1",
        baresip_port: int = 4444,
    ):
        self.logger = get_logger()
        self.server = server
        self.username = username
        self.password = password
        self.domain = domain or server
        self.ami_port = ami_port
        self.ami_username = ami_username
        self.ami_password = ami_password
        self.baresip_host = baresip_host
        self.baresip_port = baresip_port

        # State
        self._call_state = CallState.IDLE
        self._state_lock = threading.RLock()
        self._call_start_time: Optional[float] = None
        self._current_call_id: Optional[str] = None
        self._current_extension: Optional[str] = None
        self._ami_connected = True  # compat

        # baresip TCP socket
        self._bs_sock: Optional[socket.socket] = None
        self._bs_lock = threading.Lock()
        self._bs_event_thread: Optional[threading.Thread] = None
        self._bs_stop = threading.Event()

        # AMI socket (for extension registration check only)
        self._ami_socket: Optional[socket.socket] = None

        # Callbacks (same names as SIPClient)
        self._on_state_changed: Optional[Callable] = None
        self._on_dtmf_received: Optional[Callable] = None
        self._on_call_connected: Optional[Callable] = None
        self._on_call_ended: Optional[Callable] = None
        self._on_door_open: Optional[Callable] = None
        self._on_incoming_call: Optional[Callable] = None

        self.logger.info(f"BaresipClient 初始化：baresip={baresip_host}:{baresip_port}")

    # ── Setter methods (same as SIPClient) ────────────────────────────────
    def set_on_state_changed(self, cb: Callable): self._on_state_changed = cb
    def set_on_dtmf_received(self, cb: Callable): self._on_dtmf_received = cb
    def set_on_call_connected(self, cb: Callable): self._on_call_connected = cb
    def set_on_call_ended(self, cb: Callable): self._on_call_ended = cb
    def set_on_door_open(self, cb: Callable): self._on_door_open = cb
    def set_on_incoming_call(self, cb: Callable): self._on_incoming_call = cb

    # ── Connect to baresip ─────────────────────────────────────────────────
    def register(self) -> bool:
        """連接到 baresip ctrl_tcp（compat with SIPClient.register()）"""
        return self._baresip_connect(max_attempts=15)

    def _baresip_connect(self, max_attempts: int = 5) -> bool:
        for attempt in range(max_attempts):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(3)
                sock.connect((self.baresip_host, self.baresip_port))
                with self._bs_lock:
                    self._bs_sock = sock
                self._bs_stop.clear()
                self._bs_event_thread = threading.Thread(
                    target=self._event_loop, daemon=True, name='baresip-events'
                )
                self._bs_event_thread.start()
                self.logger.info(f"已連接 baresip ctrl_tcp {self.baresip_host}:{self.baresip_port}")
                return True
            except Exception as e:
                self.logger.warning(f"baresip 連線嘗試 {attempt+1}/{max_attempts}: {e}")
                time.sleep(2)
        self.logger.error("無法連接 baresip ctrl_tcp，請確認 baresip.service 已啟動")
        return False

    def _baresip_send(self, cmd: str):
        """發送命令到 baresip"""
        try:
            with self._bs_lock:
                if self._bs_sock:
                    self._bs_sock.sendall((cmd + '\n').encode())
                    self.logger.debug(f"baresip ← {cmd}")
        except Exception as e:
            self.logger.error(f"baresip send 失敗: {e}")

    # ── Call control ────────────────────────────────────────────────────────
    def call(self, extension: str) -> bool:
        """透過 baresip 撥打分機（含 H.264 視訊）"""
        with self._state_lock:
            if self._call_state not in (CallState.IDLE, CallState.DISCONNECTED):
                self.logger.warning(f"已有通話進行中 ({self._call_state.value})，拒絕重複撥號")
                return False
            self._call_state = CallState.DIALING

        self._current_extension = extension
        uri = f"sip:{extension}@{self.server}"
        self.logger.info(f"baresip 撥號: {uri}")

        if self._on_state_changed:
            self._on_state_changed(CallState.DIALING)

        self._baresip_send(f"/dial {uri}")
        return True

    def hangup(self) -> bool:
        """掛斷通話"""
        self._baresip_send("/hangup")
        if self._call_state != CallState.DISCONNECTED:
            self._update_state(CallState.DISCONNECTED)
        return True

    def answer_incoming_call(self) -> bool:
        """接聽來電"""
        self._baresip_send("/answer")
        return True

    def send_dtmf(self, digits: str) -> bool:
        """發送 DTMF（由手機端處理）"""
        self.logger.info(f"DTMF {digits} 由手機端處理")
        return True

    def reset_to_idle(self):
        """重置狀態到 IDLE"""
        with self._state_lock:
            self._call_state = CallState.IDLE
            self._current_call_id = None
            self._call_start_time = None
        self.logger.info("狀態已重置到 IDLE")

    # ── State management ───────────────────────────────────────────────────
    def _update_state(self, state: CallState):
        with self._state_lock:
            old = self._call_state
            if old == state:
                return
            self._call_state = state

        self.logger.info(f"通話狀態變更: {old.value} -> {state.value}")

        if state == CallState.CONNECTED:
            self._call_start_time = time.time()
            if self._on_call_connected:
                self._on_call_connected()

        elif state == CallState.DISCONNECTED:
            self._call_start_time = None
            self._current_call_id = None
            if self._on_call_ended:
                self._on_call_ended()

        if self._on_state_changed:
            self._on_state_changed(state)

    # ── baresip event loop ─────────────────────────────────────────────────
    def _event_loop(self):
        """持續讀取 baresip ctrl_tcp 事件"""
        buf = ""
        try:
            self._bs_sock.settimeout(1.0)
        except Exception:
            return

        while not self._bs_stop.is_set():
            try:
                data = self._bs_sock.recv(4096)
                if not data:
                    self.logger.warning("baresip socket 已關閉")
                    break
                buf += data.decode('utf-8', errors='replace')

                while '\n' in buf:
                    line, buf = buf.split('\n', 1)
                    line = line.strip()
                    if line:
                        self._handle_event(line)

            except socket.timeout:
                continue
            except Exception as e:
                if not self._bs_stop.is_set():
                    self.logger.debug(f"baresip event loop: {e}")
                break

        # 自動重連
        if not self._bs_stop.is_set():
            self.logger.info("baresip: 嘗試重連...")
            time.sleep(3)
            self._baresip_connect(max_attempts=5)

    def _handle_event(self, line: str):
        """解析並處理 baresip 事件"""
        self.logger.debug(f"baresip → {line[:200]}")

        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return  # 非 JSON 訊息（如命令確認）

        if not event.get('event'):
            return

        etype = event.get('type', '')

        if etype == 'REGISTER_OK':
            self.logger.info(f"baresip 已向 Asterisk 註冊: {event.get('accountaor','')}")

        elif etype == 'REGISTER_FAIL':
            self.logger.error(f"baresip 註冊失敗: {event.get('reason','')}")

        elif etype == 'CALL_RINGING':
            self._current_call_id = event.get('id')
            self._update_state(CallState.RINGING)

        elif etype in ('CALL_ESTABLISHED', 'CALL_ANSWERED'):
            self._current_call_id = event.get('id')
            self._update_state(CallState.CONNECTED)

        elif etype == 'CALL_CLOSED':
            reason = event.get('reason', '')
            self.logger.info(f"通話結束: {reason}")
            if self._call_state != CallState.IDLE:
                self._update_state(CallState.DISCONNECTED)

        elif etype == 'CALL_INCOMING':
            # 來電通知（有人打給 intercom 100）
            peer = event.get('peeruri', '')
            call_id = event.get('id', '')
            self.logger.info(f"來電: {peer}")
            if self._on_incoming_call:
                self._on_incoming_call(peer, call_id)

        elif etype == 'CALL_DTMF_END':
            key = event.get('key', '')
            self.logger.info(f"DTMF: {key}")
            if self._on_dtmf_received:
                self._on_dtmf_received(key)
            # 開門（#或9）
            if key in ('#', '9'):
                if self._on_door_open:
                    self._on_door_open()

    # ── Extension registration check (via AMI) ─────────────────────────────
    def check_extension_registered(self, extension: str) -> bool:
        """透過 AMI 檢查分機是否在線"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3)
            sock.connect(("127.0.0.1", self.ami_port))
            sock.recv(1024)
            sock.send(
                f"Action: Login\r\nUsername: {self.ami_username}\r\n"
                f"Secret: {self.ami_password}\r\n\r\n".encode()
            )
            time.sleep(0.3)
            sock.recv(4096)
            sock.send(b"Action: Command\r\nCommand: pjsip show contacts\r\n\r\n")
            time.sleep(0.5)
            resp = sock.recv(8192).decode('utf-8', errors='replace')
            sock.close()
            return f"/{extension}@" in resp or f" {extension} " in resp
        except Exception as e:
            self.logger.debug(f"check_extension_registered: {e}")
            return False  # 無法確認時假設在線（讓 Originate 自然失敗）

    # ── Properties ─────────────────────────────────────────────────────────
    @property
    def call_state(self) -> CallState:
        return self._call_state

    @property
    def is_in_call(self) -> bool:
        return self._call_state in (CallState.DIALING, CallState.RINGING, CallState.CONNECTED)

    @property
    def call_duration(self) -> float:
        if self._call_start_time and self._call_state == CallState.CONNECTED:
            return time.time() - self._call_start_time
        return 0.0

    def get_call_info(self) -> CallInfo:
        return CallInfo(
            state=self._call_state,
            remote_uri=self._current_extension or "",
            duration=self.call_duration,
        )

    # ── Cleanup ─────────────────────────────────────────────────────────────
    def cleanup(self):
        self._bs_stop.set()
        try:
            if self._bs_sock:
                self._bs_sock.close()
        except Exception:
            pass
        self.logger.info("BaresipClient 資源已清理")
