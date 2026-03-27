# -*- coding: utf-8 -*-
"""
SIP/VoIP 通話模組 - 使用 Asterisk AMI 發起通話

透過 Asterisk Manager Interface (AMI) 控制通話，
使用 ConfBridge + Console 通道實現雙向語音。
"""

import socket
import threading
import time
import re
from typing import Callable, Optional
from enum import Enum
from dataclasses import dataclass

import sys
sys.path.append('..')
from utils.logger import get_logger


class CallState(Enum):
    """通話狀態"""
    IDLE = "idle"           # 閒置
    DIALING = "dialing"     # 撥號中
    RINGING = "ringing"     # 響鈴中
    CONNECTED = "connected" # 通話中
    DISCONNECTED = "disconnected"  # 已斷線


@dataclass
class CallInfo:
    """通話資訊"""
    state: CallState
    remote_uri: str = ""
    duration: float = 0.0
    dtmf_digits: str = ""


class SIPClient:
    """
    SIP 通話客戶端 - 使用 Asterisk AMI + ConfBridge

    撥打流程：
    1. AMI Originate 呼叫目標分機，接聽後進入 ConfBridge 房間
    2. 偵測到通話接聽後，自動讓 Console 通道也加入同一房間
    3. 雙方透過 ConfBridge 建立雙向語音
    4. 對方按 DTMF # 或 9 觸發開門事件
    """

    def __init__(
        self,
        server: str,
        port: int = 5060,
        username: str = "",
        password: str = "",
        domain: str = None,
        ami_port: int = 5038,
        ami_username: str = "intercom",
        ami_password: str = "intercom123"
    ):
        """
        初始化 SIP 客戶端

        Args:
            server: SIP 伺服器地址
            port: SIP 伺服器埠號（未使用，保留相容性）
            username: SIP 帳號（門口機分機，如 100）
            password: SIP 密碼（未使用）
            domain: SIP 域名（未使用）
            ami_port: AMI 埠號
            ami_username: AMI 使用者名稱
            ami_password: AMI 密碼
        """
        self.logger = get_logger()
        self.server = server
        self.port = port
        self.username = username
        self.password = password
        self.domain = domain or server
        self.ami_port = ami_port
        self.ami_username = ami_username
        self.ami_password = ami_password

        self._current_call = None
        self._call_state = CallState.IDLE
        self._state_lock = threading.RLock()  # RLock：同執行緒可重入，防止 call() + _update_state() 互相死鎖
        self._call_start_time: Optional[float] = None
        self._call_channel = None  # 目前通話的 channel ID
        self._console_channel = None  # Console 通道 ID
        self._confbridge_room = None  # ConfBridge 房間名稱
        self._ami_socket = None
        self._ami_connected = False
        self._monitor_thread = None
        self._event_thread = None
        self._stop_monitor = False

        # 來電相關
        self._incoming_call_channel: Optional[str] = None

        # 回調函數
        self._on_state_changed: Optional[Callable[[CallState], None]] = None
        self._on_dtmf_received: Optional[Callable[[str], None]] = None
        self._on_call_connected: Optional[Callable] = None
        self._on_call_ended: Optional[Callable] = None
        self._on_door_open: Optional[Callable] = None  # 開門回調
        self._on_incoming_call: Optional[Callable] = None  # 來電回調 (caller_id, channel)

        self.logger.info(f"SIP 客戶端初始化 (AMI+ConfBridge 模式): {server}:{ami_port}")

    def _ami_connect(self) -> bool:
        """連線到 AMI"""
        try:
            self._ami_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._ami_socket.settimeout(5)
            # AMI 和 GUI 在同一台機器上，連到 localhost
            ami_host = "127.0.0.1"
            self._ami_socket.connect((ami_host, self.ami_port))

            # 讀取歡迎訊息
            welcome = self._ami_socket.recv(1024).decode()
            self.logger.info(f"AMI 連線: {welcome.strip()}")

            # 登入
            login_cmd = (
                f"Action: Login\r\n"
                f"Username: {self.ami_username}\r\n"
                f"Secret: {self.ami_password}\r\n"
                f"\r\n"
            )
            self._ami_socket.send(login_cmd.encode())

            # 讀取回應（可能包含多個事件，如 FullyBooted）
            response = self._ami_recv(timeout=3)
            if "Success" in response:
                self._ami_connected = True
                self.logger.info("AMI 登入成功")
                # 清空可能殘留的事件（如 FullyBooted）
                self._ami_drain()
                return True
            else:
                self.logger.error(f"AMI 登入失敗: {response}")
                return False

        except Exception as e:
            self.logger.error(f"AMI 連線失敗: {e}")
            self._ami_socket = None
            return False

    def _ami_drain(self):
        """清空 socket 緩衝區中的殘留事件"""
        try:
            self._ami_socket.settimeout(0.5)
            while True:
                try:
                    data = self._ami_socket.recv(4096)
                    if not data:
                        break
                except socket.timeout:
                    break
        except:
            pass

    def _ami_recv(self, timeout=3) -> str:
        """接收 AMI 回應"""
        try:
            self._ami_socket.settimeout(timeout)
            data = b""
            while True:
                try:
                    chunk = self._ami_socket.recv(4096)
                    if not chunk:
                        break
                    data += chunk
                    # 讀取足夠的回應（等 timeout 或收到回應）
                    if b"\r\n\r\n" in data:
                        # 繼續短暫讀取，可能還有更多事件
                        self._ami_socket.settimeout(0.3)
                except socket.timeout:
                    break
            return data.decode('utf-8', errors='replace')
        except Exception as e:
            self.logger.error(f"AMI 接收失敗: {e}")
            return ""

    def _ami_send(self, action: dict) -> str:
        """發送 AMI 命令"""
        if not self._ami_connected or not self._ami_socket:
            if not self._ami_connect():
                return ""

        try:
            cmd = ""
            for key, value in action.items():
                cmd += f"{key}: {value}\r\n"
            cmd += "\r\n"

            self._ami_socket.send(cmd.encode())
            return self._ami_recv()
        except Exception as e:
            self.logger.error(f"AMI 發送失敗: {e}")
            self._ami_connected = False
            return ""

    def _ami_disconnect(self):
        """斷開 AMI 連線"""
        if self._ami_socket:
            try:
                self._ami_send({"Action": "Logoff"})
                self._ami_socket.close()
            except:
                pass
            self._ami_socket = None
            self._ami_connected = False

    def register(self) -> bool:
        """
        註冊到 SIP 伺服器（AMI 模式下連線到 AMI）

        Returns:
            bool: 是否成功
        """
        result = self._ami_connect()
        if result:
            self.logger.info(f"已連線到 Asterisk AMI: {self.server}:{self.ami_port}")
            self._start_event_listener()
        return result

    def _start_event_listener(self):
        """啟動 AMI 事件監聽執行緒（獨立 socket，偵測來電）"""
        self._event_thread = threading.Thread(
            target=self._event_listener_loop, daemon=True
        )
        self._event_thread.start()
        self.logger.info("AMI 事件監聽執行緒已啟動")

    def _event_listener_loop(self):
        """AMI 事件監聽迴圈（持續連線，重連失敗後等待 5 秒）"""
        while not self._stop_monitor:
            try:
                ev_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                ev_sock.settimeout(5)
                ev_sock.connect(("127.0.0.1", self.ami_port))

                # 讀取歡迎訊息
                ev_sock.recv(1024)

                # 登入
                ev_sock.send((
                    f"Action: Login\r\n"
                    f"Username: {self.ami_username}\r\n"
                    f"Secret: {self.ami_password}\r\n"
                    f"\r\n"
                ).encode())

                # 等待登入回應
                resp = b""
                ev_sock.settimeout(3)
                try:
                    while b"Success" not in resp:
                        chunk = ev_sock.recv(4096)
                        if not chunk:
                            break
                        resp += chunk
                except socket.timeout:
                    pass

                if b"Success" not in resp:
                    ev_sock.close()
                    time.sleep(5)
                    continue

                # 訂閱通話相關事件
                ev_sock.send(b"Action: Events\r\nEventMask: call\r\n\r\n")
                self.logger.info("AMI 事件監聽：已連線並訂閱 call 事件")

                # 持續讀取事件
                buf = ""
                ev_sock.settimeout(1.0)
                while not self._stop_monitor:
                    try:
                        data = ev_sock.recv(4096)
                        if not data:
                            break
                        buf += data.decode('utf-8', errors='replace')
                        while "\r\n\r\n" in buf:
                            event_str, buf = buf.split("\r\n\r\n", 1)
                            self._process_ami_event(event_str)
                    except socket.timeout:
                        continue

                ev_sock.close()

            except Exception as e:
                self.logger.debug(f"AMI 事件監聽重連: {e}")
                try:
                    ev_sock.close()
                except Exception:
                    pass
                time.sleep(5)

    def _process_ami_event(self, event_str: str):
        """解析並處理單一 AMI 事件"""
        fields: dict = {}
        for line in event_str.strip().split("\r\n"):
            if ': ' in line:
                key, _, val = line.partition(': ')
                fields[key.strip()] = val.strip()

        event_name = fields.get('Event', '')

        # ── Debug：記錄所有收到的 AMI 事件（排除常見雜訊事件）──────────
        if event_name and event_name not in ('RTCPSent', 'RTCPReceived', 'VarSet',
                                              'Cdr', 'AgentCalled', 'QueueCallerJoin'):
            self.logger.debug(f"AMI事件: {event_name} ch={fields.get('Channel','')} "
                              f"exten={fields.get('Exten','')} ctx={fields.get('Context','')}")

        # ── 偵測來電到本機分機 ──────────────────────────────────────────
        if event_name == 'Newchannel':
            exten = fields.get('Exten', '')
            channel = fields.get('Channel', '')
            caller_id = fields.get('CallerIDNum', '')
            context = fields.get('Context', '')
            # 來電條件：被呼叫號碼是我們（100），且不是自己 Originate 的 intercom-answer 通道
            if (exten == self.username
                    and channel.startswith('PJSIP/')
                    and context != 'intercom-answer'
                    and self._call_state in (CallState.IDLE, CallState.DISCONNECTED)):
                self.logger.info(f"偵測到來電: {caller_id} → {exten} (channel={channel})")
                self._incoming_call_channel = channel
                if self._on_incoming_call:
                    self._on_incoming_call(caller_id, channel)

        # ── 偵測來電通道掛斷（對方未等到接聽就掛掉）─────────────────────
        if event_name == 'Hangup':
            channel = fields.get('Channel', '')
            if channel and channel == self._incoming_call_channel:
                self.logger.info(f"來電通道已掛斷: {channel}")
                self._incoming_call_channel = None
                # 如果 GUI 還在響鈴就觸發 call_ended
                if self._call_state == CallState.RINGING and self._on_call_ended:
                    self._update_state(CallState.DISCONNECTED)

    def check_extension_registered(self, extension: str) -> bool:
        """
        檢查分機是否已在 Asterisk 登錄（使用獨立臨時 AMI 連線）

        Args:
            extension: 分機號碼

        Returns:
            bool: True = 已登錄，False = 未登錄或查詢失敗
        """
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(3)
            s.connect(("127.0.0.1", self.ami_port))
            s.recv(1024)  # 歡迎訊息

            # 登入
            s.send((
                f"Action: Login\r\n"
                f"Username: {self.ami_username}\r\n"
                f"Secret: {self.ami_password}\r\n"
                f"\r\n"
            ).encode())

            resp = b""
            s.settimeout(2)
            try:
                while b"Success" not in resp:
                    chunk = s.recv(4096)
                    if not chunk:
                        break
                    resp += chunk
            except socket.timeout:
                pass

            if b"Success" not in resp:
                s.close()
                return False

            # 查詢 AOR 聯絡人
            s.send((
                f"Action: Command\r\n"
                f"Command: pjsip show aor {extension}\r\n"
                f"\r\n"
            ).encode())

            resp = b""
            s.settimeout(2)
            try:
                while b"--END COMMAND--" not in resp:
                    chunk = s.recv(4096)
                    if not chunk:
                        break
                    resp += chunk
            except socket.timeout:
                pass

            s.close()

            # 回應中含有 "sip:" 表示有活躍的 Contact URI → 已登錄
            text = resp.decode('utf-8', errors='replace')
            return 'sip:' in text or 'sips:' in text

        except Exception as e:
            self.logger.debug(f"check_extension_registered({extension}) 失敗: {e}")
            return False

    def call(self, extension: str) -> bool:
        """
        撥打電話 - 使用 AMI Originate

        流程：
        1. Originate 呼叫目標分機
        2. 對方接聽後，dialplan 自動橋接 Console/default（USB 音效卡）
        3. 雙方透過 Dial() 橋接進行雙向語音通話
        4. 對方按 DTMF # 或 9 觸發 UserEvent(DoorOpen)

        Args:
            extension: 目標分機號碼

        Returns:
            bool: 是否成功發起通話
        """
        # 原子性防護：在同一把鎖內完成「檢查 + 設為 DIALING」
        # 避免多執行緒同時通過 guard 後重複撥出
        with self._state_lock:
            if self._call_state not in (CallState.IDLE, CallState.DISCONNECTED):
                self.logger.warning(
                    f"已有通話進行中（state={self._call_state.value}），拒絕重複撥號"
                )
                return False
            # 立刻鎖定為 DIALING，後續任何執行緒再進來都會被擋住
            self._call_state = CallState.DIALING

        self._call_ended_count = 0  # 每次新撥號重設計數，避免誤報重複
        self.logger.info(f"撥打分機: {extension}")
        if self._on_state_changed:
            self._on_state_changed(CallState.DIALING)

        # 確保 AMI 已連線
        if not self._ami_connected:
            if not self._ami_connect():
                self.logger.error("AMI 未連線，無法撥號")
                self._update_state(CallState.DISCONNECTED)
                return False

        try:
            # 清空可能殘留的事件
            self._ami_drain()

            # 使用 Originate 發起通話
            # 呼叫目標分機 → 接聽後進入 intercom-answer context
            # intercom-answer 會自動 Dial(Console/default) 橋接本地音訊
            # 對方按 # 或 9 觸發 UserEvent(DoorOpen)
            response = self._ami_send({
                "Action": "Originate",
                "Channel": f"PJSIP/{extension}",
                "Context": "intercom-answer",
                "Exten": "s",
                "Priority": "1",
                "CallerID": f"Intercom <{self.username}>",
                "Timeout": "30000",
                "Async": "true"
            })

            self.logger.info(f"Originate 回應: {response[:200].strip()}")

            if "Success" in response or "Originate successfully queued" in response:
                # 偵測立即失敗：OriginateResponse: Failure 已在同一筆回應裡
                # （對方離線時 Asterisk 幾乎同步回傳 Failure）
                if "OriginateResponse" in response and "\nResponse: Failure" in response:
                    self.logger.warning(f"撥號立即失敗（{extension} 不在線）")
                    self._update_state(CallState.DISCONNECTED)
                    return False

                self._current_call = extension
                self._update_state(CallState.RINGING)

                # 啟動監控執行緒
                self._stop_monitor = False
                self._monitor_thread = threading.Thread(
                    target=self._monitor_call,
                    args=(extension,),
                    daemon=True
                )
                self._monitor_thread.start()

                return True
            else:
                self.logger.error(f"撥號失敗: {response}")
                self._update_state(CallState.DISCONNECTED)
                return False

        except Exception as e:
            self.logger.error(f"撥打失敗: {e}")
            self._update_state(CallState.DISCONNECTED)
            return False

    def _parse_active_channels(self, response: str, extension: str):
        """
        只從 CoreShowChannels 回應的 Event: CoreShowChannel block 判斷通道狀態。
        只比對 Channel: 欄位（避免 BridgedChannel:/ConnectedLineNum: 等欄位誤中）。
        忽略後續流入的 RTCPSent/VarSet/Hangup 事件（它們也含頻道名稱，會造成假陽性）。
        回傳: (has_active, is_up, full_channel_name)
        """
        has_active = False
        is_up = False
        found_channel = ""
        prefix = f"PJSIP/{extension}"
        for block in response.split("\r\n\r\n"):
            if "Event: CoreShowChannel" not in block:
                continue
            # 只檢查 Channel: 欄位（逐行比對）
            channel_val = ""
            channel_state = ""
            for line in block.split("\r\n"):
                if line.startswith("Channel: "):
                    channel_val = line[len("Channel: "):]
                elif line.startswith("ChannelStateDesc: "):
                    channel_state = line[len("ChannelStateDesc: "):]
            if not channel_val.startswith(prefix):
                continue
            has_active = True
            if not found_channel:
                found_channel = channel_val  # 記錄完整 channel name（如 PJSIP/105-00000018）
            if channel_state == "Up":
                is_up = True
                self.logger.info(f"[monitor] 通道 Up 偵測: {channel_val} state={channel_state}")
        return has_active, is_up, found_channel

    def _monitor_call(self, extension: str):
        """監控通話狀態和 DTMF 開門事件"""
        call_answered = False
        max_wait = 35  # 最多等 35 秒（未接聽前）
        start = time.time()

        while not self._stop_monitor and (time.time() - start) < max_wait:
            time.sleep(0.5)  # 更頻繁檢查

            try:
                # 使用 CoreShowChannels 查詢所有活躍通道
                response = self._ami_send({
                    "Action": "CoreShowChannels"
                })

                if not response:
                    continue

                # 只解析 Event: CoreShowChannel block，忽略殘留的 RTCP/Hangup 事件
                has_pjsip, pjsip_up, full_channel = self._parse_active_channels(response, extension)

                # 一旦偵測到完整 channel name，更新 _current_call 以便 hangup() 精確掛斷
                if full_channel and full_channel != self._current_call:
                    self._current_call = full_channel
                    self.logger.info(f"[monitor] 更新 _current_call: {full_channel}")

                # 檢查是否已接聽（PJSIP Up）
                # 前 3 秒忽略 Up 偵測（Asterisk 初始化通道時可能瞬間出現 Up）
                if pjsip_up and not call_answered and (time.time() - start) >= 3.0:
                    call_answered = True
                    self._call_start_time = time.time()
                    self._update_state(CallState.CONNECTED)
                    self.logger.info(f"通話已接聽: {extension}")
                    # 接聽後延長等待時間
                    max_wait = time.time() - start + 300  # 通話最多 5 分鐘

                # 檢查 AMI 事件中是否有 DoorOpen UserEvent
                if call_answered and "UserEvent" in response and "DoorOpen" in response:
                    self.logger.info("收到 DoorOpen 事件！觸發開門")
                    if self._on_door_open:
                        self._on_door_open()

                # 如果已接聽但 PJSIP 通道消失了（對方掛斷）
                if call_answered and not has_pjsip:
                    self.logger.info("對方已掛斷")
                    # 也掛斷 Console
                    try:
                        self._ami_send({
                            "Action": "Command",
                            "Command": "channel request hangup Console/default"
                        })
                    except:
                        pass
                    break

                # 如果還沒接聽且 PJSIP 通道消失了（對方拒接或超時）
                if not call_answered and not has_pjsip:
                    if time.time() - start > 3:
                        self.logger.info("對方未接聽或已拒接")
                        break

            except Exception as e:
                self.logger.error(f"監控通話失敗: {e}")
                break

        if self._call_state != CallState.DISCONNECTED:
            self._update_state(CallState.DISCONNECTED)
            self._current_call = None
            self._confbridge_room = None

    def hangup(self) -> bool:
        """
        掛斷電話（同時清理 Console 和 PJSIP 通道）

        Returns:
            bool: 是否成功掛斷
        """
        self._stop_monitor = True

        if self._ami_connected:
            # 掛斷 Console 通道
            try:
                response = self._ami_send({
                    "Action": "Command",
                    "Command": "channel request hangup Console/default"
                })
                self.logger.info(f"掛斷 Console: {response[:100].strip() if response else 'no response'}")
            except Exception as e:
                self.logger.error(f"掛斷 Console 失敗: {e}")

            # 掛斷 PJSIP 通道
            # _current_call 可能是 "105"（extension）或完整名稱 "PJSIP/105-00000018"
            if self._current_call:
                try:
                    # 優先使用 AMI Hangup action（需要完整 channel name）
                    if self._current_call.startswith("PJSIP/") and "-" in self._current_call:
                        # 已有完整 channel name，使用 AMI Hangup（精確）
                        response = self._ami_send({
                            "Action": "Hangup",
                            "Channel": self._current_call,
                            "Cause": "16"
                        })
                        self.logger.info(f"AMI Hangup {self._current_call}: {response[:100].strip() if response else 'no response'}")
                    else:
                        # 只有 extension，查 CoreShowChannels 找完整 channel name
                        cs_resp = self._ami_send({"Action": "CoreShowChannels"})
                        _, _, full_ch = self._parse_active_channels(cs_resp, self._current_call)
                        if full_ch:
                            response = self._ami_send({
                                "Action": "Hangup",
                                "Channel": full_ch,
                                "Cause": "16"
                            })
                            self.logger.info(f"AMI Hangup (查找) {full_ch}: {response[:100].strip() if response else 'no response'}")
                        else:
                            # fallback：CLI command
                            response = self._ami_send({
                                "Action": "Command",
                                "Command": f"channel request hangup PJSIP/{self._current_call}"
                            })
                            self.logger.info(f"掛斷通話 (fallback): {response[:100].strip() if response else 'no response'}")
                except Exception as e:
                    self.logger.error(f"掛斷失敗: {e}")

        self._current_call = None
        self._confbridge_room = None
        if self._call_state != CallState.DISCONNECTED:
            self._update_state(CallState.DISCONNECTED)

        return True

    def send_dtmf(self, digits: str) -> bool:
        """
        發送 DTMF 按鍵（AMI 模式下不支援，由對方手機端處理）

        Args:
            digits: DTMF 數字/符號

        Returns:
            bool: 是否成功發送
        """
        self.logger.info(f"DTMF 由對方手機端處理: {digits}")
        return True

    def _update_state(self, state: CallState):
        """更新通話狀態（含鎖與 idempotent 保護，防止 DISCONNECTED callback 重入）"""
        with self._state_lock:
            old_state = self._call_state
            if state == old_state:
                # 狀態未變：直接忽略，避免 DISCONNECTED/CONNECTED 重入觸發 callback
                return
            self._call_state = state

        if state == CallState.CONNECTED:
            self._call_start_time = time.time()
            if self._on_call_connected:
                self._on_call_connected()

        elif state == CallState.DISCONNECTED:
            self._call_start_time = None
            if self._on_call_ended:
                self._on_call_ended()

        if self._on_state_changed:
            self._on_state_changed(state)

        self.logger.info(f"通話狀態變更: {old_state.value} -> {state.value}")

    @property
    def call_state(self) -> CallState:
        """目前通話狀態"""
        return self._call_state

    @property
    def is_in_call(self) -> bool:
        """是否在通話中（含撥號中、響鈴中）"""
        return self._call_state in (CallState.DIALING, CallState.RINGING, CallState.CONNECTED)

    @property
    def call_duration(self) -> float:
        """通話時長 (秒)"""
        if self._call_start_time:
            return time.time() - self._call_start_time
        return 0.0

    def get_call_info(self) -> CallInfo:
        """取得通話資訊"""
        return CallInfo(
            state=self._call_state,
            duration=self.call_duration
        )

    def set_on_state_changed(self, callback: Callable[[CallState], None]):
        """設定狀態變更回調"""
        self._on_state_changed = callback

    def set_on_dtmf_received(self, callback: Callable[[str], None]):
        """設定收到 DTMF 的回調"""
        self._on_dtmf_received = callback

    def set_on_call_connected(self, callback: Callable):
        """設定通話連接時的回調"""
        self._on_call_connected = callback

    def set_on_call_ended(self, callback: Callable):
        """設定通話結束時的回調"""
        self._on_call_ended = callback

    def set_on_door_open(self, callback: Callable):
        """設定開門事件的回調"""
        self._on_door_open = callback

    def set_on_incoming_call(self, callback: Callable):
        """設定來電回調 callback(caller_id: str, channel: str)"""
        self._on_incoming_call = callback

    def answer_incoming_call(self) -> bool:
        """
        接聽來電：將來電通道重新導向到 intercom-answer dialplan context，
        讓 Asterisk 透過 AudioSocket 橋接音訊。
        """
        channel = self._incoming_call_channel
        if not channel:
            self.logger.warning("answer_incoming_call: 沒有待接來電")
            return False

        self.logger.info(f"接聽來電: {channel}")
        self._current_call = channel
        self._update_state(CallState.RINGING)

        response = self._ami_send({
            "Action": "Redirect",
            "Channel": channel,
            "Context": "intercom-answer",
            "Exten": "s",
            "Priority": "1"
        })
        self.logger.info(f"Redirect 回應: {response[:200].strip()}")

        if "Success" in response:
            # 啟動監控執行緒等待通話接通
            ext_guess = channel.split('/')[1].split('-')[0] if '/' in channel else channel
            self._stop_monitor = False
            self._monitor_thread = threading.Thread(
                target=self._monitor_call, args=(ext_guess,), daemon=True
            )
            self._monitor_thread.start()
            return True
        else:
            self._update_state(CallState.DISCONNECTED)
            self._incoming_call_channel = None
            return False

    def reset_to_idle(self):
        """
        安全地重設狀態到 IDLE（持鎖），供 GUI 在 DISCONNECTED 後呼叫。
        不觸發 call_ended callback；只做狀態清理。
        """
        with self._state_lock:
            if self._call_state == CallState.DISCONNECTED:
                self._call_state = CallState.IDLE
                self._call_start_time = None
                self._current_call = None
                self._confbridge_room = None
                self._incoming_call_channel = None

    def cleanup(self):
        """清理資源"""
        self._stop_monitor = True
        if self._current_call:
            self.hangup()
        self._ami_disconnect()
        self.logger.info("SIP 客戶端資源已清理")
        # 等待事件監聽執行緒結束（最多 3 秒）
        if self._event_thread and self._event_thread.is_alive():
            self._event_thread.join(timeout=3)


# =============================================================================
# 測試程式
# =============================================================================
if __name__ == "__main__":
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.logger import setup_logger

    # 初始化日誌
    setup_logger(log_level="DEBUG")

    print("=" * 50)
    print("SIP 通話模組測試 (AMI 模式)")
    print("=" * 50)

    # 建立 SIP 客戶端
    sip_client = SIPClient(
        server="192.168.100.163",
        username="100",
        password="password100"
    )

    # 設定回調
    sip_client.set_on_state_changed(
        lambda state: print(f">>> 狀態: {state.value}")
    )
    sip_client.set_on_call_connected(
        lambda: print(">>> 通話已連接!")
    )
    sip_client.set_on_call_ended(
        lambda: print(">>> 通話已結束!")
    )

    # 測試連線
    print("\n測試 1: 連線 AMI")
    sip_client.register()
    time.sleep(2)

    # 測試撥號
    print("\n測試 2: 撥打分機 101")
    sip_client.call("101")
    time.sleep(10)

    # 測試掛斷
    print("\n測試 3: 掛斷")
    sip_client.hangup()

    print("\n測試完成!")
    sip_client.cleanup()
