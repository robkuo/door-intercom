# -*- coding: utf-8 -*-
"""
AudioSocket 音訊橋接程式

此程式接收 Asterisk AudioSocket 連線，將音訊橋接到本地 USB 音效卡。
使用 pyalsaaudio 直接操作 ALSA，支援 8000Hz 取樣率轉換。
"""

import os
import json
import socket
import struct
import threading
import time
import logging
from typing import Optional

try:
    import alsaaudio
    ALSAAUDIO_AVAILABLE = True
except ImportError:
    ALSAAUDIO_AVAILABLE = False
    print("錯誤: 需要安裝 pyalsaaudio")

# AudioSocket 協議常數
AUDIOSOCKET_HEADER_SIZE = 3
AUDIOSOCKET_TYPE_HANGUP = 0x00
AUDIOSOCKET_TYPE_UUID = 0x01
AUDIOSOCKET_TYPE_SILENCE = 0x02
AUDIOSOCKET_TYPE_DTMF = 0x03
AUDIOSOCKET_TYPE_AUDIO = 0x10
AUDIOSOCKET_TYPE_ERROR = 0xff

# 音訊參數
SAMPLE_RATE = 8000
CHANNELS = 1
PERIOD_SIZE = 160
# 播放與錄音分開指定，避免 asound.conf plug-over-asym 無法被 alsaaudio 正確開啟
PLAYBACK_DEVICE = "plughw:Headphones"   # bcm2835 3.5mm 耳機輸出（CD002AUDIO USB 喇叭未偵測到時的備用）
CAPTURE_DEVICE  = "plughw:UACDemoV10"   # UACDemoV10 USB 麥克風（明確指定）

# ── 雜音抑制設定 ──────────────────────────────────────────────────────
# NOISE_GATE_THRESHOLD：低於此 RMS 振幅視為背景雜音，以靜音取代（0-32767）
# 建議值：500-800；若仍有雜音可提高至 1000；若語音被截斷可降低至 300
NOISE_GATE_THRESHOLD = 0

# 音訊設定檔路徑（由管理網頁寫入，每次通話開始時重新讀取）
AUDIO_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'audio_config.json')


class AudioBridge:
    def __init__(self, host: str = "127.0.0.1", port: int = 9999):
        self.host = host
        self.port = port
        self.server_socket = None
        self.running = False
        self.logger = logging.getLogger("AudioBridge")
        self.pcm_out = None
        self.pcm_in = None
        self.mic_running = False
        self.noise_gate_threshold = NOISE_GATE_THRESHOLD  # 預設值，每次通話開始時重新讀取

    def _reload_audio_config(self):
        """從 audio_config.json 讀取最新設定（每次通話建立時呼叫）"""
        try:
            with open(AUDIO_CONFIG_PATH, 'r') as f:
                cfg = json.load(f)
            self.noise_gate_threshold = int(cfg.get('noise_gate_threshold', NOISE_GATE_THRESHOLD))
            self.logger.info(f"音訊設定已載入: 噪音閘門={self.noise_gate_threshold}")
        except FileNotFoundError:
            pass  # 設定檔尚未建立，使用預設值
        except Exception as e:
            self.logger.warning(f"讀取音訊設定失敗（使用預設值）: {e}")

    def start(self):
        self.running = True
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        self.logger.info(f"AudioSocket 伺服器啟動: {self.host}:{self.port}")

        while self.running:
            try:
                self.server_socket.settimeout(1.0)
                try:
                    client_socket, addr = self.server_socket.accept()
                    self.logger.info(f"AudioSocket 連線: {addr}")
                    self._handle_connection(client_socket)
                except socket.timeout:
                    continue
            except Exception as e:
                if self.running:
                    self.logger.error(f"錯誤: {e}")
                break

    def stop(self):
        self.running = False
        if self.server_socket:
            self.server_socket.close()

    def _handle_connection(self, client_socket):
        try:
            client_socket.settimeout(0.1)
            self._reload_audio_config()   # 每次通話開始時讀取最新音訊設定
            self._init_alsa()

            # 啟動麥克風執行緒
            self.mic_running = True
            mic_thread = threading.Thread(target=self._mic_to_asterisk, args=(client_socket,), daemon=True)
            mic_thread.start()

            # 主迴圈：接收 Asterisk 音訊並播放
            while self.running:
                try:
                    header = self._recv_exact(client_socket, 3)
                    if not header:
                        break

                    msg_type = header[0]
                    length = struct.unpack(">H", header[1:3])[0]

                    payload = b""
                    if length > 0:
                        payload = self._recv_exact(client_socket, length)
                        if not payload:
                            break

                    if msg_type == AUDIOSOCKET_TYPE_UUID:
                        self.logger.info(f"UUID 收到")
                    elif msg_type == AUDIOSOCKET_TYPE_AUDIO:
                        self._play_audio(payload)
                    elif msg_type == AUDIOSOCKET_TYPE_HANGUP:
                        self.logger.info("掛斷")
                        break

                except socket.timeout:
                    continue
                except Exception as e:
                    self.logger.error(f"處理錯誤: {e}")
                    break
        finally:
            self.mic_running = False
            self._close_alsa()
            client_socket.close()
            self.logger.info("連線結束")

    def _recv_exact(self, sock, n):
        data = b""
        while len(data) < n:
            try:
                chunk = sock.recv(n - len(data))
                if not chunk:
                    return None
                data += chunk
            except socket.timeout:
                if not self.running:
                    return None
                continue
        return data

    def _init_alsa(self):
        # 播放初始化（失敗時 raise → 讓連線關閉，Asterisk 收到 EOF 後重連）
        try:
            self.pcm_out = alsaaudio.PCM(
                type=alsaaudio.PCM_PLAYBACK, mode=alsaaudio.PCM_NORMAL,
                device=PLAYBACK_DEVICE, channels=CHANNELS, rate=SAMPLE_RATE,
                format=alsaaudio.PCM_FORMAT_S16_LE, periodsize=PERIOD_SIZE
            )
            self.logger.info(f"ALSA 播放 OK: {PLAYBACK_DEVICE}")
        except Exception as e:
            self.logger.error(f"ALSA 播放失敗 ({PLAYBACK_DEVICE}): {e}")
            raise  # 讓 _handle_connection 進入 finally → 關閉連線

        # 錄音初始化（失敗時同樣 raise）
        try:
            self.pcm_in = alsaaudio.PCM(
                type=alsaaudio.PCM_CAPTURE, mode=alsaaudio.PCM_NORMAL,
                device=CAPTURE_DEVICE, channels=CHANNELS, rate=SAMPLE_RATE,
                format=alsaaudio.PCM_FORMAT_S16_LE, periodsize=PERIOD_SIZE
            )
            self.logger.info(f"ALSA 錄音 OK: {CAPTURE_DEVICE}")
        except Exception as e:
            self.logger.error(f"ALSA 錄音失敗 ({CAPTURE_DEVICE}): {e}")
            raise  # 讓 _handle_connection 進入 finally → 關閉連線

    def _close_alsa(self):
        if self.pcm_out:
            self.pcm_out.close()
            self.pcm_out = None
        if self.pcm_in:
            self.pcm_in.close()
            self.pcm_in = None

    def _play_audio(self, data):
        if self.pcm_out and data:
            self.pcm_out.write(data)

    def _apply_noise_gate(self, data: bytes) -> bytes:
        """
        軟體噪音閘門：計算音訊幀的 RMS 振幅，
        低於 self.noise_gate_threshold 的幀視為背景雜音，以靜音取代。
        threshold=0 表示關閉噪音閘門（直接回傳原始音訊）。
        """
        if self.noise_gate_threshold == 0:
            return data
        n = len(data) // 2
        if n == 0:
            return data
        samples = struct.unpack(f'<{n}h', data[:n * 2])
        rms = (sum(s * s for s in samples) / n) ** 0.5
        if rms < self.noise_gate_threshold:
            return b'\x00' * len(data)
        return data

    def _mic_to_asterisk(self, client_socket):
        """從麥克風讀取並發送到 Asterisk（ALSA 錯誤時送靜音並嘗試恢復，不中斷連線）"""
        self.logger.info("麥克風啟動")
        count = 0
        SILENCE = b'\x00' * (PERIOD_SIZE * 2)  # 320 bytes 靜音

        while self.mic_running and self.running:
            # ── 1. 從 ALSA 讀取音訊（錯誤時送靜音並嘗試重開裝置）──
            try:
                if not self.pcm_in:
                    time.sleep(0.02)
                    data = SILENCE
                else:
                    length, raw_data = self.pcm_in.read()
                    data = raw_data if (length > 0 and raw_data) else SILENCE
            except Exception as e:
                self.logger.error(f"麥克風讀取錯誤（送靜音）: {e}")
                data = SILENCE
                # 嘗試重新初始化 ALSA 錄音裝置
                try:
                    if self.pcm_in:
                        self.pcm_in.close()
                    self.pcm_in = alsaaudio.PCM(
                        type=alsaaudio.PCM_CAPTURE, mode=alsaaudio.PCM_NORMAL,
                        device=CAPTURE_DEVICE, channels=CHANNELS, rate=SAMPLE_RATE,
                        format=alsaaudio.PCM_FORMAT_S16_LE, periodsize=PERIOD_SIZE
                    )
                    self.logger.info("ALSA 錄音恢復成功")
                except Exception as e2:
                    self.logger.error(f"ALSA 錄音恢復失敗: {e2}")
                    self.pcm_in = None

            # ── 2. 噪音閘門（靜音背景雜音）──
            if data is not SILENCE:
                data = self._apply_noise_gate(data)

            # ── 3. 發送到 Asterisk（只有 Socket 錯誤才結束執行緒）──
            try:
                header = bytes([AUDIOSOCKET_TYPE_AUDIO]) + struct.pack(">H", len(data))
                client_socket.sendall(header + data)
                count += 1
                if count % 500 == 0:
                    self.logger.info(f"已發送 {count} 封包")
            except Exception as e:
                self.logger.error(f"Socket 發送失敗（結束）: {e}")
                break

        self.logger.info(f"麥克風結束，共 {count} 封包")


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(name)s | %(message)s')
    if not ALSAAUDIO_AVAILABLE:
        return
    bridge = AudioBridge()
    try:
        bridge.start()
    except KeyboardInterrupt:
        pass
    finally:
        bridge.stop()


if __name__ == "__main__":
    main()
