# -*- coding: utf-8 -*-
"""
NFC 卡片管理模組 - 使用 PN532 (I2C)
"""

import threading
import time
import sqlite3
from pathlib import Path
from typing import Callable, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

# 在非樹莓派環境下提供模擬模式
PN532_AVAILABLE = False
try:
    import board
    import busio
    from adafruit_pn532.i2c import PN532_I2C
    PN532_AVAILABLE = True
except ImportError:
    print("[模擬模式] PN532 函式庫不可用")

import sys
sys.path.append('..')
from utils.logger import get_logger


class NFCResult(Enum):
    """NFC 讀取結果"""
    SUCCESS = "success"          # 已授權卡片
    UNKNOWN = "unknown"          # 未知卡片
    NO_CARD = "no_card"          # 沒有偵測到卡片
    DISABLED = "disabled"        # 卡片已停用
    ERROR = "error"              # 讀取錯誤


@dataclass
class NFCCard:
    """NFC 卡片資料"""
    id: int
    uid: str
    name: str
    company_id: int
    card_type: str = "card"      # card / phone
    created_at: str = ""
    last_used: str = ""
    active: bool = True


class NFCManager:
    """
    NFC 卡片管理器

    使用 PN532 模組透過 I2C 連接樹莓派
    支援 MIFARE Classic、NTAG、ISO14443A 等卡片
    """

    def __init__(
        self,
        database_path: str = "nfc_cards.db",
        i2c_bus: int = 1,
        scan_interval: float = 0.3
    ):
        """
        初始化 NFC 管理器

        Args:
            database_path: 資料庫檔案路徑
            i2c_bus: I2C 匯流排編號
            scan_interval: 掃描間隔 (秒)
        """
        self.logger = get_logger()
        self.database_path = database_path
        self.i2c_bus = i2c_bus
        self.scan_interval = scan_interval

        # PN532 讀卡器
        self._pn532 = None
        self._i2c = None

        # 已授權卡片快取
        self._authorized_cards: dict = {}  # uid -> NFCCard

        # 連續掃描
        self._is_scanning = False
        self._scan_thread: Optional[threading.Thread] = None
        self._stop_scan = threading.Event()

        # 回調函數
        self._on_card_detected: Optional[Callable[[NFCCard], None]] = None
        self._on_unknown_card: Optional[Callable[[str], None]] = None

        # 初始化
        self._init_database()
        self._init_reader()
        self._load_authorized_cards()

    def _init_database(self):
        """初始化資料庫"""
        try:
            db_path = Path(self.database_path)
            db_path.parent.mkdir(parents=True, exist_ok=True)

            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()

            # 建立卡片表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS nfc_cards (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    uid TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    company_id INTEGER DEFAULT 0,
                    card_type TEXT DEFAULT 'card',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_used TIMESTAMP,
                    active INTEGER DEFAULT 1,
                    user_id INTEGER
                )
            """)

            # 建立存取記錄表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS nfc_access_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    card_id INTEGER,
                    uid TEXT,
                    result TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.commit()
            conn.close()

            self.logger.info(f"NFC 資料庫已初始化: {self.database_path}")

        except Exception as e:
            self.logger.error(f"NFC 資料庫初始化失敗: {e}")
            raise

    def _init_reader(self):
        """初始化 PN532 讀卡器"""
        if not PN532_AVAILABLE:
            self.logger.info("[模擬] NFC 讀卡器初始化")
            return

        try:
            self._i2c = busio.I2C(board.SCL, board.SDA)
            self._pn532 = PN532_I2C(self._i2c, debug=False)

            # 取得韌體版本
            ic, ver, rev, support = self._pn532.firmware_version
            self.logger.info(f"PN532 韌體版本: {ver}.{rev}")

            # 設定為等待 NFC 卡片
            self._pn532.SAM_configuration()

            self.logger.info("PN532 NFC 讀卡器已初始化")

        except Exception as e:
            self.logger.error(f"PN532 初始化失敗: {e}")
            self._pn532 = None

    def _load_authorized_cards(self):
        """載入已授權的卡片"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT id, uid, name, company_id, card_type,
                       created_at, last_used, active
                FROM nfc_cards
            """)
            rows = cursor.fetchall()
            conn.close()

            self._authorized_cards.clear()

            for row in rows:
                card = NFCCard(
                    id=row[0],
                    uid=row[1],
                    name=row[2],
                    company_id=row[3],
                    card_type=row[4] or "card",
                    created_at=row[5] or "",
                    last_used=row[6] or "",
                    active=bool(row[7])
                )
                self._authorized_cards[card.uid] = card

            self.logger.info(f"已載入 {len(self._authorized_cards)} 張授權卡片")

        except Exception as e:
            self.logger.error(f"載入卡片資料失敗: {e}")

    @property
    def is_reader_ready(self) -> bool:
        """讀卡器是否就緒"""
        if not PN532_AVAILABLE:
            return True  # 模擬模式
        return self._pn532 is not None

    @property
    def card_count(self) -> int:
        """已授權卡片數量"""
        return len(self._authorized_cards)

    def read_card_uid(self, timeout: float = 1.0) -> Optional[str]:
        """
        讀取卡片 UID

        Args:
            timeout: 等待時間 (秒)

        Returns:
            卡片 UID（十六進位字串）或 None
        """
        if not PN532_AVAILABLE:
            # 模擬模式：隨機返回
            return None

        if not self._pn532:
            return None

        try:
            # 嘗試讀取卡片
            uid = self._pn532.read_passive_target(timeout=timeout)

            if uid is not None:
                # 轉換為十六進位字串
                uid_hex = ''.join([f'{b:02x}' for b in uid])
                return uid_hex.upper()

            return None

        except Exception as e:
            self.logger.error(f"讀取卡片失敗: {e}")
            return None

    def check_card(self, uid: str) -> Tuple[NFCResult, Optional[NFCCard]]:
        """
        檢查卡片是否已授權

        Args:
            uid: 卡片 UID

        Returns:
            (結果, 卡片資料)
        """
        if uid in self._authorized_cards:
            card = self._authorized_cards[uid]

            if not card.active:
                self._log_access(card.id, uid, "disabled")
                return NFCResult.DISABLED, card

            # 更新最後使用時間
            self._update_last_used(card.id)
            self._log_access(card.id, uid, "success")

            return NFCResult.SUCCESS, card

        # 未知卡片
        self._log_access(None, uid, "unknown")
        return NFCResult.UNKNOWN, None

    def _update_last_used(self, card_id: int):
        """更新卡片最後使用時間"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE nfc_cards SET last_used = CURRENT_TIMESTAMP WHERE id = ?",
                (card_id,)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            self.logger.error(f"更新使用時間失敗: {e}")

    def _log_access(self, card_id: Optional[int], uid: str, result: str):
        """記錄存取日誌"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO nfc_access_logs (card_id, uid, result) VALUES (?, ?, ?)",
                (card_id, uid, result)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            self.logger.error(f"記錄存取日誌失敗: {e}")

    def register_card(
        self,
        name: str,
        company_id: int = 0,
        card_type: str = "card",
        timeout: float = 10.0
    ) -> Tuple[bool, str]:
        """
        登錄新卡片

        Args:
            name: 使用者名稱
            company_id: 所屬公司 ID
            card_type: 卡片類型 (card/phone)
            timeout: 等待卡片的時間

        Returns:
            (成功與否, 訊息)
        """
        self.logger.info(f"開始登錄卡片: {name} (公司 {company_id})")

        # 等待卡片
        start_time = time.time()
        uid = None

        while time.time() - start_time < timeout:
            uid = self.read_card_uid(timeout=0.5)
            if uid:
                break
            time.sleep(0.1)

        if not uid:
            return False, "未偵測到卡片，請將卡片靠近讀卡器"

        # 檢查是否已存在
        if uid in self._authorized_cards:
            existing = self._authorized_cards[uid]
            return False, f"此卡片已登錄為: {existing.name}"

        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()

            cursor.execute(
                """INSERT INTO nfc_cards (uid, name, company_id, card_type)
                   VALUES (?, ?, ?, ?)""",
                (uid, name, company_id, card_type)
            )

            conn.commit()
            conn.close()

            # 重新載入卡片
            self._load_authorized_cards()

            self.logger.info(f"卡片登錄成功: {name} (UID: {uid})")
            return True, f"卡片登錄成功！UID: {uid}"

        except Exception as e:
            self.logger.error(f"卡片登錄失敗: {e}")
            return False, f"登錄失敗: {str(e)}"

    def delete_card(self, card_id: int) -> bool:
        """
        刪除卡片

        Args:
            card_id: 卡片 ID

        Returns:
            是否成功
        """
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM nfc_cards WHERE id = ?", (card_id,))
            conn.commit()
            conn.close()

            # 重新載入卡片
            self._load_authorized_cards()

            self.logger.info(f"卡片已刪除: {card_id}")
            return True

        except Exception as e:
            self.logger.error(f"刪除卡片失敗: {e}")
            return False

    def toggle_card_active(self, card_id: int, active: bool) -> bool:
        """
        啟用/停用卡片

        Args:
            card_id: 卡片 ID
            active: 是否啟用

        Returns:
            是否成功
        """
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE nfc_cards SET active = ? WHERE id = ?",
                (1 if active else 0, card_id)
            )
            conn.commit()
            conn.close()

            # 重新載入卡片
            self._load_authorized_cards()

            status = "啟用" if active else "停用"
            self.logger.info(f"卡片已{status}: {card_id}")
            return True

        except Exception as e:
            self.logger.error(f"更新卡片狀態失敗: {e}")
            return False

    def get_all_cards(self) -> List[NFCCard]:
        """取得所有卡片"""
        return list(self._authorized_cards.values())

    def start_continuous_scan(self, callback=None):
        """
        開始連續掃描模式

        Args:
            callback: 回調函數 (result: NFCResult, card: Optional[NFCCard])
        """
        if self._is_scanning:
            return

        self._is_scanning = True
        self._stop_scan.clear()

        def scan_loop():
            last_uid = None
            last_scan_time = 0

            while not self._stop_scan.is_set():
                uid = self.read_card_uid(timeout=0.5)

                if uid:
                    # 防止同一張卡重複觸發（3秒內）
                    current_time = time.time()
                    if uid != last_uid or current_time - last_scan_time > 3:
                        last_uid = uid
                        last_scan_time = current_time

                        result, card = self.check_card(uid)

                        if callback:
                            callback(result, card)
                        elif result == NFCResult.SUCCESS and card:
                            if self._on_card_detected:
                                self._on_card_detected(card)
                        elif result == NFCResult.UNKNOWN:
                            if self._on_unknown_card:
                                self._on_unknown_card(uid)

                time.sleep(self.scan_interval)

        self._scan_thread = threading.Thread(target=scan_loop, daemon=True)
        self._scan_thread.start()
        self.logger.info("NFC 連續掃描已啟動")

    def stop_continuous_scan(self):
        """停止連續掃描"""
        if not self._is_scanning:
            return

        self._stop_scan.set()
        if self._scan_thread:
            self._scan_thread.join(timeout=2.0)
        self._is_scanning = False
        self.logger.info("NFC 連續掃描已停止")

    def set_on_card_detected(self, callback: Callable[[NFCCard], None]):
        """設定卡片識別成功的回調"""
        self._on_card_detected = callback

    def set_on_unknown_card(self, callback: Callable[[str], None]):
        """設定未知卡片的回調"""
        self._on_unknown_card = callback

    def cleanup(self):
        """清理資源"""
        self.stop_continuous_scan()

        if self._i2c:
            try:
                self._i2c.deinit()
            except:
                pass

        self.logger.info("NFC 管理器資源已清理")


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
    print("NFC 卡片管理器測試 (PN532)")
    print("=" * 50)

    # 建立管理器
    nfc_manager = NFCManager(
        database_path="test_nfc.db",
        scan_interval=0.3
    )

    print(f"\n讀卡器狀態: {'就緒' if nfc_manager.is_reader_ready else '未就緒'}")
    print(f"已授權卡片數: {nfc_manager.card_count}")

    # 列出所有卡片
    print("\n已授權卡片:")
    for card in nfc_manager.get_all_cards():
        status = "啟用" if card.active else "停用"
        print(f"  - {card.name} (UID: {card.uid}) [{status}]")

    # 測試讀取卡片
    print("\n請將卡片靠近讀卡器...")
    uid = nfc_manager.read_card_uid(timeout=5.0)
    if uid:
        print(f"讀取到卡片: {uid}")
        result, card = nfc_manager.check_card(uid)
        if card:
            print(f"卡片持有者: {card.name}")
        else:
            print(f"結果: {result.value}")
    else:
        print("未偵測到卡片")

    print("\n測試完成!")
    nfc_manager.cleanup()
