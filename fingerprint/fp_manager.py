# -*- coding: utf-8 -*-
"""
指紋辨識模組 - 支援 R307/R503 光學指紋模組
"""

import threading
import time
import sqlite3
from pathlib import Path
from typing import Callable, Optional, Tuple, List, Dict
from dataclasses import dataclass
from enum import Enum

# 在非樹莓派環境下提供模擬模式
try:
    from pyfingerprint.pyfingerprint import PyFingerprint
    SIMULATION_MODE = False
except ImportError:
    SIMULATION_MODE = True
    print("[模擬模式] pyfingerprint 不可用，使用模擬模式")

import sys
sys.path.append('..')
from utils.logger import get_logger


class FingerprintResult(Enum):
    """指紋辨識結果"""
    SUCCESS = "success"
    NOT_FOUND = "not_found"
    ERROR = "error"
    TIMEOUT = "timeout"
    NO_FINGER = "no_finger"


@dataclass
class FingerprintUser:
    """指紋使用者資料"""
    id: int
    name: str
    company_id: int
    position_id: int  # 指紋在模組中的位置
    created_at: str


class FingerprintManager:
    """
    指紋辨識管理類別

    支援 R307/R503 光學指紋模組，透過 UART 通訊
    """

    def __init__(
        self,
        port: str = "/dev/ttyS0",
        baudrate: int = 57600,
        database_path: str = "fingerprints.db"
    ):
        """
        初始化指紋管理器

        Args:
            port: 串口路徑
            baudrate: 鮑率
            database_path: 資料庫檔案路徑
        """
        self.logger = get_logger()
        self.port = port
        self.baudrate = baudrate
        self.database_path = database_path

        self._sensor: Optional[PyFingerprint] = None
        self._is_scanning = False
        self._scan_thread: Optional[threading.Thread] = None
        self._stop_scan = threading.Event()

        # 回調函數
        self._on_finger_detected: Optional[Callable[[FingerprintUser], None]] = None
        self._on_unknown_finger: Optional[Callable] = None
        self._on_scan_error: Optional[Callable[[str], None]] = None

        # 初始化
        self._init_database()
        self._connect_sensor()

    def _init_database(self):
        """初始化資料庫"""
        try:
            db_path = Path(self.database_path)
            db_path.parent.mkdir(parents=True, exist_ok=True)

            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()

            # 建立使用者表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS fingerprint_users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    company_id INTEGER NOT NULL,
                    position_id INTEGER UNIQUE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 建立存取記錄表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS access_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    user_name TEXT,
                    result TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES fingerprint_users(id)
                )
            """)

            conn.commit()
            conn.close()

            self.logger.info(f"資料庫已初始化: {self.database_path}")

        except Exception as e:
            self.logger.error(f"資料庫初始化失敗: {e}")
            raise

    def _connect_sensor(self):
        """連接指紋感測器"""
        if SIMULATION_MODE:
            self.logger.info("[模擬] 指紋感測器已連接")
            return

        try:
            self._sensor = PyFingerprint(
                self.port,
                self.baudrate,
                0xFFFFFFFF,  # 地址
                0x00000000   # 密碼
            )

            if not self._sensor.verifyPassword():
                raise ValueError("指紋模組密碼驗證失敗")

            self.logger.info("指紋感測器已連接")
            self.logger.info(f"  模組容量: {self._sensor.getStorageCapacity()} 指紋")
            self.logger.info(f"  已儲存數量: {self._sensor.getTemplateCount()} 指紋")

        except Exception as e:
            self.logger.error(f"指紋感測器連接失敗: {e}")
            self._sensor = None

    @property
    def is_connected(self) -> bool:
        """感測器是否已連接"""
        if SIMULATION_MODE:
            return True
        return self._sensor is not None

    @property
    def template_count(self) -> int:
        """已儲存的指紋數量"""
        if SIMULATION_MODE:
            return self._get_user_count()
        if self._sensor:
            return self._sensor.getTemplateCount()
        return 0

    @property
    def storage_capacity(self) -> int:
        """指紋儲存容量"""
        if SIMULATION_MODE:
            return 150
        if self._sensor:
            return self._sensor.getStorageCapacity()
        return 0

    def _get_user_count(self) -> int:
        """從資料庫取得使用者數量"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM fingerprint_users")
            count = cursor.fetchone()[0]
            conn.close()
            return count
        except:
            return 0

    def enroll_fingerprint(
        self,
        name: str,
        company_id: int,
        timeout: float = 30.0
    ) -> Tuple[bool, str, Optional[int]]:
        """
        登錄新指紋

        Args:
            name: 使用者名稱
            company_id: 所屬公司 ID
            timeout: 等待手指的超時時間

        Returns:
            Tuple[成功與否, 訊息, 位置ID]
        """
        self.logger.info(f"開始登錄指紋: {name} (公司 {company_id})")

        if SIMULATION_MODE:
            # 模擬模式：直接儲存到資料庫
            position_id = self._get_user_count()
            try:
                conn = sqlite3.connect(self.database_path)
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO fingerprint_users (name, company_id, position_id) VALUES (?, ?, ?)",
                    (name, company_id, position_id)
                )
                conn.commit()
                conn.close()
                self.logger.info(f"[模擬] 指紋已登錄: {name}, 位置 {position_id}")
                return True, "指紋登錄成功", position_id
            except Exception as e:
                return False, f"資料庫錯誤: {e}", None

        if not self._sensor:
            return False, "指紋感測器未連接", None

        try:
            # 第一次採集
            self.logger.info("請將手指放在感測器上...")
            start_time = time.time()

            while not self._sensor.readImage():
                if time.time() - start_time > timeout:
                    return False, "等待超時", None
                time.sleep(0.1)

            self._sensor.convertImage(0x01)

            # 第二次採集
            self.logger.info("請移開手指後再次放上...")
            time.sleep(1)

            start_time = time.time()
            while not self._sensor.readImage():
                if time.time() - start_time > timeout:
                    return False, "等待超時", None
                time.sleep(0.1)

            self._sensor.convertImage(0x02)

            # 比對兩次採集
            if self._sensor.compareCharacteristics() == 0:
                return False, "兩次指紋不一致，請重試", None

            # 建立模板並儲存
            self._sensor.createTemplate()
            position_id = self._sensor.storeTemplate()

            # 儲存到資料庫
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO fingerprint_users (name, company_id, position_id) VALUES (?, ?, ?)",
                (name, company_id, position_id)
            )
            conn.commit()
            conn.close()

            self.logger.info(f"指紋登錄成功: {name}, 位置 {position_id}")
            return True, "指紋登錄成功", position_id

        except Exception as e:
            self.logger.error(f"指紋登錄失敗: {e}")
            return False, f"登錄失敗: {e}", None

    def search_fingerprint(self, timeout: float = 5.0) -> Tuple[FingerprintResult, Optional[FingerprintUser]]:
        """
        搜尋指紋

        Args:
            timeout: 等待手指的超時時間

        Returns:
            Tuple[結果狀態, 使用者資料]
        """
        if SIMULATION_MODE:
            # 模擬模式：假裝找到第一個使用者
            users = self.get_all_users()
            if users:
                self.logger.info(f"[模擬] 找到使用者: {users[0].name}")
                return FingerprintResult.SUCCESS, users[0]
            return FingerprintResult.NOT_FOUND, None

        if not self._sensor:
            return FingerprintResult.ERROR, None

        try:
            # 等待手指
            start_time = time.time()
            while not self._sensor.readImage():
                if time.time() - start_time > timeout:
                    return FingerprintResult.TIMEOUT, None
                time.sleep(0.05)

            # 轉換圖像
            self._sensor.convertImage(0x01)

            # 搜尋指紋
            result = self._sensor.searchTemplate()
            position_id = result[0]

            if position_id == -1:
                self._log_access(None, None, "unknown")
                return FingerprintResult.NOT_FOUND, None

            # 從資料庫查詢使用者
            user = self._get_user_by_position(position_id)
            if user:
                self._log_access(user.id, user.name, "success")
                return FingerprintResult.SUCCESS, user
            else:
                return FingerprintResult.NOT_FOUND, None

        except Exception as e:
            self.logger.error(f"指紋搜尋失敗: {e}")
            return FingerprintResult.ERROR, None

    def _get_user_by_position(self, position_id: int) -> Optional[FingerprintUser]:
        """根據位置 ID 取得使用者"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, name, company_id, position_id, created_at FROM fingerprint_users WHERE position_id = ?",
                (position_id,)
            )
            row = cursor.fetchone()
            conn.close()

            if row:
                return FingerprintUser(
                    id=row[0],
                    name=row[1],
                    company_id=row[2],
                    position_id=row[3],
                    created_at=row[4]
                )
            return None

        except Exception as e:
            self.logger.error(f"查詢使用者失敗: {e}")
            return None

    def get_all_users(self) -> List[FingerprintUser]:
        """取得所有使用者"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, name, company_id, position_id, created_at FROM fingerprint_users ORDER BY id"
            )
            rows = cursor.fetchall()
            conn.close()

            return [
                FingerprintUser(
                    id=row[0],
                    name=row[1],
                    company_id=row[2],
                    position_id=row[3],
                    created_at=row[4]
                )
                for row in rows
            ]

        except Exception as e:
            self.logger.error(f"取得使用者列表失敗: {e}")
            return []

    def delete_fingerprint(self, user_id: int) -> bool:
        """
        刪除指紋

        Args:
            user_id: 使用者 ID

        Returns:
            bool: 是否成功刪除
        """
        try:
            # 取得使用者資料
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT position_id FROM fingerprint_users WHERE id = ?",
                (user_id,)
            )
            row = cursor.fetchone()

            if not row:
                conn.close()
                return False

            position_id = row[0]

            # 從感測器刪除
            if not SIMULATION_MODE and self._sensor:
                self._sensor.deleteTemplate(position_id)

            # 從資料庫刪除
            cursor.execute("DELETE FROM fingerprint_users WHERE id = ?", (user_id,))
            conn.commit()
            conn.close()

            self.logger.info(f"指紋已刪除: 使用者 {user_id}, 位置 {position_id}")
            return True

        except Exception as e:
            self.logger.error(f"刪除指紋失敗: {e}")
            return False

    def _log_access(self, user_id: Optional[int], user_name: Optional[str], result: str):
        """記錄存取日誌"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO access_logs (user_id, user_name, result) VALUES (?, ?, ?)",
                (user_id, user_name, result)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            self.logger.error(f"記錄存取日誌失敗: {e}")

    def start_continuous_scan(self, scan_interval: float = 0.5):
        """
        開始連續掃描模式

        Args:
            scan_interval: 掃描間隔 (秒)
        """
        if self._is_scanning:
            return

        self._is_scanning = True
        self._stop_scan.clear()

        def scan_loop():
            while not self._stop_scan.is_set():
                result, user = self.search_fingerprint(timeout=1.0)

                if result == FingerprintResult.SUCCESS and user:
                    if self._on_finger_detected:
                        self._on_finger_detected(user)

                elif result == FingerprintResult.NOT_FOUND:
                    if self._on_unknown_finger:
                        self._on_unknown_finger()

                elif result == FingerprintResult.ERROR:
                    if self._on_scan_error:
                        self._on_scan_error("掃描錯誤")

                time.sleep(scan_interval)

        self._scan_thread = threading.Thread(target=scan_loop, daemon=True)
        self._scan_thread.start()
        self.logger.info("指紋連續掃描已啟動")

    def stop_continuous_scan(self):
        """停止連續掃描"""
        if not self._is_scanning:
            return

        self._stop_scan.set()
        if self._scan_thread:
            self._scan_thread.join(timeout=2.0)
        self._is_scanning = False
        self.logger.info("指紋連續掃描已停止")

    def set_on_finger_detected(self, callback: Callable[[FingerprintUser], None]):
        """設定識別成功時的回調"""
        self._on_finger_detected = callback

    def set_on_unknown_finger(self, callback: Callable):
        """設定未知指紋的回調"""
        self._on_unknown_finger = callback

    def set_on_scan_error(self, callback: Callable[[str], None]):
        """設定掃描錯誤的回調"""
        self._on_scan_error = callback

    def cleanup(self):
        """清理資源"""
        self.stop_continuous_scan()
        self.logger.info("指紋管理器資源已清理")


# =============================================================================
# 測試程式
# =============================================================================
if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.logger import setup_logger

    # 初始化日誌
    setup_logger(log_level="DEBUG")

    print("=" * 50)
    print("指紋辨識模組測試")
    print("=" * 50)

    # 建立指紋管理器
    fp_manager = FingerprintManager(
        port="/dev/ttyS0",
        database_path="test_fingerprints.db"
    )

    print(f"\n感測器狀態: {'已連接' if fp_manager.is_connected else '未連接'}")
    print(f"儲存容量: {fp_manager.storage_capacity}")
    print(f"已儲存數量: {fp_manager.template_count}")

    # 測試登錄
    print("\n測試 1: 登錄指紋")
    success, msg, pos = fp_manager.enroll_fingerprint("測試用戶", company_id=1)
    print(f"結果: {msg}")

    # 列出所有使用者
    print("\n所有使用者:")
    for user in fp_manager.get_all_users():
        print(f"  - {user.name} (公司 {user.company_id})")

    # 測試連續掃描
    print("\n測試 2: 連續掃描 (5秒)")
    fp_manager.set_on_finger_detected(
        lambda u: print(f">>> 識別成功: {u.name}")
    )
    fp_manager.set_on_unknown_finger(
        lambda: print(">>> 未知指紋")
    )
    fp_manager.start_continuous_scan()
    time.sleep(5)
    fp_manager.stop_continuous_scan()

    print("\n測試完成!")
    fp_manager.cleanup()
