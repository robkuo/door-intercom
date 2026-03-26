# -*- coding: utf-8 -*-
"""
門鎖控制模組 - 透過 GPIO 控制繼電器開門
"""

import threading
import time
from typing import Callable, Optional

# 在非樹莓派環境下提供模擬模式
try:
    import RPi.GPIO as GPIO
    SIMULATION_MODE = False
except ImportError:
    SIMULATION_MODE = True
    print("[模擬模式] RPi.GPIO 不可用，使用模擬模式")

import sys
sys.path.append('..')
from utils.logger import get_logger


class DoorLock:
    """
    門鎖控制類別

    使用 GPIO 控制繼電器來開啟電磁鎖/電子鎖
    """

    def __init__(
        self,
        relay_pin: int = 17,
        unlock_duration: float = 5.0,
        active_low: bool = True
    ):
        """
        初始化門鎖控制器

        Args:
            relay_pin: GPIO 腳位編號 (BCM 編號)
            unlock_duration: 開門持續時間 (秒)
            active_low: True 表示低電位觸發繼電器
        """
        self.logger = get_logger()
        self.relay_pin = relay_pin
        self.unlock_duration = unlock_duration
        self.active_low = active_low
        self._is_unlocked = False
        self._lock = threading.Lock()
        self._unlock_timer: Optional[threading.Timer] = None
        self._on_unlock_callback: Optional[Callable] = None
        self._on_lock_callback: Optional[Callable] = None

        self._setup_gpio()

    def _setup_gpio(self):
        """設定 GPIO"""
        if SIMULATION_MODE:
            self.logger.info(f"[模擬] GPIO {self.relay_pin} 已設定為輸出")
            return

        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            GPIO.setup(self.relay_pin, GPIO.OUT)

            # 初始狀態為上鎖 (繼電器關閉)
            if self.active_low:
                GPIO.output(self.relay_pin, GPIO.HIGH)
            else:
                GPIO.output(self.relay_pin, GPIO.LOW)

            self.logger.info(f"GPIO {self.relay_pin} 已初始化為門鎖控制腳位")

        except Exception as e:
            self.logger.error(f"GPIO 初始化失敗: {e}")
            raise

    def unlock(self, duration: float = None) -> bool:
        """
        開門 (解鎖)

        Args:
            duration: 開門持續時間，若為 None 則使用預設值

        Returns:
            bool: 是否成功開門
        """
        with self._lock:
            if self._is_unlocked:
                self.logger.info("門已經是開啟狀態")
                return True

            duration = duration or self.unlock_duration

            try:
                if SIMULATION_MODE:
                    self.logger.info(f"[模擬] 門已開啟，將在 {duration} 秒後自動上鎖")
                else:
                    # 觸發繼電器開門
                    if self.active_low:
                        GPIO.output(self.relay_pin, GPIO.LOW)
                    else:
                        GPIO.output(self.relay_pin, GPIO.HIGH)
                    self.logger.info(f"門已開啟，將在 {duration} 秒後自動上鎖")

                self._is_unlocked = True

                # 執行回調
                if self._on_unlock_callback:
                    try:
                        self._on_unlock_callback()
                    except Exception as e:
                        self.logger.error(f"開門回調執行失敗: {e}")

                # 設定自動上鎖計時器
                self._cancel_timer()
                self._unlock_timer = threading.Timer(duration, self._auto_lock)
                self._unlock_timer.start()

                return True

            except Exception as e:
                self.logger.error(f"開門失敗: {e}")
                return False

    def lock(self) -> bool:
        """
        關門 (上鎖)

        Returns:
            bool: 是否成功上鎖
        """
        with self._lock:
            self._cancel_timer()
            return self._do_lock()

    def _do_lock(self) -> bool:
        """執行上鎖動作"""
        try:
            if SIMULATION_MODE:
                self.logger.info("[模擬] 門已上鎖")
            else:
                if self.active_low:
                    GPIO.output(self.relay_pin, GPIO.HIGH)
                else:
                    GPIO.output(self.relay_pin, GPIO.LOW)
                self.logger.info("門已上鎖")

            self._is_unlocked = False

            # 執行回調
            if self._on_lock_callback:
                try:
                    self._on_lock_callback()
                except Exception as e:
                    self.logger.error(f"上鎖回調執行失敗: {e}")

            return True

        except Exception as e:
            self.logger.error(f"上鎖失敗: {e}")
            return False

    def _auto_lock(self):
        """自動上鎖 (計時器回調)"""
        with self._lock:
            if self._is_unlocked:
                self.logger.info("自動上鎖")
                self._do_lock()

    def _cancel_timer(self):
        """取消計時器"""
        if self._unlock_timer:
            self._unlock_timer.cancel()
            self._unlock_timer = None

    @property
    def is_unlocked(self) -> bool:
        """門是否為開啟狀態"""
        return self._is_unlocked

    @property
    def is_locked(self) -> bool:
        """門是否為上鎖狀態"""
        return not self._is_unlocked

    def set_on_unlock(self, callback: Callable):
        """設定開門時的回調函數"""
        self._on_unlock_callback = callback

    def set_on_lock(self, callback: Callable):
        """設定上鎖時的回調函數"""
        self._on_lock_callback = callback

    def cleanup(self):
        """清理資源"""
        self._cancel_timer()

        if not SIMULATION_MODE:
            try:
                # 確保門是上鎖的
                if self.active_low:
                    GPIO.output(self.relay_pin, GPIO.HIGH)
                else:
                    GPIO.output(self.relay_pin, GPIO.LOW)
                GPIO.cleanup(self.relay_pin)
                self.logger.info("GPIO 資源已清理")
            except Exception as e:
                self.logger.error(f"GPIO 清理失敗: {e}")

    def __del__(self):
        """解構函數"""
        self.cleanup()


# =============================================================================
# 測試程式
# =============================================================================
if __name__ == "__main__":
    from utils.logger import setup_logger

    # 初始化日誌
    setup_logger(log_level="DEBUG")

    print("=" * 50)
    print("門鎖控制模組測試")
    print("=" * 50)

    # 建立門鎖控制器
    door = DoorLock(relay_pin=17, unlock_duration=3.0)

    # 設定回調
    door.set_on_unlock(lambda: print(">>> 回調: 門已開啟!"))
    door.set_on_lock(lambda: print(">>> 回調: 門已上鎖!"))

    try:
        print("\n測試 1: 開門 3 秒後自動上鎖")
        door.unlock()
        time.sleep(5)

        print("\n測試 2: 開門後手動上鎖")
        door.unlock()
        time.sleep(1)
        door.lock()

        print("\n測試完成!")

    finally:
        door.cleanup()
