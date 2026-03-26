# -*- coding: utf-8 -*-
"""
人臉辨識模組 - 使用 OpenCV LBPH (不需要 dlib)
"""

import threading
import time
import sqlite3
import numpy as np
import cv2
from pathlib import Path
from typing import Callable, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

# 在非樹莓派環境下提供模擬模式
try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False
    print("[模擬模式] picamera2 不可用")

import sys
sys.path.append('..')
from utils.logger import get_logger


class FaceResult(Enum):
    """人臉辨識結果"""
    SUCCESS = "success"
    NOT_FOUND = "not_found"
    NO_FACE = "no_face"
    UNKNOWN = "unknown"
    ERROR = "error"


@dataclass
class FaceUser:
    """人臉使用者資料"""
    id: int
    name: str
    company_id: int
    created_at: str = ""
    label: int = 0


class FaceManager:
    """
    人臉辨識管理類別

    使用 OpenCV LBPH 人臉辨識（不需要 dlib）
    支援 Raspberry Pi Camera Module 3 或 USB 攝影機
    """

    def __init__(
        self,
        database_path: str = "faces.db",
        confidence_threshold: float = 80.0,
        detection_interval: float = 0.5
    ):
        """
        初始化人臉管理器

        Args:
            database_path: 資料庫檔案路徑
            confidence_threshold: 信心閾值（越低越嚴格，建議 50-100）
            detection_interval: 偵測間隔 (秒)
        """
        self.logger = get_logger()
        self.database_path = database_path
        self.confidence_threshold = confidence_threshold
        self.detection_interval = detection_interval

        # 人臉偵測器 (Haar Cascade)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        # LBPH 人臉辨識器
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()

        # 攝影機
        self._camera = None
        self._camera_lock = threading.Lock()

        # 已知使用者
        self._known_users: dict = {}  # label -> FaceUser
        self._known_user_ids: List[int] = []

        # 連續掃描
        self._is_scanning = False
        self._scan_thread: Optional[threading.Thread] = None
        self._stop_scan = threading.Event()

        # 回調函數
        self._on_face_detected: Optional[Callable[[FaceUser], None]] = None
        self._on_unknown_face: Optional[Callable] = None
        self._on_no_face: Optional[Callable] = None

        # 初始化
        self._init_database()
        self._init_camera()
        self._load_known_faces()

    def _init_database(self):
        """初始化資料庫"""
        try:
            db_path = Path(self.database_path)
            db_path.parent.mkdir(parents=True, exist_ok=True)

            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()

            # 建立使用者表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS face_users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    company_id INTEGER NOT NULL,
                    label INTEGER UNIQUE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    user_id INTEGER
                )
            """)

            # 建立人臉圖片表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS face_images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    image BLOB NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES face_users(id)
                )
            """)

            # 建立存取記錄表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS access_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    user_name TEXT,
                    result TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.commit()
            conn.close()

            self.logger.info(f"資料庫已初始化: {self.database_path}")

        except Exception as e:
            self.logger.error(f"資料庫初始化失敗: {e}")
            raise

    def _init_camera(self):
        """初始化相機"""
        # 相機在需要時才啟動，這裡只記錄
        self.logger.info("相機將在首次使用時初始化")

    def _get_camera(self):
        """取得攝影機實例"""
        if self._camera is None:
            try:
                # 嘗試使用 Picamera2
                if PICAMERA_AVAILABLE:
                    self._camera = Picamera2()
                    config = self._camera.create_still_configuration(
                        main={"size": (640, 480), "format": "RGB888"}
                    )
                    self._camera.configure(config)
                    self._camera.start()
                    time.sleep(0.5)
                    self.logger.info("使用 Picamera2")
                else:
                    raise ImportError("Picamera2 不可用")
            except Exception as e:
                self.logger.warning(f"Picamera2 不可用: {e}")
                # 退回使用 USB 攝影機（依序嘗試 /dev/video0~2）
                opened = False
                for idx in range(3):
                    cap = cv2.VideoCapture(idx)
                    if cap.isOpened():
                        self._camera = cap
                        self.logger.info(f"使用 USB 攝影機 /dev/video{idx}")
                        opened = True
                        break
                    cap.release()
                if not opened:
                    self._camera = None
                    raise RuntimeError("無法開啟攝影機（/dev/video0~2 均失敗）")
        return self._camera

    def _load_known_faces(self):
        """載入已知人臉並訓練辨識器"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()

            cursor.execute('SELECT id, name, company_id, label, created_at FROM face_users')
            users = cursor.fetchall()

            faces = []
            labels = []

            self._known_users.clear()
            self._known_user_ids.clear()

            for user_id, name, company_id, label, created_at in users:
                self._known_users[label] = FaceUser(
                    id=user_id,
                    name=name,
                    company_id=company_id,
                    label=label,
                    created_at=created_at or ""
                )
                self._known_user_ids.append(user_id)

                cursor.execute('SELECT image FROM face_images WHERE user_id = ?', (user_id,))
                images = cursor.fetchall()

                for (image_blob,) in images:
                    img_array = np.frombuffer(image_blob, dtype=np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        faces.append(img)
                        labels.append(label)

            conn.close()

            # 訓練辨識器
            if faces and labels:
                self.recognizer.train(faces, np.array(labels))
                self.logger.info(f"已載入並訓練 {len(self._known_users)} 位使用者的人臉資料")
            else:
                self.logger.info("尚無已登錄的人臉")

        except Exception as e:
            self.logger.error(f"載入人臉資料失敗: {e}")

    @property
    def is_camera_ready(self) -> bool:
        """相機是否就緒"""
        try:
            self._get_camera()
            return True
        except:
            return False

    @property
    def user_count(self) -> int:
        """已登錄的人臉數量"""
        return len(self._known_user_ids)

    def capture_frame(self) -> Optional[np.ndarray]:
        """擷取一張影像"""
        with self._camera_lock:
            try:
                camera = self._get_camera()

                # Picamera2
                if hasattr(camera, 'capture_array'):
                    frame = camera.capture_array()
                    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                # OpenCV VideoCapture
                else:
                    ret, frame = camera.read()
                    return frame if ret else None

            except Exception as e:
                self.logger.error(f"擷取影像失敗: {e}")
                return None

    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """偵測人臉位置"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 100)
        )
        return faces.tolist() if len(faces) > 0 else []

    def enroll_face(
        self,
        name: str,
        company_id: int,
        frame: Optional[np.ndarray] = None,
        num_samples: int = 5
    ) -> Tuple[bool, str]:
        """
        登錄新人臉

        Args:
            name: 使用者名稱
            company_id: 所屬公司 ID
            frame: 影像 (若為 None 則自動擷取多張)
            num_samples: 要擷取的樣本數

        Returns:
            Tuple[成功與否, 訊息]
        """
        self.logger.info(f"開始登錄人臉: {name} (公司 {company_id})")

        try:
            samples = []

            if frame is not None:
                # 使用提供的單張影像
                faces = self.detect_faces(frame)
                if len(faces) == 0:
                    return False, "未偵測到人臉，請正對相機"
                if len(faces) > 1:
                    return False, "偵測到多張人臉，請確保只有一人"

                x, y, w, h = faces[0]
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face_img = gray[y:y+h, x:x+w]
                face_img = cv2.resize(face_img, (200, 200))
                samples.append(face_img)
            else:
                # 自動擷取多張
                self.logger.info(f"開始擷取 {num_samples} 張人臉照片...")

                for i in range(num_samples):
                    captured_frame = self.capture_frame()
                    if captured_frame is None:
                        return False, "無法擷取影像"

                    faces = self.detect_faces(captured_frame)

                    if len(faces) == 0:
                        return False, "未偵測到人臉，請正對相機"
                    if len(faces) > 1:
                        return False, "偵測到多張人臉，請確保只有一人"

                    x, y, w, h = faces[0]
                    gray = cv2.cvtColor(captured_frame, cv2.COLOR_BGR2GRAY)
                    face_img = gray[y:y+h, x:x+w]
                    face_img = cv2.resize(face_img, (200, 200))

                    samples.append(face_img)
                    self.logger.info(f"已擷取第 {i+1}/{num_samples} 張")
                    time.sleep(0.3)

            # 儲存到資料庫
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()

            # 取得新的 label
            cursor.execute('SELECT MAX(label) FROM face_users')
            result = cursor.fetchone()
            new_label = (result[0] or 0) + 1

            # 新增使用者
            cursor.execute(
                'INSERT INTO face_users (name, company_id, label) VALUES (?, ?, ?)',
                (name, company_id, new_label)
            )
            user_id = cursor.lastrowid

            # 儲存人臉圖片
            for face_img in samples:
                _, buffer = cv2.imencode('.png', face_img)
                cursor.execute(
                    'INSERT INTO face_images (user_id, image) VALUES (?, ?)',
                    (user_id, buffer.tobytes())
                )

            conn.commit()
            conn.close()

            # 重新載入並訓練
            self._load_known_faces()

            self.logger.info(f"人臉登錄成功: {name} (ID: {user_id})")
            return True, f"成功登錄 {name}"

        except Exception as e:
            self.logger.error(f"人臉登錄失敗: {e}")
            return False, f"登錄失敗: {str(e)}"

    def enroll_face_from_file(
        self,
        image_path: str,
        name: str,
        company_id: int
    ) -> Tuple[bool, str]:
        """
        從圖片檔案登錄人臉（用於網頁管理介面）

        Args:
            image_path: 照片檔案路徑
            name: 使用者名稱
            company_id: 所屬公司 ID

        Returns:
            Tuple[成功與否, 訊息]
        """
        self.logger.info(f"從檔案登錄人臉: {name} (公司 {company_id}) - {image_path}")

        try:
            # 讀取圖片
            frame = cv2.imread(image_path)
            if frame is None:
                return False, "無法讀取照片檔案"

            # 偵測人臉
            faces = self.detect_faces(frame)
            if len(faces) == 0:
                return False, "照片中未偵測到人臉"
            if len(faces) > 1:
                return False, "照片中偵測到多張人臉，請使用只有一人的照片"

            # 提取人臉區域
            x, y, w, h = faces[0]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (200, 200))

            # 儲存到資料庫
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()

            # 取得新的 label
            cursor.execute('SELECT MAX(label) FROM face_users')
            result = cursor.fetchone()
            new_label = (result[0] or 0) + 1

            # 新增使用者
            cursor.execute(
                'INSERT INTO face_users (name, company_id, label) VALUES (?, ?, ?)',
                (name, company_id, new_label)
            )
            user_id = cursor.lastrowid

            # 儲存人臉圖片
            _, buffer = cv2.imencode('.png', face_img)
            cursor.execute(
                'INSERT INTO face_images (user_id, image) VALUES (?, ?)',
                (user_id, buffer.tobytes())
            )

            conn.commit()
            conn.close()

            # 重新載入並訓練
            self._load_known_faces()

            self.logger.info(f"人臉登錄成功: {name} (ID: {user_id})")
            return True, f"成功登錄 {name}"

        except Exception as e:
            self.logger.error(f"人臉登錄失敗: {e}")
            return False, f"登錄失敗: {str(e)}"

    def recognize_face(self, frame: Optional[np.ndarray] = None) -> Tuple[FaceResult, Optional[FaceUser]]:
        """
        辨識人臉

        Args:
            frame: 影像 (若為 None 則自動擷取)

        Returns:
            Tuple[結果狀態, 使用者資料]
        """
        try:
            if frame is None:
                frame = self.capture_frame()
                if frame is None:
                    return FaceResult.ERROR, None

            faces = self.detect_faces(frame)

            if len(faces) == 0:
                return FaceResult.NO_FACE, None

            if not self._known_users:
                return FaceResult.UNKNOWN, None

            x, y, w, h = faces[0]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (200, 200))

            # 辨識
            label, confidence = self.recognizer.predict(face_img)

            self.logger.debug(f"辨識結果: label={label}, confidence={confidence}")

            # confidence 越低越相似
            if confidence < self.confidence_threshold:
                if label in self._known_users:
                    user = self._known_users[label]
                    self._log_access(user.id, user.name, "success")
                    return FaceResult.SUCCESS, user

            self._log_access(None, None, "unknown")
            return FaceResult.NOT_FOUND, None

        except Exception as e:
            self.logger.error(f"人臉辨識失敗: {e}")
            return FaceResult.ERROR, None

    def _get_user_by_id(self, user_id: int) -> Optional[FaceUser]:
        """根據 ID 取得使用者"""
        for user in self._known_users.values():
            if user.id == user_id:
                return user
        return None

    def get_all_users(self) -> List[FaceUser]:
        """取得所有使用者"""
        return list(self._known_users.values())

    def delete_face(self, user_id: int) -> bool:
        """刪除人臉"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()

            cursor.execute('DELETE FROM face_images WHERE user_id = ?', (user_id,))
            cursor.execute('DELETE FROM face_users WHERE id = ?', (user_id,))

            conn.commit()
            conn.close()

            # 重新載入
            self._load_known_faces()

            self.logger.info(f"人臉已刪除: 使用者 {user_id}")
            return True

        except Exception as e:
            self.logger.error(f"刪除人臉失敗: {e}")
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

    def start_continuous_scan(self, callback=None):
        """開始連續掃描模式"""
        if self._is_scanning:
            return

        self._is_scanning = True
        self._stop_scan.clear()

        def scan_loop():
            while not self._stop_scan.is_set():
                result, user = self.recognize_face()

                # 使用傳入的 callback 或預設的
                if callback:
                    callback(result, user)
                elif result == FaceResult.SUCCESS and user:
                    if self._on_face_detected:
                        self._on_face_detected(user)
                    time.sleep(3)
                elif result == FaceResult.NOT_FOUND:
                    if self._on_unknown_face:
                        self._on_unknown_face()
                    time.sleep(1)
                elif result == FaceResult.NO_FACE:
                    if self._on_no_face:
                        self._on_no_face()

                time.sleep(self.detection_interval)

        self._scan_thread = threading.Thread(target=scan_loop, daemon=True)
        self._scan_thread.start()
        self.logger.info("人臉連續掃描已啟動")

    def stop_continuous_scan(self):
        """停止連續掃描"""
        if not self._is_scanning:
            return

        self._stop_scan.set()
        if self._scan_thread:
            self._scan_thread.join(timeout=2.0)
        self._is_scanning = False
        self.logger.info("人臉連續掃描已停止")

    def set_on_face_detected(self, callback: Callable[[FaceUser], None]):
        """設定識別成功時的回調"""
        self._on_face_detected = callback

    def set_on_unknown_face(self, callback: Callable):
        """設定未知人臉的回調"""
        self._on_unknown_face = callback

    def set_on_no_face(self, callback: Callable):
        """設定無人臉的回調"""
        self._on_no_face = callback

    def cleanup(self):
        """清理資源"""
        self.stop_continuous_scan()

        with self._camera_lock:
            if self._camera:
                if hasattr(self._camera, 'stop'):
                    try:
                        self._camera.stop()
                        self._camera.close()
                    except:
                        pass
                elif hasattr(self._camera, 'release'):
                    self._camera.release()
                self._camera = None

        self.logger.info("人臉管理器資源已清理")

    # 別名方法，保持向後相容
    def close(self):
        """關閉資源（cleanup 的別名）"""
        self.cleanup()


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
    print("人臉辨識模組測試 (OpenCV LBPH)")
    print("=" * 50)

    # 建立人臉管理器
    face_manager = FaceManager(
        database_path="test_faces.db",
        confidence_threshold=80.0
    )

    print(f"\n已登錄人臉數: {face_manager.user_count}")

    # 列出所有使用者
    print("\n所有使用者:")
    for user in face_manager.get_all_users():
        print(f"  - {user.name} (公司 {user.company_id})")

    # 測試辨識
    print("\n測試辨識人臉...")
    result, user = face_manager.recognize_face()
    if user:
        print(f"識別結果: {user.name}")
    else:
        print(f"識別結果: {result.value}")

    print("\n測試完成!")
    face_manager.cleanup()
