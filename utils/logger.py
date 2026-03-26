# -*- coding: utf-8 -*-
"""
日誌記錄模組
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

_logger = None


def setup_logger(log_file: str = None, log_level: str = "INFO") -> logging.Logger:
    """
    設定並初始化日誌記錄器

    Args:
        log_file: 日誌檔案路徑，若為 None 則只輸出到終端
        log_level: 日誌等級 (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        logging.Logger: 設定好的日誌記錄器
    """
    global _logger

    if _logger is not None:
        return _logger

    # 建立 logger
    _logger = logging.getLogger("intercom_system")
    _logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # 日誌格式
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(module)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 終端輸出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    _logger.addHandler(console_handler)

    # 檔案輸出
    if log_file:
        try:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            _logger.addHandler(file_handler)
        except Exception as e:
            _logger.warning(f"無法建立日誌檔案 {log_file}: {e}")

    _logger.info("=" * 60)
    _logger.info("門口對講機系統 - 日誌記錄器已啟動")
    _logger.info(f"啟動時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    _logger.info("=" * 60)

    return _logger


def get_logger() -> logging.Logger:
    """
    取得日誌記錄器實例

    Returns:
        logging.Logger: 日誌記錄器
    """
    global _logger

    if _logger is None:
        return setup_logger()

    return _logger
