#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
門口對講機系統 - 網頁管理介面

Flask Web 伺服器，提供：
- 人臉管理（上傳照片登錄）
- NFC 卡片管理
- 系統設定
- 存取日誌
"""

import os
import sys
import re
import json
import sqlite3
import hashlib
import secrets
import subprocess
from datetime import datetime, timedelta
from functools import wraps

from flask import (
    Flask, render_template, request, jsonify, redirect,
    url_for, session, flash, send_file
)
from werkzeug.utils import secure_filename

# 加入父目錄到模組路徑
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    COMPANIES, DATABASE_PATH, NFC_DATABASE_PATH,
    NFC_ENABLED, WEB_ADMIN_PORT, WEB_SECRET_KEY,
    WEB_UPLOAD_FOLDER, WEB_ADMIN_DB_PATH
)


def create_app():
    """建立 Flask 應用程式"""
    app = Flask(__name__)

    # 設定
    app.secret_key = WEB_SECRET_KEY
    app.config['UPLOAD_FOLDER'] = WEB_UPLOAD_FOLDER
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

    # 確保上傳目錄存在
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # 初始化管理員資料庫
    init_admin_db()

    # 註冊路由
    register_routes(app)

    return app


def init_admin_db():
    """初始化管理員資料庫"""
    conn = sqlite3.connect(WEB_ADMIN_DB_PATH)
    cursor = conn.cursor()

    # 建立管理員帳號表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS admin_users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # 建立存取日誌表（整合所有開門記錄）
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS access_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            access_type TEXT NOT NULL,
            user_name TEXT,
            user_id INTEGER,
            result TEXT NOT NULL,
            details TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # 建立公司資料表（動態公司名稱）
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS companies (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            extension TEXT NOT NULL,
            floor TEXT
        )
    ''')

    # 建立密碼開門資料表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS door_passwords (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            password TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            company_id INTEGER DEFAULT 0,
            active INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_used TIMESTAMP
        )
    ''')

    # 建立密碼錯誤嘗試記錄表（用於鎖定功能）
    # 使用 strftime 確保儲存本地時間
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS password_attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            attempt_time TEXT,
            ip_address TEXT
        )
    ''')

    # 建立統一用戶表（用於關聯多種認證方式）
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            company_id INTEGER DEFAULT 0,
            phone TEXT,
            email TEXT,
            notes TEXT,
            active INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # 為 door_passwords 表新增 user_id 欄位（如果不存在）
    try:
        cursor.execute('ALTER TABLE door_passwords ADD COLUMN user_id INTEGER')
    except sqlite3.OperationalError:
        pass  # 欄位已存在

    # 檢查是否已有管理員帳號，若沒有則建立預設帳號
    cursor.execute('SELECT COUNT(*) FROM admin_users')
    count = cursor.fetchone()[0]

    if count == 0:
        # 建立預設管理員帳號 admin / admin123
        password_hash = hashlib.sha256('admin123'.encode()).hexdigest()
        cursor.execute(
            'INSERT INTO admin_users (username, password_hash) VALUES (?, ?)',
            ('admin', password_hash)
        )
        print("已建立預設管理員帳號: admin / admin123")

    # 檢查是否已有公司資料，若沒有則從 config.py 匯入
    cursor.execute('SELECT COUNT(*) FROM companies')
    company_count = cursor.fetchone()[0]

    if company_count == 0:
        # 從 config.py 匯入預設公司資料
        for company_id, info in COMPANIES.items():
            cursor.execute(
                'INSERT INTO companies (id, name, extension, floor) VALUES (?, ?, ?, ?)',
                (company_id, info['name'], info['extension'], info.get('floor', ''))
            )
        print(f"已匯入 {len(COMPANIES)} 間公司資料")

    conn.commit()
    conn.close()


def get_companies_from_db():
    """從資料庫取得公司資料"""
    try:
        conn = sqlite3.connect(WEB_ADMIN_DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('SELECT id, name, extension, floor FROM companies ORDER BY id')
        rows = cursor.fetchall()
        conn.close()

        companies = {}
        for row in rows:
            companies[row['id']] = {
                'name': row['name'],
                'extension': row['extension'],
                'floor': row['floor'] or ''
            }
        return companies
    except Exception as e:
        print(f"取得公司資料失敗: {e}")
        return COMPANIES  # 回傳 config.py 的預設值


def login_required(f):
    """登入驗證裝飾器"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


def register_routes(app):
    """註冊所有路由"""

    # =========================================================================
    # 登入/登出
    # =========================================================================
    @app.route('/login', methods=['GET', 'POST'])
    def login():
        """登入頁面"""
        if request.method == 'POST':
            username = request.form.get('username', '').strip()
            password = request.form.get('password', '')

            if not username or not password:
                flash('請輸入帳號和密碼', 'error')
                return render_template('login.html')

            # 驗證帳號密碼
            conn = sqlite3.connect(WEB_ADMIN_DB_PATH)
            cursor = conn.cursor()
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            cursor.execute(
                'SELECT id, username FROM admin_users WHERE username = ? AND password_hash = ?',
                (username, password_hash)
            )
            user = cursor.fetchone()
            conn.close()

            if user:
                session['user_id'] = user[0]
                session['username'] = user[1]
                session.permanent = True
                app.permanent_session_lifetime = timedelta(hours=8)
                return redirect(url_for('dashboard'))
            else:
                flash('帳號或密碼錯誤', 'error')

        return render_template('login.html')

    @app.route('/logout')
    def logout():
        """登出"""
        session.clear()
        return redirect(url_for('login'))

    # =========================================================================
    # 儀表板
    # =========================================================================
    @app.route('/')
    @login_required
    def dashboard():
        """儀表板首頁"""
        stats = get_system_stats()
        return render_template('dashboard.html', stats=stats, companies=COMPANIES)

    # =========================================================================
    # 人臉管理
    # =========================================================================
    @app.route('/faces')
    @login_required
    def faces():
        """人臉管理頁面"""
        return render_template('faces.html', companies=COMPANIES)

    @app.route('/api/faces', methods=['GET'])
    @login_required
    def api_get_faces():
        """取得所有已登錄人臉"""
        try:
            conn = sqlite3.connect(DATABASE_PATH)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, name, company_id, created_at
                FROM users
                ORDER BY created_at DESC
            ''')
            users = [dict(row) for row in cursor.fetchall()]
            conn.close()

            # 加入公司名稱
            for user in users:
                company = COMPANIES.get(user['company_id'], {})
                user['company_name'] = company.get('name', '未知')

            return jsonify({'success': True, 'data': users})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})

    @app.route('/api/faces/enroll', methods=['POST'])
    @login_required
    def api_enroll_face():
        """登錄新人臉（上傳照片）"""
        try:
            name = request.form.get('name', '').strip()
            company_id = request.form.get('company_id', type=int)

            if not name:
                return jsonify({'success': False, 'error': '請輸入姓名'})

            if 'photo' not in request.files:
                return jsonify({'success': False, 'error': '請上傳照片'})

            photo = request.files['photo']
            if photo.filename == '':
                return jsonify({'success': False, 'error': '請選擇照片檔案'})

            # 檢查檔案類型
            allowed_extensions = {'png', 'jpg', 'jpeg'}
            ext = photo.filename.rsplit('.', 1)[-1].lower()
            if ext not in allowed_extensions:
                return jsonify({'success': False, 'error': '只支援 PNG, JPG, JPEG 格式'})

            # 儲存照片
            filename = secure_filename(f"{name}_{datetime.now().strftime('%Y%m%d%H%M%S')}.{ext}")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            photo.save(filepath)

            # 呼叫 FaceManager 登錄人臉
            from face.face_manager import FaceManager
            face_manager = FaceManager(database_path=DATABASE_PATH)
            success, msg = face_manager.enroll_face_from_file(filepath, name, company_id or 0)
            face_manager.cleanup()

            # 記錄日誌
            log_access('face_enroll', name, None, 'success' if success else 'failed', msg)

            if success:
                return jsonify({'success': True, 'message': msg})
            else:
                return jsonify({'success': False, 'error': msg})

        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})

    @app.route('/api/faces/<int:user_id>', methods=['DELETE'])
    @login_required
    def api_delete_face(user_id):
        """刪除人臉"""
        try:
            from face.face_manager import FaceManager
            face_manager = FaceManager(database_path=DATABASE_PATH)

            # 取得使用者名稱用於日誌
            users = face_manager.get_all_users()
            user_name = next((u.name for u in users if u.id == user_id), 'unknown')

            success = face_manager.delete_face(user_id)
            face_manager.cleanup()

            # 記錄日誌
            log_access('face_delete', user_name, user_id, 'success' if success else 'failed')

            if success:
                return jsonify({'success': True, 'message': '刪除成功'})
            else:
                return jsonify({'success': False, 'error': '刪除失敗'})

        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})

    # =========================================================================
    # NFC 管理
    # =========================================================================
    @app.route('/nfc')
    @login_required
    def nfc():
        """NFC 管理頁面"""
        return render_template('nfc.html', companies=COMPANIES, nfc_enabled=NFC_ENABLED)

    @app.route('/api/nfc', methods=['GET'])
    @login_required
    def api_get_nfc_cards():
        """取得所有已登錄 NFC 卡片"""
        if not NFC_ENABLED:
            return jsonify({'success': False, 'error': 'NFC 功能未啟用'})

        try:
            conn = sqlite3.connect(NFC_DATABASE_PATH)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, uid, name, company_id, card_type, active, created_at, last_used
                FROM nfc_cards
                ORDER BY created_at DESC
            ''')
            cards = [dict(row) for row in cursor.fetchall()]
            conn.close()

            # 加入公司名稱
            for card in cards:
                company = COMPANIES.get(card['company_id'], {})
                card['company_name'] = company.get('name', '未知')

            return jsonify({'success': True, 'data': cards})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})

    @app.route('/api/nfc/register', methods=['POST'])
    @login_required
    def api_register_nfc():
        """開始登錄 NFC 卡片（等待感應）"""
        if not NFC_ENABLED:
            return jsonify({'success': False, 'error': 'NFC 功能未啟用'})

        try:
            data = request.get_json()
            name = data.get('name', '').strip()
            company_id = data.get('company_id', 0)

            if not name:
                return jsonify({'success': False, 'error': '請輸入持卡人姓名'})

            # 呼叫 NFCManager 登錄卡片
            from nfc.nfc_manager import NFCManager
            nfc_manager = NFCManager(database_path=NFC_DATABASE_PATH)
            success, msg = nfc_manager.register_card(name, company_id, timeout=15.0)
            nfc_manager.cleanup()

            # 記錄日誌
            log_access('nfc_enroll', name, None, 'success' if success else 'failed', msg)

            if success:
                return jsonify({'success': True, 'message': msg})
            else:
                return jsonify({'success': False, 'error': msg})

        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})

    @app.route('/api/nfc/<int:card_id>', methods=['DELETE'])
    @login_required
    def api_delete_nfc(card_id):
        """刪除 NFC 卡片"""
        if not NFC_ENABLED:
            return jsonify({'success': False, 'error': 'NFC 功能未啟用'})

        try:
            from nfc.nfc_manager import NFCManager
            nfc_manager = NFCManager(database_path=NFC_DATABASE_PATH)

            # 取得卡片名稱用於日誌
            cards = nfc_manager.get_all_cards()
            card_name = next((c.name for c in cards if c.id == card_id), 'unknown')

            success = nfc_manager.delete_card(card_id)
            nfc_manager.cleanup()

            # 記錄日誌
            log_access('nfc_delete', card_name, card_id, 'success' if success else 'failed')

            if success:
                return jsonify({'success': True, 'message': '刪除成功'})
            else:
                return jsonify({'success': False, 'error': '刪除失敗'})

        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})

    @app.route('/api/nfc/<int:card_id>/toggle', methods=['PATCH'])
    @login_required
    def api_toggle_nfc(card_id):
        """停用/啟用 NFC 卡片"""
        if not NFC_ENABLED:
            return jsonify({'success': False, 'error': 'NFC 功能未啟用'})

        try:
            data = request.get_json()
            active = data.get('active', True)

            from nfc.nfc_manager import NFCManager
            nfc_manager = NFCManager(database_path=NFC_DATABASE_PATH)
            success = nfc_manager.toggle_card_active(card_id, active)
            nfc_manager.cleanup()

            status = '啟用' if active else '停用'
            if success:
                return jsonify({'success': True, 'message': f'卡片已{status}'})
            else:
                return jsonify({'success': False, 'error': '操作失敗'})

        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})

    # =========================================================================
    # 系統設定
    # =========================================================================
    @app.route('/settings')
    @login_required
    def settings():
        """系統設定頁面"""
        companies = get_companies_from_db()
        return render_template('settings.html', companies=companies)

    @app.route('/api/settings', methods=['GET'])
    @login_required
    def api_get_settings():
        """取得系統設定"""
        try:
            from config import (
                SIP_SERVER, SIP_PORT, SIP_USERNAME, SIP_DOMAIN,
                GPIO_RELAY_PIN, DOOR_UNLOCK_DURATION,
                NFC_ENABLED, NFC_SCAN_INTERVAL
            )

            settings = {
                'sip': {
                    'server': SIP_SERVER,
                    'port': SIP_PORT,
                    'username': SIP_USERNAME,
                    'domain': SIP_DOMAIN
                },
                'door': {
                    'gpio_pin': GPIO_RELAY_PIN,
                    'unlock_duration': DOOR_UNLOCK_DURATION
                },
                'nfc': {
                    'enabled': NFC_ENABLED,
                    'scan_interval': NFC_SCAN_INTERVAL
                },
                'companies': COMPANIES
            }

            return jsonify({'success': True, 'data': settings})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})

    # =========================================================================
    # 公司管理 API
    # =========================================================================
    @app.route('/api/companies', methods=['GET'])
    @login_required
    def api_get_companies():
        """取得所有公司資料"""
        try:
            companies = get_companies_from_db()
            return jsonify({'success': True, 'data': companies})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})

    @app.route('/api/extension/company-name')
    def api_extension_company_name():
        """無需登入：根據分機號碼查詢公司名稱（供 Android App 使用）"""
        ext = request.args.get('ext', '').strip()
        if not ext:
            return jsonify({'error': 'missing ext'}), 400
        try:
            companies = get_companies_from_db()
            for company in companies.values():
                if company['extension'] == ext:
                    return jsonify({'name': company['name'], 'extension': ext})
        except Exception:
            pass
        return jsonify({'name': ext, 'extension': ext})  # fallback: 找不到就回傳分機號本身

    @app.route('/api/companies/<int:company_id>', methods=['PUT'])
    @login_required
    def api_update_company(company_id):
        """更新公司資料"""
        try:
            data = request.get_json()
            name = data.get('name', '').strip()
            extension = data.get('extension', '').strip()
            floor = data.get('floor', '').strip()

            if not name:
                return jsonify({'success': False, 'error': '請輸入公司名稱'})

            if not extension:
                return jsonify({'success': False, 'error': '請輸入分機號碼'})

            conn = sqlite3.connect(WEB_ADMIN_DB_PATH)
            cursor = conn.cursor()

            # 檢查公司是否存在
            cursor.execute('SELECT id FROM companies WHERE id = ?', (company_id,))
            if not cursor.fetchone():
                conn.close()
                return jsonify({'success': False, 'error': '公司不存在'})

            # 更新公司資料
            cursor.execute('''
                UPDATE companies SET name = ?, extension = ?, floor = ?
                WHERE id = ?
            ''', (name, extension, floor, company_id))

            conn.commit()
            conn.close()

            return jsonify({'success': True, 'message': '公司資料更新成功'})

        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})

    # =========================================================================
    # 音訊設定 API
    # =========================================================================
    AUDIO_CONFIG_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'audio_config.json'
    )
    MIC_DEVICE    = 'hw:UACDemoV10'
    MIC_NUMID     = '3'
    SPEAKER_DEVICE = 'hw:CD002AUDIO'
    SPEAKER_NUMID  = '3'
    ALSA_MAX      = 147

    def _amixer_get(device, numid):
        """以 amixer cget 取得 ALSA 音量值（失敗回傳 -1）"""
        try:
            r = subprocess.run(
                ['amixer', '-D', device, 'cget', f'numid={numid}'],
                capture_output=True, text=True, timeout=3
            )
            m = re.search(r':\s*values=(\d+)', r.stdout)
            return int(m.group(1)) if m else -1
        except Exception:
            return -1

    def _amixer_set(device, numid, value, stereo=False):
        """以 amixer cset 設定 ALSA 音量值"""
        val_str = f'{value},{value}' if stereo else str(value)
        subprocess.run(
            ['amixer', '-D', device, 'cset', f'numid={numid}', val_str],
            capture_output=True, timeout=3
        )

    def _read_audio_config():
        try:
            with open(AUDIO_CONFIG_PATH, 'r') as f:
                return json.load(f)
        except Exception:
            return {}

    def _write_audio_config(cfg):
        with open(AUDIO_CONFIG_PATH, 'w') as f:
            json.dump(cfg, f, indent=2)

    @app.route('/api/audio', methods=['GET'])
    @login_required
    def api_get_audio():
        """取得麥克風 / 喇叭音量及噪音閘門設定"""
        try:
            mic_vol = _amixer_get(MIC_DEVICE, MIC_NUMID)
            spk_vol = _amixer_get(SPEAKER_DEVICE, SPEAKER_NUMID)
            cfg     = _read_audio_config()
            noise_gate = int(cfg.get('noise_gate_threshold', 600))
            return jsonify({
                'success': True,
                'data': {
                    'mic_volume':           mic_vol,
                    'speaker_volume':       spk_vol,
                    'noise_gate_threshold': noise_gate,
                    'alsa_max':             ALSA_MAX,
                }
            })
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})

    @app.route('/api/audio', methods=['POST'])
    @login_required
    def api_set_audio():
        """設定麥克風 / 喇叭音量及噪音閘門"""
        try:
            data = request.get_json() or {}
            cfg  = _read_audio_config()
            changed_alsa = False

            if 'mic_volume' in data:
                val = max(0, min(ALSA_MAX, int(data['mic_volume'])))
                _amixer_set(MIC_DEVICE, MIC_NUMID, val)
                cfg['mic_volume'] = val
                changed_alsa = True

            if 'speaker_volume' in data:
                val = max(0, min(ALSA_MAX, int(data['speaker_volume'])))
                _amixer_set(SPEAKER_DEVICE, SPEAKER_NUMID, val, stereo=True)
                cfg['speaker_volume'] = val
                changed_alsa = True

            if 'noise_gate_threshold' in data:
                val = max(0, min(5000, int(data['noise_gate_threshold'])))
                cfg['noise_gate_threshold'] = val

            _write_audio_config(cfg)

            # 持久化 ALSA 設定（重開機後保留）
            if changed_alsa:
                subprocess.run(['sudo', 'alsactl', 'store'],
                               capture_output=True, timeout=5)

            return jsonify({'success': True, 'message': '音訊設定已儲存'})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})

    # =========================================================================
    # 密碼開門管理 API
    # =========================================================================
    @app.route('/passwords')
    @login_required
    def passwords():
        """密碼開門管理頁面"""
        companies = get_companies_from_db()
        return render_template('passwords.html', companies=companies)

    @app.route('/api/passwords', methods=['GET'])
    @login_required
    def api_get_passwords():
        """取得所有密碼"""
        try:
            conn = sqlite3.connect(WEB_ADMIN_DB_PATH)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                SELECT dp.id, dp.password, dp.name, dp.company_id, dp.active,
                       dp.created_at, dp.last_used, dp.user_id, u.name as user_name
                FROM door_passwords dp
                LEFT JOIN users u ON dp.user_id = u.id
                ORDER BY dp.created_at DESC
            ''')
            passwords = [dict(row) for row in cursor.fetchall()]
            conn.close()

            # 加入公司名稱
            companies = get_companies_from_db()
            for pwd in passwords:
                company = companies.get(pwd['company_id'], {})
                pwd['company_name'] = company.get('name', '無')

            return jsonify({'success': True, 'data': passwords})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})

    @app.route('/api/passwords', methods=['POST'])
    @login_required
    def api_add_password():
        """新增密碼"""
        try:
            data = request.get_json()
            password = data.get('password', '').strip()
            name = data.get('name', '').strip()
            company_id = data.get('company_id', 0)

            if not password:
                return jsonify({'success': False, 'error': '請輸入密碼'})

            if len(password) < 4:
                return jsonify({'success': False, 'error': '密碼至少需要 4 位數'})

            if not password.isdigit():
                return jsonify({'success': False, 'error': '密碼只能包含數字'})

            if not name:
                return jsonify({'success': False, 'error': '請輸入持有人名稱'})

            conn = sqlite3.connect(WEB_ADMIN_DB_PATH)
            cursor = conn.cursor()

            # 檢查密碼是否已存在
            cursor.execute('SELECT id FROM door_passwords WHERE password = ?', (password,))
            if cursor.fetchone():
                conn.close()
                return jsonify({'success': False, 'error': '此密碼已存在'})

            # 新增密碼（支援關聯用戶）
            user_id = data.get('user_id')
            cursor.execute('''
                INSERT INTO door_passwords (password, name, company_id, user_id)
                VALUES (?, ?, ?, ?)
            ''', (password, name, company_id, user_id))

            conn.commit()
            conn.close()

            # 記錄日誌
            log_access('password_add', name, None, 'success', f'新增密碼')

            return jsonify({'success': True, 'message': '密碼新增成功'})

        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})

    @app.route('/api/passwords/<int:pwd_id>', methods=['PUT'])
    @login_required
    def api_update_password(pwd_id):
        """更新密碼"""
        try:
            data = request.get_json()
            password = data.get('password', '').strip()
            name = data.get('name', '').strip()
            company_id = data.get('company_id', 0)

            if not password:
                return jsonify({'success': False, 'error': '請輸入密碼'})

            if len(password) < 4:
                return jsonify({'success': False, 'error': '密碼至少需要 4 位數'})

            if not password.isdigit():
                return jsonify({'success': False, 'error': '密碼只能包含數字'})

            if not name:
                return jsonify({'success': False, 'error': '請輸入持有人名稱'})

            conn = sqlite3.connect(WEB_ADMIN_DB_PATH)
            cursor = conn.cursor()

            # 檢查密碼是否已被其他人使用
            cursor.execute(
                'SELECT id FROM door_passwords WHERE password = ? AND id != ?',
                (password, pwd_id)
            )
            if cursor.fetchone():
                conn.close()
                return jsonify({'success': False, 'error': '此密碼已被其他人使用'})

            # 更新密碼
            cursor.execute('''
                UPDATE door_passwords SET password = ?, name = ?, company_id = ?
                WHERE id = ?
            ''', (password, name, company_id, pwd_id))

            conn.commit()
            conn.close()

            # 記錄日誌
            log_access('password_update', name, pwd_id, 'success', '更新密碼')

            return jsonify({'success': True, 'message': '密碼更新成功'})

        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})

    @app.route('/api/passwords/<int:pwd_id>', methods=['DELETE'])
    @login_required
    def api_delete_password(pwd_id):
        """刪除密碼"""
        try:
            conn = sqlite3.connect(WEB_ADMIN_DB_PATH)
            cursor = conn.cursor()

            # 取得密碼名稱用於日誌
            cursor.execute('SELECT name FROM door_passwords WHERE id = ?', (pwd_id,))
            row = cursor.fetchone()
            pwd_name = row[0] if row else 'unknown'

            cursor.execute('DELETE FROM door_passwords WHERE id = ?', (pwd_id,))
            conn.commit()
            conn.close()

            # 記錄日誌
            log_access('password_delete', pwd_name, pwd_id, 'success')

            return jsonify({'success': True, 'message': '密碼刪除成功'})

        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})

    @app.route('/api/passwords/<int:pwd_id>/toggle', methods=['PATCH'])
    @login_required
    def api_toggle_password(pwd_id):
        """停用/啟用密碼（切換狀態）"""
        try:
            conn = sqlite3.connect(WEB_ADMIN_DB_PATH)
            cursor = conn.cursor()

            # 取得當前狀態
            cursor.execute('SELECT active, name FROM door_passwords WHERE id = ?', (pwd_id,))
            row = cursor.fetchone()
            if not row:
                conn.close()
                return jsonify({'success': False, 'error': '密碼不存在'})

            current_active = row[0]
            pwd_name = row[1]
            new_active = 0 if current_active else 1

            cursor.execute(
                'UPDATE door_passwords SET active = ? WHERE id = ?',
                (new_active, pwd_id)
            )
            conn.commit()
            conn.close()

            status = '啟用' if new_active else '停用'
            log_access('password_toggle', pwd_name, pwd_id, 'success', f'密碼已{status}')
            return jsonify({'success': True, 'message': f'密碼已{status}'})

        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})

    @app.route('/api/passwords/verify', methods=['POST'])
    def api_verify_password():
        """驗證密碼（供 GUI 呼叫，不需登入）"""
        try:
            data = request.get_json()
            password = data.get('password', '').strip()
            client_ip = request.remote_addr or 'local'

            if not password:
                return jsonify({'success': False, 'error': '請輸入密碼'})

            conn = sqlite3.connect(WEB_ADMIN_DB_PATH)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # 檢查是否在鎖定期間（5 分鐘內超過 3 次錯誤）
            lockout_minutes = 5
            max_attempts = 3
            lockout_check_time = (datetime.now() - timedelta(minutes=lockout_minutes)).strftime('%Y-%m-%d %H:%M:%S')

            cursor.execute('''
                SELECT COUNT(*) as count FROM password_attempts
                WHERE attempt_time > ? AND ip_address = ?
            ''', (lockout_check_time, client_ip))

            attempt_count = cursor.fetchone()['count']

            if attempt_count >= max_attempts:
                # 計算剩餘鎖定時間
                cursor.execute('''
                    SELECT MAX(attempt_time) as last_attempt FROM password_attempts
                    WHERE ip_address = ?
                ''', (client_ip,))
                last_attempt = cursor.fetchone()['last_attempt']
                conn.close()

                if last_attempt:
                    last_time = datetime.strptime(last_attempt, '%Y-%m-%d %H:%M:%S')
                    unlock_time = last_time + timedelta(minutes=lockout_minutes)
                    remaining = unlock_time - datetime.now()
                    remaining_minutes = max(1, int(remaining.total_seconds() // 60) + 1)
                    log_access('password_unlock', None, None, 'locked', f'帳號鎖定中，剩餘 {remaining_minutes} 分鐘')
                    return jsonify({
                        'success': False,
                        'valid': False,
                        'locked': True,
                        'error': f'錯誤次數過多，請等待 {remaining_minutes} 分鐘後再試'
                    })

            cursor.execute('''
                SELECT id, name, company_id, active
                FROM door_passwords
                WHERE password = ?
            ''', (password,))

            row = cursor.fetchone()

            if not row:
                # 記錄錯誤嘗試（使用本地時間）
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                cursor.execute(
                    'INSERT INTO password_attempts (attempt_time, ip_address) VALUES (?, ?)',
                    (current_time, client_ip)
                )
                conn.commit()

                # 計算剩餘嘗試次數
                remaining_attempts = max_attempts - attempt_count - 1
                conn.close()

                log_access('password_unlock', None, None, 'failed', f'密碼不存在，剩餘 {remaining_attempts} 次嘗試')

                if remaining_attempts <= 0:
                    return jsonify({
                        'success': False,
                        'valid': False,
                        'locked': True,
                        'error': f'錯誤次數過多，請等待 {lockout_minutes} 分鐘後再試'
                    })
                else:
                    return jsonify({
                        'success': False,
                        'valid': False,
                        'error': f'密碼錯誤，剩餘 {remaining_attempts} 次嘗試'
                    })

            if not row['active']:
                conn.close()
                log_access('password_unlock', row['name'], row['id'], 'disabled', '密碼已停用')
                return jsonify({'success': False, 'valid': False, 'error': '此密碼已停用'})

            # 密碼正確，清除該 IP 的錯誤記錄
            cursor.execute(
                'DELETE FROM password_attempts WHERE ip_address = ?',
                (client_ip,)
            )

            # 更新最後使用時間
            cursor.execute(
                'UPDATE door_passwords SET last_used = CURRENT_TIMESTAMP WHERE id = ?',
                (row['id'],)
            )
            conn.commit()
            conn.close()

            # 記錄日誌
            log_access('password_unlock', row['name'], row['id'], 'success')

            return jsonify({
                'success': True,
                'valid': True,
                'name': row['name'],
                'company_id': row['company_id']
            })

        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})

    @app.route('/api/settings/password', methods=['POST'])
    @login_required
    def api_change_password():
        """修改管理員密碼"""
        try:
            data = request.get_json()
            old_password = data.get('old_password', '')
            new_password = data.get('new_password', '')

            if not old_password or not new_password:
                return jsonify({'success': False, 'error': '請輸入原密碼和新密碼'})

            if len(new_password) < 6:
                return jsonify({'success': False, 'error': '新密碼至少需要 6 個字元'})

            # 驗證原密碼
            conn = sqlite3.connect(WEB_ADMIN_DB_PATH)
            cursor = conn.cursor()
            old_hash = hashlib.sha256(old_password.encode()).hexdigest()
            cursor.execute(
                'SELECT id FROM admin_users WHERE id = ? AND password_hash = ?',
                (session['user_id'], old_hash)
            )

            if not cursor.fetchone():
                conn.close()
                return jsonify({'success': False, 'error': '原密碼錯誤'})

            # 更新密碼
            new_hash = hashlib.sha256(new_password.encode()).hexdigest()
            cursor.execute(
                'UPDATE admin_users SET password_hash = ? WHERE id = ?',
                (new_hash, session['user_id'])
            )
            conn.commit()
            conn.close()

            return jsonify({'success': True, 'message': '密碼修改成功'})

        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})

    # =========================================================================
    # 存取日誌
    # =========================================================================
    @app.route('/logs')
    @login_required
    def logs():
        """存取日誌頁面"""
        return render_template('logs.html')

    @app.route('/api/logs', methods=['GET'])
    @login_required
    def api_get_logs():
        """取得存取日誌"""
        try:
            page = request.args.get('page', 1, type=int)
            per_page = request.args.get('per_page', 50, type=int)
            access_type = request.args.get('type', '')
            start_date = request.args.get('start_date', '')
            end_date = request.args.get('end_date', '')

            conn = sqlite3.connect(WEB_ADMIN_DB_PATH)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # 建立查詢條件
            conditions = []
            params = []

            if access_type:
                conditions.append('access_type = ?')
                params.append(access_type)

            if start_date:
                conditions.append('timestamp >= ?')
                params.append(f'{start_date} 00:00:00')

            if end_date:
                conditions.append('timestamp <= ?')
                params.append(f'{end_date} 23:59:59')

            where_clause = ' AND '.join(conditions) if conditions else '1=1'

            # 計算總數
            cursor.execute(f'SELECT COUNT(*) FROM access_logs WHERE {where_clause}', params)
            total = cursor.fetchone()[0]

            # 取得分頁資料
            offset = (page - 1) * per_page
            cursor.execute(f'''
                SELECT * FROM access_logs
                WHERE {where_clause}
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
            ''', params + [per_page, offset])

            logs = [dict(row) for row in cursor.fetchall()]
            conn.close()

            return jsonify({
                'success': True,
                'data': logs,
                'pagination': {
                    'page': page,
                    'per_page': per_page,
                    'total': total,
                    'pages': (total + per_page - 1) // per_page
                }
            })
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})

    @app.route('/api/logs/export', methods=['GET'])
    @login_required
    def api_export_logs():
        """匯出存取日誌為 CSV"""
        try:
            import csv
            import io

            access_type = request.args.get('type', '')
            start_date = request.args.get('start_date', '')
            end_date = request.args.get('end_date', '')

            conn = sqlite3.connect(WEB_ADMIN_DB_PATH)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # 建立查詢條件
            conditions = []
            params = []

            if access_type:
                conditions.append('access_type = ?')
                params.append(access_type)

            if start_date:
                conditions.append('timestamp >= ?')
                params.append(f'{start_date} 00:00:00')

            if end_date:
                conditions.append('timestamp <= ?')
                params.append(f'{end_date} 23:59:59')

            where_clause = ' AND '.join(conditions) if conditions else '1=1'

            cursor.execute(f'''
                SELECT * FROM access_logs
                WHERE {where_clause}
                ORDER BY timestamp DESC
            ''', params)

            logs = cursor.fetchall()
            conn.close()

            # 建立 CSV
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(['ID', '類型', '使用者', '結果', '詳細資訊', '時間'])

            for log in logs:
                writer.writerow([
                    log['id'],
                    log['access_type'],
                    log['user_name'] or '',
                    log['result'],
                    log['details'] or '',
                    log['timestamp']
                ])

            output.seek(0)

            return send_file(
                io.BytesIO(output.getvalue().encode('utf-8-sig')),
                mimetype='text/csv',
                as_attachment=True,
                download_name=f'access_logs_{datetime.now().strftime("%Y%m%d")}.csv'
            )

        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})

    # =========================================================================
    # 統一用戶管理
    # =========================================================================
    @app.route('/users')
    @login_required
    def users():
        """用戶管理頁面"""
        companies = get_companies_from_db()
        return render_template('users.html', companies=companies)

    @app.route('/api/users', methods=['GET'])
    @login_required
    def api_get_users():
        """取得所有用戶（含認證方式統計）"""
        try:
            conn = sqlite3.connect(WEB_ADMIN_DB_PATH)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, name, company_id, phone, email, notes, active, created_at, updated_at
                FROM users
                ORDER BY created_at DESC
            ''')
            users_list = [dict(row) for row in cursor.fetchall()]
            conn.close()

            # 加入公司名稱和認證方式統計
            companies = get_companies_from_db()
            for user in users_list:
                company = companies.get(user['company_id'], {})
                user['company_name'] = company.get('name', '無')

                # 統計認證方式
                auth_methods = get_user_auth_methods(user['id'])
                user['face_count'] = len(auth_methods['faces'])
                user['nfc_count'] = len(auth_methods['nfc_cards'])
                user['password_count'] = len(auth_methods['passwords'])

            return jsonify({'success': True, 'data': users_list})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})

    @app.route('/api/users', methods=['POST'])
    @login_required
    def api_add_user():
        """新增用戶"""
        try:
            data = request.get_json()
            name = data.get('name', '').strip()
            company_id = data.get('company_id', 0)
            phone = data.get('phone', '').strip()
            email = data.get('email', '').strip()
            notes = data.get('notes', '').strip()

            if not name:
                return jsonify({'success': False, 'error': '請輸入用戶姓名'})

            conn = sqlite3.connect(WEB_ADMIN_DB_PATH)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO users (name, company_id, phone, email, notes)
                VALUES (?, ?, ?, ?, ?)
            ''', (name, company_id, phone, email, notes))

            user_id = cursor.lastrowid
            conn.commit()
            conn.close()

            log_access('user_add', name, user_id, 'success', '新增用戶')

            return jsonify({'success': True, 'message': '用戶新增成功', 'user_id': user_id})

        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})

    @app.route('/api/users/<int:user_id>', methods=['GET'])
    @login_required
    def api_get_user(user_id):
        """取得單一用戶詳情"""
        try:
            conn = sqlite3.connect(WEB_ADMIN_DB_PATH)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, name, company_id, phone, email, notes, active, created_at, updated_at
                FROM users WHERE id = ?
            ''', (user_id,))
            row = cursor.fetchone()
            conn.close()

            if not row:
                return jsonify({'success': False, 'error': '用戶不存在'})

            user = dict(row)

            # 加入公司名稱
            companies = get_companies_from_db()
            company = companies.get(user['company_id'], {})
            user['company_name'] = company.get('name', '無')

            # 加入認證方式詳情
            user['auth_methods'] = get_user_auth_methods(user_id)

            return jsonify({'success': True, 'data': user})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})

    @app.route('/api/users/<int:user_id>', methods=['PUT'])
    @login_required
    def api_update_user(user_id):
        """更新用戶資料"""
        try:
            data = request.get_json()
            name = data.get('name', '').strip()
            company_id = data.get('company_id', 0)
            phone = data.get('phone', '').strip()
            email = data.get('email', '').strip()
            notes = data.get('notes', '').strip()

            if not name:
                return jsonify({'success': False, 'error': '請輸入用戶姓名'})

            conn = sqlite3.connect(WEB_ADMIN_DB_PATH)
            cursor = conn.cursor()

            cursor.execute('''
                UPDATE users SET name = ?, company_id = ?, phone = ?, email = ?, notes = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (name, company_id, phone, email, notes, user_id))

            conn.commit()
            conn.close()

            log_access('user_update', name, user_id, 'success', '更新用戶')

            return jsonify({'success': True, 'message': '用戶資料更新成功'})

        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})

    @app.route('/api/users/<int:user_id>', methods=['DELETE'])
    @login_required
    def api_delete_user(user_id):
        """刪除用戶（同時解除所有認證方式的關聯）"""
        try:
            conn = sqlite3.connect(WEB_ADMIN_DB_PATH)
            cursor = conn.cursor()

            # 取得用戶名稱
            cursor.execute('SELECT name FROM users WHERE id = ?', (user_id,))
            row = cursor.fetchone()
            user_name = row[0] if row else 'unknown'

            # 解除密碼關聯
            cursor.execute('UPDATE door_passwords SET user_id = NULL WHERE user_id = ?', (user_id,))

            # 刪除用戶
            cursor.execute('DELETE FROM users WHERE id = ?', (user_id,))

            conn.commit()
            conn.close()

            # 解除人臉關聯
            if os.path.exists(DATABASE_PATH):
                conn = sqlite3.connect(DATABASE_PATH)
                cursor = conn.cursor()
                try:
                    cursor.execute('UPDATE face_users SET user_id = NULL WHERE user_id = ?', (user_id,))
                    conn.commit()
                except:
                    pass
                conn.close()

            # 解除 NFC 關聯
            if os.path.exists(NFC_DATABASE_PATH):
                conn = sqlite3.connect(NFC_DATABASE_PATH)
                cursor = conn.cursor()
                try:
                    cursor.execute('UPDATE nfc_cards SET user_id = NULL WHERE user_id = ?', (user_id,))
                    conn.commit()
                except:
                    pass
                conn.close()

            log_access('user_delete', user_name, user_id, 'success', '刪除用戶')

            return jsonify({'success': True, 'message': '用戶刪除成功'})

        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})

    @app.route('/api/users/<int:user_id>/toggle', methods=['PATCH'])
    @login_required
    def api_toggle_user(user_id):
        """停用/啟用用戶"""
        try:
            conn = sqlite3.connect(WEB_ADMIN_DB_PATH)
            cursor = conn.cursor()

            # 取得當前狀態
            cursor.execute('SELECT active, name FROM users WHERE id = ?', (user_id,))
            row = cursor.fetchone()
            if not row:
                conn.close()
                return jsonify({'success': False, 'error': '用戶不存在'})

            current_active = row[0]
            user_name = row[1]
            new_active = 0 if current_active else 1

            cursor.execute(
                'UPDATE users SET active = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?',
                (new_active, user_id)
            )
            conn.commit()
            conn.close()

            status = '啟用' if new_active else '停用'
            log_access('user_toggle', user_name, user_id, 'success', f'用戶已{status}')
            return jsonify({'success': True, 'message': f'用戶已{status}'})

        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})

    @app.route('/api/users/<int:user_id>/auth-methods', methods=['GET'])
    @login_required
    def api_get_user_auth_methods(user_id):
        """取得用戶的所有認證方式"""
        try:
            auth_methods = get_user_auth_methods(user_id)
            return jsonify({'success': True, 'data': auth_methods})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})

    @app.route('/api/users/<int:user_id>/auth-methods/password', methods=['POST'])
    @login_required
    def api_add_user_password(user_id):
        """為用戶新增密碼"""
        try:
            data = request.get_json()
            password = data.get('password', '').strip()

            if not password:
                return jsonify({'success': False, 'error': '請輸入密碼'})

            if len(password) < 4:
                return jsonify({'success': False, 'error': '密碼至少需要 4 位數'})

            if not password.isdigit():
                return jsonify({'success': False, 'error': '密碼只能包含數字'})

            # 取得用戶名稱
            conn = sqlite3.connect(WEB_ADMIN_DB_PATH)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute('SELECT name, company_id FROM users WHERE id = ?', (user_id,))
            user = cursor.fetchone()
            if not user:
                conn.close()
                return jsonify({'success': False, 'error': '用戶不存在'})

            # 檢查密碼是否已存在
            cursor.execute('SELECT id FROM door_passwords WHERE password = ?', (password,))
            if cursor.fetchone():
                conn.close()
                return jsonify({'success': False, 'error': '此密碼已被使用'})

            # 新增密碼
            cursor.execute('''
                INSERT INTO door_passwords (password, name, company_id, user_id)
                VALUES (?, ?, ?, ?)
            ''', (password, user['name'], user['company_id'], user_id))

            conn.commit()
            conn.close()

            return jsonify({'success': True, 'message': '密碼新增成功'})

        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})

    @app.route('/api/users/list', methods=['GET'])
    @login_required
    def api_get_users_list():
        """取得用戶列表（供下拉選單使用）"""
        try:
            conn = sqlite3.connect(WEB_ADMIN_DB_PATH)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT id, name, company_id FROM users WHERE active = 1 ORDER BY name')
            users_list = [dict(row) for row in cursor.fetchall()]
            conn.close()

            return jsonify({'success': True, 'data': users_list})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})

    # =========================================================================
    # 攝影機串流（Android App 直接存取，無需登入）
    # =========================================================================
    import cv2 as _cv2
    import threading as _threading
    import time as _time_cam

    _camera_lock = _threading.Lock()

    def _get_camera_cap():
        """嘗試開啟攝影機，最多等 5 秒（face_manager 釋放後才能開啟）"""
        for _ in range(10):
            try:
                cap = _cv2.VideoCapture('/dev/video1', _cv2.CAP_V4L2)
                if cap.isOpened():
                    cap.set(_cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(_cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(_cv2.CAP_PROP_FOURCC, _cv2.VideoWriter_fourcc(*'MJPG'))
                    cap.set(_cv2.CAP_PROP_FPS, 15)
                    return cap
                cap.release()
            except Exception:
                pass
            _time_cam.sleep(0.5)
        return None

    def _gen_mjpeg_frames():
        """產生 MJPEG 串流幀（持有攝影機直到連線中斷）"""
        with _camera_lock:
            cap = _get_camera_cap()
        if cap is None:
            return
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                ret2, buf = _cv2.imencode('.jpg', frame, [_cv2.IMWRITE_JPEG_QUALITY, 70])
                if ret2:
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                           + buf.tobytes() + b'\r\n')
                _time_cam.sleep(1 / 15)
        finally:
            cap.release()

    @app.route('/camera/stream')
    def camera_stream():
        """MJPEG 攝影機串流（Android App 來電畫面使用，無需登入）"""
        from flask import Response
        return Response(_gen_mjpeg_frames(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/camera/snapshot')
    def camera_snapshot():
        """單張快照"""
        from flask import Response
        try:
            cap = _cv2.VideoCapture('/dev/video1', _cv2.CAP_V4L2)
            if not cap.isOpened():
                return jsonify({'error': 'camera unavailable'}), 503
            ret, frame = cap.read()
            cap.release()
            if not ret:
                return jsonify({'error': 'capture failed'}), 503
            ret2, buf = _cv2.imencode('.jpg', frame, [_cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret2:
                return jsonify({'error': 'encode failed'}), 503
            return Response(buf.tobytes(), mimetype='image/jpeg')
        except Exception as e:
            return jsonify({'error': str(e)}), 503


def get_system_stats():
    """取得系統統計資訊"""
    stats = {
        'total_faces': 0,
        'total_nfc_cards': 0,
        'today_access': 0,
        'nfc_enabled': NFC_ENABLED
    }

    try:
        # 人臉數量
        if os.path.exists(DATABASE_PATH):
            conn = sqlite3.connect(DATABASE_PATH)
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM users')
            stats['total_faces'] = cursor.fetchone()[0]
            conn.close()

        # NFC 卡片數量
        if NFC_ENABLED and os.path.exists(NFC_DATABASE_PATH):
            conn = sqlite3.connect(NFC_DATABASE_PATH)
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM nfc_cards')
            stats['total_nfc_cards'] = cursor.fetchone()[0]
            conn.close()

        # 今日存取次數
        if os.path.exists(WEB_ADMIN_DB_PATH):
            conn = sqlite3.connect(WEB_ADMIN_DB_PATH)
            cursor = conn.cursor()
            today = datetime.now().strftime('%Y-%m-%d')
            cursor.execute(
                "SELECT COUNT(*) FROM access_logs WHERE timestamp >= ? AND result = 'success'",
                (f'{today} 00:00:00',)
            )
            stats['today_access'] = cursor.fetchone()[0]
            conn.close()

    except Exception as e:
        print(f"取得統計資訊失敗: {e}")

    return stats


def log_access(access_type, user_name, user_id, result, details=None):
    """記錄存取日誌"""
    try:
        conn = sqlite3.connect(WEB_ADMIN_DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO access_logs (access_type, user_name, user_id, result, details)
            VALUES (?, ?, ?, ?, ?)
        ''', (access_type, user_name, user_id, result, details))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"記錄日誌失敗: {e}")


def get_user_auth_methods(user_id):
    """取得用戶的所有認證方式"""
    auth_methods = {
        'faces': [],
        'nfc_cards': [],
        'passwords': []
    }

    # 人臉
    try:
        if os.path.exists(DATABASE_PATH):
            conn = sqlite3.connect(DATABASE_PATH)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT id, name, created_at FROM face_users WHERE user_id = ?', (user_id,))
            auth_methods['faces'] = [dict(row) for row in cursor.fetchall()]
            conn.close()
    except Exception as e:
        print(f"取得人臉資料失敗: {e}")

    # NFC 卡片
    try:
        if os.path.exists(NFC_DATABASE_PATH):
            conn = sqlite3.connect(NFC_DATABASE_PATH)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT id, uid, name, card_type, active, created_at FROM nfc_cards WHERE user_id = ?', (user_id,))
            auth_methods['nfc_cards'] = [dict(row) for row in cursor.fetchall()]
            conn.close()
    except Exception as e:
        print(f"取得 NFC 資料失敗: {e}")

    # 密碼
    try:
        conn = sqlite3.connect(WEB_ADMIN_DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('SELECT id, name, active, created_at FROM door_passwords WHERE user_id = ?', (user_id,))
        auth_methods['passwords'] = [dict(row) for row in cursor.fetchall()]
        conn.close()
    except Exception as e:
        print(f"取得密碼資料失敗: {e}")

    return auth_methods


# 主程式入口
if __name__ == '__main__':
    app = create_app()
    print(f"網頁管理介面已啟動: http://0.0.0.0:{WEB_ADMIN_PORT}")
    print("預設帳號: admin / admin123")
    app.run(host='0.0.0.0', port=WEB_ADMIN_PORT, debug=False)
