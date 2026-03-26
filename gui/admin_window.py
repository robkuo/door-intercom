# -*- coding: utf-8 -*-
"""
管理介面 - 人臉登錄與 NFC 卡片管理
"""

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from typing import Callable, Optional, List
import sys
sys.path.append('..')


class AdminWindow:
    """
    管理介面類別

    用於登錄新人臉、管理 NFC 卡片、檢視/刪除已登錄的項目
    """

    def __init__(
        self,
        root: tk.Tk,
        companies: dict,
        on_enroll: Optional[Callable[[str, int], None]] = None,
        on_delete: Optional[Callable[[int], None]] = None,
        on_close: Optional[Callable] = None,
        get_users: Optional[Callable] = None,
        # NFC 相關
        nfc_manager=None,
        on_enroll_nfc: Optional[Callable[[str, int], None]] = None,
        on_delete_nfc: Optional[Callable[[int], None]] = None,
        get_nfc_cards: Optional[Callable] = None
    ):
        """
        初始化管理介面

        Args:
            root: Tkinter 根視窗
            companies: 公司資料字典
            on_enroll: 登錄人臉時的回調 (name, company_id)
            on_delete: 刪除人臉時的回調 (user_id)
            on_close: 關閉介面時的回調
            get_users: 取得使用者列表的函數
            nfc_manager: NFC 管理器實例
            on_enroll_nfc: 登錄 NFC 卡片時的回調 (name, company_id)
            on_delete_nfc: 刪除 NFC 卡片時的回調 (card_id)
            get_nfc_cards: 取得 NFC 卡片列表的函數
        """
        self.root = root
        self.companies = companies
        self.on_enroll = on_enroll
        self.on_delete = on_delete
        self.on_close = on_close
        self.get_users = get_users

        # NFC 相關
        self.nfc_manager = nfc_manager
        self.on_enroll_nfc = on_enroll_nfc
        self.on_delete_nfc = on_delete_nfc
        self.get_nfc_cards = get_nfc_cards
        self.nfc_enabled = nfc_manager is not None

        self.frame: Optional[tk.Frame] = None
        self._user_listbox: Optional[tk.Listbox] = None
        self._nfc_listbox: Optional[tk.Listbox] = None
        self._password_verified = False
        self._current_tab = "face"  # face 或 nfc

        self._setup_styles()
        self._create_widgets()

    def _setup_styles(self):
        """設定樣式"""
        self.colors = {
            'bg': '#1a1a2e',
            'card': '#16213e',
            'text': '#ffffff',
            'text_secondary': '#a0a0a0',
            'accent': '#0f3460',
            'button': '#e94560',
            'success': '#00d9a0',
            'nfc': '#4a90d9',  # NFC 專用顏色
        }

        self.fonts = {
            'title': ('Microsoft JhengHei', 20, 'bold'),
            'normal': ('Microsoft JhengHei', 12),
            'button': ('Microsoft JhengHei', 14),
            'list': ('Microsoft JhengHei', 11),
            'tab': ('Microsoft JhengHei', 13, 'bold'),
        }

    def _create_widgets(self):
        """建立介面元件"""
        # 主框架
        self.frame = tk.Frame(self.root, bg=self.colors['bg'])

        # 標題列
        header = tk.Frame(self.frame, bg=self.colors['bg'])
        header.pack(fill=tk.X, pady=(20, 10), padx=20)

        title = tk.Label(
            header,
            text="系統管理",
            font=self.fonts['title'],
            fg=self.colors['text'],
            bg=self.colors['bg']
        )
        title.pack(side=tk.LEFT)

        close_btn = tk.Button(
            header,
            text="✕ 返回",
            font=self.fonts['normal'],
            fg=self.colors['text'],
            bg=self.colors['accent'],
            activebackground=self.colors['card'],
            activeforeground=self.colors['text'],
            bd=0,
            padx=15,
            pady=5,
            cursor='hand2',
            command=self._on_close_click
        )
        close_btn.pack(side=tk.RIGHT)

        # 分頁選擇
        if self.nfc_enabled:
            tab_frame = tk.Frame(self.frame, bg=self.colors['bg'])
            tab_frame.pack(fill=tk.X, padx=20, pady=(0, 10))

            self._face_tab_btn = tk.Button(
                tab_frame,
                text="👤 人臉辨識",
                font=self.fonts['tab'],
                fg=self.colors['text'],
                bg=self.colors['success'],
                activebackground=self.colors['success'],
                activeforeground=self.colors['text'],
                bd=0,
                padx=20,
                pady=8,
                cursor='hand2',
                command=lambda: self._switch_tab("face")
            )
            self._face_tab_btn.pack(side=tk.LEFT, padx=(0, 5))

            self._nfc_tab_btn = tk.Button(
                tab_frame,
                text="💳 NFC 卡片",
                font=self.fonts['tab'],
                fg=self.colors['text'],
                bg=self.colors['accent'],
                activebackground=self.colors['nfc'],
                activeforeground=self.colors['text'],
                bd=0,
                padx=20,
                pady=8,
                cursor='hand2',
                command=lambda: self._switch_tab("nfc")
            )
            self._nfc_tab_btn.pack(side=tk.LEFT)

        # 內容區域容器
        self._content_container = tk.Frame(self.frame, bg=self.colors['bg'])
        self._content_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # 建立人臉管理分頁
        self._create_face_tab()

        # 建立 NFC 管理分頁
        if self.nfc_enabled:
            self._create_nfc_tab()

    def _create_face_tab(self):
        """建立人臉管理分頁"""
        self._face_frame = tk.Frame(self._content_container, bg=self.colors['bg'])

        # 左側：登錄新人臉
        left_panel = tk.Frame(self._face_frame, bg=self.colors['card'], padx=20, pady=20)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        tk.Label(
            left_panel,
            text="登錄新人臉",
            font=self.fonts['title'],
            fg=self.colors['text'],
            bg=self.colors['card']
        ).pack(pady=(0, 20))

        # 姓名輸入
        tk.Label(
            left_panel,
            text="姓名：",
            font=self.fonts['normal'],
            fg=self.colors['text'],
            bg=self.colors['card']
        ).pack(anchor='w')

        self._name_entry = tk.Entry(
            left_panel,
            font=self.fonts['normal'],
            width=25
        )
        self._name_entry.pack(fill=tk.X, pady=(5, 15))

        # 公司選擇
        tk.Label(
            left_panel,
            text="所屬公司：",
            font=self.fonts['normal'],
            fg=self.colors['text'],
            bg=self.colors['card']
        ).pack(anchor='w')

        self._company_var = tk.StringVar()
        company_names = [f"{cid}. {info['name']}" for cid, info in self.companies.items()]
        self._company_combo = ttk.Combobox(
            left_panel,
            textvariable=self._company_var,
            values=company_names,
            font=self.fonts['normal'],
            state='readonly',
            width=23
        )
        self._company_combo.pack(fill=tk.X, pady=(5, 20))
        if company_names:
            self._company_combo.current(0)

        # 登錄按鈕
        enroll_btn = tk.Button(
            left_panel,
            text="📷  拍照登錄人臉",
            font=self.fonts['button'],
            fg=self.colors['text'],
            bg=self.colors['success'],
            activebackground='#00b386',
            activeforeground=self.colors['text'],
            bd=0,
            padx=20,
            pady=12,
            cursor='hand2',
            command=self._on_enroll_click
        )
        enroll_btn.pack(pady=20)

        self._status_label = tk.Label(
            left_panel,
            text="",
            font=self.fonts['normal'],
            fg=self.colors['text_secondary'],
            bg=self.colors['card'],
            wraplength=250
        )
        self._status_label.pack()

        # 右側：已登錄列表
        right_panel = tk.Frame(self._face_frame, bg=self.colors['card'], padx=20, pady=20)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))

        tk.Label(
            right_panel,
            text="已登錄人臉",
            font=self.fonts['title'],
            fg=self.colors['text'],
            bg=self.colors['card']
        ).pack(pady=(0, 10))

        # 列表框
        list_frame = tk.Frame(right_panel, bg=self.colors['card'])
        list_frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._user_listbox = tk.Listbox(
            list_frame,
            font=self.fonts['list'],
            bg=self.colors['accent'],
            fg=self.colors['text'],
            selectbackground=self.colors['button'],
            selectforeground=self.colors['text'],
            yscrollcommand=scrollbar.set,
            height=10
        )
        self._user_listbox.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self._user_listbox.yview)

        # 刪除按鈕
        delete_btn = tk.Button(
            right_panel,
            text="🗑  刪除選取",
            font=self.fonts['normal'],
            fg=self.colors['text'],
            bg=self.colors['button'],
            activebackground='#ff6b6b',
            activeforeground=self.colors['text'],
            bd=0,
            padx=15,
            pady=8,
            cursor='hand2',
            command=self._on_delete_click
        )
        delete_btn.pack(pady=(15, 0))

    def _create_nfc_tab(self):
        """建立 NFC 管理分頁"""
        self._nfc_frame = tk.Frame(self._content_container, bg=self.colors['bg'])

        # 左側：登錄新卡片
        left_panel = tk.Frame(self._nfc_frame, bg=self.colors['card'], padx=20, pady=20)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        tk.Label(
            left_panel,
            text="登錄新 NFC 卡片",
            font=self.fonts['title'],
            fg=self.colors['text'],
            bg=self.colors['card']
        ).pack(pady=(0, 20))

        # 姓名輸入
        tk.Label(
            left_panel,
            text="持卡人姓名：",
            font=self.fonts['normal'],
            fg=self.colors['text'],
            bg=self.colors['card']
        ).pack(anchor='w')

        self._nfc_name_entry = tk.Entry(
            left_panel,
            font=self.fonts['normal'],
            width=25
        )
        self._nfc_name_entry.pack(fill=tk.X, pady=(5, 15))

        # 公司選擇
        tk.Label(
            left_panel,
            text="所屬公司：",
            font=self.fonts['normal'],
            fg=self.colors['text'],
            bg=self.colors['card']
        ).pack(anchor='w')

        self._nfc_company_var = tk.StringVar()
        company_names = [f"{cid}. {info['name']}" for cid, info in self.companies.items()]
        self._nfc_company_combo = ttk.Combobox(
            left_panel,
            textvariable=self._nfc_company_var,
            values=company_names,
            font=self.fonts['normal'],
            state='readonly',
            width=23
        )
        self._nfc_company_combo.pack(fill=tk.X, pady=(5, 20))
        if company_names:
            self._nfc_company_combo.current(0)

        # 登錄按鈕
        enroll_nfc_btn = tk.Button(
            left_panel,
            text="💳  感應登錄卡片",
            font=self.fonts['button'],
            fg=self.colors['text'],
            bg=self.colors['nfc'],
            activebackground='#3a7bc8',
            activeforeground=self.colors['text'],
            bd=0,
            padx=20,
            pady=12,
            cursor='hand2',
            command=self._on_enroll_nfc_click
        )
        enroll_nfc_btn.pack(pady=20)

        self._nfc_status_label = tk.Label(
            left_panel,
            text="",
            font=self.fonts['normal'],
            fg=self.colors['text_secondary'],
            bg=self.colors['card'],
            wraplength=250
        )
        self._nfc_status_label.pack()

        # 右側：已登錄卡片列表
        right_panel = tk.Frame(self._nfc_frame, bg=self.colors['card'], padx=20, pady=20)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))

        tk.Label(
            right_panel,
            text="已登錄 NFC 卡片",
            font=self.fonts['title'],
            fg=self.colors['text'],
            bg=self.colors['card']
        ).pack(pady=(0, 10))

        # 列表框
        list_frame = tk.Frame(right_panel, bg=self.colors['card'])
        list_frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._nfc_listbox = tk.Listbox(
            list_frame,
            font=self.fonts['list'],
            bg=self.colors['accent'],
            fg=self.colors['text'],
            selectbackground=self.colors['nfc'],
            selectforeground=self.colors['text'],
            yscrollcommand=scrollbar.set,
            height=10
        )
        self._nfc_listbox.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self._nfc_listbox.yview)

        # 按鈕區
        btn_frame = tk.Frame(right_panel, bg=self.colors['card'])
        btn_frame.pack(pady=(15, 0))

        # 刪除按鈕
        delete_nfc_btn = tk.Button(
            btn_frame,
            text="🗑  刪除",
            font=self.fonts['normal'],
            fg=self.colors['text'],
            bg=self.colors['button'],
            activebackground='#ff6b6b',
            activeforeground=self.colors['text'],
            bd=0,
            padx=15,
            pady=8,
            cursor='hand2',
            command=self._on_delete_nfc_click
        )
        delete_nfc_btn.pack(side=tk.LEFT, padx=(0, 10))

        # 停用/啟用按鈕
        toggle_btn = tk.Button(
            btn_frame,
            text="🔒  停用/啟用",
            font=self.fonts['normal'],
            fg=self.colors['text'],
            bg=self.colors['accent'],
            activebackground=self.colors['card'],
            activeforeground=self.colors['text'],
            bd=0,
            padx=15,
            pady=8,
            cursor='hand2',
            command=self._on_toggle_nfc_click
        )
        toggle_btn.pack(side=tk.LEFT)

    def _switch_tab(self, tab: str):
        """切換分頁"""
        self._current_tab = tab

        # 隱藏所有分頁
        self._face_frame.pack_forget()
        if self.nfc_enabled:
            self._nfc_frame.pack_forget()

        # 更新分頁按鈕樣式
        if self.nfc_enabled:
            if tab == "face":
                self._face_tab_btn.configure(bg=self.colors['success'])
                self._nfc_tab_btn.configure(bg=self.colors['accent'])
            else:
                self._face_tab_btn.configure(bg=self.colors['accent'])
                self._nfc_tab_btn.configure(bg=self.colors['nfc'])

        # 顯示對應分頁
        if tab == "face":
            self._face_frame.pack(fill=tk.BOTH, expand=True)
            self.refresh_user_list()
        else:
            self._nfc_frame.pack(fill=tk.BOTH, expand=True)
            self.refresh_nfc_list()

    def show(self):
        """顯示管理介面"""
        # 先驗證密碼
        if not self._verify_password():
            return

        self._switch_tab("face")
        self.frame.pack(fill=tk.BOTH, expand=True)

    def hide(self):
        """隱藏管理介面"""
        if self.frame:
            self.frame.pack_forget()
        self._password_verified = False

    def _verify_password(self) -> bool:
        """驗證管理員密碼"""
        password = simpledialog.askstring(
            "管理員驗證",
            "請輸入管理密碼：",
            show='*'
        )

        # 預設密碼：admin (實際使用時應改為更安全的方式)
        if password == "admin":
            self._password_verified = True
            return True
        elif password is not None:
            messagebox.showerror("錯誤", "密碼錯誤")

        return False

    def refresh_user_list(self):
        """重新整理使用者列表"""
        if not self._user_listbox:
            return

        self._user_listbox.delete(0, tk.END)

        if self.get_users:
            users = self.get_users()
            for user in users:
                company_name = self.companies.get(user.company_id, {}).get('name', '未知')
                self._user_listbox.insert(
                    tk.END,
                    f"[{user.id}] {user.name} - {company_name}"
                )

    def refresh_nfc_list(self):
        """重新整理 NFC 卡片列表"""
        if not self._nfc_listbox or not self.get_nfc_cards:
            return

        self._nfc_listbox.delete(0, tk.END)

        cards = self.get_nfc_cards()
        for card in cards:
            company_name = self.companies.get(card.company_id, {}).get('name', '未知')
            status = "✓" if card.active else "✗"
            self._nfc_listbox.insert(
                tk.END,
                f"[{card.id}] {status} {card.name} - {company_name} ({card.uid[:8]}...)"
            )

    def _on_enroll_click(self):
        """登錄按鈕點擊處理"""
        name = self._name_entry.get().strip()
        if not name:
            self.set_status("請輸入姓名", "error")
            return

        company_selection = self._company_var.get()
        if not company_selection:
            self.set_status("請選擇公司", "error")
            return

        # 解析公司 ID
        try:
            company_id = int(company_selection.split('.')[0])
        except:
            self.set_status("公司選擇無效", "error")
            return

        self.set_status("請面對相機，3 秒後拍照...")

        if self.on_enroll:
            self.on_enroll(name, company_id)

    def _on_delete_click(self):
        """刪除按鈕點擊處理"""
        selection = self._user_listbox.curselection()
        if not selection:
            messagebox.showwarning("提示", "請先選擇要刪除的項目")
            return

        item_text = self._user_listbox.get(selection[0])

        if messagebox.askyesno("確認刪除", f"確定要刪除以下人臉？\n\n{item_text}"):
            # 解析 user_id
            try:
                user_id = int(item_text.split(']')[0].replace('[', ''))
                if self.on_delete:
                    self.on_delete(user_id)
                self.refresh_user_list()
                self.set_status("刪除成功", "success")
            except Exception as e:
                self.set_status(f"刪除失敗: {e}", "error")

    def _on_enroll_nfc_click(self):
        """NFC 登錄按鈕點擊處理"""
        name = self._nfc_name_entry.get().strip()
        if not name:
            self.set_nfc_status("請輸入持卡人姓名", "error")
            return

        company_selection = self._nfc_company_var.get()
        if not company_selection:
            self.set_nfc_status("請選擇公司", "error")
            return

        # 解析公司 ID
        try:
            company_id = int(company_selection.split('.')[0])
        except:
            self.set_nfc_status("公司選擇無效", "error")
            return

        self.set_nfc_status("請將卡片靠近讀卡器...")

        if self.on_enroll_nfc:
            self.on_enroll_nfc(name, company_id)

    def _on_delete_nfc_click(self):
        """NFC 刪除按鈕點擊處理"""
        selection = self._nfc_listbox.curselection()
        if not selection:
            messagebox.showwarning("提示", "請先選擇要刪除的卡片")
            return

        item_text = self._nfc_listbox.get(selection[0])

        if messagebox.askyesno("確認刪除", f"確定要刪除以下卡片？\n\n{item_text}"):
            try:
                card_id = int(item_text.split(']')[0].replace('[', ''))
                if self.on_delete_nfc:
                    self.on_delete_nfc(card_id)
                self.refresh_nfc_list()
                self.set_nfc_status("刪除成功", "success")
            except Exception as e:
                self.set_nfc_status(f"刪除失敗: {e}", "error")

    def _on_toggle_nfc_click(self):
        """NFC 停用/啟用按鈕點擊處理"""
        selection = self._nfc_listbox.curselection()
        if not selection:
            messagebox.showwarning("提示", "請先選擇要操作的卡片")
            return

        item_text = self._nfc_listbox.get(selection[0])

        try:
            card_id = int(item_text.split(']')[0].replace('[', ''))

            # 取得目前狀態
            is_active = "✓" in item_text

            if self.nfc_manager:
                self.nfc_manager.toggle_card_active(card_id, not is_active)
                self.refresh_nfc_list()

                status = "停用" if is_active else "啟用"
                self.set_nfc_status(f"卡片已{status}", "success")

        except Exception as e:
            self.set_nfc_status(f"操作失敗: {e}", "error")

    def _on_close_click(self):
        """關閉按鈕點擊處理"""
        self.hide()
        if self.on_close:
            self.on_close()

    def set_status(self, message: str, status_type: str = "info"):
        """
        設定人臉狀態訊息

        Args:
            message: 訊息內容
            status_type: 類型 (info, success, error)
        """
        color = {
            'info': self.colors['text_secondary'],
            'success': self.colors['success'],
            'error': self.colors['button']
        }.get(status_type, self.colors['text_secondary'])

        self._status_label.configure(text=message, fg=color)

    def set_nfc_status(self, message: str, status_type: str = "info"):
        """
        設定 NFC 狀態訊息

        Args:
            message: 訊息內容
            status_type: 類型 (info, success, error)
        """
        if not hasattr(self, '_nfc_status_label'):
            return

        color = {
            'info': self.colors['text_secondary'],
            'success': self.colors['success'],
            'error': self.colors['button']
        }.get(status_type, self.colors['text_secondary'])

        self._nfc_status_label.configure(text=message, fg=color)

    def clear_inputs(self):
        """清除輸入欄位"""
        self._name_entry.delete(0, tk.END)
        if self._company_combo['values']:
            self._company_combo.current(0)

        if hasattr(self, '_nfc_name_entry'):
            self._nfc_name_entry.delete(0, tk.END)
        if hasattr(self, '_nfc_company_combo') and self._nfc_company_combo['values']:
            self._nfc_company_combo.current(0)


# =============================================================================
# 測試程式
# =============================================================================
if __name__ == "__main__":
    from dataclasses import dataclass

    @dataclass
    class MockUser:
        id: int
        name: str
        company_id: int

    @dataclass
    class MockNFCCard:
        id: int
        uid: str
        name: str
        company_id: int
        active: bool = True

    # 測試用資料
    test_companies = {
        1: {"name": "公司 A", "extension": "101"},
        2: {"name": "公司 B", "extension": "102"},
        3: {"name": "公司 C", "extension": "103"},
    }

    mock_users = [
        MockUser(1, "張三", 1),
        MockUser(2, "李四", 2),
        MockUser(3, "王五", 1),
    ]

    mock_nfc_cards = [
        MockNFCCard(1, "A1B2C3D4E5F6", "陳小明", 1, True),
        MockNFCCard(2, "1234567890AB", "林小華", 2, True),
        MockNFCCard(3, "AABBCCDD1122", "王大明", 1, False),
    ]

    def on_enroll(name, cid):
        print(f"登錄人臉: {name}, 公司 {cid}")
        admin_win.set_status("登錄成功！", "success")

    def on_delete(uid):
        print(f"刪除人臉: {uid}")

    def get_users():
        return mock_users

    def on_enroll_nfc(name, cid):
        print(f"登錄 NFC 卡片: {name}, 公司 {cid}")
        admin_win.set_nfc_status("登錄成功！", "success")

    def on_delete_nfc(card_id):
        print(f"刪除 NFC 卡片: {card_id}")

    def get_nfc_cards():
        return mock_nfc_cards

    def on_close():
        print("關閉管理介面")

    # 模擬 NFC Manager
    class MockNFCManager:
        def toggle_card_active(self, card_id, active):
            print(f"切換卡片 {card_id} 狀態為 {active}")

    # 建立視窗
    root = tk.Tk()
    root.title("管理介面")
    root.geometry("800x480")
    root.configure(bg='#1a1a2e')

    # 建立管理介面
    admin_win = AdminWindow(
        root,
        test_companies,
        on_enroll=on_enroll,
        on_delete=on_delete,
        on_close=on_close,
        get_users=get_users,
        nfc_manager=MockNFCManager(),
        on_enroll_nfc=on_enroll_nfc,
        on_delete_nfc=on_delete_nfc,
        get_nfc_cards=get_nfc_cards
    )

    admin_win.show()

    root.mainloop()
