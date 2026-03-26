# -*- coding: utf-8 -*-
"""
主介面 - 選擇公司撥號
"""

import tkinter as tk
from tkinter import ttk, font
from typing import Callable, Dict, Optional
import sys
sys.path.append('..')


class MainWindow:
    """
    主介面類別

    顯示 8 間公司的按鈕，點擊後撥打對應分機
    管理介面已移至網頁版
    """

    def __init__(
        self,
        root: tk.Tk,
        companies: Dict,
        on_company_selected: Optional[Callable[[int, Dict], None]] = None,
        on_password_click: Optional[Callable[[], None]] = None
    ):
        """
        初始化主介面

        Args:
            root: Tkinter 根視窗
            companies: 公司資料字典 {id: {name, extension, floor}}
            on_company_selected: 選擇公司時的回調 (company_id, company_info)
        """
        self.root = root
        self.companies = companies
        self.on_company_selected = on_company_selected
        self.on_password_click = on_password_click

        self.frame: Optional[tk.Frame] = None
        self._setup_styles()
        self._create_widgets()

    def _setup_styles(self):
        """設定樣式"""
        self.colors = {
            'bg': '#1a1a2e',           # 深藍背景
            'card': '#16213e',          # 卡片背景
            'accent': '#0f3460',        # 強調色
            'text': '#ffffff',          # 主文字
            'text_secondary': '#a0a0a0', # 次要文字
            'button': '#e94560',        # 按鈕色
            'button_hover': '#ff6b6b',  # 按鈕 hover
            'success': '#00d9a0',       # 成功色
        }

        # 自訂字型 - 使用 Noto Sans CJK TC（繁體中文）
        # 螢幕解析度 1920x1080，字體需要夠大才看得清楚
        _font_family = 'Noto Sans CJK TC'
        self.fonts = {
            'title': (_font_family, 42, 'bold'),
            'subtitle': (_font_family, 22),
            'button': (_font_family, 36, 'bold'),     # 公司名字 - 大字
            'button_sub': (_font_family, 20),          # 樓層資訊
            'small': (_font_family, 18),               # 分機號碼
        }

    def _create_widgets(self):
        """建立介面元件"""
        # 主框架
        self.frame = tk.Frame(self.root, bg=self.colors['bg'])
        self.frame.pack(fill=tk.BOTH, expand=True)

        # 標題區域
        header = tk.Frame(self.frame, bg=self.colors['bg'])
        header.pack(fill=tk.X, pady=(20, 10))

        title = tk.Label(
            header,
            text="歡迎光臨",
            font=self.fonts['title'],
            fg=self.colors['text'],
            bg=self.colors['bg']
        )
        title.pack()

        subtitle = tk.Label(
            header,
            text="請選擇要聯繫的公司",
            font=self.fonts['subtitle'],
            fg=self.colors['text_secondary'],
            bg=self.colors['bg']
        )
        subtitle.pack()

        # 公司按鈕區域
        button_frame = tk.Frame(self.frame, bg=self.colors['bg'])
        button_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # 4x2 網格配置
        for i in range(4):
            button_frame.columnconfigure(i, weight=1, uniform='col')
        for i in range(2):
            button_frame.rowconfigure(i, weight=1, uniform='row')

        # 建立 8 個公司按鈕
        for idx, (company_id, info) in enumerate(self.companies.items()):
            row = idx // 4
            col = idx % 4

            btn = self._create_company_button(
                button_frame,
                company_id,
                info['name'],
                info.get('floor', ''),
                info['extension']
            )
            btn.grid(row=row, column=col, padx=8, pady=8, sticky='nsew')

        # 底部區域
        footer = tk.Frame(self.frame, bg=self.colors['bg'])
        footer.pack(fill=tk.X, pady=10)

        # 密碼開門按鈕
        password_btn = tk.Button(
            footer,
            text="🔐 密碼開門",
            font=self.fonts['button_sub'],
            fg=self.colors['text'],
            bg=self.colors['accent'],
            activebackground=self.colors['button'],
            activeforeground=self.colors['text'],
            width=12,
            height=1,
            relief='flat',
            cursor='hand2',
            command=self._on_password_click
        )
        password_btn.pack(pady=(0, 10))

        # 提示訊息（管理介面已移至網頁版）
        fp_hint = tk.Label(
            footer,
            text="💡 住戶請使用人臉辨識、NFC 或密碼開門",
            font=self.fonts['small'],
            fg=self.colors['text_secondary'],
            bg=self.colors['bg']
        )
        fp_hint.pack(padx=20)

        # SIP 分機狀態列（由 set_sip_offline_hint() 更新）
        self.status_label = tk.Label(
            footer,
            text="系統就緒",
            font=self.fonts['small'],
            fg=self.colors['text_secondary'],
            bg=self.colors['bg']
        )
        self.status_label.pack(pady=(4, 0))

    def _create_company_button(
        self,
        parent: tk.Frame,
        company_id: int,
        name: str,
        floor: str,
        extension: str
    ) -> tk.Frame:
        """
        建立公司按鈕

        Args:
            parent: 父容器
            company_id: 公司 ID
            name: 公司名稱
            floor: 樓層
            extension: 分機號碼

        Returns:
            tk.Frame: 按鈕框架
        """
        # 按鈕框架
        btn_frame = tk.Frame(
            parent,
            bg=self.colors['card'],
            cursor='hand2'
        )

        # 公司名稱（置中）
        name_label = tk.Label(
            btn_frame,
            text=name,
            font=self.fonts['button'],
            fg=self.colors['text'],
            bg=self.colors['card'],
            anchor='center',
            justify='center'
        )
        name_label.pack(pady=(15, 5), fill=tk.X)

        # 樓層資訊（置中）
        if floor:
            floor_label = tk.Label(
                btn_frame,
                text=floor,
                font=self.fonts['button_sub'],
                fg=self.colors['text_secondary'],
                bg=self.colors['card'],
                anchor='center',
                justify='center'
            )
            floor_label.pack(fill=tk.X)

        # 分機號碼（置中）
        ext_label = tk.Label(
            btn_frame,
            text=f"分機 {extension}",
            font=self.fonts['small'],
            fg=self.colors['accent'],
            bg=self.colors['card'],
            anchor='center',
            justify='center'
        )
        ext_label.pack(pady=(5, 15), fill=tk.X)

        # 綁定點擊事件
        def on_click(event=None):
            self._on_company_click(company_id)

        def on_enter(event):
            btn_frame.configure(bg=self.colors['accent'])
            for child in btn_frame.winfo_children():
                child.configure(bg=self.colors['accent'])

        def on_leave(event):
            btn_frame.configure(bg=self.colors['card'])
            for child in btn_frame.winfo_children():
                child.configure(bg=self.colors['card'])

        # 綁定到所有子元件
        for widget in [btn_frame, name_label, ext_label]:
            widget.bind('<Button-1>', on_click)
            widget.bind('<Enter>', on_enter)
            widget.bind('<Leave>', on_leave)

        if floor:
            floor_label.bind('<Button-1>', on_click)
            floor_label.bind('<Enter>', on_enter)
            floor_label.bind('<Leave>', on_leave)

        return btn_frame

    def _on_company_click(self, company_id: int):
        """公司按鈕點擊處理"""
        if self.on_company_selected and company_id in self.companies:
            self.on_company_selected(company_id, self.companies[company_id])

    def _on_password_click(self):
        """密碼開門按鈕點擊處理"""
        if self.on_password_click:
            self.on_password_click()

    def show(self):
        """顯示主介面"""
        if self.frame:
            self.frame.pack(fill=tk.BOTH, expand=True)

    def hide(self):
        """隱藏主介面"""
        if self.frame:
            self.frame.pack_forget()

    def show_message(self, message: str, message_type: str = "info"):
        """
        顯示訊息提示

        Args:
            message: 訊息內容
            message_type: 訊息類型 (info, success, error)
        """
        color = {
            'info': self.colors['text'],
            'success': self.colors['success'],
            'error': self.colors['button']
        }.get(message_type, self.colors['text'])

        # 建立訊息標籤
        msg_label = tk.Label(
            self.frame,
            text=message,
            font=self.fonts['subtitle'],
            fg=color,
            bg=self.colors['bg']
        )
        msg_label.place(relx=0.5, rely=0.5, anchor='center')

        # 2 秒後移除
        self.root.after(2000, msg_label.destroy)

    def set_sip_offline_hint(self, offline_list: list):
        """
        更新 SIP 分機離線狀態列

        Args:
            offline_list: 離線分機顯示名稱清單（空白 = 全部在線）
        """
        if not hasattr(self, 'status_label') or self.status_label is None:
            return
        if not offline_list:
            self.status_label.config(
                text="系統就緒",
                fg=self.colors['text_secondary']
            )
        else:
            names = "、".join(offline_list)
            self.status_label.config(
                text=f"⚠ {names} 目前離線",
                fg=self.colors['button']   # '#e94560' 紅色
            )

    def update_companies(self, new_companies: Dict):
        """
        動態更新公司資料（從資料庫讀取後呼叫）

        Args:
            new_companies: 新的公司資料字典
        """
        # 檢查是否有變化
        if new_companies == self.companies:
            return  # 沒有變化，不需更新

        self.companies = new_companies

        # 重新建立整個介面
        if self.frame:
            self.frame.destroy()

        self._create_widgets()


# =============================================================================
# 測試程式
# =============================================================================
if __name__ == "__main__":
    # 測試用公司資料
    test_companies = {
        1: {"name": "公司 A", "extension": "101", "floor": "1F"},
        2: {"name": "公司 B", "extension": "102", "floor": "2F"},
        3: {"name": "公司 C", "extension": "103", "floor": "3F"},
        4: {"name": "公司 D", "extension": "104", "floor": "4F"},
        5: {"name": "公司 E", "extension": "105", "floor": "5F"},
        6: {"name": "公司 F", "extension": "106", "floor": "6F"},
        7: {"name": "公司 G", "extension": "107", "floor": "7F"},
        8: {"name": "公司 H", "extension": "108", "floor": "8F"},
    }

    def on_select(cid, info):
        print(f"選擇了: {info['name']} (分機 {info['extension']})")

    # 建立視窗
    root = tk.Tk()
    root.title("門口對講機")
    root.geometry("800x480")
    root.configure(bg='#1a1a2e')

    # 建立主介面（管理介面已移至網頁版）
    main_win = MainWindow(
        root,
        test_companies,
        on_company_selected=on_select
    )

    root.mainloop()
