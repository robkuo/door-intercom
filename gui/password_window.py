# -*- coding: utf-8 -*-
"""
密碼鍵盤介面 - 輸入密碼開門
"""

import tkinter as tk
from typing import Callable, Optional


class PasswordWindow:
    """
    密碼鍵盤介面類別

    顯示數字鍵盤，讓使用者輸入密碼開門
    """

    def __init__(
        self,
        root: tk.Tk,
        on_password_submit: Optional[Callable[[str], None]] = None,
        on_cancel: Optional[Callable[[], None]] = None
    ):
        """
        初始化密碼鍵盤介面

        Args:
            root: Tkinter 根視窗
            on_password_submit: 密碼提交回調 (password)
            on_cancel: 取消回調
        """
        self.root = root
        self.on_password_submit = on_password_submit
        self.on_cancel = on_cancel

        self.frame: Optional[tk.Frame] = None
        self.password_var = tk.StringVar()
        self.message_label: Optional[tk.Label] = None

        self._setup_styles()
        self._create_widgets()

    def _setup_styles(self):
        """設定樣式"""
        self.colors = {
            'bg': '#1a1a2e',
            'card': '#16213e',
            'accent': '#0f3460',
            'text': '#ffffff',
            'text_secondary': '#a0a0a0',
            'button': '#e94560',
            'button_hover': '#ff6b6b',
            'success': '#00d9a0',
            'key': '#2d3a4f',
            'key_hover': '#3d4a5f',
        }

        self.fonts = {
            'title': ('Noto Sans CJK TC', 36, 'bold'),
            'display': ('Consolas', 48, 'bold'),
            'key': ('Noto Sans CJK TC', 36, 'bold'),
            'button': ('Noto Sans CJK TC', 22, 'bold'),
            'message': ('Noto Sans CJK TC', 18),
        }

    def _create_widgets(self):
        """建立介面元件"""
        # 主框架
        self.frame = tk.Frame(self.root, bg=self.colors['bg'])

        # 標題
        title = tk.Label(
            self.frame,
            text="🔐 請輸入密碼",
            font=self.fonts['title'],
            fg=self.colors['text'],
            bg=self.colors['bg']
        )
        title.pack(pady=(30, 20))

        # 密碼顯示框
        display_frame = tk.Frame(self.frame, bg=self.colors['card'], padx=20, pady=15)
        display_frame.pack(pady=10)

        self.display_label = tk.Label(
            display_frame,
            textvariable=self.password_var,
            font=self.fonts['display'],
            fg=self.colors['text'],
            bg=self.colors['card'],
            width=12,
            anchor='center'
        )
        self.display_label.pack()

        # 訊息標籤
        self.message_label = tk.Label(
            self.frame,
            text="",
            font=self.fonts['message'],
            fg=self.colors['text_secondary'],
            bg=self.colors['bg']
        )
        self.message_label.pack(pady=5)

        # 數字鍵盤
        keypad_frame = tk.Frame(self.frame, bg=self.colors['bg'])
        keypad_frame.pack(pady=20)

        # 鍵盤配置 (4x3)
        keys = [
            ['1', '2', '3'],
            ['4', '5', '6'],
            ['7', '8', '9'],
            ['C', '0', '⌫']
        ]

        for row_idx, row in enumerate(keys):
            for col_idx, key in enumerate(row):
                btn = self._create_key_button(keypad_frame, key)
                btn.grid(row=row_idx, column=col_idx, padx=5, pady=5)

        # 底部按鈕
        button_frame = tk.Frame(self.frame, bg=self.colors['bg'])
        button_frame.pack(pady=20)

        # 取消按鈕
        cancel_btn = tk.Button(
            button_frame,
            text="取消",
            font=self.fonts['button'],
            fg=self.colors['text'],
            bg=self.colors['accent'],
            activebackground=self.colors['key_hover'],
            activeforeground=self.colors['text'],
            width=10,
            height=2,
            relief='flat',
            cursor='hand2',
            command=self._on_cancel_click
        )
        cancel_btn.pack(side=tk.LEFT, padx=10)

        # 確認按鈕
        submit_btn = tk.Button(
            button_frame,
            text="確認",
            font=self.fonts['button'],
            fg=self.colors['text'],
            bg=self.colors['success'],
            activebackground='#00b080',
            activeforeground=self.colors['text'],
            width=10,
            height=2,
            relief='flat',
            cursor='hand2',
            command=self._on_submit_click
        )
        submit_btn.pack(side=tk.LEFT, padx=10)

    def _create_key_button(self, parent: tk.Frame, key: str) -> tk.Button:
        """建立鍵盤按鈕"""
        # 特殊鍵顏色
        if key == 'C':
            bg_color = self.colors['button']
            hover_color = self.colors['button_hover']
        elif key == '⌫':
            bg_color = self.colors['accent']
            hover_color = self.colors['key_hover']
        else:
            bg_color = self.colors['key']
            hover_color = self.colors['key_hover']

        btn = tk.Button(
            parent,
            text=key,
            font=self.fonts['key'],
            fg=self.colors['text'],
            bg=bg_color,
            activebackground=hover_color,
            activeforeground=self.colors['text'],
            width=3,
            height=1,
            relief='flat',
            cursor='hand2',
            command=lambda k=key: self._on_key_press(k)
        )

        # Hover 效果
        btn.bind('<Enter>', lambda e, b=btn, c=hover_color: b.configure(bg=c))
        btn.bind('<Leave>', lambda e, b=btn, c=bg_color: b.configure(bg=c))

        return btn

    def _on_key_press(self, key: str):
        """按鍵處理"""
        current = self.password_var.get()

        if key == 'C':
            # 清除
            self.password_var.set('')
            self._clear_message()
        elif key == '⌫':
            # 刪除最後一個字元
            self.password_var.set(current[:-1])
            self._clear_message()
        else:
            # 輸入數字（最多 8 位）
            if len(current) < 8:
                # 顯示星號而非實際密碼
                self.password_var.set(current + '●')
                # 儲存實際密碼
                if not hasattr(self, '_actual_password'):
                    self._actual_password = ''
                self._actual_password += key

    def _on_submit_click(self):
        """確認按鈕點擊"""
        password = getattr(self, '_actual_password', '')

        if not password:
            self._show_message("請輸入密碼", "error")
            return

        if len(password) < 4:
            self._show_message("密碼至少需要 4 位數", "error")
            return

        if self.on_password_submit:
            self.on_password_submit(password)

    def _on_cancel_click(self):
        """取消按鈕點擊"""
        self._reset()
        if self.on_cancel:
            self.on_cancel()

    def _show_message(self, message: str, msg_type: str = "info"):
        """顯示訊息"""
        color = {
            'info': self.colors['text_secondary'],
            'success': self.colors['success'],
            'error': self.colors['button']
        }.get(msg_type, self.colors['text_secondary'])

        if self.message_label:
            self.message_label.configure(text=message, fg=color)

    def _clear_message(self):
        """清除訊息"""
        if self.message_label:
            self.message_label.configure(text="")

    def _reset(self):
        """重設狀態"""
        self.password_var.set('')
        self._actual_password = ''
        self._clear_message()

    def show(self):
        """顯示密碼鍵盤介面"""
        self._reset()
        if self.frame:
            self.frame.pack(fill=tk.BOTH, expand=True)

    def hide(self):
        """隱藏密碼鍵盤介面"""
        self._reset()
        if self.frame:
            self.frame.pack_forget()

    def show_success(self, name: str):
        """顯示成功訊息"""
        self._show_message(f"歡迎 {name}！", "success")

    def show_error(self, message: str = "密碼錯誤"):
        """顯示錯誤訊息"""
        self._show_message(message, "error")
        # 清除輸入
        self.password_var.set('')
        self._actual_password = ''


# =============================================================================
# 測試程式
# =============================================================================
if __name__ == "__main__":
    def on_submit(password):
        print(f"提交密碼: {password}")

    def on_cancel():
        print("取消")

    root = tk.Tk()
    root.title("密碼開門測試")
    root.geometry("800x480")
    root.configure(bg='#1a1a2e')

    pwd_win = PasswordWindow(
        root,
        on_password_submit=on_submit,
        on_cancel=on_cancel
    )
    pwd_win.show()

    root.mainloop()
