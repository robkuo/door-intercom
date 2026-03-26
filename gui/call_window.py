# -*- coding: utf-8 -*-
"""
通話介面 - 顯示撥號狀態與通話中資訊
"""

import tkinter as tk
from typing import Callable, Optional
import time
import sys
sys.path.append('..')


class CallWindow:
    """
    通話介面類別

    顯示撥號中、響鈴中、通話中等狀態（含來電接聽模式）
    """

    def __init__(
        self,
        root: tk.Tk,
        on_hangup: Optional[Callable] = None,
        on_answer: Optional[Callable] = None
    ):
        """
        初始化通話介面

        Args:
            root: Tkinter 根視窗
            on_hangup: 掛斷時的回調
            on_answer: 接聽來電時的回調
        """
        self.root = root
        self.on_hangup = on_hangup
        self.on_answer = on_answer

        self.frame: Optional[tk.Frame] = None
        self._status_label: Optional[tk.Label] = None
        self._company_label: Optional[tk.Label] = None
        self._timer_label: Optional[tk.Label] = None
        self._hangup_btn: Optional[tk.Button] = None
        self._answer_btn: Optional[tk.Button] = None

        self._call_start_time: Optional[float] = None
        self._timer_running = False
        self._timer_id = None

        self._setup_styles()
        self._create_widgets()

    def _setup_styles(self):
        """設定樣式"""
        self.colors = {
            'bg': '#1a1a2e',
            'text': '#ffffff',
            'text_secondary': '#a0a0a0',
            'calling': '#ffa500',      # 撥號中 - 橘色
            'ringing': '#00bcd4',      # 響鈴中 - 青色
            'connected': '#00d9a0',    # 通話中 - 綠色
            'hangup': '#e94560',       # 掛斷 - 紅色
            'answer': '#00d9a0',       # 接聽 - 綠色
        }

        self.fonts = {
            'status': ('Noto Sans CJK TC', 36, 'bold'),
            'company': ('Noto Sans CJK TC', 48, 'bold'),
            'timer': ('Noto Sans CJK TC', 72, 'bold'),
            'button': ('Noto Sans CJK TC', 24, 'bold'),
        }

    def _create_widgets(self):
        """建立介面元件"""
        # 主框架
        self.frame = tk.Frame(self.root, bg=self.colors['bg'])

        # 狀態標籤
        self._status_label = tk.Label(
            self.frame,
            text="撥號中...",
            font=self.fonts['status'],
            fg=self.colors['calling'],
            bg=self.colors['bg']
        )
        self._status_label.pack(pady=(60, 20))

        # 公司名稱
        self._company_label = tk.Label(
            self.frame,
            text="",
            font=self.fonts['company'],
            fg=self.colors['text'],
            bg=self.colors['bg']
        )
        self._company_label.pack(pady=10)

        # 通話計時器
        self._timer_label = tk.Label(
            self.frame,
            text="00:00",
            font=self.fonts['timer'],
            fg=self.colors['text'],
            bg=self.colors['bg']
        )
        self._timer_label.pack(pady=30)

        # 按鈕列（接聽 + 掛斷）
        btn_row = tk.Frame(self.frame, bg=self.colors['bg'])
        btn_row.pack(pady=20)

        # 接聽按鈕（來電模式下顯示）
        self._answer_btn = tk.Button(
            btn_row,
            text="🟢  接聽",
            font=self.fonts['button'],
            fg=self.colors['text'],
            bg=self.colors['answer'],
            activebackground='#00ffbb',
            activeforeground=self.colors['text'],
            bd=0,
            padx=30,
            pady=15,
            cursor='hand2',
            command=self._on_answer_click
        )
        # 預設隱藏，來電時才顯示
        self._answer_btn.pack(side=tk.LEFT, padx=15)
        self._answer_btn.pack_forget()

        # 掛斷按鈕
        self._hangup_btn = tk.Button(
            btn_row,
            text="🔴  掛斷",
            font=self.fonts['button'],
            fg=self.colors['text'],
            bg=self.colors['hangup'],
            activebackground='#ff6b6b',
            activeforeground=self.colors['text'],
            bd=0,
            padx=40,
            pady=15,
            cursor='hand2',
            command=self._on_hangup_click
        )
        self._hangup_btn.pack(side=tk.LEFT, padx=15)

        # 提示文字
        hint_label = tk.Label(
            self.frame,
            text="對方可按 # 鍵為您開門",
            font=('Noto Sans CJK TC', 18),
            fg=self.colors['text_secondary'],
            bg=self.colors['bg']
        )
        hint_label.pack(pady=10)

    def is_visible(self) -> bool:
        """回傳通話介面是否目前顯示中"""
        if self.frame is None:
            return False
        try:
            return self.frame.winfo_ismapped()
        except Exception:
            return False

    def show(self, company_name: str = ""):
        """
        顯示通話介面（撥出模式）

        Args:
            company_name: 公司名稱
        """
        self._company_label.configure(text=company_name)
        self.set_status("dialing")
        # 撥出模式：隱藏接聽按鈕
        if self._answer_btn:
            self._answer_btn.pack_forget()
        self.frame.pack(fill=tk.BOTH, expand=True)

    def show_incoming(self, caller_name: str = ""):
        """
        顯示來電介面（來電模式：顯示接聽 + 掛斷按鈕）

        Args:
            caller_name: 來電方名稱或號碼
        """
        self._company_label.configure(text=caller_name or "來電")
        self.set_status("ringing")
        # 來電模式：顯示接聽按鈕
        if self._answer_btn:
            self._answer_btn.pack(side=tk.LEFT, padx=15)
        self.frame.pack(fill=tk.BOTH, expand=True)

    def hide(self):
        """隱藏通話介面"""
        self._stop_timer()
        if self.frame:
            self.frame.pack_forget()

    def set_status(self, status: str):
        """
        設定通話狀態

        Args:
            status: 狀態 (dialing, ringing, connected, disconnected)
        """
        status_config = {
            'dialing': {
                'text': '撥號中...',
                'color': self.colors['calling']
            },
            'ringing': {
                'text': '響鈴中...',
                'color': self.colors['ringing']
            },
            'connected': {
                'text': '通話中',
                'color': self.colors['connected']
            },
            'disconnected': {
                'text': '通話結束',
                'color': self.colors['text_secondary']
            },
            'unavailable': {
                'text': '⚠ 目前不在線',
                'color': self.colors['hangup']
            },
        }

        config = status_config.get(status, status_config['dialing'])
        self._status_label.configure(
            text=config['text'],
            fg=config['color']
        )

        if status == 'connected':
            self._start_timer()
        elif status == 'disconnected':
            self._stop_timer()

    def _start_timer(self):
        """開始計時"""
        self._call_start_time = time.time()
        self._timer_running = True
        self._update_timer()

    def _stop_timer(self):
        """停止計時"""
        self._timer_running = False
        if self._timer_id:
            self.root.after_cancel(self._timer_id)
            self._timer_id = None

    def _update_timer(self):
        """更新計時器顯示"""
        if not self._timer_running:
            return

        if self._call_start_time:
            elapsed = int(time.time() - self._call_start_time)
            minutes = elapsed // 60
            seconds = elapsed % 60
            self._timer_label.configure(text=f"{minutes:02d}:{seconds:02d}")

        self._timer_id = self.root.after(1000, self._update_timer)

    def reset_timer(self):
        """重置計時器"""
        self._timer_label.configure(text="00:00")
        self._call_start_time = None

    def _on_hangup_click(self):
        """掛斷按鈕點擊處理"""
        if self.on_hangup:
            self.on_hangup()

    def _on_answer_click(self):
        """接聽按鈕點擊處理"""
        # 接聽後隱藏接聽按鈕
        if self._answer_btn:
            self._answer_btn.pack_forget()
        if self.on_answer:
            self.on_answer()

    def show_door_opened(self):
        """顯示開門成功提示"""
        # 建立提示框
        popup = tk.Frame(
            self.frame,
            bg='#00d9a0',
            padx=30,
            pady=20
        )
        popup.place(relx=0.5, rely=0.7, anchor='center')

        label = tk.Label(
            popup,
            text="✓ 門已開啟",
            font=('Noto Sans CJK TC', 18, 'bold'),
            fg='white',
            bg='#00d9a0'
        )
        label.pack()

        # 2 秒後移除
        self.root.after(2000, popup.destroy)


# =============================================================================
# 測試程式
# =============================================================================
if __name__ == "__main__":
    def on_hangup():
        print("掛斷通話")
        call_win.hide()
        root.after(1000, root.destroy)

    # 建立視窗
    root = tk.Tk()
    root.title("通話中")
    root.geometry("800x480")
    root.configure(bg='#1a1a2e')

    # 建立通話介面
    call_win = CallWindow(root, on_hangup=on_hangup)
    call_win.show("公司 A")

    # 模擬狀態變化
    root.after(2000, lambda: call_win.set_status('ringing'))
    root.after(4000, lambda: call_win.set_status('connected'))
    root.after(6000, call_win.show_door_opened)

    root.mainloop()
