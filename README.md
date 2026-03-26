# 門口對講機 + 指紋門禁系統

基於樹莓派的門口對講機系統，支援 8 間公司呼叫及指紋開門功能。

## 功能

- 觸控螢幕選擇公司，透過 SIP 撥打分機
- 指紋辨識開門
- 通話中遠端開門（對方按 # 鍵）
- 管理介面：登錄/刪除指紋

## 硬體需求

| 品項 | 規格 |
|------|------|
| 樹莓派 | Pi 4 (4GB+) |
| 觸控螢幕 | 7吋 800x480 |
| 指紋模組 | R307 或 R503 |
| USB 音效卡 | - |
| 喇叭 + 麥克風 | - |
| 繼電器模組 | 5V 1路 |
| 電磁鎖 | 12V DC |

## GPIO 接線

```
指紋模組 (R307):
  VCC → 3.3V (Pin 1)
  GND → GND (Pin 6)
  TX  → GPIO 15 (Pin 10)
  RX  → GPIO 14 (Pin 8)

繼電器:
  VCC → 5V (Pin 2)
  GND → GND (Pin 14)
  IN  → GPIO 17 (Pin 11)
```

## 安裝

### 1. 安裝系統依賴

```bash
sudo apt update
sudo apt install python3-pip python3-tk

# PJSIP (SIP 通話)
sudo apt install libpjproject-dev
```

### 2. 安裝 Python 套件

```bash
pip3 install -r requirements.txt
```

### 3. 設定 UART (指紋模組)

```bash
sudo raspi-config
# Interface Options → Serial Port
# Login shell: No
# Serial hardware: Yes
```

### 4. 修改設定檔

編輯 `config.py`，設定 SIP 伺服器資訊和公司分機。

## 執行

```bash
python3 main.py
```

## 設定 SIP 伺服器

建議使用 Asterisk 自建 SIP 伺服器。分機規劃：

```
100 - 門口對講機
101~108 - 公司 1~8
```

各公司使用 Zoiper 或 Linphone App 接聽。

## 開機自動啟動

建立 systemd 服務：

```bash
sudo nano /etc/systemd/system/intercom.service
```

內容：

```ini
[Unit]
Description=Intercom System
After=network.target

[Service]
ExecStart=/usr/bin/python3 /home/pi/intercom_system/main.py
WorkingDirectory=/home/pi/intercom_system
User=pi
Restart=always

[Install]
WantedBy=multi-user.target
```

啟用服務：

```bash
sudo systemctl enable intercom
sudo systemctl start intercom
```

## 管理密碼

預設密碼：`admin`

請在 `gui/admin_window.py` 中修改。
