import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(3)
try:
    s.connect(("127.0.0.1", 5038))
    welcome = s.recv(1024).decode().strip()
    print("Connected!", welcome)
    login = "Action: Login\r\nUsername: intercom\r\nSecret: intercom123\r\n\r\n"
    s.send(login.encode())
    import time
    time.sleep(1)
    resp = s.recv(4096).decode().strip()
    print("Response:", resp)
    s.close()
except Exception as e:
    print("Error:", e)
