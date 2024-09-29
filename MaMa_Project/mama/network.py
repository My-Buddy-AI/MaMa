import socket
import json

def send_message(port: int, message: dict):
    """Send a message to a specific port."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(('localhost', port))
    s.sendall(json.dumps(message).encode('utf-8'))
    s.close()

def receive_message(port: int):
    """Receive a message on a specific port."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('localhost', port))
    s.listen(1)
    conn, addr = s.accept()
    data = conn.recv(1024)
    conn.close()
    return json.loads(data.decode('utf-8'))
