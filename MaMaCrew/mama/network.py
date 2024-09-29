import socket
import json


def find_available_port(start_port: int = 8000, max_retries: int = 100) -> int:
    """
    Find an available port starting from `start_port`. If the port is in use, 
    try the next port until a free one is found or until `max_retries` is reached.
    
    Args:
        start_port (int): The starting port to check.
        max_retries (int): The number of ports to check before failing.
        
    Returns:
        int: The available port number.
    """
    for port in range(start_port, start_port + max_retries):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind(('localhost', port))  # Attempt to bind to the port
            s.close()
            return port  # Port is available
        except OSError:
            continue  # Try the next port
    raise OSError(f"Could not find available port after {max_retries} attempts")


def send_message(port: int, message: dict):
    """
    Send a message to a specific port. Retry sending if the port is not available.
    
    Args:
        port (int): The port to send the message to.
        message (dict): The message to send, which will be converted to JSON format.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect(('localhost', port))
        s.send(json.dumps(message).encode())
    except socket.error as e:
        print(f"Failed to send message to port {port}: {e}")
    finally:
        s.close()


def receive_message(start_port: int = 8000, max_retries: int = 100) -> dict:
    """
    Receive a message on an available port. If the default port is in use, 
    find the next available port within a given range.
    
    Args:
        start_port (int): The starting port to bind.
        max_retries (int): The number of ports to check if one is already in use.
        
    Returns:
        dict: The received message in JSON format.
    """
    port = find_available_port(start_port, max_retries)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('localhost', port))
    s.listen(1)
    print(f"Listening on port {port} for incoming messages...")
    conn, addr = s.accept()
    data = conn.recv(1024)
    conn.close()
    s.close()
    return json.loads(data.decode())
