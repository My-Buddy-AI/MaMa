import socket
import json
import time

def send_message(port, message, retries=3, wait_time=0.02):
    """
    Send a JSON message to the specified port. Retries if the connection fails, with a wait time between retries.
    
    Args:
        port (int): The port to send the message to.
        message (dict): The message to send.
        retries (int): Number of retry attempts.
        wait_time (float): Time (in seconds) to wait between retries. Default is 20ms (0.02 seconds).
    """
    for attempt in range(retries):
        try:
            # Create a socket connection
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect(('localhost', port))

            # Send the message
            s.sendall(json.dumps(message).encode('utf-8'))

            # Close the socket after sending
            s.close()

            print(f"Message successfully sent to port {port}")
            return  # Exit the function if the message is sent successfully

        except socket.error as e:
            print(f"Failed to send message to port {port}: {e}")
            if attempt < retries - 1:
                print(f"Retrying in {wait_time * 1000}ms... (Attempt {attempt + 1} of {retries})")
                time.sleep(wait_time)  # Wait for the specified wait time before retrying
            else:
                print(f"All {retries} attempts to send the message failed.")

def receive_message(port):
    """
    Listen for incoming messages on the specified port.
    
    Args:
        port (int): The port to listen on.

    Returns:
        dict: The received message, parsed from JSON format.
    """
    try:
        # Create a socket to listen for incoming connections
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(('localhost', port))
        s.listen(1)

        print(f"Listening for messages on port {port}...")

        conn, addr = s.accept()
        with conn:
            print(f"Connection established with {addr}")
            data = conn.recv(1024)
            if data:
                message = json.loads(data.decode('utf-8'))
                print(f"Message received: {message}")
                return message

    except socket.error as e:
        print(f"Failed to receive message on port {port}: {e}")

    finally:
        s.close()
