import aiohttp
import asyncio

async def send_message(port: int, message: dict):
    """
    Sends an asynchronous HTTP POST request to a specified port with a given message.
    
    Args:
        port (int): The port number where the message should be sent.
        message (dict): The message to send in JSON format.
    
    Raises:
        Exception: If the message cannot be sent after retries.
    """
    url = f'http://localhost:{port}'

    retries = 0
    max_retries = 10
    wait_time = 0.1  # 100ms

    while retries < max_retries:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=message) as response:
                    if response.status == 200:
                        print(f"Message sent successfully to port {port}.")
                        return
                    else:
                        raise Exception(f"Failed to send message to port {port}. Status: {response.status}, Reason: {response.reason}")
        except Exception as e:
            retries += 1
            print(f"Failed to send message to port {port}. Attempt {retries}/{max_retries}. Error: {e}")

            if retries < max_retries:
                await asyncio.sleep(wait_time)
            else:
                raise Exception(f"Exceeded max retries. Could not send message to port {port}.")

async def receive_message(port: int):
    """
    Asynchronously listen on a specified port to receive a message.
    
    Args:
        port (int): The port number to listen on for incoming messages.
    
    Returns:
        dict: The received message as a dictionary.
    
    Raises:
        Exception: If the message cannot be received.
    """
    retries = 0
    max_retries = 10
    wait_time = 0.1  # 100ms

    server = None
    while retries < max_retries:
        try:
            # Define a coroutine that will be used to handle connections
            async def handle_connection(reader, writer):
                data = await reader.read(1024)
                message = data.decode()
                print(f"Received message: {message}")
                return message

            # Create a TCP server to listen on the specified port
            server = await asyncio.start_server(handle_connection, host='0.0.0.0', port=port)
            async with server:
                await server.serve_forever()

        except Exception as e:
            retries += 1
            print(f"Failed to receive message on port {port}. Attempt {retries}/{max_retries}. Error: {e}")

            if retries < max_retries:
                await asyncio.sleep(wait_time)
            else:
                raise Exception(f"Exceeded max retries. Could not receive message on port {port}.")
        finally:
            if server:
                server.close()
                await server.wait_closed()
