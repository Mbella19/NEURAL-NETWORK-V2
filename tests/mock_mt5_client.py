
import socket
import json
import struct
import time
import random
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MockClient")

HOST = "127.0.0.1"
PORT = 5555

def generate_random_ohlcv(count=60):
    data = []
    t = int(time.time())
    price = 1.1000
    for i in range(count):
        t_bar = t - (count - i) * 60 * 15
        o = price + random.uniform(-0.0005, 0.0005)
        c = o + random.uniform(-0.0005, 0.0005)
        h = max(o, c) + random.uniform(0, 0.0002)
        l = min(o, c) - random.uniform(0, 0.0002)
        v = int(random.uniform(100, 1000))
        data.append([t_bar, o, h, l, c, v])
        price = c
    return data

def run_mock_client():
    logger.info(f"Connecting to {HOST}:{PORT}...")
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((HOST, PORT))
        logger.info("Connected!")
        
        # Prepare Payload
        payload = {
            "rates": {
                "15m": generate_random_ohlcv(200),
                "1h": generate_random_ohlcv(100),
                "4h": generate_random_ohlcv(50)
            },
            "position": {
                "type": -1, # None
                "volume": 0.0,
                "price": 0.0,
                "sl": 0.0,
                "tp": 0.0,
                "profit": 0.0
            },
            "account": {
                "balance": 10000.0,
                "equity": 10000.0
            }
        }
        
        json_str = json.dumps(payload)
        json_bytes = json_str.encode('utf-8')
        msg_len = len(json_bytes)
        
        logger.info(f"Sending {msg_len} bytes...")
        
        # Send Header + Body
        sock.sendall(struct.pack('>I', msg_len))
        sock.sendall(json_bytes)
        
        # Receive Response
        header = sock.recv(4)
        if not header:
            logger.error("No response header received")
            return
            
        resp_len = struct.unpack('>I', header)[0]
        logger.info(f"Response length: {resp_len}")
        
        resp_bytes = sock.recv(resp_len)
        resp_str = resp_bytes.decode('utf-8')
        
        logger.info(f"Response: {resp_str}")
        
        resp_json = json.loads(resp_str)
        if "action" in resp_json:
            logger.info("✓ Test Passed: Valid Action received")
        else:
            logger.error("✗ Test Failed: No action in response")
            
    except ConnectionRefusedError:
        logger.error("Connection failed. Is the server running?")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    finally:
        sock.close()

if __name__ == "__main__":
    run_mock_client()
