
import socket
import json
import struct
import logging
import time
import sys
from typing import Optional

from config.settings import config
from src.live.bridge import LiveSession

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("LiveServer")

class TradingServer:
    """
    TCP Socket Server for communicating with MT5 EA.
    
    Protocol:
    1. Client connects.
    2. Client sends 4-byte integer (Big Endian) = Length of JSON payload.
    3. Client sends JSON payload string.
    4. Server processes and sends response (Length + JSON).
    """
    
    def __init__(self, host: str = "127.0.0.1", port: int = 5555):
        self.host = host
        self.port = port
        self.sock: Optional[socket.socket] = None
        self.session: Optional[LiveSession] = None
        
    def start(self):
        """Start the server and live session."""
        logger.info(f"Starting Trading Server on {self.host}:{self.port}...")
        
        # Initialize Brain
        try:
            self.session = LiveSession(config)
            logger.info("✓ Brain (LiveSession) initialized successfully.")
        except Exception as e:
            logger.critical(f"Failed to initialize Brain: {e}")
            sys.exit(1)
            
        # Bind Socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            self.sock.bind((self.host, self.port))
            self.sock.listen(1)
            logger.info("Waiting for MT5 connection...")
            
            while True:
                conn, addr = self.sock.accept()
                logger.info(f"Client connected from {addr}")
                self._handle_client(conn)
                logger.warning("Client disconnected. Waiting for reconnection...")
                
        except KeyboardInterrupt:
            logger.info("Server stopping...")
        except Exception as e:
            logger.error(f"Server error: {e}", exc_info=True)
        finally:
            if self.sock:
                self.sock.close()

    def _handle_client(self, conn: socket.socket):
        """Handle individual client connection loop."""
        buffer = b""
        
        try:
            while True:
                # 1. Read Message Length (4 bytes)
                header = self._recv_exact(conn, 4)
                if not header:
                    break
                    
                msg_len = struct.unpack('>I', header)[0]
                
                # 2. Read JSON Payload
                payload_bytes = self._recv_exact(conn, msg_len)
                if not payload_bytes:
                    break
                    
                payload_str = payload_bytes.decode('utf-8')
                # logger.debug(f"Received: {payload_str[:100]}...")
                
                try:
                    data = json.loads(payload_str)
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON received: {e}")
                    logger.error(f"Raw Payload: {payload_str}")
                    continue
                
                # 3. Process with Brain
                # Measure latency
                t0 = time.time()
                response_data = self.session.on_tick(data)
                dt = (time.time() - t0) * 1000
                
                # logger.info(f"Processed in {dt:.1f}ms. Action: {response_data.get('action')}")
                
                # 4. Send Response
                resp_str = json.dumps(response_data)
                resp_bytes = resp_str.encode('utf-8')
                
                # Send Length + Data
                conn.sendall(struct.pack('>I', len(resp_bytes)))
                conn.sendall(resp_bytes)
                
        except Exception as e:
            logger.error(f"Connection error: {e}")
        finally:
            conn.close()

    def _recv_exact(self, conn: socket.socket, n: int) -> Optional[bytes]:
        """Receive exactly n bytes."""
        data = b""
        while len(data) < n:
            try:
                packet = conn.recv(n - len(data))
                if not packet:
                    return None
                data += packet
            except ConnectionError:
                return None
        return data

if __name__ == "__main__":
    server = TradingServer(
        host=config.live.host,
        port=config.live.port
    )
    server.start()
