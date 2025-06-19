import socket
import pickle
import threading
import struct
import cv2
from collections import defaultdict
from violation_detections import FrameBatchProcessor  # Assuming you have this module
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad
from Crypto.Random import get_random_bytes


PORT = 9999

# Key variable
AES_KEY = b"this_is_a_32_byte_super_secret_key!!"

def recv_exact(conn, length):
    """Ensure exactly 'length' bytes are read from the TCP stream"""
    data = b""
    while len(data) < length:
        packet = conn.recv(length - len(data))
        if not packet:
            return None
        data += packet
    return data

class ClientHandler:
    def __init__(self, addr, conn):
        self.addr = addr
        self.conn = conn
        print(f"[+] New client connected: {addr}")

    def handle_client(self):
        
        # Recive position of the stop line
        raw_stop_line_positionlen = recv_exact(self.conn, 4)
        stop_line_positionlen = struct.unpack('!I', raw_stop_line_positionlen)[0]
        raw_stop_line_position = recv_exact(self.conn, stop_line_positionlen).decode()
        stop_line_position = raw_stop_line_position.split("#")[:-1]
        print(stop_line_position)
        processor = FrameBatchProcessor(stop_line_position)

        while True:
           # try:
                # Receive frame
                raw_msglen = recv_exact(self.conn, 4)
                if not raw_msglen:
                    break
                msglen = struct.unpack('!I', raw_msglen)[0]
                data = recv_exact(self.conn, msglen)
                if data is None:
                    break

                frame = pickle.loads(data)

                # Process frame (returns only when batch is complete)
                violations, violation_frames = processor.add_frame(frame)
                
                # Handle results when we have violations
                if violations and violation_frames:
                    for violation, violation_frame in zip(violations, violation_frames):
                        plate_text = violation["license_plate_number"]
                        

                        msg = f"🚦 Violation | Plate: {plate_text} | Reason: {violation['violation']} | Points: {violation['points']}"
                        print(msg)
                        violation_data = {
                            "message": msg,
                            "frame": violation_frame
                        }
                        raw_data = pickle.dumps(violation_data)

                        # Encrypt
                        iv = get_random_bytes(16)
                        cipher = AES.new(AES_KEY, AES.MODE_CBC, iv)
                        encrypted_data = cipher.encrypt(pad(raw_data, AES.block_size))

                        encrypted_msg_len = struct.pack('!I', len(encrypted_data))
                        try:
                            self.conn.sendall(iv + encrypted_msg_len + encrypted_data)
                        except Exception as e:
                            print(f"[!] Failed to send violation data to client {self.addr}: {e}")

                else:
                    cv2.imshow("frame",frame)
                if cv2.waitKey(1) == ord('q'):
                    break

            #except Exception as e:
               # print(f"[!] Error handling client {self.addr}: {e}")
               # break

        self.conn.close()
        print(f"[-] Client disconnected: {self.addr}")

def main_server():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("0.0.0.0", PORT))
    sock.listen(5)
    print(f"🚦 Listening on TCP port {PORT}...")

    while True:
        conn, addr = sock.accept()
        handler = ClientHandler(addr, conn)
        threading.Thread(target=handler.handle_client, daemon=True).start()

if __name__ == "__main__":
    main_server()