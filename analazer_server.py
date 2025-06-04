import socket
import pickle
import threading
import struct
import pytesseract
import cv2
from collections import defaultdict
from violation_detections import FrameBatchProcessor  # Assuming you have this module

PORT = 9999

def extract_license_plate_text(roi):
    try:
        config = '--psm 8 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        return pytesseract.image_to_string(roi, config=config).strip()
    except Exception as e:
        return "[Error]"

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
        self.processor = FrameBatchProcessor(batch_size=60)
        print(f"[+] New client connected: {addr}")

    def handle_client(self):
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
                violations, violation_frames = self.processor.add_frame(frame)
                
                # Handle results when we have violations
                if violations and violation_frames:
                    for violation, violation_frame in zip(violations, violation_frames):
                        l, t, r, b = violation['position']
                        plate_roi = violation_frame[t:b, l:r]
                        plate_text = extract_license_plate_text(plate_roi)
                        
                        print(f"🚦 Violation | ID: {violation['track_id']} | Plate: {plate_text} | "
                              f"Reason: {violation['violation']} | Points: {violation['points']}")
                        
                        cv2.imshow(f"Violation {self.addr}", violation_frame)

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