import cv2
import socket
import pickle
import struct

server_ip = "127.0.0.1"
server_port = 9999

# Create TCP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((server_ip, server_port))

cap = cv2.VideoCapture("traffic_video.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Serialize the frame
    data = pickle.dumps(frame)
    
    # Pack the length of data into 4 bytes
    data_length = struct.pack("!I", len(data))
    
    # Send length followed by the actual data
    sock.sendall(data_length + data)

cap.release()
sock.close()
