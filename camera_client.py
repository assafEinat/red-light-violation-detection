import cv2
import socket
import pickle
import struct
import threading
from tkinter import Tk, Button, Label, filedialog, messagebox, Text, END
from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad

# GUI variables
selected_file = None
coordinates = []
violation_text = None

# Networking setup
server_ip = "127.0.0.1"
server_port = 9999

# Key variable
AES_KEY = b"this_is_a_32_byte_super_secret_key!!"

def recv_exact(sock, length):
    """Ensure exactly 'length' bytes are read from the TCP stream"""
    data = b""
    while len(data) < length:
        more = sock.recv(length - len(data))
        if not more:
            return None
        data += more
    return data

def choose_video():
    global selected_file
    selected_file = filedialog.askopenfilename(
        title="Select Traffic Video",
        filetypes=[("Video Files", "*.mp4 *.avi *.mov")]
    )
    if selected_file:
        status_label.config(text=f"Video selected: {selected_file.split('/')[-1]}")
        choose_coords_button.config(state="normal")

def mouse_callback(event, x, y, flags, param):
    global coordinates
    if event == cv2.EVENT_LBUTTONDOWN and len(coordinates) < 2:
        coordinates.append((x, y))
        print(f"Selected point: ({x}, {y})")

def choose_coordinates():
    global coordinates
    coordinates.clear()
    cap = cv2.VideoCapture(selected_file)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        messagebox.showerror("Error", "Failed to read the video.")
        return

    cv2.namedWindow("Select Stop Line")
    cv2.setMouseCallback("Select Stop Line", mouse_callback)

    while True:
        temp_frame = frame.copy()
        if len(coordinates) == 1:
            cv2.circle(temp_frame, coordinates[0], 5, (0, 255, 0), -1)
        elif len(coordinates) == 2:
            cv2.rectangle(temp_frame, coordinates[0], coordinates[1], (0, 255, 0), 2)
        cv2.imshow("Select Stop Line", temp_frame)

        key = cv2.waitKey(1)
        if len(coordinates) == 2:
            break
        if key == 27:  # ESC to cancel
            coordinates.clear()
            break

    cv2.destroyAllWindows()

    if len(coordinates) == 2:
        status_label.config(text="Stop line selected.")
        start_button.config(state="normal")

def start_stream(root):
    global selected_file, coordinates, violation_text

    if not selected_file or len(coordinates) != 2:
        messagebox.showerror("Error", "Please select a video and stop line.")
        return

    x1 = min(coordinates[0][0], coordinates[1][0])
    y1 = min(coordinates[0][1], coordinates[1][1])
    x2 = max(coordinates[0][0], coordinates[1][0])
    y2 = max(coordinates[0][1], coordinates[1][1])
    stop_line_position = f"{x1}#{y1}#{x2}#{y2}#"

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((server_ip, server_port))

        # Send stop line position
        stop_line_len = struct.pack("!I", len(stop_line_position))
        sock.sendall(stop_line_len + stop_line_position.encode())

        # Thread to receive violations
        def receive_violations():
            while True:
                try:
                    # Receive IV
                    iv = recv_exact(sock, 16)
                    if not iv:
                        break

                    # Receive encrypted length
                    raw_msglen = recv_exact(sock, 4)
                    if not raw_msglen:
                        break
                    msglen = struct.unpack('!I', raw_msglen)[0]

                    # Receive ciphertext
                    encrypted_data = recv_exact(sock, msglen)
                    if not encrypted_data:
                        break

                    # Decrypt
                    cipher = AES.new(AES_KEY, AES.MODE_CBC, iv)
                    decrypted = unpad(cipher.decrypt(encrypted_data), AES.block_size)
                    violation_data = pickle.loads(decrypted)

                    # Show result
                    frame = violation_data["frame"]
                    msg = violation_data["message"]
                    cv2.imshow("Violation Frame", frame)

                    def update_gui():
                        violation_text.insert(END, msg + "\n")
                        violation_text.see(END)
                    root.after(0, update_gui)

                    if cv2.waitKey(1) == ord('q'):
                        break

                except Exception as e:
                    print(f"Error receiving or decrypting violation: {e}")
                    break

        threading.Thread(target=receive_violations, daemon=True).start()

        # Start sending video frames
        cap = cv2.VideoCapture(selected_file)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            data = pickle.dumps(frame)
            data_len = struct.pack("!I", len(data))
            sock.sendall(data_len + data)

            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()
        sock.close()
        messagebox.showinfo("Done", "Video streaming finished.")
    except Exception as e:
        messagebox.showerror("Error", f"Connection failed: {e}")

def start_gui():
    global choose_coords_button, start_button, status_label, violation_text

    root = Tk()
    root.title("Traffic Video Sender")
    root.geometry("400x400")

    Label(root, text="Traffic Violation Tool", font=("Helvetica", 16, "bold")).pack(pady=15)

    Button(root, text="Choose Video", command=choose_video, width=20).pack(pady=5)

    choose_coords_button = Button(root, text="Choose Stop Line", command=choose_coordinates, state="disabled", width=20)
    choose_coords_button.pack(pady=5)

    start_button = Button(root, text="Start Streaming", command=lambda: threading.Thread(target=start_stream, args=(root,)).start(), state="disabled", width=20)
    start_button.pack(pady=5)

    status_label = Label(root, text="No video selected.", fg="gray")
    status_label.pack(pady=10)

    Label(root, text="Violations:", font=("Helvetica", 12)).pack()
    violation_text = Text(root, height=10, width=45)
    violation_text.pack(pady=5)

    root.mainloop()

if __name__ == "__main__":
    start_gui()
