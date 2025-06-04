
from collections import deque
import cv2
import numpy as np


class LineDetector:
    def __init__(self, num_frames_avg=10):
        # Initialize two deque queues to hold y-coordinate values across frames
        self.y_start_queue = deque(maxlen=num_frames_avg)
        self.y_end_queue = deque(maxlen=num_frames_avg)

    
    def detect_white_line(self, frame, color, 
                          slope1=0.03, intercept1=460, slope2=0.03, intercept2=385, slope3=-0.8, intercept3=1075):
        
        # Function to map color names to BGR values
        def get_color_code(color_name):
            color_codes = {
                'red': (0, 0, 255),
                'green': (0, 255, 0),
                'yellow': (0, 255, 255)
                 }
            return color_codes.get(color_name.lower())

        frame_org = frame.copy()
        
        # Line equations for defining region of interest (ROI)
        def line1(x): return slope1 * x + intercept1
        def line2(x): return slope2 * x + intercept2
        def line3(x): return slope3 * x + intercept3

        height, width, _ = frame.shape
        
        # Create a mask to spotlight the line's desired area
        mask1 = frame.copy()
        # Set pixels below the first line to black in mask1
        for x in range(width):
            y_line = line1(x)
            mask1[int(y_line):, x] = 0

        mask2 = mask1.copy()
        # Set pixels above the second line to black in mask2
        for x in range(width):
            y_line = line2(x)
            mask2[:int(y_line), x] = 0


        # Convert the mask to grayscale
        gray = cv2.cvtColor(mask2, cv2.COLOR_BGR2GRAY)

        # Apply a Gaussian filter to the ROI
        blurred_gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # Apply CLAHE to equalize the histogram
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray_clahe = clahe.apply(blurred_gray)

        

        # Perform edge detection
        edges = cv2.Canny(gray, 30, 100)

        
        # Perform a dilation and erosion to close gaps in between object edges
        dilated_edges = cv2.dilate(edges, None, iterations=1)
        edges = cv2.erode(dilated_edges, None, iterations=1)

        # Perform Hough Line Transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 70, minLineLength=100, maxLineGap=7)
        
        print(lines)
        # Calculate x coordinates for the start and end of the line
        x_start = 0
        x_end = width - 1

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]

                # Calculate line parameters
                slope = (y2 - y1) / (x2 - x1 + np.finfo(float).eps)  # Add a small number to avoid division by zero
                intercept = y1 - slope * x1

                # Calculate corresponding y coordinates
                y_start = int(slope * x_start + intercept)
                y_end = int(slope * x_end + intercept)

                # Add the y_start and y_end values to the queues
                self.y_start_queue.append(y_start)
                self.y_end_queue.append(y_end)

        # Compute the average y_start and y_end values
        avg_y_start = int(sum(self.y_start_queue) / len(self.y_start_queue)) if self.y_start_queue else 0
        avg_y_end = int(sum(self.y_end_queue) / len(self.y_end_queue)) if self.y_end_queue else 0

        
        # Draw the line
        line_start_ratio=0.32
        x_start_adj = x_start + int(line_start_ratio * (x_end - x_start))  # Adjusted x_start
        avg_y_start_adj = avg_y_start + int(line_start_ratio * (avg_y_end - avg_y_start))  # Adjusted avg_y_start

        # Create a mask with the same size as the frame and all zeros (black)
        mask = np.zeros_like(frame)

        
        # Draw the line on the mask
        cv2.line(mask, (x_start_adj, avg_y_start_adj), (x_end, avg_y_end), (255, 255, 255), 4)

        
        # Determine which color channel(s) to change based on the color argument
        color_code = get_color_code(color)
        if color_code == (0, 255, 0):  # Green
            channel_indices = [1]
        elif color_code == (0, 0, 255):  # Red
            channel_indices = [2]
        elif color_code == (0, 255, 255):  # Yellow
            # Yellow in BGR is a combination of green and red channels.
            # Here we modify both green and red channels.
            channel_indices = [1, 2]
        else:
            raise ValueError('Unsupported color')

        # Change the specified color channels of the frame where the mask is white
       # for channel_index in channel_indices:
        #    frame[mask[:,:,channel_index] == 255, channel_index] = 255
        

        # Calculate slope and intercept for the average green line
        #slope_avg = (avg_y_end - avg_y_start) / (x_end - x_start + np.finfo(float).eps)
        #intercept_avg = avg_y_start - slope_avg * x_start

        # Create a mask with the pixels above the green line set to black
        #mask_line = np.copy(frame_org)
        #for x in range(width):
        #    y_line = slope_avg * x + intercept_avg - 35
        #    mask_line[:int(y_line), x] = 0  # set pixels above the line to black
        
        
        return (x_start_adj, avg_y_start_adj), (x_end, avg_y_end), channel_indices