import cv2

class Line:
    def __init__(self, position):

        # Insert line coordinates
        self.x1 = int(position[0])
        self.y1 = int(position[1])
        self.x2 = int(position[2])
        self.y2 = int(position[3])

    def is_car_touch_line(self, car_x1, car_y2, car_x2, original_shape):
        """
        Check if the car is completely inside the stop line area.
        """
        # Target resolution (used for detection)
        target_width = 640
        target_height = 480

        original_width = original_shape[1]
        original_height = original_shape[0]

        # Compute scale factors to map from original resolution to 640x480
        scale_x = target_width / original_width
        scale_y = target_height / original_height

        print(scale_x, scale_y)

        #check by detection scale if a car touches the line
        if (
            car_x1 > int(self.x1 * scale_x) and
            car_x2 < int(self.x2 * scale_x) and
            car_y2 > int(self.y1 * scale_y) and
            car_y2 < int(self.y2 * scale_y)
        ):
            return True
        return False

    def draw_line(self, frame, color=(0, 255, 0), thickness=2):
        """
        Draws the scaled stop line (as a rectangle) on the frame.
        """

        cv2.rectangle(frame, (self.x1, self.y1), (self.x2, self.y2), color, thickness)
        return frame
