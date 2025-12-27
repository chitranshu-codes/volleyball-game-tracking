import cv2
import numpy as np
import config

class MiniCourt():
    def __init__(self, frame):
        self.drawing_rectangle_width = 250
        self.drawing_rectangle_height = 500 
        self.buffer = 50 
        
        self.padding_court = 50
        
        self.set_canvas_background_box_position(frame)
        self.mini_court_img = self.set_court_drawing_key_points()
        self.set_court_lines_and_color()

    def set_canvas_background_box_position(self, frame):
        frame_h, frame_w = frame.shape[:2]
        self.start_x = frame_w - self.drawing_rectangle_width - self.buffer
        self.start_y = self.buffer
        self.end_x = self.start_x + self.drawing_rectangle_width
        self.end_y = self.start_y + self.drawing_rectangle_height

    def set_court_drawing_key_points(self):
        drawing_court_img = np.ones((self.drawing_rectangle_height, self.drawing_rectangle_width, 3), dtype=np.uint8) * 255
        
        self.court_start_x = self.padding_court
        self.court_start_y = self.padding_court
        self.court_end_x = self.drawing_rectangle_width - self.padding_court
        self.court_end_y = self.drawing_rectangle_height - self.padding_court
        
        self.court_drawing_width = self.court_end_x - self.court_start_x
        self.court_drawing_height = self.court_end_y - self.court_start_y
        
        return drawing_court_img

    def set_court_lines_and_color(self):
        color = (0, 0, 0)
        thick = 2
        img = self.mini_court_img.copy()
        
        # Outer Boundary
        cv2.rectangle(img, (self.court_start_x, self.court_start_y), (self.court_end_x, self.court_end_y), color, thick)
        
        # Net
        mid_y = int(self.court_start_y + (self.court_drawing_height / 2))
        cv2.line(img, (self.court_start_x, mid_y), (self.court_end_x, mid_y), color, thick)
        
        # Attack Lines
        pixels_per_meter = self.court_drawing_height / 18.0
        attack_line_pixel_dist = 3 * pixels_per_meter
        y_attack_top = int(mid_y - attack_line_pixel_dist)
        y_attack_bot = int(mid_y + attack_line_pixel_dist)
        cv2.line(img, (self.court_start_x, y_attack_top), (self.court_end_x, y_attack_top), color, 1)
        cv2.line(img, (self.court_start_x, y_attack_bot), (self.court_end_x, y_attack_bot), color, 1)
        
        self.mini_court_img = img

    def draw_background_rectangle(self, frame):
        shapes = np.zeros_like(frame, np.uint8)
        cv2.rectangle(shapes, (self.start_x, self.start_y), (self.end_x, self.end_y), (255, 255, 255), cv2.FILLED)
        out = frame.copy()
        alpha = 0.5
        mask = shapes.astype(bool)
        out[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]
        return out

    def draw_court(self, frame):
        frame[self.start_y:self.end_y, self.start_x:self.end_x] = self.mini_court_img
        return frame

    def draw_points_on_mini_court(self, frame, positions, color=(0, 0, 255)):
        for pos in positions:
            x_meters, y_meters = pos
            
            # 1. Map Meters to Local Pixels (Relative to White Box)
            # x_meters (Length 0-18) -> Y axis on map
            # y_meters (Width 0-9)   -> X axis on map
            
            x_norm = x_meters / 18.0
            y_norm = y_meters / 9.0

            # Calculate Local Pixels (inside the white box)
            mini_y = int(self.court_start_y + (x_norm * self.court_drawing_height))
            mini_x = int(self.court_start_x + (y_norm * self.court_drawing_width))

            # 2. Filter outliers
            # We check if the point is inside the local box dimensions (0 to width/height)
            if mini_x < 0 or mini_x > self.drawing_rectangle_width:
                continue
            if mini_y < 0 or mini_y > self.drawing_rectangle_height:
                continue

            # 3. Convert to Global Pixels (Main Frame)
            global_x = self.start_x + mini_x
            global_y = self.start_y + mini_y
            
            # 4. Draw
            cv2.circle(frame, (global_x, global_y), 5, color, -1)
            
        return frame