import numpy as np 
import cv2
import json
import os

class ViewTransformer():
    def __init__(self):
        # VOLLEYBALL COURT DIMENSIONS (Meters)
        self.court_width = 9.0
        self.court_length = 18.0

        # 1. Try to load from the automated file
        if os.path.exists('court_config.json'):
            try:
                with open('court_config.json', 'r') as f:
                    loaded_points = json.load(f)
                self.pixel_vertices = np.array(loaded_points)
                print("[INFO] ViewTransformer: Loaded custom court coordinates.")
            except Exception as e:
                print(f"[ERROR] Failed to load court_config.json: {e}")
                self.pixel_vertices = self.get_default_vertices()
        else:
            print("[WARNING] ViewTransformer: Using DEFAULT coordinates (Run get_court_coordinates.py to fix).")
            self.pixel_vertices = self.get_default_vertices()
        
        # REAL WORLD MAP (Top-Down View)
        self.target_vertices = np.array([
            [0, 0],                         # Top-Left (0,0)
            [self.court_length, 0],         # Top-Right (18,0)
            [self.court_length, self.court_width], # Bottom-Right (18,9)
            [0, self.court_width]           # Bottom-Left (0,9)
        ])

        self.pixel_vertices = self.pixel_vertices.astype(np.float32)
        self.target_vertices = self.target_vertices.astype(np.float32)

        self.perspective_transformer = cv2.getPerspectiveTransform(self.pixel_vertices, self.target_vertices)

    def get_default_vertices(self):
        # Fallback placeholders
        return np.array([
            [250, 400], [1100, 400], [1300, 900], [100, 900]
        ])

    def transform_points(self, points):
        if points is None or len(points) == 0:
            return []

        p = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(p, self.perspective_transformer)
        return transformed_points.reshape(-1, 2)