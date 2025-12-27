from sklearn.cluster import KMeans
import numpy as np
import cv2

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}
        self.kmeans = None
        self.left_samples = []
        self.right_samples = []
        self.trained = False

    def get_player_color(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        w = x2 - x1
        h = y2 - y1
        
        center_x = int(x1 + w / 2)
        center_y = int(y1 + h * 0.2)
        patch_w = int(w * 0.2)
        patch_h = int(h * 0.2) 
        
        patch_x1 = max(0, center_x - patch_w // 2)
        patch_x2 = min(frame.shape[1], center_x + patch_w // 2)
        patch_y1 = max(0, center_y - patch_h // 2)
        patch_y2 = min(frame.shape[0], center_y + patch_h // 2)
        
        image = frame[patch_y1:patch_y2, patch_x1:patch_x2]
        
        if image.size == 0: return None

        # Convert to HSV (Hue, Saturation, Value)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mean_hsv = np.mean(hsv_image, axis=(0, 1))
        
        return np.array([mean_hsv[0], mean_hsv[1]])

    def collect_samples(self, frame, player_detections):
        """
        Phase 1: Spatially separate samples.
        Players on Left -> Left Bucket
        Players on Right -> Right Bucket
        """
        frame_width = frame.shape[1]
        center_x = frame_width / 2

        for bbox in player_detections.xyxy:
            color = self.get_player_color(frame, bbox)
            if color is None: continue

            # Determine side based on player center
            p_x = (bbox[0] + bbox[2]) / 2
            
            if p_x < center_x:
                self.left_samples.append(color)
            else:
                self.right_samples.append(color)

    def get_dominant_color(self, samples):
        """
        Aggressive Filtering:
        Finds the MAJORITY color in a list of samples.
        Ignores Liberos and outliers.
        """
        if len(samples) < 5: return np.mean(samples, axis=0)

        # Cluster samples into 2 groups (Main Jersey vs Libero/Noise)
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(samples)
        
        # Count how many samples are in each cluster
        labels = kmeans.labels_
        count_0 = np.sum(labels == 0)
        count_1 = np.sum(labels == 1)

        # The cluster with MORE samples is the Main Team Color
        if count_0 > count_1:
            return kmeans.cluster_centers_[0]
        else:
            return kmeans.cluster_centers_[1]

    def fit_model(self):
        """Phase 2: Train."""
        print(f"[INFO] Left Samples: {len(self.left_samples)} | Right Samples: {len(self.right_samples)}")
        
        if len(self.left_samples) == 0 or len(self.right_samples) == 0:
            print("[WARNING] Missing data for one side! Cannot train aggressively.")
            return False

        # 1. Find pure colors for each side (Filtering out Liberos)
        color_left = self.get_dominant_color(self.left_samples)
        color_right = self.get_dominant_color(self.right_samples)

        # 2. Train a final classifier on these two clean colors
        # We create a synthetic dataset of just these 2 pure colors
        # This forces the decision boundary to be exactly between them
        training_data = np.array([color_left, color_right])
        
        self.kmeans = KMeans(n_clusters=2, init=training_data, n_init=1) # Force init at calculated centers
        self.kmeans.fit(training_data)
        
        # Map cluster centers back to Team IDs
        self.team_colors[1] = self.kmeans.cluster_centers_[0]
        self.team_colors[2] = self.kmeans.cluster_centers_[1]
        
        self.trained = True
        return True

    def get_player_team(self, frame, bbox, player_id):
        if not self.trained: return 0
        if player_id in self.player_team_dict: return self.player_team_dict[player_id]

        color = self.get_player_color(frame, bbox)
        if color is None: return 0

        if not self.trained or self.kmeans is None:
         return 0
        # Predict
        team_id = self.kmeans.predict(color.reshape(1, -1))[0]
        team_id += 1 

        self.player_team_dict[player_id] = team_id
        return team_id