import cv2
import numpy as np
import pandas as pd
import config

def interpolate_ball_positions(ball_detections):
    """
    Takes a list of bounding boxes (some might be None/Empty).
    Returns a list of interpolated bounding boxes.
    """
    # Convert input to a format Pandas likes
    data = []
    for bbox in ball_detections:
        if bbox is None or len(bbox) == 0:
            data.append([np.nan, np.nan, np.nan, np.nan])
        else:
            data.append(bbox)

    # Create DataFrame
    df = pd.DataFrame(data, columns=['x1', 'y1', 'x2', 'y2'])

    # Interpolate (fill gaps BETWEEN detections)
    # This connects the dots when the ball is temporarily lost
    df = df.interpolate()
    
    # Convert back to list of lists
    return df.to_numpy().tolist()

def draw_triangle(image, xyxy_box):
    """
    Draws an inverted triangle above a bounding box (usually for the ball).
    """
    x1, y1, x2, y2 = map(int, xyxy_box)
    
    # Calculate the center of the bounding box
    cx = int((x1 + x2) / 2)
    top_y = y1

    # Define the 3 points of the inverted triangle
    # Point 1: The tip pointing down (just above the object)
    tip = [cx, top_y - config.TRI_OFFSET]
    
    # Point 2: Top Left corner
    top_left = [cx - (config.TRI_WIDTH // 2), top_y - config.TRI_OFFSET - config.TRI_HEIGHT]
    
    # Point 3: Top Right corner
    top_right = [cx + (config.TRI_WIDTH // 2), top_y - config.TRI_OFFSET - config.TRI_HEIGHT]

    # Convert to numpy array of points
    triangle_cnt = np.array([tip, top_left, top_right], np.int32)
    
    # Draw filled triangle
    cv2.fillPoly(image, [triangle_cnt], color=config.TRI_COLOR_BGR)
    
    return image

