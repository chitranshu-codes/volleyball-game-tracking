import cv2
import config
import json

# List to store points
points = []
DISPLAY_WIDTH = 1280 

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        scale = original_width / DISPLAY_WIDTH
        orig_x = int(x * scale)
        orig_y = int(y * scale)
        
        print(f"Captured: [{orig_x}, {orig_y}]")
        points.append([orig_x, orig_y])
        
        cv2.circle(resized_img, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(resized_img, f"{len(points)}", (x+10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow('Select 4 Corners', resized_img)

cap = cv2.VideoCapture(config.VIDEO_SOURCE)
ret, img = cap.read()
cap.release()

if not ret:
    print("Failed to read video!")
    exit()

original_height, original_width = img.shape[:2]
aspect_ratio = original_height / original_width
display_height = int(DISPLAY_WIDTH * aspect_ratio)
resized_img = cv2.resize(img, (DISPLAY_WIDTH, display_height))

print("--- INSTRUCTIONS ---")
print("1. Click the 4 corners: TOP-LEFT -> TOP-RIGHT -> BOTTOM-RIGHT -> BOTTOM-LEFT")
print("2. Press ANY KEY to save and exit.")

cv2.imshow('Select 4 Corners', resized_img)
cv2.setMouseCallback('Select 4 Corners', click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()

# SAVE TO JSON FILE ---
if len(points) == 4:
    with open('court_config.json', 'w') as f:
        json.dump(points, f)
    print(f"\n[SUCCESS] Coordinates saved to 'court_config.json': {points}")
    print("You can now run main.py immediately.")
else:
    print(f"\n[ERROR] You selected {len(points)} points. Need exactly 4. Not saved.")