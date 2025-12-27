import supervision as sv

# --- PATHS ---
VIDEO_SOURCE = "/content/Volleyball/Video5.mp4"
VIDEO_TARGET = "/content/1.mp4"
MODEL_PATH = "/content/YOUR_MODEL.pt"

# --- IDS ---
ID_PLAYER = 1
ID_REF = 2
ID_BALL = 0

CALIBRATION_FRAMES = 60

# --- OUTPUT COLORS ---
# --- COURT CONFIGURATION  ---
# Real world dimensions of a Volleyball Court (in meters)
COURT_WIDTH_METERS = 9
COURT_LENGTH_METERS = 18


# --- COLORS ---
COLOR_TEAM_1 = sv.ColorPalette.from_hex(["#00FFFF"]) # Cyan
COLOR_TEAM_2 = sv.ColorPalette.from_hex(["#D200D2"]) # Magenta
COLOR_REF = sv.ColorPalette.from_hex(["#6200FF"])    
COLOR_BALL = sv.ColorPalette.from_hex(['#FFFF00'])   
COLOR_BOARD = sv.ColorPalette.from_hex(['#222222'])  # Dark Grey for Minimap background   

# --- TRIANGLE ---
TRI_COLOR_BGR = (0, 255, 255) 
TRI_HEIGHT = 30
TRI_WIDTH = 40
TRI_OFFSET = 15
