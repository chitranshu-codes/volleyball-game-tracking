import numpy as np
import cv2
from ultralytics import YOLO
import supervision as sv
import config
import utils
from team_assigner import TeamAssigner
from view_transformer import ViewTransformer
from mini_court import MiniCourt 

#  COLOR EXTRACTION ---
def get_color_tuple(color_obj):
    if hasattr(color_obj, 'by_idx'):
        return color_obj.by_idx(0).as_bgr()
    elif hasattr(color_obj, 'as_bgr'):
        return color_obj.as_bgr()
    return color_obj

# --- SETUP ---
print("[INFO] Loading YOLO model...")
model = YOLO(config.MODEL_PATH)

print("[INFO] Initializing Trackers...")
tracker = sv.ByteTrack(lost_track_buffer=60, minimum_matching_threshold=0.8)

team_assigner = TeamAssigner()
view_transformer = ViewTransformer()

# --- ANNOTATORS ---
ell_annotator_1 = sv.EllipseAnnotator(color=config.COLOR_TEAM_1, thickness=2)
ell_annotator_2 = sv.EllipseAnnotator(color=config.COLOR_TEAM_2, thickness=2)
ell_annotator_neutral = sv.EllipseAnnotator(color=sv.ColorPalette.from_hex(['#888888']), thickness=2)
ref_annotator = sv.BoxAnnotator(color=config.COLOR_REF, thickness=4)

# Annotator for Real-World Coordinates
coord_annotator = sv.LabelAnnotator(
    color=sv.Color.WHITE, 
    text_color=sv.Color.BLACK, 
    text_position=sv.Position.TOP_CENTER,
    text_scale=0.5,
    text_padding=5
)

# Annotators for Player IDs
label_annotator_1 = sv.LabelAnnotator(
    color=config.COLOR_TEAM_1, text_color=sv.Color.BLACK, text_position=sv.Position.BOTTOM_CENTER)
label_annotator_2 = sv.LabelAnnotator(
    color=config.COLOR_TEAM_2, text_color=sv.Color.WHITE, text_position=sv.Position.BOTTOM_CENTER)
label_annotator_ref = sv.LabelAnnotator(
    color=config.COLOR_REF, text_color=sv.Color.WHITE, text_position=sv.Position.TOP_CENTER)

trace_annotator = sv.TraceAnnotator(color=config.COLOR_BALL, trace_length=20)
dot_annotator = sv.DotAnnotator(color=config.COLOR_BALL, radius=4)

CALIBRATION_FRAMES = 60

def main():
    # ---------------------------------------------------------
    # PASS 1: DETECTION & DATA COLLECTION
    # ---------------------------------------------------------
    print("[INFO] PASS 1: Running Inference on all frames...")
    
    cap = cv2.VideoCapture(config.VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {config.VIDEO_SOURCE}")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    all_player_detections = [] 
    all_ref_detections = []    
    all_ball_bboxes = []       
    
    ret, first_frame = cap.read()
    if not ret: return
    mini_court = MiniCourt(first_frame) 
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 
    
    current_frame = 0
    while True:
        ret, frame = cap.read()
        if not ret: break

        results = model.predict(frame, device=0, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = tracker.update_with_detections(detections)

        players = detections[detections.class_id == config.ID_PLAYER]
        referees = detections[detections.class_id == config.ID_REF]
        balls = detections[detections.class_id == config.ID_BALL]
        
        all_player_detections.append(players)
        all_ref_detections.append(referees)

        if len(balls) > 0:
            best_ball = balls.xyxy[0] 
            all_ball_bboxes.append(best_ball)
        else:
            all_ball_bboxes.append(None)
            
        current_frame += 1
        if current_frame % 100 == 0:
            print(f"      Processed {current_frame}/{frame_count} frames...")
            
    cap.release()

    # ---------------------------------------------------------
    # PASS 2: INTERPOLATION
    # ---------------------------------------------------------
    print("[INFO] PASS 2: Interpolating Ball Positions...")
    interpolated_ball_bboxes = utils.interpolate_ball_positions(all_ball_bboxes)
    
    # ---------------------------------------------------------
    # PASS 3: RENDERING & MINI-MAP
    # ---------------------------------------------------------
    print("[INFO] PASS 3: Rendering Final Video...")
    
    cap = cv2.VideoCapture(config.VIDEO_SOURCE)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(config.VIDEO_TARGET, fourcc, fps, (width, height))
    
    team_assigner = TeamAssigner()
    
    for i, ball_box in enumerate(interpolated_ball_bboxes):
        ret, frame = cap.read()
        if not ret: break

        # 1. Retrieve Data
        players = all_player_detections[i]
        referees = all_ref_detections[i]
        
        # 2. Setup Ball
        if np.isnan(ball_box[0]):
            balls = sv.Detections.empty()
        else:
            xyxy = np.array([ball_box], dtype=np.float32)
            class_id = np.array([config.ID_BALL])
            confidence = np.array([1.0])
            tracker_id = np.array([1]) 
            balls = sv.Detections(xyxy=xyxy, class_id=class_id, confidence=confidence, tracker_id=tracker_id)

        # 3. Team Assignment
        players_1 = sv.Detections.empty()
        players_2 = sv.Detections.empty()
        players_neutral = sv.Detections.empty()
        
        if len(players) > 0:
            if i < CALIBRATION_FRAMES:
                team_assigner.collect_samples(frame, players)
                players_neutral = players
                cv2.putText(frame, f"Calibrating... {i}/{CALIBRATION_FRAMES}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            else:
                if not team_assigner.trained:
                    team_assigner.fit_model()
                
                team_ids = []
                for idx, bbox in enumerate(players.xyxy):
                    p_id = players.tracker_id[idx] if players.tracker_id is not None else idx
                    t_id = team_assigner.get_player_team(frame, bbox, p_id)
                    team_ids.append(t_id)
                
                team_ids = np.array(team_ids)
                players_1 = players[team_ids == 1]
                players_2 = players[team_ids == 2]

        # 4. VIEW TRANSFORMATION
        # Calculate real-world positions (Meters)
        points_feet_1 = players_1.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        points_feet_2 = players_2.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        points_ref = referees.get_anchors_coordinates(sv.Position.BOTTOM_CENTER) # Referees
        
        # Transform Points
        transformed_p1 = view_transformer.transform_points(points_feet_1)   
        transformed_p2 = view_transformer.transform_points(points_feet_2)   
        transformed_ref = view_transformer.transform_points(points_ref) # Transform Referees
        
        # Ball position
        points_ball = None
        if len(balls) > 0:
            b_box = balls.xyxy[0]
            b_center = np.array([[(b_box[0]+b_box[2])/2, (b_box[1]+b_box[3])/2]])
            points_ball = view_transformer.transform_points(b_center)

        # Coordinate Labels (Combine all players for text)
        all_points = []
        if len(transformed_p1) > 0: all_points.extend(transformed_p1)
        if len(transformed_p2) > 0: all_points.extend(transformed_p2)
        
        coord_labels = []
        for point in all_points:
            x, y = point
            coord_labels.append(f"({x:.1f}m, {y:.1f}m)")

        # --- DRAWING ---
        annotated_frame = frame.copy()
        
        # A. Mini Court
        annotated_frame = mini_court.draw_background_rectangle(annotated_frame)
        annotated_frame = mini_court.draw_court(annotated_frame)
        
        if i >= CALIBRATION_FRAMES:
            # Team 1 Dots
            annotated_frame = mini_court.draw_points_on_mini_court(
                annotated_frame, transformed_p1, color=get_color_tuple(config.COLOR_TEAM_1))
            # Team 2 Dots
            annotated_frame = mini_court.draw_points_on_mini_court(
                annotated_frame, transformed_p2, color=get_color_tuple(config.COLOR_TEAM_2))
            # Referee Dots (ADDED)
            annotated_frame = mini_court.draw_points_on_mini_court(
                annotated_frame, transformed_ref, color=get_color_tuple(config.COLOR_REF))
            # Ball Dot (ADDED)
            if points_ball is not None:
                annotated_frame = mini_court.draw_points_on_mini_court(
                    annotated_frame, points_ball, color=get_color_tuple(config.COLOR_BALL))

        # B. Standard Annotations
        
        # Referees
        annotated_frame = ref_annotator.annotate(scene=annotated_frame, detections=referees)
        if referees.tracker_id is not None:
            labels = [f"Ref {i}" for i in referees.tracker_id]
            annotated_frame = label_annotator_ref.annotate(scene=annotated_frame, detections=referees, labels=labels)

        # Team 1
        annotated_frame = ell_annotator_1.annotate(scene=annotated_frame, detections=players_1)
        if players_1.tracker_id is not None:
            labels = [f"#{i}" for i in players_1.tracker_id]
            annotated_frame = label_annotator_1.annotate(scene=annotated_frame, detections=players_1, labels=labels)

        # Team 2
        annotated_frame = ell_annotator_2.annotate(scene=annotated_frame, detections=players_2)
        if players_2.tracker_id is not None:
            labels = [f"#{i}" for i in players_2.tracker_id]
            annotated_frame = label_annotator_2.annotate(scene=annotated_frame, detections=players_2, labels=labels)

        # C. Coordinates Overlay
        if len(players) > 0 and i >= CALIBRATION_FRAMES:
             # Create combined detections to match label length
             all_players = sv.Detections.merge([players_1, players_2])
             # Ensure labels match the merged detections order
             if len(all_players) == len(coord_labels):
                 annotated_frame = coord_annotator.annotate(
                     scene=annotated_frame, 
                     detections=all_players, 
                     labels=coord_labels
                 )

        # D. Ball Tracing
        if len(balls) > 0:
            annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=balls)
            annotated_frame = dot_annotator.annotate(scene=annotated_frame, detections=balls)
            for box in balls.xyxy:
                annotated_frame = utils.draw_triangle(annotated_frame, box)

        writer.write(annotated_frame)
        
        if i % 100 == 0:
            print(f"      Rendered {i}/{frame_count} frames...")

    cap.release()
    writer.release()
    print(f"[INFO] Done! Output saved to {config.VIDEO_TARGET}")

if __name__ == "__main__":
    main()