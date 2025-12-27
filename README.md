# AI-Powered Volleyball Tactical Analysis System 🏐📊

## Overview

This project is a computer vision-based application designed to perform **tactical analysis** on volleyball match footage. Utilizing state-of-the-art Deep Learning models (YOLO) and classic Computer Vision techniques (Optical Flow, Perspective Transformation), the system detects players, referees, and the ball, tracks their movements, and projects their positions onto a 2D tactical map (Mini-Map) in real-time.

Unlike standard sports analytics tools that rely on expensive multi-camera setups (like Hawk-Eye), this system aims to extract **"2D Hawk-Eye" style data** using only a **single-camera video feed**.

---

## 🏗️ System Architecture

The project follows a modular pipeline approach, processing the video frame-by-frame.

### Core Modules:

1. Dataset Construction & Model Training:

* Custom Dataset: Due to the lack of public datasets for volleyball, a custom dataset was created from scratch.
* Annotation: Images were manually annotated to define bounding boxes for players, referees, and the volleyball.
* Training: The YOLOv8 model was fine-tuned on this custom dataset to specifically recognize the unique features of a volleyball court (e.g., distinguishing a fast-moving ball from noise).
  
2. **`main.py` (The Pipeline Conductor):**
* Orchestrates the data flow.
* Loads the video into memory (or streams it) to prevent RAM overload.
* Integrates detection, tracking, and visualization loops.


3. **`Object Detection` (YOLOv8 + Custom Weights):**
* We utilize a fine-tuned YOLO model (`your_model.pt`) specifically trained to identify:
* **Class 0:** Ball 🏐
* **Class 1:** Player 🏃
* **Class 2:** Referee 👮




4. **`Object Tracking` (ByteTrack):**
* Assigns a unique **Stable ID** to every detected entity.
* Handles **Occlusion**: When players cross each other or cluster at the net, the tracker maintains their identity using a motion buffer.


5. **`TeamAssigner` (K-Means Clustering):**
* Extracts pixel data from player bounding boxes.
* Uses **K-Means Clustering** to separate jersey colors from the background.
* Classifies players into **Team 1** or **Team 2** automatically.


6. **`ViewTransformer` (Perspective Transformation):**
* The "Mathematical Brain" of the system.
* Uses **Homography** to map 2D video pixels (trapezoid shape of the court) to Real-World Meters (rectangular 18m x 9m court).
* Calculates the exact  coordinates of every player on the floor.


7. **`MiniCourt` (Tactical Visualization):**
* Draws a synchronized top-down view of the match.
* Plots the real-time positions of players, referees, and the ball.
* Includes logic to clamp objects within boundaries to handle perspective distortion.



---

## 🚀 Salient Features

* **Accurate Entity Detection:** robust detection of small objects (like the ball) even during rapid movement.
* **Smart Team Classification:** Automatically distinguishes between teams based on jersey color without manual input.
* **Interpolation of Ball Trajectory:** Uses pandas interpolation to fill in the gaps when the ball is momentarily lost due to motion blur or occlusion.
* **Real-World Coordinate System:** Displays the physical position of players in meters (e.g., `(5.2m, 3.1m)`).
* **Synchronized Tactical Board:** A frame-by-frame Mini-Map visualization that updates perfectly in sync with the video.

---

## 🆚 Comparison: Why Volleyball is Harder than Football

Developing a computer vision system for Volleyball is significantly more challenging than for Football (Soccer) due to the inherent nature of the sport.

| Feature | Football (Soccer) ⚽ | Volleyball 🏐 |
| --- | --- | --- |
| **Field of Play** | Massive (100m x 60m). Players are spread out. | Tiny (18m x 9m). 12 players are packed into a small box. |
| **Occlusion** | Occasional overlap of players. | **Constant occlusion.** Players constantly jump in front of each other and block the camera view. |
| **The Net** | No visual obstruction in the middle of the field. | A large net splits the view, often cutting off player bodies or the ball in detection. |
| **Ball Dynamics** | **Mostly 2D.** The ball rolls on the grass 80% of the time. | **Inherently 3D.** The ball is in the air 90% of the time. |

**The "Z-Axis" Problem:**
In computer vision, **Homography** assumes the world is a flat 2D plane (). In Football, tracking players' feet works perfectly because they are on the ground.
In Volleyball, the ball flies 5-10 meters in the air. A single camera cannot tell the difference between "High Up" and "Far Away". This causes the ball to mathematically "fly out of the stadium" on the tactical map, requiring complex clamping logic to fix.

---

## ⚠️ Limitations & Challenges Faced

While the objectives were achieved, the system has certain limitations due to hardware and data constraints:

### 1. The Single Camera Constraint (No Depth Data)

Since we are using a standard 2D video feed, we lack **Depth (Z-axis)** data.

* *Issue:* When the ball is spiked high, the `ViewTransformer` interprets it as being very far back on the floor.
* *Result:* The ball sometimes appears to drift "out of bounds" on the tactical map during high arcs.

### 2. Dataset Scarcity

There is a lack of high-quality, annotated datasets for Volleyball compared to Football.

* Our model struggles occasionally with **Motion Blur** (when the ball travels at >100 km/h).
* Referees standing still are sometimes confused with background elements (like poles).

### 3. Camera Angles (Side vs. Broadcast)

* **Side View (Success):** The system performs excellently on side-view footage where the teams are separated Left vs. Right.
* **Broadcast/Third-Person View (Failure):** In standard TV angles (from behind the server), the depth compression makes it nearly impossible for a single camera to accurately measure distance. Players in the back row look like giants compared to players near the net due to perspective distortion.

### 4. High-Resolution Processing

Processing 4K footage frame-by-frame with Optical Flow and Deep Learning is computationally expensive. To ensure the system runs on standard consumer hardware, we had to optimize the tracking buffers and limit the resolution of the analysis stream. This also makes tracking the ball difficult.

---

## 🛠️ Installation & Usage

### Prerequisites

* Python 3.8+
* GPU (Recommended for YOLO inference)

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/chitranshu-codes/volleyball-game-tracking.git
cd volleyball-analysis

```


2. **Install dependencies:**
```bash
pip install ultralytics supervision opencv-python numpy pandas

```


3. **Calibrate the Court:**
Run the coordinate capture script to click the 4 corners of the court in your video.
```bash
python get_court_coordinates.py

```


4. **Run the Analysis:**
```bash
python main.py

```



---

## 📈 Objectives Achieved

Despite the complexity of the sport, this project successfully delivers:
✅ A working end-to-end pipeline from raw video to analytical output.
✅ Stable tracking of 12+ entities simultaneously.
✅ A visually coherent Tactical Map that mimics professional broadcast tools.
✅ A proof-of-concept that single-camera analytics is viable for Volleyball with proper constraints.

---

**Author:** Chitranshu Singh Rawat
**Domain:** Mechanical Engineering / Computer Vision / AI
**Location:** India
