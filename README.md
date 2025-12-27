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

## 🧠 Approach & Key Assumptions

Building a tactical analysis system with a single camera requires balancing mathematical precision with practical approximations. Below is the breakdown of the methodology used to solve the core challenges.

### 1. The Pipeline Approach

The problem was decomposed into sequential stages to ensure modularity:

* **Detection First:** We prioritize detecting every object in every frame independently before attempting to track them. This avoids "drift" errors where a tracker might latch onto the background.
* **Color-Based Clustering:** Instead of training a model to recognize specific teams (which would fail if jerseys change), we used **K-Means Clustering**. This allows the system to dynamically adapt to *any* match color scheme by analyzing the pixel histogram within player bounding boxes.
* **Homography for Mapping:** We assumed the court is a planar surface. By manually defining the 4 corners of the court, we calculated a **Transformation Matrix** that mathematically converts any pixel  in the video to a metric coordinate  on the floor.

### 2. Critical Assumptions

To make the system feasible with a single video feed, the following engineering assumptions were made:

**Assumption 1: The "Flat Earth" Model (Z = 0)**
* *Logic:* We assume all relevant gameplay happens on the floor (2D plane).
* *Implication:* This works perfectly for player feet. However, it introduces error for the **ball**, which moves in 3D space. A ball 5 meters in the air is mathematically interpreted as being "further away" on the ground.
* *Correction:* We implemented a clamping mechanism in the Mini-Map visualization to constrain the ball within reasonable bounds, preventing it from "flying out of the stadium" visually.


**Assumption 2: Static Camera Position**
* *Logic:* The system assumes the camera does not zoom or pan aggressively during the clip.
* *Implication:* If the camera moves significantly, the coordinate system (homography matrix) would shift, causing the tactical map to become inaccurate.
* *Note:* While we experimented with **Optical Flow** to estimate camera movement, the most stable results for this version were achieved using static camera clips.


**Assumption 3: Jersey Uniformity**
* *Logic:* We assume that the majority of a player's upper body (torso) represents their team color.
* *Implication:* The system automatically crops the top half of the bounding box to sample colors, avoiding confusion with shorts or shoes that might be neutral colors (black/white).


**Assumption 4: Linear Ball Motion (Interpolation)**
* *Logic:* When the ball disappears (due to motion blur or occlusion), we assume it travels in a relatively straight line between the last seen point and the next reappearance.
* *Correction:* We utilized **Pandas Interpolation** to mathematically fill the gaps, creating a smooth trajectory even when the detector misses a few frames.
  
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
* **Broadcast/Third-Person View (Failure):** In standard TV angles (from behind the server), the depth compression makes it nearly impossible for a single camera to accurately measure distance. Players in the back row look like giants compared to players near the net due to perspective distortion. Tactical map creation was not possible with only behind the server view.

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
