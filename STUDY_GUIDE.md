# ?? Multi-Camera Person Tracking & Re-Identification (Re-ID) — Study Guide

Built for the **Era Biz (Kottawa) Computer Vision & AI Intern** Position.
Use this guide to understand the complete codebase and master the interview questions.

---

## ?? Table of Contents
1. [Core Libraries & Tech Stack](#1-core-libraries--tech-stack)
2. [Step 1: Model Optimization & ONNX Export (`src/export_onnx.py`)](#2-step-1-model-optimization--onnx-export)
3. [Step 2: Single Camera Detection vs Tracking (ByteTrack)](#3-step-2-single-camera-detection-vs-tracking-bytetrack)
4. [Step 3: Deep Person Re-Identification (Re-ID Math)](#4-step-3-deep-person-re-identification-re-id-math)
5. [Step 4: Multi-Camera Gallery & Camera Transitions](#5-step-4-multi-camera-gallery--camera-transitions)
6. [Step 5: Real-Time Streamlit Dashboard UI](#6-step-5-real-time-streamlit-dashboard-ui)
7. [?? Era Biz Interview Questions & Answers Cheat Sheet](#7--era-biz-interview-questions--answers-cheat-sheet)

---

## 1. Core Libraries & Tech Stack

| Library | Role in Project | Key Interview Concept |
| :--- | :--- | :--- |
| **`ultralytics`** | Pretrained YOLOv8 Nano object detector | Single-stage anchor-free detector. Detects class 0 (`person`) bounding boxes. |
| **`opencv-python`** | Video stream decoding & frame annotation | Reads FPS, resizes frames, draws bounding boxes, calculates timestamps. |
| **`torch` & `torchvision`** | MobileNetV3 deep feature extractor | Produces 512-dim embedding vector from person crops. |
| **`onnxruntime` & `onnx`** | High-performance inference engine | Graph optimization & operator fusion. Runs on CPU at 20+ FPS. |
| **`lap`** | Linear Assignment Problem solver | Hungarian Algorithm used by ByteTrack to associate detections across frames. |
| **`streamlit`** | Real-time CCTV analytics web dashboard | Side-by-side video streaming, live visitor metrics, and alert logs. |
| **`numpy` & `scipy`** | Vector math & metric calculation | Calculates L2-Norm and Cosine Similarity between embeddings. |

---

## 2. Step 1: Model Optimization & ONNX Export

### Why not run `.pt` (PyTorch) directly in production?
- PyTorch `.pt` models are interpreted by Python, which introduces runtime overhead.
- An ONNX (`.onnx`) graph combines adjacent neural layers (Convolution + Batch Normalization + ReLU) into a single optimized instruction.
- Using `onnxslim`, redundant dead weights and operations are pruned.
- Result: **3x to 5x faster CPU inference**, allowing laptops without dedicated NVIDIA GPUs to process live video at 20-30+ FPS.

---

## 3. Step 2: Single Camera Detection vs Tracking (ByteTrack)

### What is the difference between Detection and Tracking?
- **Detection (YOLO)**: Finds objects in every frame independently. It has NO memory of past frames. (Frame 1: "Person", Frame 2: "Person" — YOLO does not know if they are the same person).
- **Tracking (ByteTrack)**: Gives each detected person a persistent **Local ID** (e.g., `Track #1`, `Track #2`) across consecutive frames.

### How does ByteTrack work?
1. **Kalman Filter**: Predicts where the person will move in the next frame based on their velocity.
2. **First Association**: Matches high-confidence detections with existing tracks using IoU (Intersection over Union).
3. **Second Association**: Matches remaining unmatched tracks with *low-confidence* detections (crucial for occlusions or motion blur).

---

## 4. Step 3: Deep Person Re-Identification (Re-ID Math)

### How do we identify a person across two disconnected cameras?
Single-camera tracking fails when a person walks out of Camera 1 and enters Camera 2 (the cameras don't share pixels). Re-ID solves this using **Deep Metric Learning**:

1. **Crop**: Extract the bounding box containing the person $(H \times W \times 3)$.
2. **Deep Backbone**: Pass crop through MobileNetV3 to extract feature embedding vector $\mathbf{f} \in \mathbb{R}^{512}$.
3. **L2 Normalization**:
   $$\mathbf{e} = \frac{\mathbf{f}}{\|\mathbf{f}\|_2}$$
   Ensures the embedding has a length of 1.0 (points on a unit sphere).
4. **Cosine Similarity**:
   $$\text{Similarity}(\mathbf{u}, \mathbf{v}) = \mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^{512} u_i v_i$$
   - If $\text{Similarity} \ge 0.62$, the two crops belong to the **same person**!
   - If $\text{Similarity} < 0.62$, it is a **new person**.

---

## 5. Step 4: Multi-Camera Gallery & Camera Transitions

- The **Global Identity Gallery** maintains a database:
  `{ Global_ID: { mean_embedding, last_seen_camera, last_seen_time } }`
- **Spatial Exclusion Rule**: The same Global ID cannot appear in two different bounding boxes in the exact same frame of the same camera.
- **Cross-Camera Transition**: When a person previously seen in `CAM_01` is detected in `CAM_02` with high similarity, a transition alert is triggered:
  `"Global Person #1 moved from CAM_01 -> CAM_02 (Sim: 0.89)"`
- **Exponential Moving Average (EMA)**:
  $$\mathbf{e}_{\text{gallery}} \leftarrow \alpha \mathbf{e}_{\text{gallery}} + (1 - \alpha) \mathbf{e}_{\text{new}}$$
  Updates the person's appearance as lighting or perspective changes.

---

## 6. Step 5: Real-Time Streamlit Dashboard UI
- Dual-feed synchronized layout.
- Real-time KPI metrics: Unique Person Count, Cam 1 Count, Cam 2 Count, Pipeline FPS.
- Live Transition Event Log table.
