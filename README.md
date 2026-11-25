# Multi-Camera Person Tracking & Re-Identification (Re-ID) Video Analytics

[![Python](https://img.shields.io/badge/Python-3.12-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF.svg?style=flat)](https://github.com/ultralytics/ultralytics)
[![ONNX Runtime](https://img.shields.io/badge/ONNX-Runtime%20Accelerated-005CED.svg?style=flat)](https://onnxruntime.ai/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Interactive%20Dashboard-FF4B4B.svg?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)

A production-grade, real-time Computer Vision and deep learning pipeline designed for multi-camera CCTV environments. Features synchronized object detection, temporal tracking, deep appearance feature extraction (Re-ID), cross-camera identity association, and ONNX Runtime inference optimization.

---

## ?? Key Architecture & Highlights

```
+-------------------+      +-------------------+
|  Camera 01 Stream |      |  Camera 02 Stream |
+---------+---------+      +---------+---------+
          |                          |
          v                          v
+----------------------------------------------+
|     YOLOv8 Object Detection (Person Class)   |
|         [PyTorch / ONNX Runtime CPU]         |
+----------------------------------------------+
|       ByteTrack Local Temporal Tracking      |
+----------------------------------------------+
|  Re-ID Feature Extractor (L2-Norm Embedding) |
+----------------------------------------------+
                       |
                       v
+----------------------------------------------+
|        Global Identity Gallery & Matcher     |
|         Cosine Similarity >= Threshold       |
|    Exponential Moving Average (EMA) Updates  |
+----------------------------------------------+
                       |
                       v
+----------------------------------------------+
|     Streamlit Real-Time Analytics UI         |
|  - Dual Video Feeds with Global Person Tags  |
|  - Real-time FPS & Unique Count Metrics      |
|  - Cross-Camera Transition Event Log         |
+----------------------------------------------+
```

1. **High-Throughput Real-Time Detection**: Powered by YOLOv8 nano with ByteTrack for consistent bounding boxes even under partial occlusions.
2. **Inference Acceleration via ONNX**: Converts raw PyTorch models into optimized ONNX computation graphs with graph simplifications, achieving 25-40+ FPS on standard CPU hardware.
3. **Deep Person Re-Identification (Re-ID)**: Extracts 512-dimensional deep appearance embeddings from person crops and maps them using Cosine Similarity against an active cross-camera gallery.
4. **Spatial & Temporal Consistency**: Prevents identity collision (ensuring the same person cannot simultaneously occupy two distinct positions in the same camera view).
5. **Interactive Dashboard & Event Logging**: Real-time Streamlit application displaying dual feeds, cross-camera transition alerts, and unique visitor metrics.

---

## ?? Repository Structure

```
+-- models/
¦   +-- yolov8n.onnx            # Optimized ONNX model file
+-- videos/
¦   +-- cam1.mp4                # Camera 1 test video (East Corridor)
¦   +-- cam2.mp4                # Camera 2 test video (Central Lobby)
+-- src/
¦   +-- get_samples.py          # Synthetic multi-camera CCTV video generator
¦   +-- export_onnx.py          # ONNX conversion and CPU benchmark script
¦   +-- reid_extractor.py       # Deep feature extractor & cosine similarity
¦   +-- multi_cam_manager.py    # Multi-camera pipeline & global identity gallery
+-- app.py                      # Streamlit interactive live dashboard
+-- requirements.txt            # Python dependencies
+-- Dockerfile                  # Containerized deployment config
+-- README.md                   # Technical documentation
```

---

## ?? Quickstart Guide

### 1. Clone & Set Up Virtual Environment

```bash
git clone https://github.com/<your-username>/multi-cam-reid-analytics.git
cd multi-cam-reid-analytics

# Create virtual environment (Python 3.12 recommended)
py -3.12 -m venv venv
venv\Scripts\activate      # On Windows
# source venv/bin/activate # On Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 2. Export ONNX Optimized Model (Inference Acceleration)

```bash
python src/export_onnx.py
```
*Exports `yolov8n.pt` into `models/yolov8n.onnx` and benchmarks latency.*

### 3. Generate Sample CCTV Feeds (if not using own videos)

```bash
python src/get_samples.py
```
*Creates `videos/cam1.mp4` and `videos/cam2.mp4` with coordinated pedestrian movements.*

### 4. Launch Real-Time Analytics Dashboard

```bash
streamlit run app.py
```
*Open `http://localhost:8501` in your browser and click **"?? Start Stream"**.*

---

## ?? Docker Deployment

To build and run as a containerized edge application:

```bash
docker build -t multi-cam-reid-analytics .
docker run -p 8501:8501 multi-cam-reid-analytics
```

---

## ?? Computer Vision Concepts & Interview Cheat Sheet

### 1. YOLOv8 (You Only Look Once)
- **Single-Stage Detector**: Predicts bounding box coordinates and class probabilities simultaneously in one forward pass, unlike two-stage detectors (Faster R-CNN) that first generate region proposals.
- **Anchor-Free Architecture**: Predicts the distance from bounding box centers directly rather than relying on predefined anchor box ratios, reducing hyperparameters and accelerating inference.
- **NMS (Non-Maximum Suppression)**: Filters redundant overlapping bounding boxes by keeping the highest confidence box and suppressing boxes with Intersection over Union (IoU) above a threshold.

### 2. ByteTrack Multi-Object Tracking
- Standard trackers (like SORT) discard low-confidence detections, losing track during occlusions or motion blur.
- ByteTrack retains both high-confidence and low-confidence detections, first associating high-score boxes with existing tracklets using the Hungarian Algorithm and Kalman Filter velocity predictions, then associating unmatched tracklets with low-score boxes.

### 3. Person Re-Identification (Re-ID) & Metric Learning
- **Embedding Vector**: The model maps high-dimensional pixel crops $(H \times W \times 3)$ to a compact latent vector $\mathbf{f} \in \mathbb{R}^{512}$.
- **L2-Normalization**: $\mathbf{e} = \frac{\mathbf{f}}{\|\mathbf{f}\|_2}$, ensuring all identity representations lie on a unit hypersphere.
- **Cosine Similarity**: Measures directional alignment between two embeddings:
  $$\text{CosineSim}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$
- **Gallery Updating with Exponential Moving Average (EMA)**:
  $$\mathbf{e}_{\text{gallery}} \leftarrow \alpha \mathbf{e}_{\text{gallery}} + (1 - \alpha) \mathbf{e}_{\text{new}}$$
  Maintains smooth identity representations as person angle and lighting vary across camera fields of view.

### 4. Real-Time Optimization Strategies
- **Graph Optimization & ONNX Runtime**: Fuses redundant layers (Conv + BatchNorm + ReLU into a single operator), minimizing memory access overhead.
- **Dynamic Resolution / Quantization**: Converting FP32 weights to FP16 or INT8 for high-throughput edge deployment.

---

## ?? License
This project is open-source under the MIT License.
