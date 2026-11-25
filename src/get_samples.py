"""
Step 2: Generate Realistic Multi-Camera CCTV Streams for Tracking & Re-ID Testing.
Requirement Match: "Developing and optimizing real-time video analytics pipelines"
"""
import os
import cv2
import numpy as np

def generate_cctv_streams(output_dir="videos", duration_sec=10, fps=20):
    os.makedirs(output_dir, exist_ok=True)
    cam1_path = os.path.join(output_dir, "cam1.mp4")
    cam2_path = os.path.join(output_dir, "cam2.mp4")
    
    # Real pedestrians la inna image ekak load kirima (bus.jpg)
    bus_path = "bus.jpg"
    if not os.path.exists(bus_path):
        import urllib.request
        print("[INFO] Downloading reference pedestrian image...")
        urllib.request.urlretrieve("https://ultralytics.com/images/bus.jpg", bus_path)
        
    img = cv2.imread(bus_path)
    if img is None:
        raise FileNotFoundError(f"Could not load {bus_path}")
        
    # Crop distinct real people from the image
    p1 = cv2.resize(img[380:880, 40:240], (110, 260))   # Person A
    p2 = cv2.resize(img[390:850, 220:340], (100, 250))  # Person B
    p3 = cv2.resize(img[400:870, 660:800], (110, 260))  # Person C
    
    # Coordinated movement trajectory across Camera 1 and Camera 2
    people = [
        {"crop": p1, "cam1_start": 0, "cam2_start": int(fps * 3.5), "speed": 4.0},
        {"crop": p2, "cam1_start": int(fps * 1.5), "cam2_start": int(fps * 5.0), "speed": 3.2},
        {"crop": p3, "cam1_start": int(fps * 4.0), "cam2_start": int(fps * 7.5), "speed": 3.8}
    ]
    
    width, height = 640, 480
    total_frames = duration_sec * fps
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    out1 = cv2.VideoWriter(cam1_path, fourcc, fps, (width, height))
    out2 = cv2.VideoWriter(cam2_path, fourcc, fps, (width, height))
    
    print(f"[INFO] Generating synthetic CCTV feeds ({total_frames} frames)...")
    
    for f in range(total_frames):
        # Background canvas (Cam 1: Corridor, Cam 2: Lobby)
        f1 = np.full((height, width, 3), 230, dtype=np.uint8)
        f2 = np.full((height, width, 3), 225, dtype=np.uint8)
        
        # Camera labels
        cv2.rectangle(f1, (10, 10), (220, 40), (30, 30, 30), -1)
        cv2.putText(f1, "CAM_01 [East Corridor]", (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 120), 2)
        cv2.rectangle(f2, (10, 10), (220, 40), (30, 30, 30), -1)
        cv2.putText(f2, "CAM_02 [Central Lobby]", (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 120), 2)
        
        # Place people walking across cameras
        for p in people:
            # Camera 1: Person walks left-to-right
            if f >= p["cam1_start"]:
                el1 = f - p["cam1_start"]
                x1 = int(-100 + el1 * p["speed"])
                y1 = 180 + int(np.sin(el1 * 0.1) * 6)
                ch, cw = p["crop"].shape[:2]
                if 0 <= x1 < width - cw and 0 <= y1 < height - ch:
                    f1[y1:y1+ch, x1:x1+cw] = p["crop"]
                    
            # Camera 2: Person enters after delay (Transition)
            if f >= p["cam2_start"]:
                el2 = f - p["cam2_start"]
                x2 = int(width + 20 - el2 * p["speed"])
                y2 = 185 + int(np.sin(el2 * 0.1) * 6)
                ch, cw = p["crop"].shape[:2]
                if 0 <= x2 < width - cw and 0 <= y2 < height - ch:
                    f2[y2:y2+ch, x2:x2+cw] = p["crop"]
                    
        # Timestamp
        ts = f"2026-11-24 11:55:{f//fps:02d}.{int((f%fps)*(1000/fps)):03d}"
        cv2.putText(f1, ts, (width - 240, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1)
        cv2.putText(f2, ts, (width - 240, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1)
        
        out1.write(f1)
        out2.write(f2)
        
    out1.release()
    out2.release()
    print(f"[SUCCESS] Sample videos created:\n  - {cam1_path}\n  - {cam2_path}")

if __name__ == "__main__":
    generate_cctv_streams()
