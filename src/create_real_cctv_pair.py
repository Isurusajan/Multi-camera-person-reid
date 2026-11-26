"""
Generate high-definition realistic dual-camera surveillance videos:
- Camera 01: Inside retail store aisle (Person walking in yellow jacket)
- Camera 02: Outside street sidewalk (Exact same person walking past storefront)

Synchronized with realistic CCTV timecode, natural perspective walking motion,
and sensor simulation for flawless Multi-Camera Person Re-ID demonstration.
"""
import os
import cv2
import numpy as np

def generate_surveillance_pair():
    os.makedirs("videos", exist_ok=True)
    cam1_out = os.path.join("videos", "street_store_cam1_inside.mp4")
    cam2_out = os.path.join("videos", "street_store_cam2_outside.mp4")
    
    img1_path = "assets/person_inside_shop.jpg"
    img2_path = "assets/person_outside_street.jpg"
    
    if not os.path.exists(img1_path) or not os.path.exists(img2_path):
        raise FileNotFoundError("Reference assets not found in assets/")
        
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    # Target video specs: 768x432 at 24 FPS, 12 seconds (288 frames)
    out_w, out_h = 768, 432
    fps = 24
    total_frames = 288
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    writer1 = cv2.VideoWriter(cam1_out, fourcc, fps, (out_w, out_h))
    writer2 = cv2.VideoWriter(cam2_out, fourcc, fps, (out_w, out_h))
    
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    print(f"[INFO] Generating Camera 01 (Store Interior) & Camera 02 (Street Exterior)...")
    
    for f in range(total_frames):
        t_sec = f / float(fps)
        
        # ----------------- CAMERA 01 (Store Interior) -----------------
        # Person walks forward down aisle from t=0s to t=8.5s, then exits frame by t=12s
        if t_sec <= 8.5:
            progress = t_sec / 8.5
            scale = 1.0 + progress * 0.18
            crop_w = int(w1 / scale)
            crop_h = int(h1 / scale)
            
            sway_x = int(np.sin(f * 0.3) * 5)
            bob_y = int(abs(np.sin(f * 0.35)) * 4)
            
            cx = w1 // 2 + sway_x
            cy = h1 // 2 - int(progress * 40) + bob_y
            
            x1 = max(0, min(w1 - crop_w, cx - crop_w // 2))
            y1 = max(0, min(h1 - crop_h, cy - crop_h // 2))
            
            sub1 = img1[y1:y1+crop_h, x1:x1+crop_w]
            frame1 = cv2.resize(sub1, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        else:
            exit_prog = (t_sec - 8.5) / 3.5
            scale = 1.18 + exit_prog * 0.15
            crop_w = int(w1 / scale)
            crop_h = int(h1 / scale)
            
            cx = w1 // 2
            cy = h1 // 2 - 40 - int(exit_prog * 180)
            
            x1 = max(0, min(w1 - crop_w, cx - crop_w // 2))
            y1 = max(0, min(h1 - crop_h, cy - crop_h // 2))
            
            sub1 = img1[y1:y1+crop_h, x1:x1+crop_w]
            frame1 = cv2.resize(sub1, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
            
        # ----------------- CAMERA 02 (Street Exterior) -----------------
        # Frame 0 to 2.5s: Camera monitors street sidewalk (other pedestrians)
        # Frame 2.5s to 9.5s: Subject walks out along sidewalk past storefront
        if t_sec < 2.5:
            scale = 1.0
            crop_w = int(w2 / scale)
            crop_h = int(h2 / scale)
            x1 = 0
            y1 = 0
            sub2 = img2[y1:y1+crop_h, x1:x1+crop_w]
            frame2 = cv2.resize(sub2, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        elif t_sec <= 9.5:
            walk_prog = (t_sec - 2.5) / 7.0
            scale = 1.0 + walk_prog * 0.16
            crop_w = int(w2 / scale)
            crop_h = int(h2 / scale)
            
            sway_x = int(np.sin(f * 0.28) * 4)
            bob_y = int(abs(np.sin(f * 0.35)) * 4)
            
            cx = w2 // 2 + sway_x
            cy = h2 // 2 - int(walk_prog * 35) + bob_y
            
            x1 = max(0, min(w2 - crop_w, cx - crop_w // 2))
            y1 = max(0, min(h2 - crop_h, cy - crop_h // 2))
            
            sub2 = img2[y1:y1+crop_h, x1:x1+crop_w]
            frame2 = cv2.resize(sub2, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        else:
            exit_prog2 = (t_sec - 9.5) / 2.5
            scale = 1.16 + exit_prog2 * 0.12
            crop_w = int(w2 / scale)
            crop_h = int(h2 / scale)
            
            cx = w2 // 2
            cy = h2 // 2 - 35 - int(exit_prog2 * 140)
            
            x1 = max(0, min(w2 - crop_w, cx - crop_w // 2))
            y1 = max(0, min(h2 - crop_h, cy - crop_h // 2))
            
            sub2 = img2[y1:y1+crop_h, x1:x1+crop_w]
            frame2 = cv2.resize(sub2, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

        # ----------------- Authentic CCTV OSD Overlays -----------------
        s1 = 12 + int(t_sec)
        m1 = 30 + (s1 // 60)
        s1 = s1 % 60
        ms1 = int((t_sec % 1.0) * 100)
        ts1_str = f"2026-11-26 {14:02d}:{m1:02d}:{s1:02d}.{ms1:02d} GMT"
        
        cv2.rectangle(frame1, (10, 10), (out_w - 10, 36), (15, 23, 42), -1)
        cv2.rectangle(frame1, (10, 10), (out_w - 10, 36), (255, 255, 255), 1)
        cv2.putText(frame1, "CAM 01 [STORE_INTERIOR]", (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 120), 1)
        cv2.putText(frame1, ts1_str, (out_w - 240, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        cv2.circle(frame1, (out_w - 250, 25), 4, (0, 0, 255), -1)
        
        s2 = 16 + int(t_sec)
        m2 = 30 + (s2 // 60)
        s2 = s2 % 60
        ms2 = int((t_sec % 1.0) * 100)
        ts2_str = f"2026-11-26 {14:02d}:{m2:02d}:{s2:02d}.{ms2:02d} GMT"
        
        cv2.rectangle(frame2, (10, 10), (out_w - 10, 36), (15, 23, 42), -1)
        cv2.rectangle(frame2, (10, 10), (out_w - 10, 36), (255, 255, 255), 1)
        cv2.putText(frame2, "CAM 02 [STREET_EXTERIOR]", (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 120), 1)
        cv2.putText(frame2, ts2_str, (out_w - 240, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        cv2.circle(frame2, (out_w - 250, 25), 4, (0, 0, 255), -1)
        
        writer1.write(frame1)
        writer2.write(frame2)
        
    writer1.release()
    writer2.release()
    print(f"[SUCCESS] Multi-Camera Surveillance Videos generated:")
    print(f"  - {cam1_out} ({total_frames} frames, {total_frames/fps:.1f}s)")
    print(f"  - {cam2_out} ({total_frames} frames, {total_frames/fps:.1f}s)")

if __name__ == "__main__":
    generate_surveillance_pair()
