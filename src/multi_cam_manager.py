"""
Step 4: Real-Time Multi-Camera Tracking & Global Identity Re-ID Pipeline.
Optimized for 20-30+ FPS Real-Time Inference on CPU.
Uses imgsz=320 resolution, ByteTrack, and Track-Level Embedding Caching.
"""
import os
import sys
import time
from collections import defaultdict
import cv2
import numpy as np
import torch

# Prevent CPU thread contention on Intel 13th Gen hybrid cores
torch.set_num_threads(4)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from src.reid_extractor import ReIDFeatureExtractor
except ImportError:
    from reid_extractor import ReIDFeatureExtractor


class GlobalIdentityGallery:
    """
    Central database that associates local camera tracks to unified Global Person IDs.
    """
    def __init__(self, similarity_threshold=0.60, alpha=0.85):
        self.similarity_threshold = similarity_threshold
        self.alpha = alpha
        self.next_global_id = 1
        self.identities = {}
        self.active_in_cam = defaultdict(dict)
        self.track_history_cam = defaultdict(dict)  # cam_id -> {local_id: (gid, timestamp)}
        self.transition_log = []

    def reset(self):
        """Instantaneous reset without releasing models or memory"""
        self.next_global_id = 1
        self.identities.clear()
        self.active_in_cam.clear()
        self.track_history_cam.clear()
        self.transition_log.clear()

    def associate_or_register(self, local_id, cam_id, embedding=None, timestamp=None, crop=None):
        if timestamp is None:
            timestamp = time.time()
        time_str = time.strftime("%H:%M:%S", time.localtime(timestamp))

        # 1. Continuous active track in this camera
        if local_id in self.active_in_cam[cam_id]:
            gid = self.active_in_cam[cam_id][local_id]
            self.track_history_cam[cam_id][local_id] = (gid, timestamp)

            if embedding is not None:
                templates = self.identities[gid].setdefault("embeddings", [self.identities[gid]["embedding"]])
                sims = [ReIDFeatureExtractor.cosine_similarity(embedding, t) for t in templates]
                max_sim = max(sims) if sims else 0.0

                # Only update running profile if current frame genuinely matches
                if max_sim >= 0.65:
                    old_emb = self.identities[gid]["embedding"]
                    new_emb = self.alpha * old_emb + (1 - self.alpha) * embedding
                    norm = np.linalg.norm(new_emb[:240])
                    if norm > 1e-6:
                        self.identities[gid]["embedding"][:240] = new_emb[:240] / norm
                        self.identities[gid]["embedding"][240:] = embedding[240:]

                    # Periodically save diverse appearance angles (up to 5 templates per person)
                    if len(templates) < 5 and min(sims) < 0.85:
                        templates.append(embedding.copy())

            self.identities[gid]["last_cam"] = cam_id
            self.identities[gid]["last_time"] = timestamp
            self.identities[gid]["count"] += 1

            # Periodically accumulate multi-angle snapshots (up to 6 distinct poses)
            if crop is not None and crop.size > 0 and len(self.identities[gid]["snapshots"]) < 6:
                if (self.identities[gid]["count"] % 25 == 0):
                    snap = cv2.resize(crop, (140, 280))
                    self.identities[gid]["snapshots"].append({
                        "image": snap,
                        "cam": cam_id,
                        "time": time_str
                    })
            return gid, 1.0, False

        # 2. Brief occlusion recovery in the same camera (< 1.5 seconds)
        if local_id in self.track_history_cam[cam_id]:
            old_gid, last_time = self.track_history_cam[cam_id][local_id]
            if (timestamp - last_time) < 1.5 and old_gid in self.identities:
                self.active_in_cam[cam_id][local_id] = old_gid
                self.track_history_cam[cam_id][local_id] = (old_gid, timestamp)
                self.identities[old_gid]["last_cam"] = cam_id
                self.identities[old_gid]["last_time"] = timestamp
                self.identities[old_gid]["count"] += 1
                return old_gid, 1.0, False

        # If this is a new track and embedding is not ready yet
        if embedding is None:
            return 1, 1.0, False

        # 3. Match against known global identities using Multi-Template Matching
        best_gid = None
        best_score = -1.0
        currently_in_same_cam = set(self.active_in_cam[cam_id].values())

        for gid, info in self.identities.items():
            if gid in currently_in_same_cam:
                continue

            templates = info.get("embeddings", [info["embedding"]])
            scores = [ReIDFeatureExtractor.cosine_similarity(embedding, t) for t in templates]
            sim = max(scores) if scores else 0.0
            if sim > best_score:
                best_score = sim
                best_gid = gid

        # Cross-Camera or Re-identification Match Found
        if best_gid is not None and best_score >= self.similarity_threshold:
            prev_cam = self.identities[best_gid]["last_cam"]
            is_transition = (prev_cam is not None and prev_cam != cam_id)

            old_emb = self.identities[best_gid]["embedding"]
            new_emb = self.alpha * old_emb + (1 - self.alpha) * embedding
            norm = np.linalg.norm(new_emb[:240])
            if norm > 1e-6:
                self.identities[best_gid]["embedding"][:240] = new_emb[:240] / norm
                self.identities[best_gid]["embedding"][240:] = embedding[240:]

            templates = self.identities[best_gid].setdefault("embeddings", [self.identities[best_gid]["embedding"]])
            if len(templates) < 5:
                min_s = min(ReIDFeatureExtractor.cosine_similarity(embedding, t) for t in templates)
                if min_s < 0.85:
                    templates.append(embedding.copy())

            self.identities[best_gid]["last_cam"] = cam_id
            self.identities[best_gid]["last_time"] = timestamp
            self.identities[best_gid]["count"] += 1

            # Save snapshot from this new camera viewpoint
            if crop is not None and crop.size > 0:
                snap = cv2.resize(crop, (140, 280))
                self.identities[best_gid]["thumbnail"] = snap
                if len(self.identities[best_gid]["snapshots"]) < 8:
                    self.identities[best_gid]["snapshots"].append({
                        "image": snap,
                        "cam": cam_id,
                        "time": time_str
                    })

            self.active_in_cam[cam_id][local_id] = best_gid
            self.track_history_cam[cam_id][local_id] = (best_gid, timestamp)

            if is_transition:
                event = {
                    "time": time_str,
                    "global_id": best_gid,
                    "from_cam": prev_cam,
                    "to_cam": cam_id,
                    "similarity": round(best_score, 2),
                    "event_type": "Cross-Camera Transition"
                }
                self.transition_log.append(event)
                self.identities[best_gid]["history"].append({
                    "time": time_str,
                    "cam": cam_id,
                    "event": f"Cross-Camera Transition: {prev_cam} -> {cam_id} (Sim: {best_score:.2f})"
                })
                print(f"[RE-ID EVENT] Person #{best_gid} transitioned: {prev_cam} -> {cam_id} (Sim: {best_score:.2f})")
            else:
                prev_time = self.identities[best_gid].get("last_time", 0)
                time_gap = max(0.0, timestamp - prev_time)
                # Only log distinct re-entry event if absent for more than 2.5 seconds or first re-ID
                if time_gap > 2.5 or len(self.identities[best_gid]["history"]) <= 1:
                    event = {
                        "time": time_str,
                        "global_id": best_gid,
                        "from_cam": cam_id,
                        "to_cam": cam_id,
                        "similarity": round(best_score, 2),
                        "event_type": "Customer Re-entry / Re-identified"
                    }
                    self.transition_log.append(event)
                    self.identities[best_gid]["history"].append({
                        "time": time_str,
                        "cam": cam_id,
                        "event": f"Re-identified after absence ({time_gap:.1f}s) in {cam_id} (Sim: {best_score:.2f})"
                    })
                    print(f"[RE-ID EVENT] Person #{best_gid} re-identified in {cam_id} (Sim: {best_score:.2f})")

            return best_gid, best_score, True

        # 4. New Person: Register new Global ID with initial snapshot
        new_gid = self.next_global_id
        self.next_global_id += 1
        thumbnail = cv2.resize(crop, (140, 280)) if (crop is not None and crop.size > 0) else None
        initial_snapshots = []
        if thumbnail is not None:
            initial_snapshots.append({
                "image": thumbnail,
                "cam": cam_id,
                "time": time_str
            })

        self.identities[new_gid] = {
            "embedding": embedding,
            "embeddings": [embedding.copy()],
            "last_cam": cam_id,
            "last_time": timestamp,
            "count": 1,
            "thumbnail": thumbnail,
            "snapshots": initial_snapshots,
            "history": [{
                "time": time_str,
                "cam": cam_id,
                "event": f"First detected & registered in {cam_id}"
            }]
        }
        self.transition_log.append({
            "time": time_str,
            "global_id": new_gid,
            "from_cam": cam_id,
            "to_cam": cam_id,
            "similarity": 1.0,
            "event_type": "New Person Registered"
        })
        self.active_in_cam[cam_id][local_id] = new_gid
        self.track_history_cam[cam_id][local_id] = (new_gid, timestamp)
        return new_gid, 1.0, False

    def cleanup_inactive(self, cam_id, current_local_ids):
        for local_id in list(self.active_in_cam[cam_id].keys()):
            if local_id not in current_local_ids:
                del self.active_in_cam[cam_id][local_id]


class MultiCameraPipeline:
    """
    High-Throughput Multi-Camera Tracking Pipeline.
    Runs independent ByteTrack instances per camera at imgsz=320 for 25+ FPS real-time CPU performance.
    """
    def __init__(self, model_path="yolov8n.pt", reid_threshold=0.60, conf=0.25):
        from ultralytics import YOLO
        print(f"[INFO] Initializing Optimized Pipeline with {model_path} (imgsz=320)...")
        self.model_path = model_path
        self.models = {
            "CAM_01": YOLO(model_path),
            "CAM_02": YOLO(model_path)
        }
        self.conf = conf
        self.extractor = ReIDFeatureExtractor()
        self.gallery = GlobalIdentityGallery(similarity_threshold=reid_threshold)
        self.fps_records = defaultdict(float)
        self.frame_counts = defaultdict(int)

    def reset(self):
        """Reset tracking states and identity gallery instantly without reloading models"""
        self.gallery.reset()
        self.fps_records.clear()
        self.frame_counts.clear()
        for m in self.models.values():
            if hasattr(m, "predictor") and m.predictor is not None:
                if hasattr(m.predictor, "trackers"):
                    m.predictor.trackers.clear()

    def process_camera_frame(self, frame_bgr, cam_id="CAM_01"):
        start_t = time.perf_counter()
        h, w = frame_bgr.shape[:2]
        self.frame_counts[cam_id] += 1
        cur_frame_num = self.frame_counts[cam_id]

        if cam_id not in self.models:
            from ultralytics import YOLO
            self.models[cam_id] = YOLO(self.model_path)

        # Fast tracking with imgsz=288 for maximum CPU throughput
        tracker_path = os.path.join(os.path.dirname(__file__), "..", "configs", "fast_bytetrack.yaml")
        tracker_cfg = tracker_path if os.path.exists(tracker_path) else "bytetrack.yaml"
        results = self.models[cam_id].track(
            frame_bgr,
            persist=True,
            classes=[0],
            conf=self.conf,
            imgsz=288,
            verbose=False,
            tracker=tracker_cfg
        )

        annotated = frame_bgr.copy()
        current_local_ids = set()
        active_objects = []

        if results and len(results) > 0 and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()

            # Pre-collect valid current local IDs and purge dead tracks before association
            for box, local_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = [int(v) for v in box]
                if (x2 - x1) >= 15 and (y2 - y1) >= 30:
                    current_local_ids.add(int(local_id))

            self.gallery.cleanup_inactive(cam_id, current_local_ids)

            for box, local_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = [int(v) for v in box]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if (x2 - x1) < 15 or (y2 - y1) < 30:
                    continue

                # Track-level embedding caching: compute only if new track or every 20 frames
                is_known_track = local_id in self.gallery.active_in_cam[cam_id]
                needs_embedding = (not is_known_track) or (cur_frame_num % 20 == 0)

                crop = frame_bgr[y1:y2, x1:x2]
                embedding = None
                if needs_embedding:
                    embedding = self.extractor.extract_embedding(crop)

                global_id, sim, is_trans = self.gallery.associate_or_register(
                    local_id=local_id,
                    cam_id=cam_id,
                    embedding=embedding,
                    crop=crop
                )

                # Color coding
                color_hue = (global_id * 47) % 180
                hsv_color = np.uint8([[[color_hue, 220, 240]]])
                bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
                box_color = (int(bgr_color[0]), int(bgr_color[1]), int(bgr_color[2]))

                # Draw bounding box & label tag (safely bounded on screen)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
                tag = f"Global #{global_id} (L:{local_id})"
                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                y_tag = max(th + 6, y1)
                cv2.rectangle(annotated, (x1, y_tag - th - 6), (x1 + tw + 6, y_tag), box_color, -1)
                cv2.putText(annotated, tag, (x1 + 3, y_tag - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

                active_objects.append({
                    "global_id": global_id,
                    "local_id": int(local_id),
                    "camera": cam_id
                })

        self.gallery.cleanup_inactive(cam_id, current_local_ids)

        # FPS calculation
        latency = time.perf_counter() - start_t
        fps = 1.0 / max(latency, 1e-4)
        self.fps_records[cam_id] = fps
        cv2.putText(annotated, f"FPS: {fps:.1f}", (annotated.shape[1] - 90, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return annotated, active_objects