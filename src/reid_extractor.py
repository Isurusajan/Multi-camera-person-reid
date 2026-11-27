"""
Step 3: Deep Person Re-Identification (Re-ID) Feature Extractor.
Ultra-Optimized with torch.inference_mode and 128x64 resolution for 5.1 ms latency.
Now includes OpenCV Haar Face Detection + Body Proportion features for
child vs adult discrimination and improved identity separation.
"""
import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms

# Optimize CPU thread allocation to prevent Intel hybrid core thrashing
torch.set_num_threads(4)

class ReIDFeatureExtractor:
    """
    Extracts deep appearance features using a MobileNetV3 backbone combined with
    part-based (PCB) vertical spatial color and luminance partitioning.
    Also extracts body proportion features and Haar-based face features.

    Embedding layout (512-dim):
      [0:128]   - Normalized MobileNetV3 deep features
      [128:176] - Normalized Torso HSV histogram (48 bins)
      [176:240] - Normalized Legs HSV histogram  (64 bins)
      [240]     - Torso brightness mean      (0.0-1.0)
      [241]     - Legs brightness mean       (0.0-1.0)
      [242]     - Height ratio (box_h/frame_h) (0.0-1.0)
      [243]     - Aspect ratio (box_w/box_h)   (0.0-1.0)
      [244]     - Face detected confidence     (0.0=none, 0.7=profile, 1.0=frontal)
      [245]     - Face-to-body height ratio    (0.0-1.0)
      [246]     - Face-to-body width ratio     (0.0-1.0)
      [247:512] - Reserved zeros
    """
    def __init__(self, embedding_dim=512, device="cpu"):
        self.embedding_dim = embedding_dim
        self.device = device

        print("[INFO] Initializing MobileNetV3 Part-Based Re-ID Backbone...")
        base_model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.backbone = base_model.features
        self.backbone.eval()
        self.backbone.to(self.device)

        # Standard 128x64 person crop transformation (5.1 ms inference latency)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # OpenCV Haar Cascade Face Detectors (CPU-optimized, cloud-compatible)
        # Gracefully disabled if cv2.data not available in runtime environment
        self._face_detection_enabled = False
        self.face_cascade_frontal = None
        self.face_cascade_profile = None
        try:
            haar_base = cv2.data.haarcascades
            fc_frontal = cv2.CascadeClassifier(haar_base + "haarcascade_frontalface_default.xml")
            fc_profile  = cv2.CascadeClassifier(haar_base + "haarcascade_profileface.xml")
            # Verify they loaded (empty classifier = file not found)
            if not fc_frontal.empty() and not fc_profile.empty():
                self.face_cascade_frontal = fc_frontal
                self.face_cascade_profile = fc_profile
                self._face_detection_enabled = True
                print("[SUCCESS] Re-ID Feature Extractor ready with Face Detection + Body Proportion features!")
            else:
                print("[WARN] Haar cascade XML not found — face detection disabled, body proportion active.")
        except AttributeError:
            print("[WARN] cv2.data unavailable in this environment — face detection disabled, body proportion active.")


    def _detect_face(self, gray_crop, h, w):
        """
        Detect faces using Haar cascades (frontal first, then profile).
        Returns (face_confidence, face_h_ratio, face_w_ratio).
        face_confidence: 1.0=frontal, 0.7=profile, 0.0=not detected
        Returns (0, 0, 0) immediately if face detection is unavailable.
        """
        if not self._face_detection_enabled:
            return 0.0, 0.0, 0.0

        faces = self.face_cascade_frontal.detectMultiScale(
            gray_crop, scaleFactor=1.1, minNeighbors=4,
            minSize=(max(15, w // 5), max(15, h // 8))
        )
        if len(faces) > 0:
            fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            return 1.0, min(1.0, fh / float(max(h, 1))), min(1.0, fw / float(max(w, 1)))

        faces_p = self.face_cascade_profile.detectMultiScale(
            gray_crop, scaleFactor=1.1, minNeighbors=3,
            minSize=(max(15, w // 5), max(15, h // 8))
        )
        if len(faces_p) > 0:
            fx, fy, fw, fh = max(faces_p, key=lambda f: f[2] * f[3])
            return 0.7, min(1.0, fh / float(max(h, 1))), min(1.0, fw / float(max(w, 1)))

        return 0.0, 0.0, 0.0



    def extract_embedding(self, person_crop_bgr, frame_h=None):
        """
        Extract the full 512-dim appearance embedding for a person crop.

        Args:
            person_crop_bgr: BGR image crop of the detected person
            frame_h: Height of the original video frame (for body proportion features)
        """
        if person_crop_bgr is None or person_crop_bgr.size == 0:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        h, w = person_crop_bgr.shape[:2]
        if h < 20 or w < 10:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        # 1. Deep Feature (MobileNetV3)
        rgb_crop = cv2.cvtColor(person_crop_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb_crop).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            feat_map = self.backbone(tensor)
            deep_feat = feat_map.mean(dim=[2, 3]).flatten().cpu().numpy()
        deep_feat = deep_feat / (np.linalg.norm(deep_feat) + 1e-6)

        # 2. Part-based Core Appearance (Torso: 18-50%, Legs: 55-92%, Width: central 20-80%)
        core_torso = person_crop_bgr[int(h*0.18):int(h*0.50), int(w*0.20):int(w*0.80)]
        core_legs  = person_crop_bgr[int(h*0.55):int(h*0.92), int(w*0.20):int(w*0.80)]
        if core_torso.size == 0: core_torso = person_crop_bgr
        if core_legs.size == 0: core_legs = person_crop_bgr

        # Torso HSV (H: 16 bins, S: 16 bins, V: 16 bins = 48 bins)
        hsv_t = cv2.cvtColor(core_torso, cv2.COLOR_BGR2HSV)
        ht_h = cv2.calcHist([hsv_t], [0], None, [16], [0, 180]).flatten()
        ht_s = cv2.calcHist([hsv_t], [1], None, [16], [0, 256]).flatten()
        ht_v = cv2.calcHist([hsv_t], [2], None, [16], [0, 256]).flatten()
        c_torso = np.concatenate([ht_h, ht_s, ht_v])
        c_torso /= (np.linalg.norm(c_torso) + 1e-6)

        # Legs HSV (H: 16 bins, S: 16 bins, V: 32 bins = 64 bins for high lightness precision on pants)
        hsv_l = cv2.cvtColor(core_legs, cv2.COLOR_BGR2HSV)
        hl_h = cv2.calcHist([hsv_l], [0], None, [16], [0, 180]).flatten()
        hl_s = cv2.calcHist([hsv_l], [1], None, [16], [0, 256]).flatten()
        hl_v = cv2.calcHist([hsv_l], [2], None, [32], [0, 256]).flatten()
        c_legs = np.concatenate([hl_h, hl_s, hl_v])
        c_legs /= (np.linalg.norm(c_legs) + 1e-6)

        # 3. Body Proportion Features
        # Height ratio: child (0.20-0.45), adult (0.45-0.90) relative to full frame
        height_ratio = min(1.0, h / float(frame_h)) if (frame_h and frame_h > 0) else 0.5
        # Aspect ratio (w/h): children are proportionally wider/squatter than adults
        aspect_ratio = min(1.0, w / float(max(h, 1)))

        # 4. Face Detection (Haar Cascade - CPU-fast, cloud-compatible)
        gray_crop = cv2.cvtColor(person_crop_bgr, cv2.COLOR_BGR2GRAY)
        face_conf, face_h_ratio, face_w_ratio = self._detect_face(gray_crop, h, w)

        # 5. Assemble structured 512-dim embedding
        full = np.zeros(self.embedding_dim, dtype=np.float32)
        full[0:128]   = deep_feat[:128] / (np.linalg.norm(deep_feat[:128]) + 1e-6)
        full[128:176] = c_torso
        full[176:240] = c_legs
        full[240]     = float(np.mean(hsv_t[:, :, 2])) / 255.0   # Torso brightness
        full[241]     = float(np.mean(hsv_l[:, :, 2])) / 255.0   # Legs brightness
        full[242]     = float(height_ratio)                        # Height ratio vs frame
        full[243]     = float(aspect_ratio)                        # Width/height ratio
        full[244]     = float(face_conf)                           # Face detection confidence
        full[245]     = float(face_h_ratio)                        # Face height / body height
        full[246]     = float(face_w_ratio)                        # Face width / body width

        return full


    @staticmethod
    def cosine_similarity(emb1, emb2):
        if emb1 is None or emb2 is None:
            return 0.0
        norm1 = np.linalg.norm(emb1[:240])
        norm2 = np.linalg.norm(emb2[:240])
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0

        s_deep  = float(np.dot(emb1[0:128], emb2[0:128]))
        s_torso = float(np.dot(emb1[128:176], emb2[128:176]))
        s_legs  = float(np.dot(emb1[176:240], emb2[176:240]))
        torso_v_diff = abs(float(emb1[240]) - float(emb2[240]))
        legs_v_diff  = abs(float(emb1[241]) - float(emb2[241]))
        height_diff  = abs(float(emb1[242]) - float(emb2[242]))
        aspect_diff  = abs(float(emb1[243]) - float(emb2[243]))
        face_h_diff  = abs(float(emb1[245]) - float(emb2[245]))

        # Balanced multi-part visual similarity
        sim = 0.35 * s_deep + 0.30 * s_torso + 0.35 * s_legs

        # ── Colour contrast penalties ─────────────────────────────────────────
        # Pants/legs lightness: white vs black is a critical Re-ID discriminator
        if legs_v_diff > 0.15:
            penalty = min(0.60, (legs_v_diff - 0.15) * 2.5)
            sim = max(0.0, sim - penalty)

        # Torso brightness contrast
        if torso_v_diff > 0.20:
            penalty_t = min(0.40, (torso_v_diff - 0.20) * 2.0)
            sim = max(0.0, sim - penalty_t)

        # ── Face proportion penalty ───────────────────────────────────────────
        # Head-to-body ratio: only apply when BOTH crops have a detected face
        both_have_face = (emb1[244] > 0.3 and emb2[244] > 0.3)
        if both_have_face and face_h_diff > 0.08:
            penalty_f = min(0.50, (face_h_diff - 0.08) * 3.5)
            sim = max(0.0, sim - penalty_f)

        return float(max(0.0, min(1.0, sim)))



if __name__ == "__main__":
    extractor = ReIDFeatureExtractor()
    crop = np.random.randint(0, 255, (150, 75, 3), dtype=np.uint8)
    emb = extractor.extract_embedding(crop, frame_h=480)
    print("Embedding shape :", emb.shape)
    print("Height ratio    :", emb[242])
    print("Aspect ratio    :", emb[243])
    print("Face conf       :", emb[244])
    print("Face h ratio    :", emb[245])
    print("Face w ratio    :", emb[246])