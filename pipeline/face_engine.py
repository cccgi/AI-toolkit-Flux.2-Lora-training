import os
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFilter

# Portable model resolution: models live in <repo_root>/models/face/, downloaded
# by setup/download_models.py. Can be overridden with env vars for advanced users.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_YOLO_PATH = os.environ.get(
    "PIPELINE_YOLOFACE_PATH",
    os.path.join(_REPO_ROOT, "models", "face", "yoloface_8n.onnx"),
)
DEFAULT_ARCFACE_PATH = os.environ.get(
    "PIPELINE_ARCFACE_PATH",
    os.path.join(_REPO_ROOT, "models", "face", "arcface_w600k_r50.onnx"),
)


class FaceEngine:
    def __init__(self, yolo_path=DEFAULT_YOLO_PATH, arcface_path=DEFAULT_ARCFACE_PATH, device="cpu"):
        for p, label in [(yolo_path, "YOLOFace"), (arcface_path, "ArcFace")]:
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"{label} model not found at {p}\n"
                    f"Run 'python setup/download_models.py' first (or set "
                    f"PIPELINE_YOLOFACE_PATH / PIPELINE_ARCFACE_PATH env vars)."
                )
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        # Fallback cleanly to CPU if CUDA provider not available
        self.yolo_sess = ort.InferenceSession(yolo_path, providers=["CPUExecutionProvider"])
        self.arc_sess = ort.InferenceSession(arcface_path, providers=["CPUExecutionProvider"])

    def detect_face(self, img_bgr_or_rgb, is_rgb=False, min_conf=0.45):
        """
        Detects primary face in image.
        Returns dict with: box (x1, y1, x2, y2), conf, landmarks (5, 2), angle
        """
        if is_rgb:
            bgr = cv2.cvtColor(img_bgr_or_rgb, cv2.COLOR_RGB2BGR)
        else:
            bgr = img_bgr_or_rgb

        h_orig, w_orig = bgr.shape[:2]
        scale = min(640.0 / w_orig, 640.0 / h_orig)
        nw, nh = int(w_orig * scale), int(h_orig * scale)
        resized = cv2.resize(bgr, (nw, nh))
        canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
        dx = (640 - nw) // 2
        dy = (640 - nh) // 2
        canvas[dy:dy+nh, dx:dx+nw] = resized

        inp = (canvas[:, :, ::-1].astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis, ...]
        out = self.yolo_sess.run(None, {'input': inp})[0]
        preds = out[0].T  # [8400, 20]
        boxes = preds[:, :4]
        confs = preds[:, 4]
        kpts = preds[:, 5:]  # 5 keypoints (x, y, conf)

        valid = confs > min_conf
        if not np.any(valid):
            return None

        # Select highest confidence face
        valid_indices = np.where(valid)[0]
        best_idx = valid_indices[np.argmax(confs[valid_indices])]
        conf = float(confs[best_idx])
        bx, by, bw, bh = boxes[best_idx]

        bx = (bx - dx) / scale
        by = (by - dy) / scale
        bw /= scale
        bh /= scale

        x1 = max(0, int(bx - bw / 2.0))
        y1 = max(0, int(by - bh / 2.0))
        x2 = min(w_orig, int(bx + bw / 2.0))
        y2 = min(h_orig, int(by + bh / 2.0))

        # Rescale landmarks: 5 points (left eye, right eye, nose, left mouth, right mouth)
        raw_kpts = kpts[best_idx].reshape(-1, 3)
        rescaled_kpts = []
        for p in raw_kpts:
            px = (p[0] - dx) / scale
            py = (p[1] - dy) / scale
            rescaled_kpts.append([px, py])
        landmarks = np.array(rescaled_kpts, dtype=np.float32)

        # Classify pose/angle: based on eye distance to nose tip
        left_eye, right_eye, nose = landmarks[0], landmarks[1], landmarks[2]
        d_left = abs(nose[0] - left_eye[0])
        d_right = abs(right_eye[0] - nose[0])
        total_eye_w = abs(right_eye[0] - left_eye[0]) + 1e-6
        ratio = (d_left - d_right) / total_eye_w

        if ratio > 0.35:
            pose = "profile_right"
        elif ratio > 0.15:
            pose = "three_quarter_right"
        elif ratio < -0.35:
            pose = "profile_left"
        elif ratio < -0.15:
            pose = "three_quarter_left"
        else:
            pose = "frontal"

        return {
            "box": (x1, y1, x2, y2),
            "confidence": conf,
            "landmarks": landmarks,
            "pose": pose,
            "width": w_orig,
            "height": h_orig,
        }

    def extract_embedding(self, img_bgr_or_rgb, box=None, is_rgb=False):
        """
        Extracts 512-dim ArcFace embedding.
        """
        if is_rgb:
            bgr = cv2.cvtColor(img_bgr_or_rgb, cv2.COLOR_RGB2BGR)
        else:
            bgr = img_bgr_or_rgb

        h, w = bgr.shape[:2]
        if box is not None:
            x1, y1, x2, y2 = box
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            face_img = bgr[y1:y2, x1:x2]
        else:
            face_img = bgr

        if face_img.size == 0:
            return None

        # Standard ArcFace preprocessing: 112x112, norm [-1, 1]
        face_resized = cv2.resize(face_img, (112, 112))
        inp = ((face_resized[:, :, ::-1].astype(np.float32) - 127.5) / 128.0).transpose(2, 0, 1)[np.newaxis, ...]
        emb = self.arc_sess.run(None, {'input': inp})[0][0]
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb

    @staticmethod
    def compute_similarity(emb1, emb2):
        """
        Computes cosine similarity between two unit-normalized embeddings.
        Returns float between -1.0 and 1.0 (typically 0.3-0.8 for human faces).
        """
        if emb1 is None or emb2 is None:
            return 0.0
        return float(np.dot(emb1, emb2))

    def generate_face_mask(self, img_pil, box, landmarks=None, feather_radius=25):
        """
        Generates feathered 8-bit grayscale face mask matching AI-Toolkit regional loss requirements.
        White (255) = Face (100% loss weight)
        Black (0) = Background (scaled by mask_min_value in trainer)
        """
        w, h = img_pil.size
        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)

        x1, y1, x2, y2 = box
        bw = x2 - x1
        bh = y2 - y1

        # Use facial landmarks if available for anatomically tight bounding ellipse
        if landmarks is not None and len(landmarks) >= 5:
            center_x = float(landmarks[2][0])  # Nose tip
            center_y = float((landmarks[0][1] + landmarks[3][1]) / 2.0)  # Midpoint between eyes and mouth
            rad_x = bw * 0.52
            rad_y = bh * 0.62
        else:
            center_x = x1 + bw / 2.0
            center_y = y1 + bh / 2.0
            rad_x = bw * 0.50
            rad_y = bh * 0.58

        bbox = [
            int(center_x - rad_x),
            int(center_y - rad_y),
            int(center_x + rad_x),
            int(center_y + rad_y)
        ]
        draw.ellipse(bbox, fill=255)

        # Apply Gaussian blur for soft gradient falloff
        mask = mask.filter(ImageFilter.GaussianBlur(radius=feather_radius))
        return mask
