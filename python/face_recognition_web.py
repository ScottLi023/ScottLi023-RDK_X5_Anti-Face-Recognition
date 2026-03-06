import cv2
import numpy as np
import os
import pickle
import time
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, Response, render_template_string

from utils.load_lib import FaceSystem
from utils.face_utils import compute_similarity

# ── Config ────────────────────────────────────────────────────────────────────
THRESHOLD   = 0.6
CAMERA_ID   = 0
FRAME_W     = 640
FRAME_H     = 480
DB_PATH     = "registered_faces.pkl"

app = Flask(__name__)

# Globals
_system: FaceSystem = None
_database: dict     = {}
_camera             = None

# ── Chinese text helper ───────────────────────────────────────────────────────

_FONT = None
_FONT_PATHS = [
    "simhei.ttf",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
]

def _get_font(size: int = 28):
    global _FONT
    if _FONT is not None:
        return _FONT
    for p in _FONT_PATHS:
        if os.path.exists(p):
            try:
                _FONT = ImageFont.truetype(p, size)
                return _FONT
            except Exception:
                pass
    return ImageFont.load_default()


def put_text_cn(img: np.ndarray, text: str, pos,
                color=(0, 255, 0), size: int = 28) -> np.ndarray:
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    draw.text(pos, text, fill=color, font=_get_font(size))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# ── Init ──────────────────────────────────────────────────────────────────────

def init_resources() -> bool:
    global _system, _database, _camera

    print("[INFO] Initializing resources...")
    try:
        _system = FaceSystem()

        if os.path.exists(DB_PATH):
            with open(DB_PATH, 'rb') as f:
                _database = pickle.load(f)
            print(f"[INFO] Loaded {len(_database)} registered face(s): "
                  f"{list(_database.keys())}")
        else:
            print("[WARN] No face database — all faces will show as unknown.")

        _camera = cv2.VideoCapture(CAMERA_ID)
        _camera.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
        _camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        if not _camera.isOpened():
            print("[ERROR] Cannot open camera.")
            return False

    except Exception as e:
        print(f"[ERROR] Init failed: {e}")
        return False

    print("[INFO] Resources ready.")
    return True


# ── Frame generator ───────────────────────────────────────────────────────────

def generate_frames():
    fps_count = 0
    fps_start = time.time()
    fps       = 0.0

    while True:
        ok, frame = _camera.read()
        if not ok:
            break

        frame   = cv2.flip(frame, 1)
        display = frame.copy()

        # ── Detection ────────────────────────────────────────────────────
        try:
            faces = _system.detect_from_frame(frame)
        except Exception as e:
            print(f"[WARN] Detection error: {e}")
            faces = []

        # ── Per-face logic ────────────────────────────────────────────────
        for face in faces:
            if not face["is_real"]:
                # Fake face → draw nothing (anti-spoofing, silent ignore)
                continue

            x, y, w, h = (int(v) for v in face["bbox"])
            landmarks   = np.array(face["landmarks"], dtype=np.float32)

            # Extract embedding
            try:
                embedding = _system.get_embedding(frame, landmarks)
            except Exception as e:
                print(f"[WARN] Embedding error: {e}")
                continue

            # Identity matching
            best_name  = None
            best_score = 0.0
            for name, ref_emb in _database.items():
                score = compute_similarity(embedding, ref_emb)
                if score > best_score:
                    best_score = score
                    best_name  = name

            if best_score >= THRESHOLD:
                # Known person → green box
                color = (0, 255, 0)
                cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
                label = f"{best_name} ({best_score:.2f})"
                display = put_text_cn(display, label, (x, max(0, y - 36)),
                                      color=(0, 255, 0))
            else:
                # Unknown real person → red box
                color = (0, 0, 255)
                cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)

        # ── FPS overlay ───────────────────────────────────────────────────
        fps_count += 1
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            fps       = fps_count / elapsed
            fps_count = 0
            fps_start = time.time()
        cv2.putText(display, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # ── Encode ────────────────────────────────────────────────────────
        _, buf = cv2.imencode('.jpg', display, [cv2.IMWRITE_JPEG_QUALITY, 85])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
               + buf.tobytes() + b'\r\n')


# ── Flask routes ──────────────────────────────────────────────────────────────

_HTML = '''
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>人脸识别监控 - RDK X5 V2</title>
  <style>
    body { background:#1a1a1a; color:#fff; text-align:center; font-family:sans-serif; margin:0; }
    h1   { margin:16px 0 8px; font-size:1.4em; }
    .video-wrap { display:inline-block; border:3px solid #444; }
    img  { display:block; max-width:100%; }
    .legend { margin:10px; font-size:0.9em; }
    .g { color:#00ff00; font-weight:bold; }
    .r { color:#ff4444; font-weight:bold; }
  </style>
</head>
<body>
  <h1>📸 实时人脸识别监控  —  RDK X5 V2</h1>
  <div class="video-wrap">
    <img src="/video_feed">
  </div>
  <div class="legend">
    <span class="g">■ 绿框</span>：已知真人 &nbsp;|&nbsp;
    <span class="r">■ 红框</span>：未知真人 &nbsp;|&nbsp;
    无框：假脸（照片/视频攻击被拦截）
  </div>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(_HTML)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# ── Entry ─────────────────────────────────────────────────────────────────────

def main():
    if not init_resources():
        print("[ERROR] Startup failed.")
        return

    print("\n" + "=" * 50)
    print("  Web server running.")
    print("  Open: http://0.0.0.0:5000")
    print("=" * 50 + "\n")

    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    finally:
        if _camera and _camera.isOpened():
            _camera.release()
        if _system:
            _system.release()


if __name__ == "__main__":
    main()
