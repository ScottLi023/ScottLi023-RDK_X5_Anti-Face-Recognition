import os
import cv2
import numpy as np
import pickle
import datetime

from utils.load_lib import FaceSystem
from utils.face_utils import compute_similarity

THRESHOLD = 0.35


def load_database(db_path: str = "registered_faces.pkl") -> dict:
    if not os.path.exists(db_path):
        print("[WARN] No face database found. Run face_register.py first.")
        return {}
    with open(db_path, 'rb') as f:
        db = pickle.load(f)
    print(f"[INFO] Loaded {len(db)} registered face(s): {list(db.keys())}")
    return db


def recognize(system: FaceSystem, photo_path: str, database: dict,
              threshold: float = THRESHOLD):
    """
    识别单张照片中的人脸。
    Returns: (best_name | None, similarity)
    """
    t0 = datetime.datetime.now()

    if not os.path.exists(photo_path):
        print(f"[ERROR] Image not found: {photo_path}")
        return None, 0.0

    if not database:
        print("[ERROR] Database is empty.")
        return None, 0.0

    # 1. Detect + liveness
    print("[1/3] Detecting face...")
    faces = system.detect(photo_path)
    if not faces:
        print("[WARN] No face detected.")
        return None, 0.0

    real_face = next((f for f in faces if f["is_real"]), None)
    if real_face is None:
        print("[WARN] No real face detected (anti-spoofing blocked).")
        return None, 0.0
    print(f"      → {real_face['label']} (conf={real_face['confidence']:.3f})")

    # 2. Extract embedding
    print("[2/3] Extracting embedding...")
    frame     = cv2.imread(photo_path)
    landmarks = np.array(real_face["landmarks"], dtype=np.float32)
    embedding = system.get_embedding(frame, landmarks)

    # 3. Match
    print("[3/3] Matching...")
    best_name  = None
    best_score = 0.0
    for name, ref_emb in database.items():
        score = compute_similarity(embedding, ref_emb)
        mark  = " ✓" if score >= threshold else ""
        print(f"      {name}: {score:.4f}{mark}")
        if score > best_score:
            best_score = score
            best_name  = name

    elapsed = (datetime.datetime.now() - t0).total_seconds()
    print(f"\n  Elapsed : {elapsed:.3f}s")
    print(f"  Threshold: {threshold}")

    if best_score >= threshold:
        print(f"  Result  : ✓ {best_name}  (similarity={best_score:.4f})")
        return best_name, best_score
    else:
        print("  Result  : ✗ Unknown person")
        return None, best_score


def main():
    print("\n" + "=" * 50)
    print("  人脸识别系统  —  RDK X5 V2")
    print("=" * 50 + "\n")

    system   = FaceSystem()
    database = load_database()

    if not database:
        return

    print(f"\n[INFO] Similarity threshold: {THRESHOLD}\n")

    while True:
        print("-" * 50)
        path = input("照片路径 (q 退出): ").strip()
        if path.lower() == 'q':
            break
        if not path:
            continue
        recognize(system, path, database)


if __name__ == "__main__":
    main()
