import os
import cv2
import numpy as np
import pickle

from utils.load_lib import FaceSystem

CAMERA_ID = 0
FRAME_W   = 640
FRAME_H   = 480


class FaceRegistration:
    def __init__(self, db_path: str = "registered_faces.pkl"):
        self.db_path  = db_path
        self.system   = FaceSystem()
        self.database = self._load_db()

    # ── DB helpers ────────────────────────────────────────────────────────

    def _load_db(self) -> dict:
        if os.path.exists(self.db_path):
            with open(self.db_path, 'rb') as f:
                db = pickle.load(f)
            print(f"[INFO] Loaded {len(db)} registered face(s): {list(db.keys())}")
            return db
        return {}

    def _save_db(self):
        with open(self.db_path, 'wb') as f:
            pickle.dump(self.database, f)
        print(f"[INFO] Database saved to {self.db_path}")

    # ── Camera registration ───────────────────────────────────────────────

    def register_from_camera(self, name: str) -> bool:
        """
        打开摄像头预览，实时显示检测框。
        按 SPACE 或 ENTER 抓取当前帧进行注册；按 q/ESC 取消。
        只接受通过活体检测的真人脸。
        """
        cap = cv2.VideoCapture(CAMERA_ID)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        if not cap.isOpened():
            print("[ERROR] Cannot open camera.")
            return False

        print(f"\n[INFO] Camera ready. Registering: '{name}'")
        print("       SPACE / ENTER  →  capture & register")
        print("       q / ESC        →  cancel\n")

        result = False
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] Failed to read frame.")
                break

            frame = cv2.flip(frame, 1)
            display = frame.copy()

            # Live preview: run detection and draw boxes
            try:
                faces = self.system.detect_from_frame(frame)
            except Exception:
                faces = []

            for face in faces:
                x, y, w, h = (int(v) for v in face["bbox"])
                if face["is_real"]:
                    color = (0, 255, 0)
                    label = f"Real ({face['confidence']:.2f})"
                else:
                    color = (0, 0, 255)
                    label = f"Fake ({face['confidence']:.2f})"
                cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
                cv2.putText(display, label, (x, max(0, y - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Prompt overlay
            cv2.putText(display, f"Name: {name}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(display, "SPACE=capture  q=cancel", (10, FRAME_H - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

            cv2.imshow("Face Registration", display)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), 27):   # q or ESC → cancel
                print("[INFO] Registration cancelled.")
                break

            if key in (ord(' '), 13):   # SPACE or ENTER → capture
                print("[INFO] Frame captured, processing...")
                result = self._process_frame(frame, name)
                if result:
                    # Show success feedback for 1.5 s
                    ok_frame = frame.copy()
                    cv2.putText(ok_frame, f"OK: {name} registered!",
                                (10, FRAME_H // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                    cv2.imshow("Face Registration", ok_frame)
                    cv2.waitKey(1500)
                else:
                    # Show failure feedback for 1.5 s
                    fail_frame = frame.copy()
                    cv2.putText(fail_frame, "Failed! Try again.",
                                (10, FRAME_H // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                    cv2.imshow("Face Registration", fail_frame)
                    cv2.waitKey(1500)
                    # Don't break — let user retry
                    continue
                break

        cap.release()
        cv2.destroyAllWindows()
        return result

    def _process_frame(self, frame: np.ndarray, name: str) -> bool:
        """检测帧中的真人脸并提取 embedding 存入数据库。"""
        try:
            faces = self.system.detect_from_frame(frame)
        except Exception as e:
            print(f"[ERROR] Detection failed: {e}")
            return False

        if not faces:
            print("[WARN] No face detected in captured frame.")
            return False

        real_face = next((f for f in faces if f["is_real"]), None)
        if real_face is None:
            print("[WARN] No real face detected — anti-spoofing rejected.")
            return False

        landmarks = np.array(real_face["landmarks"], dtype=np.float32)
        try:
            embedding = self.system.get_embedding(frame, landmarks)
        except Exception as e:
            print(f"[ERROR] Embedding extraction failed: {e}")
            return False

        self.database[name] = embedding
        self._save_db()
        print(f"[OK] '{name}' registered successfully.")
        return True

    # ── Other helpers ─────────────────────────────────────────────────────

    def list_registered(self):
        print(f"[INFO] Registered users ({len(self.database)}): "
              f"{list(self.database.keys())}")

    def delete(self, name: str):
        if name in self.database:
            del self.database[name]
            self._save_db()
            print(f"[OK] '{name}' removed from database.")
        else:
            print(f"[WARN] '{name}' not found in database.")


def main():
    print("\n" + "=" * 50)
    print("  人脸注册系统  —  RDK X5 V2")
    print("=" * 50 + "\n")

    reg = FaceRegistration()

    while True:
        print("\n[1] 注册人脸  [2] 查看已注册  [3] 删除  [q] 退出")
        choice = input("选择: ").strip().lower()

        if choice == 'q':
            break
        elif choice == '1':
            name = input("请输入姓名: ").strip()
            if name:
                reg.register_from_camera(name)
        elif choice == '2':
            reg.list_registered()
        elif choice == '3':
            name = input("要删除的姓名: ").strip()
            if name:
                reg.delete(name)


if __name__ == "__main__":
    main()
