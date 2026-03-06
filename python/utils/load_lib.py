import ctypes
from ctypes import c_char_p, c_int, c_float, POINTER, byref
import atexit
import os
import numpy as np

# ── Load shared library ────────────────────────────────────────────────────────
_LIB_PATH = "/home/sunrise/Code/RDK_X5_Anti-Face-Recognition_V2/build/libface_liveness.so"

_lib = ctypes.CDLL(_LIB_PATH)

FACE_EMBEDDING_SIZE = 512

# ── C struct mirror ────────────────────────────────────────────────────────────

class FaceDetectionResult(ctypes.Structure):
    _fields_ = [
        ("is_real",          c_int),
        ("bbox",             c_float * 4),
        ("confidence",       c_float),
        ("label",            ctypes.c_char * 32),
        ("landmarks",        (c_float * 2) * 5),
        ("landmark_scores",  c_float * 5),
    ]

# ── Function signatures ────────────────────────────────────────────────────────

_lib.initialize_models.argtypes  = [c_char_p, c_char_p, c_char_p]
_lib.initialize_models.restype   = c_int

_lib.detect_faces_liveness.argtypes = [
    c_char_p, POINTER(FaceDetectionResult), c_int, POINTER(c_int)]
_lib.detect_faces_liveness.restype = c_int

_lib.detect_faces_liveness_from_buffer.argtypes = [
    POINTER(ctypes.c_ubyte), c_int, c_int,
    POINTER(FaceDetectionResult), c_int, POINTER(c_int)]
_lib.detect_faces_liveness_from_buffer.restype = c_int

_lib.get_face_embedding.argtypes = [
    POINTER(ctypes.c_ubyte), c_int, c_int,
    POINTER(c_float), POINTER(c_float), c_int]
_lib.get_face_embedding.restype = c_int

_lib.release_models.argtypes = []
_lib.release_models.restype  = None


# ── FaceSystem (Singleton) ─────────────────────────────────────────────────────

class FaceSystem:
    """
    Python wrapper for libface_liveness.so.
    Provides:
      - detect(image_path)            → list of face dicts (with liveness)
      - detect_from_frame(bgr_array)  → list of face dicts (with liveness)
      - get_embedding(bgr_array, landmarks_np)  → np.ndarray (512,)
    """

    _instance    = None
    _initialized = False

    YOLO_MODEL_PATH = "/home/sunrise/Code/RDK_X5_Anti-Face-Recognition_V2/weights/yolov8n-face.bin"
    LIVENESS_MODEL_PATH = "/home/sunrise/Code/RDK_X5_Anti-Face-Recognition_V2/weights/anti-face.bin"
    RECOG_MODEL_PATH = (
        "/home/sunrise/Code/RDK_X5_Anti-Face-Recognition_V2/weights/"
        "mobilenetv3_large.bin"
    )

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self,
                 yolo_model_path=None,
                 liveness_model_path=None,
                 recog_model_path=None):
        if FaceSystem._initialized:
            return

        yolo_path    = yolo_model_path    or self.YOLO_MODEL_PATH
        liveness_path = liveness_model_path or self.LIVENESS_MODEL_PATH
        recog_path   = recog_model_path   or self.RECOG_MODEL_PATH

        for path in (yolo_path, liveness_path, recog_path):
            if not os.path.exists(path):
                raise FileNotFoundError(f"[ERROR] Model not found: {path}")

        print("[INFO] Initializing FaceSystem (YOLO + Liveness + MobileNetV3)...")
        ret = _lib.initialize_models(
            yolo_path.encode(),
            liveness_path.encode(),
            recog_path.encode(),
        )
        if ret != 0:
            raise RuntimeError(f"[ERROR] initialize_models failed: {ret}")

        FaceSystem._initialized = True
        print("[INFO] FaceSystem ready.")
        atexit.register(self._cleanup)

    # ── Detection ─────────────────────────────────────────────────────────

    def detect(self, image_path: str, max_faces: int = 10) -> list:
        """从图片文件检测人脸+活体状态"""
        self._check_init()
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        results   = (FaceDetectionResult * max_faces)()
        num_faces = c_int(0)
        ret = _lib.detect_faces_liveness(
            image_path.encode(), results, max_faces, byref(num_faces))
        if ret != 0:
            raise RuntimeError(f"detect_faces_liveness failed: {ret}")
        return self._parse(results, num_faces.value)

    def detect_from_frame(self, frame: np.ndarray, max_faces: int = 10) -> list:
        """从内存 BGR numpy 帧检测人脸+活体状态（视频流专用）"""
        self._check_init()
        if not isinstance(frame, np.ndarray):
            raise TypeError("frame must be a numpy ndarray (BGR)")

        frame = np.ascontiguousarray(frame)
        h, w  = frame.shape[:2]
        ptr   = frame.ctypes.data_as(POINTER(ctypes.c_ubyte))

        results   = (FaceDetectionResult * max_faces)()
        num_faces = c_int(0)
        ret = _lib.detect_faces_liveness_from_buffer(
            ptr, w, h, results, max_faces, byref(num_faces))
        if ret != 0:
            raise RuntimeError(f"detect_from_buffer failed: {ret}")
        return self._parse(results, num_faces.value)

    # ── Embedding ─────────────────────────────────────────────────────────

    def get_embedding(self, frame: np.ndarray,
                      landmarks: np.ndarray) -> np.ndarray:
        """
        提取人脸 512-dim embedding。

        Args:
            frame:     BGR numpy 数组（原图，任意尺寸）
            landmarks: shape (5, 2) 的关键点数组

        Returns:
            np.ndarray shape (512,), float32
        """
        self._check_init()
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        h, w  = frame.shape[:2]

        lm = np.asarray(landmarks, dtype=np.float32).reshape(10)
        lm_ptr  = lm.ctypes.data_as(POINTER(c_float))
        emb_buf = np.zeros(FACE_EMBEDDING_SIZE, dtype=np.float32)
        emb_ptr = emb_buf.ctypes.data_as(POINTER(c_float))
        img_ptr = frame.ctypes.data_as(POINTER(ctypes.c_ubyte))

        ret = _lib.get_face_embedding(
            img_ptr, w, h, lm_ptr, emb_ptr, FACE_EMBEDDING_SIZE)
        if ret != 0:
            raise RuntimeError(f"get_face_embedding failed: {ret}")
        return emb_buf

    # ── Internals ─────────────────────────────────────────────────────────

    def _parse(self, results, n: int) -> list:
        faces = []
        for i in range(n):
            r = results[i]
            faces.append({
                "is_real":         bool(r.is_real),
                "confidence":      float(r.confidence),
                "label":           r.label.decode(),
                "bbox":            list(r.bbox),
                "landmarks":       [[r.landmarks[j][0], r.landmarks[j][1]]
                                    for j in range(5)],
                "landmark_scores": list(r.landmark_scores),
            })
        return faces

    def _check_init(self):
        if not FaceSystem._initialized:
            raise RuntimeError("FaceSystem not initialized.")

    def _cleanup(self):
        if FaceSystem._initialized:
            print("[INFO] Releasing FaceSystem resources...")
            _lib.release_models()
            FaceSystem._initialized = False

    def release(self):
        self._cleanup()
        FaceSystem._instance = None
