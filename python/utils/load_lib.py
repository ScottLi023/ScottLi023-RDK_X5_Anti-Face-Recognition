import ctypes
from ctypes import c_char_p, c_int, POINTER, byref
import atexit
import os
import numpy as np

# 加载共享库
lib = ctypes.CDLL('/home/sunrise/Code/Anti-Face-Recognition/build/libface_liveness.so')

# 定义C结构体：FaceDetectionResult
class FaceDetectionResult(ctypes.Structure):
    _fields_ = [
        ("is_real", c_int),
        ("bbox", ctypes.c_float * 4),
        ("confidence", ctypes.c_float),
        ("label", ctypes.c_char * 32),
        ("landmarks", (ctypes.c_float * 2) * 5),
        ("landmark_scores", ctypes.c_float * 5)
    ]

# 配置C函数原型
lib.initialize_models.argtypes = [c_char_p, c_char_p]
lib.initialize_models.restype = c_int

lib.detect_faces_liveness.argtypes = [
    c_char_p,
    POINTER(FaceDetectionResult),
    c_int,
    POINTER(c_int)
]
lib.detect_faces_liveness.restype = c_int

lib.detect_faces_liveness_from_buffer.argtypes = [
    POINTER(ctypes.c_ubyte),
    c_int,
    c_int,
    POINTER(FaceDetectionResult),
    c_int,
    POINTER(c_int)
]
lib.detect_faces_liveness_from_buffer.restype = c_int

lib.release_models.argtypes = []
lib.release_models.restype = None


class FaceLiveness:
    _instance = None
    _initialized = False
    _ref_count = 0  # 引用计数

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(FaceLiveness, cls).__new__(cls)
        return cls._instance

    def __init__(self, 
                 yolo_model_path="/home/sunrise/Code/Anti-Face-Recognition/weights/yolov8n-face.bin", 
                 liveness_model_path="/home/sunrise/Code/Anti-Face-Recognition/weights/anti-face.bin"):
        
        # 增加引用计数
        FaceLiveness._ref_count += 1
        
        # 只初始化一次
        if FaceLiveness._initialized:
            # print(f"[INFO] 模型已初始化，引用计数: {FaceLiveness._ref_count}")
            return
            
        # 检查模型文件
        if not os.path.exists(yolo_model_path):
            raise FileNotFoundError(f"[ERROR] YOLO模型文件不存在: {yolo_model_path}")
        if not os.path.exists(liveness_model_path):
            raise FileNotFoundError(f"[ERROR] 活体检测模型文件不存在: {liveness_model_path}")
        
        self.yolo_model_path = yolo_model_path.encode('utf-8')
        self.liveness_model_path = liveness_model_path.encode('utf-8')

        # 初始化模型
        print(f"[INFO] 正在初始化模型...")
        # print(f"  - YOLO模型: {yolo_model_path}")
        # print(f"  - 活体模型: {liveness_model_path}")
        
        ret = lib.initialize_models(self.yolo_model_path, self.liveness_model_path)
        if ret != 0:
            raise RuntimeError(f"[ERROR] 模型初始化失败，错误代码: {ret}")
        
        FaceLiveness._initialized = True
        print("[INFO] 模型初始化成功")
        
        # 注册程序退出时的清理函数
        atexit.register(self._cleanup_on_exit)

    def detect(self, image_path, max_faces=10):
        """检测图片中的人脸及其活体状态"""
        if not FaceLiveness._initialized:
            raise RuntimeError("[ERROR] 模型未初始化，请先创建FaceLiveness实例")
        
        # 检查图片文件是否存在
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"[ERROR] 图片文件不存在: {image_path}")
        
        # 分配结果数组
        results = (FaceDetectionResult * max_faces)()
        num_faces = c_int(0)

        # 调用C++函数
        ret = lib.detect_faces_liveness(
            image_path.encode('utf-8'), 
            results, 
            max_faces, 
            byref(num_faces)
        )
        
        if ret != 0:
            raise RuntimeError(f"[ERROR] 人脸检测失败，错误代码: {ret}")

        return self._parse_results(results, num_faces)

    def detect_from_frame(self, frame, max_faces=10):
        """
        从内存中的图像帧检测人脸
        frame: numpy array (BGR format)
        """
        if not FaceLiveness._initialized:
            raise RuntimeError("[ERROR] 模型未初始化，请先创建FaceLiveness实例")
        
        if frame is None or not isinstance(frame, np.ndarray):
            raise ValueError("[ERROR] 无效的图像帧")

        height, width = frame.shape[:2]
        
        # 确保数据是连续的，并获取指针
        if not frame.flags['C_CONTIGUOUS']:
            frame = np.ascontiguousarray(frame)
        
        frame_ptr = frame.ctypes.data_as(POINTER(ctypes.c_ubyte))

        # 分配结果数组
        results = (FaceDetectionResult * max_faces)()
        num_faces = c_int(0)

        # 调用C++函数
        ret = lib.detect_faces_liveness_from_buffer(
            frame_ptr,
            width,
            height,
            results, 
            max_faces, 
            byref(num_faces)
        )
        
        if ret != 0:
            raise RuntimeError(f"[ERROR] 人脸检测失败，错误代码: {ret}")

        return self._parse_results(results, num_faces)

    def _parse_results(self, results, num_faces):
        # 解析结果
        detected_faces = []
        for i in range(num_faces.value):
            face = results[i]
            face_info = {
                "is_real": bool(face.is_real),
                "confidence": face.confidence,
                "label": face.label.decode('utf-8'),
                "bbox": list(face.bbox),
                "landmarks": [[face.landmarks[j][0], face.landmarks[j][1]] for j in range(5)],
                "landmark_scores": list(face.landmark_scores)
            }
            detected_faces.append(face_info)

        return detected_faces

    @classmethod
    def _cleanup_on_exit(cls):
        """程序退出时自动清理资源"""
        if cls._initialized:
            print("[INFO] 程序退出，自动释放模型资源...")
            try:
                lib.release_models()
                cls._initialized = False
                print("[INFO] 模型资源已释放")
            except Exception as e:
                print(f"[WARNING] 释放资源时出错: {e}")

    def release(self):
        """显式释放C++模型资源（不推荐手动调用，使用atexit自动处理）"""
        FaceLiveness._ref_count -= 1
        # print(f"[INFO] 减少引用计数: {FaceLiveness._ref_count}")
        
        if FaceLiveness._ref_count <= 0 and FaceLiveness._initialized:
            lib.release_models()
            FaceLiveness._initialized = False
            FaceLiveness._instance = None
            print("[INFO] 模型资源已显式释放")

    def __del__(self):
        # 不在__del__中释放资源，避免与垃圾回收冲突
        pass
