import os
import cv2
import numpy as np
import pickle
from utils.face_utils import face_alignment
from utils.load_lib import FaceLiveness
from ctypes import c_int, byref

class SimpleFaceRegistration:
    def __init__(self, model_path="/home/sunrise/Code/Anti-Face-Recognition/weights/mobilenetv3_small_mcp.tflite"):
        """
        初始化简单人脸注册系统
        
        Args:
            model_path: TFLite模型路径
        """
        # 初始化活体检测
        self.face_liveness = FaceLiveness()

        # 初始化人脸识别引擎
        from models import TFLiteFaceEngine
        self.recognizer = TFLiteFaceEngine(model_path)
        
        # 加载已保存的人脸数据（如果存在）
        self.registered_faces = self.load_registered_faces()
        
    def load_registered_faces(self, filepath="registered_faces.pkl"):
        """
        从文件加载已注册的人脸数据
        
        Args:
            filepath: 保存人脸数据的文件路径
        """
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        else:
            return {}
    
    def save_registered_faces(self, filepath="registered_faces.pkl"):
        """
        保存注册的人脸数据到文件
        
        Args:
            filepath: 保存人脸数据的文件路径
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self.registered_faces, f)
        print(f"人脸数据已保存到 {filepath}")
    
    def register_face_from_photo(self, name, photo_path):
        """
        从照片注册人脸
        
        Args:
            name: 被注册人的姓名
            photo_path: 照片路径
        """
        if not os.path.exists(photo_path):
            print(f"照片路径 {photo_path} 不存在")
            return

        try:
            detected_faces = self.face_liveness.detect(photo_path)
        except RuntimeError as e:
            print(e)
            return

        if not detected_faces:
            print("未检测到人脸")
            return

        # 遍历检测结果，找到第一个真人脸
        for face in detected_faces:
            if face["is_real"]:
                # 提取人脸特征
                landmarks = np.array(face["landmarks"], dtype=np.float32)
                frame = cv2.imread(photo_path)
                embedding = self.recognizer.get_embedding(frame, landmarks)

                # 保存到注册人脸字典
                self.registered_faces[name] = [embedding]
                print(f"已注册 {name} 的人脸")

                # 保存注册的人脸数据
                self.save_registered_faces()
                return

        print("未检测到真人脸，无法注册")

def main():
    """
    主函数
    """
    print("简单人脸注册系统 - 照片模式")

    # 获取要注册的人名
    name = input("请输入要注册的人名: ").strip()
    if not name:
        print("人名不能为空")
        return

    # 获取照片路径
    photo_path = input("请输入照片路径: ").strip()
    if not photo_path:
        print("照片路径不能为空")
        return

    # 创建并运行人脸注册系统
    face_reg = SimpleFaceRegistration()
    face_reg.register_face_from_photo(name, photo_path)

if __name__ == "__main__":
    main()

#   /home/sunrise/Code/Anti-Face-Recognition/images/image_T1.jpg