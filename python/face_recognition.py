import os
import sys
import cv2
import numpy as np
import pickle
import datetime
import warnings
warnings.filterwarnings("ignore")

# 全局变量存储模型实例
_face_liveness = None
_recognizer = None

def init_face_liveness():
    """初始化人脸检测模块"""
    global _face_liveness
    if _face_liveness is None:
        print("[INFO] 初始化人脸检测与活体识别模块...")
        from utils.load_lib import FaceLiveness
        _face_liveness = FaceLiveness()
        print("[INFO] ✅ 人脸检测模块初始化成功")
    return _face_liveness

def init_recognizer():
    """初始化人脸识别模块"""
    global _recognizer
    if _recognizer is None:
        model_path = "/home/sunrise/Code/Anti-Face-Recognition/weights/mobilenetv3_small_mcp.tflite"
        print(f"[INFO] 初始化人脸识别模块...")
        print(f"[INFO]   模型路径: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"[ERROR] TFLite模型文件不存在: {model_path}")
        
        from models.tflite_model import TFLiteFaceEngine
        _recognizer = TFLiteFaceEngine(model_path)
        print("[INFO] ✅ 人脸识别模块初始化成功")
    return _recognizer

def load_registered_faces(filepath="registered_faces.pkl"):
    """从文件加载已注册的人脸数据"""
    if os.path.exists(filepath):
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            print(f"[INFO] 已加载 {len(data)} 个注册人脸: {list(data.keys())}")
            return data
        except Exception as e:
            print(f"[WARNING] 加载人脸数据失败: {e}")
            return {}
    else:
        print("[WARNING] 未找到已注册人脸数据库文件")
        print("[提示] 请先使用 face_register.py 注册人脸")
        return {}

def compute_similarity(feat1: np.ndarray, feat2) -> float:
    """计算两个人脸特征之间的相似度"""
    # 如果 feat2 是列表，转换为 numpy 数组
    if isinstance(feat2, list):
        feat2 = np.array(feat2)
    
    feat1 = feat1.ravel()
    feat2 = feat2.ravel()
    
    # 余弦相似度
    similarity = np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))
    return float(similarity)

def recognize_face_from_photo(photo_path, registered_faces, threshold=0.35):
    """
    从照片识别人脸
    
    Args:
        photo_path: 照片路径
        registered_faces: 已注册的人脸数据库
        threshold: 相似度阈值（默认0.35）
    """
    start_time = datetime.datetime.now()
    current_time = start_time.strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"\n{'='*60}")
    print(f"[{current_time}] 开始人脸识别...")
    print(f"{'='*60}")

    # 检查照片是否存在
    if not os.path.exists(photo_path):
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{current_time}] ❌ 照片路径不存在: {photo_path}")
        total_time = datetime.datetime.now() - start_time
        print(f"总耗时: {total_time.total_seconds():.4f}秒")
        return None

    # 检查是否有已注册的人脸
    if not registered_faces:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{current_time}] ❌ 人脸数据库为空，请先注册人脸")
        total_time = datetime.datetime.now() - start_time
        print(f"总耗时: {total_time.total_seconds():.4f}秒")
        return None

    # 1. 人脸检测
    print(f"[步骤1/4] 检测人脸...")
    try:
        detector = init_face_liveness()
        detect_start = datetime.datetime.now()
        detected_faces = detector.detect(photo_path)
        detect_time = (datetime.datetime.now() - detect_start).total_seconds()
        print(f"[INFO] ✅ 检测到 {len(detected_faces)} 个人脸")
        print(f"人脸检测耗时: {detect_time:.4f}秒")
    except Exception as e:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{current_time}] ❌ 人脸检测失败: {e}")
        import traceback
        traceback.print_exc()
        total_time = datetime.datetime.now() - start_time
        print(f"总耗时: {total_time.total_seconds():.4f}秒")
        return None

    if not detected_faces:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{current_time}] ⚠️ 未检测到人脸")
        total_time = datetime.datetime.now() - start_time
        print(f"总耗时: {total_time.total_seconds():.4f}秒")
        return None

    # 2. 验证活体
    print(f"[步骤2/4] 验证活体...")
    real_face = None
    verify_start = datetime.datetime.now()  # 开始计时
    for idx, face in enumerate(detected_faces):
        print(f"  人脸 {idx+1}: {face['label']} (置信度: {face['confidence']:.3f})")
        if face["is_real"]:
            real_face = face
            print(f"[INFO] ✅ 发现真人脸")
            break
    verify_time = (datetime.datetime.now() - verify_start).total_seconds()  # 计算耗时
    print(f"活体验证耗时: {verify_time:.4f}秒")

    if real_face is None:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{current_time}] ⚠️ 未检测到真人脸，无法识别")
        print("[提示] 请使用真人照片，而非屏幕照片或打印照片")
        total_time = datetime.datetime.now() - start_time
        print(f"总耗时: {total_time.total_seconds():.4f}秒")
        return None

    # 3. 加载识别模型并提取特征
    print(f"[步骤3/4] 提取人脸特征...")
    try:
        recognizer = init_recognizer()
        
        landmarks = np.array(real_face["landmarks"], dtype=np.float32)
        frame = cv2.imread(photo_path)
        if frame is None:
            print(f"[ERROR] ❌ 无法读取图片")
            total_time = datetime.datetime.now() - start_time
            print(f"总耗时: {total_time.total_seconds():.4f}秒")
            return None

        extract_start = datetime.datetime.now()
        embedding = recognizer.get_embedding(frame, landmarks)
        extract_time = (datetime.datetime.now() - extract_start).total_seconds()
        print(f"特征提取耗时: {extract_time:.4f}秒")
    except Exception as e:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{current_time}] ❌ 特征提取失败: {e}")
        import traceback
        traceback.print_exc()
        total_time = datetime.datetime.now() - start_time
        print(f"总耗时: {total_time.total_seconds():.4f}秒")
        return None

    # 4. 人脸匹配
    print(f"[步骤4/4] 人脸匹配...")
    print(f"[INFO] 数据库中共有 {len(registered_faces)} 个人脸")
    
    match_start = datetime.datetime.now()
    best_match = None
    best_similarity = 0.0
    
    for name, reg_embedding in registered_faces.items():
        similarity = compute_similarity(embedding, reg_embedding)
        print(f"  - {name}: 相似度 = {similarity:.4f}", end="")
        
        if similarity > threshold:
            print(" ✅ 匹配成功")
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = name
        else:
            print(f" (低于阈值 {threshold:.2f})")
    
    match_time = (datetime.datetime.now() - match_start).total_seconds()
    print(f"人脸匹配耗时: {match_time:.4f}秒")
    
    # 输出结果
    print(f"\n{'='*60}")
    if best_match:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{current_time}] ✅ 识别成功: {best_match}, 相似度: {best_similarity:.4f}")
    else:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{current_time}] ⚠️ 未找到匹配的人脸")
        print(f"[提示] 相似度阈值: {threshold:.2f}")
    
    total_time = (datetime.datetime.now() - start_time).total_seconds()
    print(f"总耗时: {total_time:.4f}秒")
    print(f"{'='*60}\n")
    
    return best_match


def main():
    """主函数"""
    print("\n" + "="*60)
    print("  人脸识别系统 - 照片模式")
    print("  地平线 RDK X5 平台")
    print("="*60 + "\n")

    try:
        # 1. 初始化人脸检测模块
        print("[初始化] 正在启动系统...\n")
        init_face_liveness()
        
        # 2. 加载已注册人脸数据库
        registered_faces = load_registered_faces()
        
        if not registered_faces:
            print("\n[ERROR] 人脸数据库为空，程序退出")
            print("[提示] 请先运行 face_register.py 注册人脸")
            return
        
        print(f"\n[INFO] ========== 系统初始化完成 ==========\n")
        
        # 3. 循环识别（固定阈值0.35）
        threshold = 0.35
        print(f"[INFO] 相似度阈值: {threshold}\n")
        
        while True:
            print("-"*60)
            photo_path = input("请输入照片路径 (输入 'q' 退出): ").strip()
            
            if photo_path.lower() == 'q':
                print("[INFO] 退出程序")
                break
            
            if not photo_path:
                print("[ERROR] 照片路径不能为空，请重新输入")
                continue

            # 执行识别
            recognize_face_from_photo(photo_path, registered_faces, threshold)
            
    except KeyboardInterrupt:
        print("\n\n[INFO] 用户中断操作 (Ctrl+C)")
    except Exception as e:
        print(f"\n[ERROR] 程序异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n" + "="*60)
        print("  程序结束，感谢使用")
        print("="*60 + "\n")


if __name__ == "__main__":
    main()

#    /home/sunrise/Code/Anti-Face-Recognition/images/image_T1.jpg