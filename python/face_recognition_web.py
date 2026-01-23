
import cv2
import numpy as np
import os
import time
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, Response, render_template_string

# 导入现有模块
from utils.load_lib import FaceLiveness
from models.tflite_model import TFLiteFaceEngine
from face_recognition import load_registered_faces, compute_similarity

app = Flask(__name__)

# 全局变量
camera = None
liveness_detector = None
recognizer = None
registered_faces = {}

def cv2_add_chinese_text(img, text, position, text_color=(0, 255, 0), text_size=30):
    """
    使用PIL在OpenCV图像上绘制中文
    """
    if (isinstance(img, np.ndarray)):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    draw = ImageDraw.Draw(img)
    
    # 尝试加载中文字体
    font_paths = [
        "simhei.ttf",  # 常见Windows/Linux字体
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf", # Ubuntu/Debian常见
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",   # Arch/Fedora常见
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"               # 常见开源中文字体
    ]
    
    font = None
    for path in font_paths:
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, text_size)
                break
            except:
                continue
                
    if font is None:
        font = ImageFont.load_default()

    draw.text(position, text, fill=text_color, font=font)
    
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def init_resources():
    global liveness_detector, recognizer, registered_faces, camera
    print("[INFO] 正在初始化模型...")
    
    try:
        # 初始化检测器 (使用优化后的内存接口)
        liveness_detector = FaceLiveness()
        
        # 初始化识别器 (MobileNetV3)
        rec_model_path = "/home/sunrise/Code/Anti-Face-Recognition/weights/mobilenetv3_small_mcp.tflite"
        recognizer = TFLiteFaceEngine(rec_model_path)
        
        # 加载注册人脸库
        registered_faces = load_registered_faces("registered_faces.pkl")
        if not registered_faces:
            print("[WARNING] 人脸库为空，所有人将被标记为未知身份")

        # 初始化摄像头
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
    except Exception as e:
        print(f"[ERROR] 初始化失败: {e}")
        return False
    return True

def generate_frames():
    global camera, liveness_detector, recognizer, registered_faces
    
    fps_count = 0
    fps_start = time.time()
    fps = 0

    while True:
        success, frame = camera.read()
        if not success:
            break
            
        # 镜像翻转
        frame = cv2.flip(frame, 1)
        display_frame = frame.copy()

        # --- 人脸检测 (使用内存直传接口) ---
        detected_faces = []
        try:
            detected_faces = liveness_detector.detect_from_frame(frame)
        except Exception as e:
            # 兼容旧接口的fallback，如果detect_from_frame不可用
            # print(f"[WARNING] 内存检测失败，尝试文件模式: {e}")
            try:
                temp_path = "temp_flask_frame.jpg"
                cv2.imwrite(temp_path, frame)
                detected_faces = liveness_detector.detect(temp_path)
            except Exception as e2:
                print(f"[WARNING] 检测失败: {e2}")

        # --- 处理检测结果 ---
        for face in detected_faces:
            # 要求1: 假脸 -> 忽略 (直接显示原图，不绘制任何框)
            if not face['is_real']:
                continue

            # 真人脸处理
            bbox = face['bbox'] # [x, y, w, h]
            x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            
            # 特征提取
            landmarks = np.array(face['landmarks'], dtype=np.float32)
            embedding = recognizer.get_embedding(frame, landmarks)
            
            # 身份匹配
            best_match_name = None
            max_similarity = 0.0
            threshold = 0.35
            
            for name, reg_embedding in registered_faces.items():
                similarity = compute_similarity(embedding, reg_embedding)
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_match_name = name

            # 绘制结果
            if max_similarity > threshold:
                # 要求2: 真人 + 已知 -> 绿框 + 中文信息
                color = (0, 255, 0) # 绿色
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
                
                text_content = f"{best_match_name} ({max_similarity:.2f})"
                display_frame = cv2_add_chinese_text(
                    display_frame, 
                    text_content, 
                    (x, y - 35), 
                    text_color=color,
                    text_size=30
                )
            else:
                # 要求3: 真人 + 未知 -> 红框
                color = (0, 0, 255) # 红色
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)

        # 计算并显示FPS
        fps_count += 1
        if time.time() - fps_start > 1.0:
            fps = fps_count / (time.time() - fps_start)
            fps_count = 0
            fps_start = time.time()
        
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 编码为JPEG流
        ret, buffer = cv2.imencode('.jpg', display_frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template_string('''
        <html>
          <head>
            <title>实时人脸识别监控</title>
            <style>
              body { background-color: #222; color: white; text-align: center; font-family: sans-serif; }
              h1 { margin-top: 20px; }
              .video-container { margin: 20px auto; border: 5px solid #444; display: inline-block; max-width: 100%; }
              img { width: 100%; max-width: 800px; height: auto; }
              .legend { margin-top: 10px; }
              .green { color: #00ff00; font-weight: bold; }
              .red { color: #ff0000; font-weight: bold; }
            </style>
          </head>
          <body>
            <h1>📸 实时人脸识别监控系统</h1>
            <div class="video-container">
              <img src="{{ url_for('video_feed') }}">
            </div>
            <div class="legend">
              <p><span class="green">绿色边框</span>: 已知身份 (真人) &nbsp;|&nbsp; <span class="red">红色边框</span>: 未知身份 (真人)</p>
              <p>无边框: 假脸 (照片/视频攻击) 或者 未检测到人脸</p>
            </div>
          </body>
        </html>
    ''')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def main():
    if init_resources():
        print("="*60)
        print("  Web服务器启动成功！")
        print("  请访问: http://0.0.0.0:5000")
        print("="*60)
        try:
            app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
        finally:
            if camera and camera.isOpened():
                camera.release()
            if liveness_detector:
                try:
                    liveness_detector.release()
                except:
                    pass
    else:
        print("[ERROR] 系统初始化失败")

if __name__ == "__main__":
    main()
