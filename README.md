# 地平线X5 静默人脸活体检测与人脸识别系统

## 1. 项目概述

本项目是一个基于地平线（Horizon）RDK X5平台的人脸防伪与识别系统，旨在高效地防止使用照片、视频或屏幕等方式冒充真实人脸的攻击行为，并对通过活体检测的真人进行身份识别。

系统集成了高性能的人脸检测（YOLOv8-Face）、静默活体检测（MiniFASNetV2）和人脸特征提取（MobileNetV3）功能，通过C++实现核心算法以保证性能，并由Python提供易于使用的上层应用接口，包括静态图片识别、实时视频流识别和人脸注册功能。

## 2. 系统架构

### 2.1 整体架构
```
┌─────────────────────────────────────────────────────────┐
│                   Python Application Layer              │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────┐  │
│  │  face_recog-    │  │  face_recog-    │  │  face_  │  │
│  │  nition.py      │  │  nition_web.py  │  │  regist │  │
│  └─────────────────┘  └─────────────────┘  │  er.py  │  │
│                                            └─────────┘  │
├─────────────────────────────────────────────────────────┤
│              Python Utility Layer                       │
│  ┌─────────────┐  ┌─────────────────┐  ┌─────────────┐  │
│  │  models/    │  │   utils/        │  │ ctypes      │  │
│  │  tflite     │  │   load_lib.py   │  │  binding    │  │
│  │  _model.py  │  │   face_utils.py │  │             │  │
│  └─────────────┘  └─────────────────┘  └─────────────┘  │
├─────────────────────────────────────────────────────────┤
│              C++ Core Library Layer                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────┐  │
│  │  face_liveness  │  │  Horizon DNN    │  │ OpenCV  │  │
│  │  .cpp/.h        │  │  SDK            │  │         │  │
│  └─────────────────┘  └─────────────────┘  └─────────┘  │
└─────────────────────────────────────────────────────────┘
```

### 2.2 文件结构
```
Anti-Face-Recognition/
├── build/                      # CMake构建输出目录 (生成 libface_liveness.so)
├── weights/                    # 存放模型文件
│   ├── yolov8n-face.bin        # 人脸检测模型
│   ├── anti-face.bin           # 活体检测模型
│   └── mobilenetv3_small_mcp.tflite # 人脸识别模型
├── python/                     # Python应用层
│   ├── models/                 # 人脸识别模型封装
│   │   └── tflite_model.py     # TFLite人脸识别引擎 (MobileNetV3)
│   ├── utils/                  # 工具函数
│   │   ├── face_utils.py       # 人脸对齐工具
│   │   └── load_lib.py         # C++动态库加载器 (ctypes)
│   ├── face_recognition.py     # 命令行静态图片人脸识别应用
│   ├── face_recognition_web.py # 实时视频流人脸识别Web应用 (Flask)
│   ├── face_register.py        # 命令行交互式人脸注册应用
│   ├── requirements.txt        # Python依赖
│   └── 1.sh                    # Python虚拟环境启动脚本
├── face_liveness.cpp           # C++核心实现：人脸检测与活体判断
├── face_liveness.h             # C++核心库头文件
├── main.cc                     # C++库测试程序
└── CMakeLists.txt              # CMake构建配置
```

## 3. 核心组件说明

### 3.1 C++核心库 (face_liveness.cpp/.h)

编译为 `libface_liveness.so` 动态库，提供高性能的人脸检测和活体判断能力。

#### 主要功能
- **人脸检测**: 使用 **YOLOv8-Face** 模型在图像中快速定位人脸。
- **关键点定位**: 定位人脸的5个关键点（双眼、鼻尖、嘴角），为后续人脸对齐做准备。
- **活体检测**: 基于 **MiniFASNetV2** 模型，判断检测到的人脸是真实人脸还是伪造人脸（如纸质照片、电子屏幕等）。

#### 核心API
```cpp
/**
 * @brief 初始化模型
 * @param yolo_model_path yolov8n-face.bin模型路径
 * @param liveness_model_path anti-face.bin模型路径
 */
int initialize_models(const char* yolo_model_path, const char* liveness_model_path);

/**
 * @brief 从图片文件检测人脸及活体状态
 */
int detect_faces_liveness(const char* image_path, FaceDetectionResult* results, int max_faces, int* num_faces);

/**
 * @brief 从内存中的图像数据检测人脸及活体状态 (用于视频流)
 */
int detect_faces_liveness_from_buffer(const unsigned char* image_data, int image_width, int image_height, FaceDetectionResult* results, int max_faces, int* num_faces);

/**
 * @brief 释放模型资源
 */
void release_models();
```

### 3.2 Python应用层

#### 3.2.1 C++库加载器 (utils/load_lib.py)
使用 `ctypes` 标准库加载C++编译的 `libface_liveness.so`，并将其API封装成易于调用的 `FaceLiveness` Python类。
- **单例模式**: 确保模型在全局只被初始化一次，节省资源。
- **内存直传**: 提供了 `detect_from_frame` 方法，可以直接处理 `numpy` 图像数组，避免了不必要的磁盘I/O，是实时视频处理的关键。
- **自动资源管理**: 通过 `atexit` 模块注册清理函数，确保程序退出时能自动释放C++库分配的内存和模型资源。

#### 3.2.2 人脸识别引擎 (models/tflite_model.py)
封装了基于 TensorFlow Lite 的 **MobileNetV3** 人脸识别模型。它接收经过活体检测和对齐后的人脸图像，提取出高维特征向量（Embedding），用于后续的身份比对。

#### 3.2.3 人脸注册 (face_register.py)
一个命令行交互式应用，用于采集和注册用户人脸。
1.  输入要注册的用户名和照片路径。
2.  调用C++库进行人脸检测和活体判断。
3.  **只对通过活体检测的真人脸**进行处理。
4.  提取人脸特征并与用户名关联，保存到 `registered_faces.pkl` 文件中。

#### 3.2.4 静态图片识别 (face_recognition.py)
一个命令行应用，用于识别单张图片中的人脸。
1.  加载 `registered_faces.pkl` 中的人脸数据库。
2.  对输入图片进行人脸检测和活体判断。
3.  如果检测到真人脸，则提取其特征。
4.  将提取的特征与数据库中的所有特征进行比对，找出最相似的用户。

#### 3.2.5 实时Web识别 (face_recognition_web.py)
一个基于 Flask 的Web应用，通过摄像头实现实时人脸识别。
1.  从摄像头逐帧读取画面。
2.  调用 `detect_from_frame` 接口进行高效的活体检测。
3.  **业务逻辑**:
    -   **假脸攻击**: 直接忽略，画面上不显示任何框。
    -   **真人 & 已注册**: 显示 **绿色** 边框，并标注用户名和相似度。
    -   **真人 & 未注册**: 显示 **红色** 边框，作为未知身份提醒。
4.  通过HTTP流将处理后的视频画面推送到浏览器，并附有图例说明。

## 4. 环境准备与使用

### 4.1 环境要求
- 地平线RDK X5开发板
- 系统镜像自带Python 3.10.12
- 已安装OpenCV和地平线DNN（`hb_dnn`）相关依赖

### 4.2 步骤1：编译C++核心库
首先，需要编译C++代码以生成 `libface_liveness.so` 动态库。
```bash
# 进入项目根目录
cd /path/to/Anti-Face-Recognition

# 创建构建目录
mkdir build && cd build
cmake ..
make
```
编译成功后，`build` 目录下会生成 `libface_liveness.so` 文件。

### 4.3 步骤2：准备Python环境
项目使用Python虚拟环境以隔离依赖。
```bash
# 进入python目录
cd ../python

# 激活虚拟环境 (如果1.sh脚本配置正确)
source 1.sh

# 安装所有依赖
pip install -r requirements.txt
```

### 4.4 步骤3：运行应用

#### 4.4.1 注册人脸
在运行识别程序前，必须先注册至少一个用户的脸。
```bash
# 确保你在python目录下并且虚拟环境已激活
python face_register.py
```
根据提示输入用户名和包含清晰、真实人脸的**照片路径**。
> 示例照片路径: `/home/sunrise/Code/Anti-Face-Recognition/images/image_T1.jpg`

成功后会在 `python` 目录下生成 `registered_faces.pkl` 文件。

#### 4.4.2 运行静态图片识别
```bash
# 确保你在python目录下并且虚拟环境已激活
python face_recognition.py
```
根据提示输入待识别的照片路径，程序将输出识别结果。

#### 4.4.3 运行实时Web应用
```bash
# 确保你在python目录下并且虚拟环境已激活
python face_recognition_web.py
```
服务启动后，在浏览器中访问 `http://<你的开发板IP>:5000` 即可看到实时识别画面。

## 5. 注意事项
- **模型路径**: 所有模型文件的路径都在代码中硬编码，请确保 `weights` 目录位置正确。
- **性能**: C++库的性能很大程度上依赖于地平线DNN库的优化。Python应用的帧率受摄像头、模型复杂度和Web传输等多种因素影响。
- **识别阈值**: 人脸识别的相似度阈值（默认为0.35）可在 `face_recognition.py` 和 `face_recognition_web.py` 中调整，以平衡准确率和召回率。
- **中文字体**: Web应用中的中文显示依赖于系统是否安装了常见的中文字体（如 `simhei.ttf`）。如果显示为方框，请检查字体文件是否存在。
