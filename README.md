# 地平线 RDK X5 静默人脸活体检测与人脸识别系统 V2

## 1. 项目概述

本项目是基于地平线（Horizon）RDK X5 平台的高性能人脸防伪与识别系统。

**与V1版本的主要区别**：
V2版本将 **人脸识别 (Feature Extraction)** 模块从 Python TFLite 迁移到了 **C++ BPU (Horizon DNN)**。这意味着**人脸检测、活体检测、人脸识别**三个核心模型全部在 C++ 层利用 BPU 硬件加速运行，Python 层仅作为轻量级的业务逻辑胶水。极大降低了 CPU 占用，提升了整体推理效率，并移除了笨重的 TensorFlow 依赖。

主要功能：
- **人脸检测**: YOLOv8-Face (C++ BPU)
- **静默活体**: MiniFASNetV2 (C++ BPU)，防御照片/视频/屏幕攻击
- **人脸识别**: MobileNetV3 (C++ BPU)，提取 512维特征向量
- **交互应用**: 提供 Python 命令行注册、静态图片比对、实时 Web 监控

## 2. 系统架构

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                   Python Application Layer                  │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐  │
│  │ face_recognition│  │ face_recognition│  │ face_regist │  │
│  │      .py        │  │     _web.py     │  │    er.py    │  │
│  └─────────────────┘  └─────────────────┘  └─────────────┘  │
│           │                    │                    │       │
├───────────┼────────────────────┼────────────────────┼───────┤
│           ▼                    ▼                    ▼       │
│                Python FaceSystem (ctypes wrapper)           │
│           (不再包含 TFLite 引擎，仅负责调用 C++ 接口)       │
├─────────────────────────────────────────────────────────────┤
│                C++ Core Library (libface_liveness.so)       │
│ ┌───────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│ │  Face Detect  │  │  Liveness Check │  │ Face Recognition│ │
│ │ (YOLOv8-Face) │  │  (MiniFASNetV2) │  │  (MobileNetV3)  │ │
│ └───────┬───────┘  └────────┬────────┘  └────────┬────────┘ │
│         │                   │                    │          │
│         ▼                   ▼                    ▼          │
│    Horizon BPU         Horizon BPU          Horizon BPU     │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 文件结构

```
RDK_X5_Anti-Face-Recognition_V2/
├── build/                          # 编译输出目录 (libface_liveness.so)
├── weights/                        # 模型文件 (全部为 .bin 格式)
│   ├── yolov8n-face.bin            # 人脸检测
│   ├── anti-face.bin               # 活体检测
│   └── mobilenetv3_small_..._nv12.bin # 人脸识别 (BPU版)
├── face_liveness.cpp               # C++ 核心实现 (全流程)
├── face_liveness.h                 # C++ 头文件
├── main.cc                         # C++ 纯代码测试程序
├── CMakeLists.txt                  # 构建脚本
└── python/                         # Python 应用
    ├── utils/
    │   ├── load_lib.py             # ctypes 封装 (FaceSystem)
    │   └── face_utils.py           # 相似度计算
    ├── face_register.py            # 人脸注册 (支持摄像头交互)
    ├── face_recognition.py         # 静态图片识别
    ├── face_recognition_web.py     # 实时 Web 监控
    └── requirements.txt            # 依赖列表 (Flask, OpenCV)
```

## 3. 核心技术升级

### 3.1 C++ 全流程推理
V2 版本在 C++ 层实现了完整的推理流水线：
1. **预处理**: BGR 转 NV12 (BPU 原生格式)。
2. **检测**: YOLOv8-Face 输出人脸框和5个关键点。
3. **活体**: 裁剪 ROI，进行 MiniFASNetV2 推理。
4. **对齐**: 使用 `cv::estimateAffinePartial2D` 进行仿射变换，将人脸对齐到标准 112x112 模板。
5. **识别**: 将对齐后的人脸输入 MobileNetV3 (BPU)，输出 512维 Embedding。

### 3.2 Python 轻量化
移除了 `tensorflow` 和 `tflite-runtime` 库。Python 环境非常干净，只需要：
- `flask`: Web 服务
- `opencv-python`: 图像 IO
- `numpy`: 数据处理
- `pillow`: 中文绘制

## 4. 环境准备与编译

### 4.1 环境要求
- 地平线 RDK X5 开发板 (Ubuntu 22.04)
- 已安装 `hbdk-dnn` (地平线 DNN 运行时)
- OpenCV (系统自带或自行编译)

### 4.2 步骤1：编译 C++ 核心库
```bash
cd /path/to/RDK_X5_Anti-Face-Recognition_V2
mkdir build && cd build
cmake ..
make -j4
```
编译成功后，`build` 目录下会生成 `libface_liveness.so`。
*(注意：`python/utils/load_lib.py` 中已配置为绝对路径加载该 so 文件，请确保路径正确)*

### 4.3 步骤2：安装 Python 依赖
```bash
cd ../python
pip install -r requirements.txt
```

## 5. 运行应用

### 5.1 人脸注册 (交互式)
V2 版本支持直接调用摄像头进行注册。
```bash
python face_register.py
```
- 选择 `[1] 注册人脸`
- 输入姓名
- 系统自动打开摄像头
- **绿框**表示真人，**红框**表示假人
- 按 **空格键 (Space)** 抓拍并注册，按 **Q** 退出

### 5.2 实时 Web 监控
```bash
python face_recognition_web.py
```
- 启动 Flask 服务器
- 浏览器访问 `http://<RDK-IP>:5000`
- **功能**：
    - **绿框**: 识别通过（数据库中已存在的真人）
    - **红框**: 陌生人（数据库中不存在的真人）
    - **无框**: 攻击拦截（照片/视频/屏幕）

### 5.3 静态图片测试
```bash
python face_recognition.py
```
- 输入图片路径，输出识别结果和相似度。

## 6. 开发接口说明

如果你想基于本项目开发自己的应用，可以使用 `FaceSystem` 类：

```python
from utils.load_lib import FaceSystem
import cv2

# 初始化 (单例模式)
system = FaceSystem()

# 1. 检测 (包含活体信息)
# 返回 list of dict: [{'bbox':..., 'is_real': True, 'landmarks': ...}, ...]
img = cv2.imread("test.jpg")
faces = system.detect_from_frame(img)

# 2. 提取特征 (只对真人脸操作)
for face in faces:
    if face['is_real']:
        embedding = system.get_embedding(img, face['landmarks'])
        # embedding 是 512维 numpy 数组
```
