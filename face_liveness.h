#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#define FACE_EMBEDDING_SIZE 512

/**
 * 人脸检测结果（C-style，兼容 Python ctypes）
 */
struct FaceDetectionResult {
    int   is_real;              // 1: 真人脸  0: 假人脸
    float bbox[4];              // [x, y, width, height]（原图坐标）
    float confidence;           // 活体置信度
    char  label[32];            // "Real Face" / "Paper Photo" / "Screen Photo"
    float landmarks[5][2];      // 5个关键点 [x, y]（原图坐标）
    float landmark_scores[5];   // 关键点置信度
};

/**
 * @brief 加载三个模型
 * @param yolo_model_path     yolov8n-face.bin
 * @param liveness_model_path anti-face.bin
 * @param recog_model_path    mobilenetv3_small_mcp_bayese_112x112_nv12.bin
 * @return 0 成功，非0 失败
 */
int initialize_models(const char* yolo_model_path,
                      const char* liveness_model_path,
                      const char* recog_model_path);

/**
 * @brief 从图片文件检测人脸 + 活体判断
 */
int detect_faces_liveness(const char* image_path,
                          FaceDetectionResult* results,
                          int max_faces,
                          int* num_faces);

/**
 * @brief 从内存 BGR 帧检测人脸 + 活体判断（用于视频流）
 * @param image_data BGR 格式，连续内存，H×W×3
 */
int detect_faces_liveness_from_buffer(const unsigned char* image_data,
                                      int image_width,
                                      int image_height,
                                      FaceDetectionResult* results,
                                      int max_faces,
                                      int* num_faces);

/**
 * @brief 提取人脸 embedding（ArcFace 对齐 → NV12 → MobileNetV3 BPU 推理）
 * @param image_data    BGR 格式原图，连续内存
 * @param image_width   原图宽
 * @param image_height  原图高
 * @param landmarks_10  5个关键点，展平为10个float: [x0,y0, x1,y1, ..., x4,y4]
 * @param embedding_out 输出 FACE_EMBEDDING_SIZE 个 float
 * @param embedding_size 必须等于 FACE_EMBEDDING_SIZE（用于校验）
 * @return 0 成功，非0 失败
 */
int get_face_embedding(const unsigned char* image_data,
                       int image_width,
                       int image_height,
                       const float* landmarks_10,
                       float* embedding_out,
                       int embedding_size);

/**
 * @brief 释放全部模型资源
 */
void release_models();

#ifdef __cplusplus
}
#endif
