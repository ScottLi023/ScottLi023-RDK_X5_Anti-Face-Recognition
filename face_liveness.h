#pragma once

#include <string>

#ifdef __cplusplus
extern "C" {
#endif

// The result structure for detection (C-style for Python compatibility)
struct FaceDetectionResult {
    int is_real;              // 1: Real, 0: Fake
    float bbox[4];            // Bounding box: [x, y, width, height]
    float confidence;         // Liveness confidence
    char label[32];           // Liveness label: [Real Face/Paper Photo/Screen Photo]
    float landmarks[5][2];    // 5 keypoints coordinates
    float landmark_scores[5]; // Confidence scores for 5 keypoints
};

/**
 * @brief Initialize YOLO and Liveness models.
 * @param yolo_model_path Path to the yolov8-face.bin model.
 * @param liveness_model_path Path to the MiniFASNetV2.bin model.
 * @return 0 on success, otherwise a non-zero error code.
 */
int initialize_models(const char* yolo_model_path, const char* liveness_model_path);

/**
 * @brief Perform face detection, landmark extraction, and liveness detection.
 * @param image_path Path to the input image.
 * @param results Pointer to an array of FaceDetectionResult to store the findings.
 * @param max_faces The maximum number of faces the results array can hold.
 * @param num_faces Pointer to an integer that will be filled with the number of detected faces.
 * @return 0 on success, otherwise a non-zero error code.
 */
int detect_faces_liveness(const char* image_path, FaceDetectionResult* results, int max_faces, int* num_faces);

/**
 * @brief Perform face detection, landmark extraction, and liveness detection from memory buffer.
 * @param image_data Pointer to the raw image data (BGR format).
 * @param image_width Width of the image.
 * @param image_height Height of the image.
 * @param results Pointer to an array of FaceDetectionResult to store the findings.
 * @param max_faces The maximum number of faces the results array can hold.
 * @param num_faces Pointer to an integer that will be filled with the number of detected faces.
 * @return 0 on success, otherwise a non-zero error code.
 */
int detect_faces_liveness_from_buffer(const unsigned char* image_data, int image_width, int image_height, FaceDetectionResult* results, int max_faces, int* num_faces);

/**
 * @brief Release all model resources.
 */
void release_models();

#ifdef __cplusplus
}
#endif