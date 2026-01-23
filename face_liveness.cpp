#include "face_liveness.h"
#include "dnn/hb_dnn.h"
#include "dnn/hb_dnn_ext.h"
#include <iostream>
#include <vector>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <opencv2/opencv.hpp>

// ============================================================================
// Macros & Constants
// ============================================================================

#define CHECK_HB_SUCCESS(value, errmsg)                                     \
    do {                                                                    \
        auto ret_code = value;                                              \
        if (ret_code != 0) {                                                \
            std::cerr << "[ERROR] " << errmsg << ", Error code: " << ret_code << std::endl; \
            return ret_code;                                                \
        }                                                                   \
    } while (0)

// YOLOv8-Face model constants
#define YOLO_RESIZE_TYPE 0
#define YOLO_LETTERBOX_TYPE 1
#define YOLO_PREPROCESS_TYPE YOLO_RESIZE_TYPE
#define YOLO_NMS_THRESHOLD 0.45
#define YOLO_SCORE_THRESHOLD 0.25
#define YOLO_KPT_SCORE_THRESHOLD 0.5
#define YOLO_REG 16
#define YOLO_KPT_NUM 5
#define YOLO_KPT_ENCODE 3

// Liveness model constants
#define LIVENESS_SCALE 2.7f
#define LIVENESS_INPUT_H 80
#define LIVENESS_INPUT_W 80

// ============================================================================
// Internal Structures
// ============================================================================

// For YOLOv8-Face post-processing
struct FaceLandmark {
    cv::Rect2d bbox;
    float score;
    std::vector<cv::Point2f> landmarks;
    std::vector<float> landmark_scores;
};

// For Liveness post-processing
struct FaceObject {
    cv::Rect2d bbox;
    int liveness_label_idx;
    std::string liveness_text;
    float liveness_confidence;
};

// ============================================================================
// Global Model Handles
// ============================================================================

static hbPackedDNNHandle_t yolo_packed_handle = nullptr;
static hbDNNHandle_t yolo_handle = nullptr;
static hbPackedDNNHandle_t liveness_packed_handle = nullptr;
static hbDNNHandle_t liveness_handle = nullptr;

// ============================================================================
// Utility Functions
// ============================================================================

// BGR to NV12 conversion
cv::Mat bgr2nv12(const cv::Mat& bgr_img) {
    int height = bgr_img.rows;
    int width = bgr_img.cols;
    cv::Mat yuv_mat;
    cv::cvtColor(bgr_img, yuv_mat, cv::COLOR_BGR2YUV_I420);
    uint8_t* yuv = yuv_mat.ptr<uint8_t>();
    cv::Mat nv12_img(height * 3 / 2, width, CV_8UC1);
    uint8_t* nv12 = nv12_img.ptr<uint8_t>();
    int y_size = height * width;
    memcpy(nv12, yuv, y_size);
    int uv_height = height / 2;
    int uv_width = width / 2;
    uint8_t* nv12_uv = nv12 + y_size;
    uint8_t* u_data = yuv + y_size;
    uint8_t* v_data = u_data + uv_height * uv_width;
    for (int i = 0; i < uv_width * uv_height; i++) {
        *nv12_uv++ = *u_data++;
        *nv12_uv++ = *v_data++;
    }
    return nv12_img;
}

// Softmax for DFL and Liveness
void softmax(float* input, float* output, int length) {
    float max_val = -FLT_MAX;
    for (int i = 0; i < length; i++) {
        if (input[i] > max_val) max_val = input[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < length; i++) {
        output[i] = std::exp(input[i] - max_val);
        sum += output[i];
    }
    for (int i = 0; i < length; i++) {
        output[i] /= sum;
    }
}

// YOLO Preprocessing
cv::Mat preprocess_yolo_image(const cv::Mat& img, int input_h, int input_w, float& x_scale, float& y_scale, int& x_shift, int& y_shift) {
    cv::Mat result;
    if (YOLO_PREPROCESS_TYPE == YOLO_LETTERBOX_TYPE) {
        x_scale = std::min(1.0f * input_h / img.rows, 1.0f * input_w / img.cols);
        y_scale = x_scale;
        int new_w = static_cast<int>(img.cols * x_scale);
        int new_h = static_cast<int>(img.rows * y_scale);
        x_shift = (input_w - new_w) / 2;
        y_shift = (input_h - new_h) / 2;
        cv::resize(img, result, cv::Size(new_w, new_h));
        cv::copyMakeBorder(result, result, y_shift, input_h - new_h - y_shift, x_shift, input_w - new_w - x_shift, cv::BORDER_CONSTANT, cv::Scalar(127, 127, 127));
    } else {
        cv::resize(img, result, cv::Size(input_w, input_h));
        x_scale = 1.0f * input_w / img.cols;
        y_scale = 1.0f * input_h / img.rows;
        x_shift = 0;
        y_shift = 0;
    }
    return result;
}

// Liveness ROI cropping
cv::Rect get_liveness_box(int src_w, int src_h, cv::Rect bbox, float scale) {
    float new_width = bbox.width * scale;
    float new_height = bbox.height * scale;
    float center_x = bbox.x + bbox.width / 2.0f;
    float center_y = bbox.y + bbox.height / 2.0f;
    int x1 = static_cast<int>(center_x - new_width / 2);
    int y1 = static_cast<int>(center_y - new_height / 2);
    int x2 = static_cast<int>(center_x + new_width / 2);
    int y2 = static_cast<int>(center_y + new_height / 2);
    x1 = std::max(0, std::min(src_w - 1, x1));
    y1 = std::max(0, std::min(src_h - 1, y1));
    x2 = std::max(0, std::min(src_w - 1, x2));
    y2 = std::max(0, std::min(src_h - 1, y2));
    return cv::Rect(x1, y1, x2 - x1, y2 - y1);
}

// ============================================================================
// Core API Implementation
// ============================================================================

int initialize_models(const char* yolo_model_path, const char* liveness_model_path) {
    // 1. Initialize YOLO model
    CHECK_HB_SUCCESS(hbDNNInitializeFromFiles(&yolo_packed_handle, &yolo_model_path, 1), "Failed to load YOLO model");
    const char** yolo_model_names;
    int yolo_model_count = 0;
    CHECK_HB_SUCCESS(hbDNNGetModelNameList(&yolo_model_names, &yolo_model_count, yolo_packed_handle), "Failed to get YOLO model names");
    CHECK_HB_SUCCESS(hbDNNGetModelHandle(&yolo_handle, yolo_packed_handle, yolo_model_names[0]), "Failed to get YOLO model handle");

    // 2. Initialize Liveness model
    CHECK_HB_SUCCESS(hbDNNInitializeFromFiles(&liveness_packed_handle, &liveness_model_path, 1), "Failed to load Liveness model");
    const char** liveness_model_names;
    int liveness_model_count = 0;
    CHECK_HB_SUCCESS(hbDNNGetModelNameList(&liveness_model_names, &liveness_model_count, liveness_packed_handle), "Failed to get Liveness model names");
    CHECK_HB_SUCCESS(hbDNNGetModelHandle(&liveness_handle, liveness_packed_handle, liveness_model_names[0]), "Failed to get Liveness model handle");

    return 0;
}

void release_models() {
    if (yolo_handle) hbDNNRelease(yolo_packed_handle);
    if (liveness_handle) hbDNNRelease(liveness_packed_handle);
    yolo_handle = nullptr;
    liveness_handle = nullptr;
    yolo_packed_handle = nullptr;
    liveness_packed_handle = nullptr;
}

// 核心处理逻辑：传入cv::Mat进行检测
int detect_process(const cv::Mat& img, FaceDetectionResult* results, int max_faces, int* num_faces) {
    // 2. Get YOLO input properties
    hbDNNTensorProperties input_props_yolo;
    CHECK_HB_SUCCESS(hbDNNGetInputTensorProperties(&input_props_yolo, yolo_handle, 0), "YOLO get input props failed");
    int input_h_yolo = input_props_yolo.validShape.dimensionSize[2];
    int input_w_yolo = input_props_yolo.validShape.dimensionSize[3];

    // 3. Preprocess for YOLO
    float x_scale, y_scale;
    int x_shift, y_shift;
    cv::Mat preprocessed_yolo = preprocess_yolo_image(img, input_h_yolo, input_w_yolo, x_scale, y_scale, x_shift, y_shift);
    cv::Mat nv12_yolo = bgr2nv12(preprocessed_yolo);

    // 4. Prepare YOLO input tensor
    hbDNNTensor yolo_input;
    yolo_input.properties = input_props_yolo;
    hbSysAllocCachedMem(&yolo_input.sysMem[0], input_props_yolo.alignedByteSize);
    memcpy(yolo_input.sysMem[0].virAddr, nv12_yolo.data, input_props_yolo.alignedByteSize);
    hbSysFlushMem(&yolo_input.sysMem[0], HB_SYS_MEM_CACHE_CLEAN);

    // 5. Prepare YOLO output tensors
    int yolo_output_count;
    hbDNNGetOutputCount(&yolo_output_count, yolo_handle);
    hbDNNTensor* yolo_outputs = new hbDNNTensor[yolo_output_count];
    for (int i = 0; i < yolo_output_count; i++) {
        hbDNNGetOutputTensorProperties(&yolo_outputs[i].properties, yolo_handle, i);
        hbSysAllocCachedMem(&yolo_outputs[i].sysMem[0], yolo_outputs[i].properties.alignedByteSize);
    }

    // 6. Run YOLO Inference
    hbDNNTaskHandle_t yolo_task = nullptr;
    hbDNNInferCtrlParam yolo_ctrl_param;
    HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&yolo_ctrl_param);
    CHECK_HB_SUCCESS(hbDNNInfer(&yolo_task, &yolo_outputs, &yolo_input, yolo_handle, &yolo_ctrl_param), "YOLO inference failed");
    CHECK_HB_SUCCESS(hbDNNWaitTaskDone(yolo_task, 0), "YOLO wait task failed");

    // 7. Post-process YOLO results
    std::vector<FaceLandmark> detections;
    const int strides[3] = {8, 16, 32};
    for (int scale = 0; scale < 3; scale++) {
        int box_idx = scale * 2, cls_idx = scale * 2 + 1, kpt_idx = scale + 6;
        int grid_h = input_h_yolo / strides[scale], grid_w = input_w_yolo / strides[scale];
        
        hbSysFlushMem(&yolo_outputs[box_idx].sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE);
        hbSysFlushMem(&yolo_outputs[cls_idx].sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE);
        hbSysFlushMem(&yolo_outputs[kpt_idx].sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE);

        float* box_raw = reinterpret_cast<float*>(yolo_outputs[box_idx].sysMem[0].virAddr);
        float* cls_raw = reinterpret_cast<float*>(yolo_outputs[cls_idx].sysMem[0].virAddr);
        float* kpt_raw = reinterpret_cast<float*>(yolo_outputs[kpt_idx].sysMem[0].virAddr);

        for (int h = 0; h < grid_h; h++) {
            for (int w = 0; w < grid_w; w++) {
                int offset = h * grid_w + w;
                if (1.0f / (1.0f + std::exp(-cls_raw[offset])) < YOLO_SCORE_THRESHOLD) continue;

                FaceLandmark det;
                det.score = 1.0f / (1.0f + std::exp(-cls_raw[offset]));

                // Bbox
                float ltrb[4] = {0.0f};
                for (int i = 0; i < 4; i++) {
                    float dfl_softmax[YOLO_REG];
                    softmax(box_raw + offset * (4 * YOLO_REG) + i * YOLO_REG, dfl_softmax, YOLO_REG);
                    for (int j = 0; j < YOLO_REG; j++) ltrb[i] += dfl_softmax[j] * j;
                }
                float cx = (w + 0.5f) * strides[scale], cy = (h + 0.5f) * strides[scale];
                float x1 = cx - ltrb[0] * strides[scale], y1 = cy - ltrb[1] * strides[scale];
                float x2 = cx + ltrb[2] * strides[scale], y2 = cy + ltrb[3] * strides[scale];
                det.bbox = cv::Rect2d(x1, y1, x2 - x1, y2 - y1);

                // Landmarks
                det.landmarks.resize(YOLO_KPT_NUM);
                det.landmark_scores.resize(YOLO_KPT_NUM);
                for (int k = 0; k < YOLO_KPT_NUM; k++) {
                    float kpt_x = (kpt_raw[offset * (YOLO_KPT_NUM * 3) + k * 3] * 2.0f + w) * strides[scale];
                    float kpt_y = (kpt_raw[offset * (YOLO_KPT_NUM * 3) + k * 3 + 1] * 2.0f + h) * strides[scale];
                    float kpt_conf = 1.0f / (1.0f + std::exp(-kpt_raw[offset * (YOLO_KPT_NUM * 3) + k * 3 + 2]));
                    det.landmarks[k] = cv::Point2f(kpt_x, kpt_y);
                    det.landmark_scores[k] = kpt_conf;
                }
                detections.push_back(det);
            }
        }
    }

    // 8. NMS for YOLO results
    std::vector<cv::Rect2d> nms_boxes;
    std::vector<float> nms_scores;
    std::vector<int> nms_indices;
    for (const auto& det : detections) {
        nms_boxes.push_back(det.bbox);
        nms_scores.push_back(det.score);
    }
    if (!nms_boxes.empty()) {
        cv::dnn::NMSBoxes(nms_boxes, nms_scores, YOLO_SCORE_THRESHOLD, YOLO_NMS_THRESHOLD, nms_indices);
    }

    // 9. Process each valid face
    float inv_x_scale = 1.0f / x_scale, inv_y_scale = 1.0f / y_scale;
    for (int idx : nms_indices) {
        if (*num_faces >= max_faces) break;

        FaceLandmark& det = detections[idx];
        
        // Scale bbox and landmarks back to original image size
        det.bbox.x = (det.bbox.x - x_shift) * inv_x_scale;
        det.bbox.y = (det.bbox.y - y_shift) * inv_y_scale;
        det.bbox.width *= inv_x_scale;
        det.bbox.height *= inv_y_scale;
        for (auto& kpt : det.landmarks) {
            kpt.x = (kpt.x - x_shift) * inv_x_scale;
            kpt.y = (kpt.y - y_shift) * inv_y_scale;
        }

        // 10. Liveness Detection
        FaceObject face_obj;
        face_obj.bbox = det.bbox;
        
        // Crop and resize for liveness model
        cv::Rect liveness_roi_box = get_liveness_box(img.cols, img.rows, face_obj.bbox, LIVENESS_SCALE);
        if (liveness_roi_box.width <= 0 || liveness_roi_box.height <= 0) continue;
        cv::Mat liveness_roi = img(liveness_roi_box);
        cv::Mat resized_liveness;
        cv::resize(liveness_roi, resized_liveness, cv::Size(LIVENESS_INPUT_W, LIVENESS_INPUT_H));
        cv::Mat nv12_liveness = bgr2nv12(resized_liveness);

        // Prepare liveness input tensor
        hbDNNTensorProperties liveness_input_props;
        hbDNNGetInputTensorProperties(&liveness_input_props, liveness_handle, 0);
        hbDNNTensor liveness_input;
        liveness_input.properties = liveness_input_props;
        hbSysAllocCachedMem(&liveness_input.sysMem[0], liveness_input_props.alignedByteSize);
        memcpy(liveness_input.sysMem[0].virAddr, nv12_liveness.data, liveness_input_props.alignedByteSize);
        hbSysFlushMem(&liveness_input.sysMem[0], HB_SYS_MEM_CACHE_CLEAN);

        // Prepare liveness output tensor
        hbDNNTensorProperties liveness_output_props;
        hbDNNGetOutputTensorProperties(&liveness_output_props, liveness_handle, 0);
        hbDNNTensor liveness_output;
        liveness_output.properties = liveness_output_props;
        hbSysAllocCachedMem(&liveness_output.sysMem[0], liveness_output_props.alignedByteSize);
        
        // Run Liveness Inference
        hbDNNTaskHandle_t liveness_task = nullptr;
        hbDNNInferCtrlParam liveness_ctrl_param;
        HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&liveness_ctrl_param);
        hbDNNTensor* liveness_output_ptr = &liveness_output;
        hbDNNInfer(&liveness_task, &liveness_output_ptr, &liveness_input, liveness_handle, &liveness_ctrl_param);
        hbDNNWaitTaskDone(liveness_task, 0);

        // Post-process Liveness
        hbSysFlushMem(&liveness_output.sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE);
        float* raw_output = reinterpret_cast<float*>(liveness_output.sysMem[0].virAddr);
        float probs[3];
        softmax(raw_output, probs, 3);
        int max_idx = std::distance(probs, std::max_element(probs, probs + 3));
        face_obj.liveness_confidence = probs[max_idx];
        face_obj.liveness_label_idx = max_idx;
        const char* labels[] = {"Paper Photo", "Real Face", "Screen Photo"};
        face_obj.liveness_text = labels[max_idx];

        // 11. Populate final result structure
        FaceDetectionResult& final_result = results[*num_faces];
        final_result.is_real = (face_obj.liveness_label_idx == 1);
        final_result.confidence = face_obj.liveness_confidence;
        strncpy(final_result.label, face_obj.liveness_text.c_str(), sizeof(final_result.label) - 1);
        final_result.bbox[0] = det.bbox.x;
        final_result.bbox[1] = det.bbox.y;
        final_result.bbox[2] = det.bbox.width;
        final_result.bbox[3] = det.bbox.height;
        for (int k = 0; k < YOLO_KPT_NUM; ++k) {
            final_result.landmarks[k][0] = det.landmarks[k].x;
            final_result.landmarks[k][1] = det.landmarks[k].y;
            final_result.landmark_scores[k] = det.landmark_scores[k];
        }
        (*num_faces)++;

        // Cleanup liveness tensors
        hbSysFreeMem(&liveness_input.sysMem[0]);
        hbSysFreeMem(&liveness_output.sysMem[0]);
        hbDNNReleaseTask(liveness_task);
    }

    // 12. Cleanup YOLO tensors
    hbSysFreeMem(&yolo_input.sysMem[0]);
    for (int i = 0; i < yolo_output_count; i++) {
        hbSysFreeMem(&yolo_outputs[i].sysMem[0]);
    }
    delete[] yolo_outputs;
    hbDNNReleaseTask(yolo_task);

    return 0;
}

int detect_faces_liveness(const char* image_path, FaceDetectionResult* results, int max_faces, int* num_faces) {
    *num_faces = 0;

    // 1. Load Image
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "[ERROR] Failed to load image: " << image_path << std::endl;
        return -1;
    }

    return detect_process(img, results, max_faces, num_faces);
}

int detect_faces_liveness_from_buffer(const unsigned char* image_data, int image_width, int image_height, FaceDetectionResult* results, int max_faces, int* num_faces) {
    *num_faces = 0;
    // 构造 cv::Mat wrapper，不复制数据
    cv::Mat img(image_height, image_width, CV_8UC3, (void*)image_data);
    if (img.empty()) {
        std::cerr << "[ERROR] Failed to create Mat from buffer." << std::endl;
        return -1;
    }
    
    // 如果需要拷贝数据来保证安全性，可以用 .clone()，但在追求性能时通常直接用
    // 注意：这里假设输入数据是BGR格式，如果是RGB需要在 detect_process 里处理或在这里转换
    // 通常OpenCV读图是BGR，Python cv2也是BGR，所以大概率是一致的
    
    return detect_process(img, results, max_faces, num_faces);
}