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
// Macros
// ============================================================================

#define CHECK_HB(value, msg)                                                        \
    do {                                                                            \
        auto _ret = (value);                                                        \
        if (_ret != 0) {                                                            \
            std::cerr << "[ERROR] " << (msg) << ", code=" << _ret << std::endl;    \
            return _ret;                                                            \
        }                                                                           \
    } while (0)

// ============================================================================
// YOLO Constants
// ============================================================================

#define YOLO_SCORE_THRESHOLD  0.25f
#define YOLO_NMS_THRESHOLD    0.45f
#define YOLO_REG              16      // DFL regression bins
#define YOLO_KPT_NUM          5

// Preprocessing mode: 0=resize, 1=letterbox
#define YOLO_PREPROCESS_RESIZE    0
#define YOLO_PREPROCESS_LETTERBOX 1
#define YOLO_PREPROCESS_TYPE      YOLO_PREPROCESS_RESIZE

// ============================================================================
// Liveness Constants
// ============================================================================

#define LIVENESS_SCALE    2.7f
#define LIVENESS_INPUT_H  80
#define LIVENESS_INPUT_W  80

// ============================================================================
// Internal Structures
// ============================================================================

struct FaceLandmarkDet {
    cv::Rect2d               bbox;
    float                    score;
    std::vector<cv::Point2f> landmarks;      // 5 points
    std::vector<float>       landmark_scores;
};

struct FaceObject {
    cv::Rect2d  bbox;
    int         liveness_label_idx;
    std::string liveness_text;
    float       liveness_confidence;
};

// ============================================================================
// Global Model Handles
// ============================================================================

static hbPackedDNNHandle_t g_yolo_packed     = nullptr;
static hbDNNHandle_t       g_yolo_handle     = nullptr;
static hbPackedDNNHandle_t g_liveness_packed = nullptr;
static hbDNNHandle_t       g_liveness_handle = nullptr;
static hbPackedDNNHandle_t g_recog_packed    = nullptr;
static hbDNNHandle_t       g_recog_handle    = nullptr;

// ============================================================================
// ArcFace Reference Landmarks (112×112)
// Mirrors face_utils.py reference_alignment
// ============================================================================

static const cv::Point2f ARCFACE_REF[5] = {
    {38.2946f, 51.6963f},   // left eye
    {73.5318f, 51.5014f},   // right eye
    {56.0252f, 71.7366f},   // nose tip
    {41.5493f, 92.3655f},   // left mouth corner
    {70.7299f, 92.2041f}    // right mouth corner
};

// ============================================================================
// Utility: BGR → NV12
// ============================================================================

static cv::Mat bgr2nv12(const cv::Mat& bgr) {
    int h = bgr.rows, w = bgr.cols;
    cv::Mat yuv_i420;
    cv::cvtColor(bgr, yuv_i420, cv::COLOR_BGR2YUV_I420);

    cv::Mat nv12(h * 3 / 2, w, CV_8UC1);
    uint8_t* y_src = yuv_i420.ptr<uint8_t>();
    uint8_t* u_src = y_src + h * w;
    uint8_t* v_src = u_src + (h / 2) * (w / 2);
    uint8_t* dst   = nv12.ptr<uint8_t>();

    memcpy(dst, y_src, h * w);
    dst += h * w;
    int uv_len = (h / 2) * (w / 2);
    for (int i = 0; i < uv_len; i++) {
        *dst++ = u_src[i];
        *dst++ = v_src[i];
    }
    return nv12;
}

// ============================================================================
// Utility: Softmax
// ============================================================================

static void softmax(const float* in, float* out, int n) {
    float max_v = *std::max_element(in, in + n);
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { out[i] = std::exp(in[i] - max_v); sum += out[i]; }
    for (int i = 0; i < n; i++) out[i] /= sum;
}

// ============================================================================
// Utility: YOLO input preprocessing (resize or letterbox)
// ============================================================================

static cv::Mat preprocess_yolo(const cv::Mat& img, int in_h, int in_w,
                                float& x_scale, float& y_scale,
                                int& x_shift,   int& y_shift) {
    cv::Mat result;
    if (YOLO_PREPROCESS_TYPE == YOLO_PREPROCESS_LETTERBOX) {
        x_scale = std::min(1.0f * in_h / img.rows, 1.0f * in_w / img.cols);
        y_scale = x_scale;
        int new_w = static_cast<int>(img.cols * x_scale);
        int new_h = static_cast<int>(img.rows * y_scale);
        x_shift   = (in_w - new_w) / 2;
        y_shift   = (in_h - new_h) / 2;
        cv::resize(img, result, cv::Size(new_w, new_h));
        cv::copyMakeBorder(result, result,
                           y_shift, in_h - new_h - y_shift,
                           x_shift, in_w - new_w - x_shift,
                           cv::BORDER_CONSTANT, cv::Scalar(127, 127, 127));
    } else {
        cv::resize(img, result, cv::Size(in_w, in_h));
        x_scale = 1.0f * in_w / img.cols;
        y_scale = 1.0f * in_h / img.rows;
        x_shift = 0;
        y_shift = 0;
    }
    return result;
}

// ============================================================================
// Utility: expand bbox for liveness ROI
// ============================================================================

static cv::Rect get_liveness_roi(int src_w, int src_h, cv::Rect bbox, float scale) {
    float cx = bbox.x + bbox.width  * 0.5f;
    float cy = bbox.y + bbox.height * 0.5f;
    float nw = bbox.width  * scale;
    float nh = bbox.height * scale;
    int x1 = std::max(0, std::min(src_w - 1, (int)(cx - nw / 2)));
    int y1 = std::max(0, std::min(src_h - 1, (int)(cy - nh / 2)));
    int x2 = std::max(0, std::min(src_w - 1, (int)(cx + nw / 2)));
    int y2 = std::max(0, std::min(src_h - 1, (int)(cy + nh / 2)));
    return cv::Rect(x1, y1, x2 - x1, y2 - y1);
}

// ============================================================================
// Utility: ArcFace alignment  →  112×112 BGR
// Mirrors face_utils.py: estimate_norm() + cv2.warpAffine()
// ============================================================================

static cv::Mat align_face_112(const cv::Mat& img,
                               const std::vector<cv::Point2f>& kpts) {
    std::vector<cv::Point2f> ref(ARCFACE_REF, ARCFACE_REF + 5);
    cv::Mat M = cv::estimateAffinePartial2D(kpts, ref, cv::noArray(),
                                            cv::RANSAC, 3.0);
    if (M.empty()) {
        // Fallback to 3-point affine
        std::vector<cv::Point2f> s3(kpts.begin(), kpts.begin() + 3);
        std::vector<cv::Point2f> d3(ref.begin(),  ref.begin()  + 3);
        M = cv::getAffineTransform(s3, d3);
    }
    cv::Mat aligned;
    cv::warpAffine(img, aligned, M, cv::Size(112, 112),
                   cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));
    return aligned;
}

// ============================================================================
// initialize_models
// ============================================================================

int initialize_models(const char* yolo_path,
                      const char* liveness_path,
                      const char* recog_path) {
    // YOLO
    CHECK_HB(hbDNNInitializeFromFiles(&g_yolo_packed, &yolo_path, 1),
             "Failed to load YOLO model");
    const char** yolo_names; int yolo_cnt = 0;
    CHECK_HB(hbDNNGetModelNameList(&yolo_names, &yolo_cnt, g_yolo_packed),
             "Failed to get YOLO model names");
    CHECK_HB(hbDNNGetModelHandle(&g_yolo_handle, g_yolo_packed, yolo_names[0]),
             "Failed to get YOLO handle");
    std::cout << "[INFO] YOLO model loaded: " << yolo_names[0] << std::endl;

    // Liveness
    CHECK_HB(hbDNNInitializeFromFiles(&g_liveness_packed, &liveness_path, 1),
             "Failed to load liveness model");
    const char** liv_names; int liv_cnt = 0;
    CHECK_HB(hbDNNGetModelNameList(&liv_names, &liv_cnt, g_liveness_packed),
             "Failed to get liveness model names");
    CHECK_HB(hbDNNGetModelHandle(&g_liveness_handle, g_liveness_packed, liv_names[0]),
             "Failed to get liveness handle");
    std::cout << "[INFO] Liveness model loaded: " << liv_names[0] << std::endl;

    // Recognition (MobileNetV3)
    CHECK_HB(hbDNNInitializeFromFiles(&g_recog_packed, &recog_path, 1),
             "Failed to load recognition model");
    const char** rec_names; int rec_cnt = 0;
    CHECK_HB(hbDNNGetModelNameList(&rec_names, &rec_cnt, g_recog_packed),
             "Failed to get recognition model names");
    CHECK_HB(hbDNNGetModelHandle(&g_recog_handle, g_recog_packed, rec_names[0]),
             "Failed to get recognition handle");
    std::cout << "[INFO] Recognition model loaded: " << rec_names[0] << std::endl;

    return 0;
}

// ============================================================================
// Core detect logic (shared by file and buffer APIs)
// ============================================================================

static int detect_process(const cv::Mat& img,
                           FaceDetectionResult* results,
                           int max_faces, int* num_faces) {
    *num_faces = 0;

    // ── 1. YOLO input properties ──────────────────────────────────────────
    hbDNNTensorProperties inp_yolo;
    CHECK_HB(hbDNNGetInputTensorProperties(&inp_yolo, g_yolo_handle, 0),
             "YOLO get input props failed");
    int in_h = inp_yolo.validShape.dimensionSize[2];
    int in_w = inp_yolo.validShape.dimensionSize[3];

    // ── 2. Preprocess + NV12 ──────────────────────────────────────────────
    float x_scale, y_scale;
    int   x_shift, y_shift;
    cv::Mat preprocessed = preprocess_yolo(img, in_h, in_w,
                                            x_scale, y_scale, x_shift, y_shift);
    cv::Mat nv12_yolo = bgr2nv12(preprocessed);

    // ── 3. YOLO input tensor ──────────────────────────────────────────────
    hbDNNTensor yolo_input;
    yolo_input.properties = inp_yolo;
    hbSysAllocCachedMem(&yolo_input.sysMem[0], inp_yolo.alignedByteSize);
    memcpy(yolo_input.sysMem[0].virAddr, nv12_yolo.data, inp_yolo.alignedByteSize);
    hbSysFlushMem(&yolo_input.sysMem[0], HB_SYS_MEM_CACHE_CLEAN);

    // ── 4. YOLO output tensors ────────────────────────────────────────────
    int yolo_out_cnt = 0;
    hbDNNGetOutputCount(&yolo_out_cnt, g_yolo_handle);
    hbDNNTensor* yolo_outputs = new hbDNNTensor[yolo_out_cnt];
    for (int i = 0; i < yolo_out_cnt; i++) {
        hbDNNGetOutputTensorProperties(&yolo_outputs[i].properties, g_yolo_handle, i);
        hbSysAllocCachedMem(&yolo_outputs[i].sysMem[0],
                            yolo_outputs[i].properties.alignedByteSize);
    }

    // ── 5. YOLO inference ─────────────────────────────────────────────────
    hbDNNTaskHandle_t yolo_task = nullptr;
    hbDNNInferCtrlParam yolo_ctrl;
    HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&yolo_ctrl);
    CHECK_HB(hbDNNInfer(&yolo_task, &yolo_outputs, &yolo_input,
                         g_yolo_handle, &yolo_ctrl),
             "YOLO inference failed");
    CHECK_HB(hbDNNWaitTaskDone(yolo_task, 0), "YOLO wait task failed");

    // ── 6. YOLO post-process (DFL decode) ─────────────────────────────────
    std::vector<FaceLandmarkDet> dets;
    const int strides[3] = {8, 16, 32};
    for (int s = 0; s < 3; s++) {
        int box_idx = s * 2,     cls_idx = s * 2 + 1,  kpt_idx = s + 6;
        int grid_h  = in_h / strides[s], grid_w = in_w / strides[s];

        hbSysFlushMem(&yolo_outputs[box_idx].sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE);
        hbSysFlushMem(&yolo_outputs[cls_idx].sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE);
        hbSysFlushMem(&yolo_outputs[kpt_idx].sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE);

        float* box_ptr = reinterpret_cast<float*>(yolo_outputs[box_idx].sysMem[0].virAddr);
        float* cls_ptr = reinterpret_cast<float*>(yolo_outputs[cls_idx].sysMem[0].virAddr);
        float* kpt_ptr = reinterpret_cast<float*>(yolo_outputs[kpt_idx].sysMem[0].virAddr);

        for (int gh = 0; gh < grid_h; gh++) {
            for (int gw = 0; gw < grid_w; gw++) {
                int off  = gh * grid_w + gw;
                float conf = 1.0f / (1.0f + std::exp(-cls_ptr[off]));
                if (conf < YOLO_SCORE_THRESHOLD) continue;

                FaceLandmarkDet det;
                det.score = conf;

                // DFL bbox decode
                float ltrb[4] = {};
                for (int i = 0; i < 4; i++) {
                    float dfl[YOLO_REG], sm[YOLO_REG];
                    memcpy(dfl, box_ptr + off * (4 * YOLO_REG) + i * YOLO_REG,
                           YOLO_REG * sizeof(float));
                    softmax(dfl, sm, YOLO_REG);
                    for (int j = 0; j < YOLO_REG; j++) ltrb[i] += sm[j] * j;
                }
                float cx = (gw + 0.5f) * strides[s];
                float cy = (gh + 0.5f) * strides[s];
                float x1 = (cx - ltrb[0] * strides[s] - x_shift) / x_scale;
                float y1 = (cy - ltrb[1] * strides[s] - y_shift) / y_scale;
                float x2 = (cx + ltrb[2] * strides[s] - x_shift) / x_scale;
                float y2 = (cy + ltrb[3] * strides[s] - y_shift) / y_scale;
                det.bbox = cv::Rect2d(x1, y1, x2 - x1, y2 - y1);

                // Landmark decode
                det.landmarks.resize(YOLO_KPT_NUM);
                det.landmark_scores.resize(YOLO_KPT_NUM);
                for (int k = 0; k < YOLO_KPT_NUM; k++) {
                    float kx = (kpt_ptr[off * YOLO_KPT_NUM * 3 + k * 3 + 0] * 2.0f + gw)
                               * strides[s] / x_scale;
                    float ky = (kpt_ptr[off * YOLO_KPT_NUM * 3 + k * 3 + 1] * 2.0f + gh)
                               * strides[s] / y_scale;
                    float ks = 1.0f / (1.0f + std::exp(
                               -kpt_ptr[off * YOLO_KPT_NUM * 3 + k * 3 + 2]));
                    det.landmarks[k]       = {kx, ky};
                    det.landmark_scores[k] = ks;
                }
                dets.push_back(det);
            }
        }
    }

    // ── 7. NMS ────────────────────────────────────────────────────────────
    std::vector<cv::Rect2d> nms_boxes;
    std::vector<float>      nms_scores;
    for (auto& d : dets) { nms_boxes.push_back(d.bbox); nms_scores.push_back(d.score); }
    std::vector<int> nms_idx;
    if (!nms_boxes.empty())
        cv::dnn::NMSBoxes(nms_boxes, nms_scores,
                          YOLO_SCORE_THRESHOLD, YOLO_NMS_THRESHOLD, nms_idx);

    // ── 8. Per-face liveness ──────────────────────────────────────────────
    const char* liveness_labels[] = {"Paper Photo", "Real Face", "Screen Photo"};

    for (int idx : nms_idx) {
        if (*num_faces >= max_faces) break;
        FaceLandmarkDet& det = dets[idx];

        // Liveness ROI
        cv::Rect roi = get_liveness_roi(img.cols, img.rows,
                                        (cv::Rect)det.bbox, LIVENESS_SCALE);
        if (roi.width <= 0 || roi.height <= 0) continue;
        cv::Mat roi_resized;
        cv::resize(img(roi), roi_resized, cv::Size(LIVENESS_INPUT_W, LIVENESS_INPUT_H));
        cv::Mat nv12_liv = bgr2nv12(roi_resized);

        // Liveness input tensor
        hbDNNTensorProperties liv_inp_props;
        hbDNNGetInputTensorProperties(&liv_inp_props, g_liveness_handle, 0);
        hbDNNTensor liv_input;
        liv_input.properties = liv_inp_props;
        hbSysAllocCachedMem(&liv_input.sysMem[0], liv_inp_props.alignedByteSize);
        memcpy(liv_input.sysMem[0].virAddr, nv12_liv.data, liv_inp_props.alignedByteSize);
        hbSysFlushMem(&liv_input.sysMem[0], HB_SYS_MEM_CACHE_CLEAN);

        // Liveness output tensor
        hbDNNTensorProperties liv_out_props;
        hbDNNGetOutputTensorProperties(&liv_out_props, g_liveness_handle, 0);
        hbDNNTensor liv_output;
        liv_output.properties = liv_out_props;
        hbSysAllocCachedMem(&liv_output.sysMem[0], liv_out_props.alignedByteSize);

        // Liveness inference
        hbDNNTaskHandle_t liv_task = nullptr;
        hbDNNInferCtrlParam liv_ctrl;
        HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&liv_ctrl);
        hbDNNTensor* liv_out_ptr = &liv_output;
        hbDNNInfer(&liv_task, &liv_out_ptr, &liv_input, g_liveness_handle, &liv_ctrl);
        hbDNNWaitTaskDone(liv_task, 0);

        // Liveness post-process
        hbSysFlushMem(&liv_output.sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE);
        float* raw = reinterpret_cast<float*>(liv_output.sysMem[0].virAddr);
        float probs[3];
        softmax(raw, probs, 3);
        int max_idx = (int)(std::max_element(probs, probs + 3) - probs);

        // Fill result
        FaceDetectionResult& r = results[*num_faces];
        r.is_real     = (max_idx == 1);
        r.confidence  = probs[max_idx];
        strncpy(r.label, liveness_labels[max_idx], sizeof(r.label) - 1);
        r.bbox[0] = (float)det.bbox.x;
        r.bbox[1] = (float)det.bbox.y;
        r.bbox[2] = (float)det.bbox.width;
        r.bbox[3] = (float)det.bbox.height;
        for (int k = 0; k < YOLO_KPT_NUM; k++) {
            r.landmarks[k][0]    = det.landmarks[k].x;
            r.landmarks[k][1]    = det.landmarks[k].y;
            r.landmark_scores[k] = det.landmark_scores[k];
        }
        (*num_faces)++;

        hbSysFreeMem(&liv_input.sysMem[0]);
        hbSysFreeMem(&liv_output.sysMem[0]);
        hbDNNReleaseTask(liv_task);
    }

    // ── 9. YOLO cleanup ───────────────────────────────────────────────────
    hbSysFreeMem(&yolo_input.sysMem[0]);
    for (int i = 0; i < yolo_out_cnt; i++)
        hbSysFreeMem(&yolo_outputs[i].sysMem[0]);
    delete[] yolo_outputs;
    hbDNNReleaseTask(yolo_task);

    return 0;
}

// ============================================================================
// Public: detect_faces_liveness  (file API)
// ============================================================================

int detect_faces_liveness(const char* image_path,
                           FaceDetectionResult* results,
                           int max_faces, int* num_faces) {
    *num_faces = 0;
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "[ERROR] Cannot load image: " << image_path << std::endl;
        return -1;
    }
    return detect_process(img, results, max_faces, num_faces);
}

// ============================================================================
// Public: detect_faces_liveness_from_buffer  (video stream API)
// ============================================================================

int detect_faces_liveness_from_buffer(const unsigned char* image_data,
                                       int image_width, int image_height,
                                       FaceDetectionResult* results,
                                       int max_faces, int* num_faces) {
    *num_faces = 0;
    cv::Mat img(image_height, image_width, CV_8UC3,
                const_cast<unsigned char*>(image_data));
    if (img.empty()) {
        std::cerr << "[ERROR] Failed to create Mat from buffer." << std::endl;
        return -1;
    }
    return detect_process(img, results, max_faces, num_faces);
}

// ============================================================================
// Public: get_face_embedding
// ============================================================================

int get_face_embedding(const unsigned char* image_data,
                        int image_width, int image_height,
                        const float* landmarks_10,
                        float* embedding_out,
                        int embedding_size) {
    if (embedding_size != FACE_EMBEDDING_SIZE) {
        std::cerr << "[ERROR] embedding_size must be " << FACE_EMBEDDING_SIZE << std::endl;
        return -1;
    }
    if (!g_recog_handle) {
        std::cerr << "[ERROR] Recognition model not initialized." << std::endl;
        return -1;
    }

    // Wrap raw BGR buffer (no copy)
    cv::Mat img(image_height, image_width, CV_8UC3,
                const_cast<unsigned char*>(image_data));

    // Reconstruct 5 keypoints from flat array
    std::vector<cv::Point2f> kpts(5);
    for (int i = 0; i < 5; i++)
        kpts[i] = {landmarks_10[i * 2], landmarks_10[i * 2 + 1]};

    // ArcFace alignment → 112×112
    cv::Mat aligned = align_face_112(img, kpts);
    if (aligned.empty()) {
        std::cerr << "[ERROR] Face alignment failed." << std::endl;
        return -1;
    }

    // BGR → NV12
    cv::Mat nv12 = bgr2nv12(aligned);

    // Input tensor
    hbDNNTensorProperties inp_props;
    hbDNNGetInputTensorProperties(&inp_props, g_recog_handle, 0);
    hbDNNTensor recog_input;
    recog_input.properties = inp_props;
    hbSysAllocCachedMem(&recog_input.sysMem[0], inp_props.alignedByteSize);
    size_t nv12_bytes = (size_t)nv12.rows * nv12.cols; // 112*168 bytes
    memcpy(recog_input.sysMem[0].virAddr, nv12.data, nv12_bytes);
    hbSysFlushMem(&recog_input.sysMem[0], HB_SYS_MEM_CACHE_CLEAN);

    // Output tensor
    hbDNNTensorProperties out_props;
    hbDNNGetOutputTensorProperties(&out_props, g_recog_handle, 0);
    hbDNNTensor recog_output;
    recog_output.properties = out_props;
    hbSysAllocCachedMem(&recog_output.sysMem[0], out_props.alignedByteSize);

    // Inference
    hbDNNTaskHandle_t task = nullptr;
    hbDNNInferCtrlParam ctrl;
    HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&ctrl);
    hbDNNTensor* out_ptr = &recog_output;
    int ret = hbDNNInfer(&task, &out_ptr, &recog_input, g_recog_handle, &ctrl);
    if (ret != 0) {
        std::cerr << "[ERROR] Recognition inference failed: " << ret << std::endl;
        hbSysFreeMem(&recog_input.sysMem[0]);
        hbSysFreeMem(&recog_output.sysMem[0]);
        return ret;
    }
    hbDNNWaitTaskDone(task, 0);

    // Copy embedding
    hbSysFlushMem(&recog_output.sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE);
    memcpy(embedding_out,
           reinterpret_cast<float*>(recog_output.sysMem[0].virAddr),
           FACE_EMBEDDING_SIZE * sizeof(float));

    hbSysFreeMem(&recog_input.sysMem[0]);
    hbSysFreeMem(&recog_output.sysMem[0]);
    hbDNNReleaseTask(task);
    return 0;
}

// ============================================================================
// release_models
// ============================================================================

void release_models() {
    if (g_yolo_packed)     { hbDNNRelease(g_yolo_packed);     g_yolo_packed     = nullptr; }
    if (g_liveness_packed) { hbDNNRelease(g_liveness_packed); g_liveness_packed = nullptr; }
    if (g_recog_packed)    { hbDNNRelease(g_recog_packed);    g_recog_packed     = nullptr; }
    g_yolo_handle = g_liveness_handle = g_recog_handle = nullptr;
    std::cout << "[INFO] All models released." << std::endl;
}
