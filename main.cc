#include "face_liveness.h"
#include <iostream>
#include <iomanip>
#include <opencv2/opencv.hpp>

// ============================================================
// Default model paths — adjust if weights live elsewhere
// ============================================================
static const char* YOLO_PATH =
    "/home/sunrise/Code/RDK_X5_Anti-Face-Recognition_V2/weights/yolov8n-face.bin";
static const char* LIVENESS_PATH =
    "/home/sunrise/Code/RDK_X5_Anti-Face-Recognition_V2/weights/anti-face.bin";
static const char* RECOG_PATH =
    "/home/sunrise/Code/RDK_X5_Anti-Face-Recognition_V2/weights/"
    "mobilenetv3_small_mcp_bayese_112x112_nv12.bin";

static const float THRESHOLD = 0.35f;

// ── helpers ──────────────────────────────────────────────────────────────────

static void draw_result(cv::Mat& img, const FaceDetectionResult& r) {
    cv::Rect   bbox(r.bbox[0], r.bbox[1], r.bbox[2], r.bbox[3]);
    cv::Scalar color = r.is_real ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
    cv::rectangle(img, bbox, color, 2);
    std::string txt = std::string(r.label) + " " +
                      std::to_string(r.confidence).substr(0, 4);
    cv::putText(img, txt, {bbox.x, bbox.y - 8},
                cv::FONT_HERSHEY_SIMPLEX, 0.55, color, 2);
    for (int k = 0; k < 5; k++) {
        if (r.landmark_scores[k] > 0.5f)
            cv::circle(img, {(int)r.landmarks[k][0], (int)r.landmarks[k][1]},
                       3, cv::Scalar(255, 255, 0), -1);
    }
}

static bool extract_embedding(const char* image_path,
                               const FaceDetectionResult& face,
                               float* emb) {
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) return false;

    float lm10[10];
    for (int i = 0; i < 5; i++) {
        lm10[i * 2]     = face.landmarks[i][0];
        lm10[i * 2 + 1] = face.landmarks[i][1];
    }
    return get_face_embedding(img.data, img.cols, img.rows,
                               lm10, emb, FACE_EMBEDDING_SIZE) == 0;
}

// ── main ─────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <image1> <image2> [output1.jpg] [output2.jpg]" << std::endl;
        return -1;
    }
    const char* img1_path = argv[1];
    const char* img2_path = argv[2];
    const char* out1_path = (argc > 3) ? argv[3] : nullptr;
    const char* out2_path = (argc > 4) ? argv[4] : nullptr;

    std::cout << "\n================================================" << std::endl;
    std::cout << "  Anti-Face-Recognition V2  (full C++ pipeline)" << std::endl;
    std::cout << "================================================\n" << std::endl;

    // ── Step 1: Init models ───────────────────────────────────────────────
    std::cout << "[1/5] Loading models..." << std::endl;
    if (initialize_models(YOLO_PATH, LIVENESS_PATH, RECOG_PATH) != 0) {
        std::cerr << "[ERROR] Model init failed." << std::endl;
        return -1;
    }

    // ── Step 2: Detect faces in image 1 ──────────────────────────────────
    std::cout << "\n[2/5] Detecting in image1: " << img1_path << std::endl;
    const int MAX_FACES = 10;
    FaceDetectionResult results1[MAX_FACES];
    int n1 = 0;
    if (detect_faces_liveness(img1_path, results1, MAX_FACES, &n1) != 0 || n1 == 0) {
        std::cerr << "[ERROR] No face detected in image1." << std::endl;
        release_models(); return -1;
    }
    // Find the first real face; if none, use the best-confidence face
    int face1_idx = -1;
    for (int i = 0; i < n1; i++) if (results1[i].is_real) { face1_idx = i; break; }
    if (face1_idx < 0) {
        std::cout << "[WARN] No real face in image1, using highest-confidence face." << std::endl;
        face1_idx = 0;
        for (int i = 1; i < n1; i++)
            if (results1[i].confidence > results1[face1_idx].confidence)
                face1_idx = i;
    }
    std::cout << "  Detected " << n1 << " face(s). Selected face " << face1_idx + 1
              << ": " << results1[face1_idx].label
              << " (conf=" << std::fixed << std::setprecision(3)
              << results1[face1_idx].confidence << ")" << std::endl;

    // ── Step 3: Detect faces in image 2 ──────────────────────────────────
    std::cout << "\n[3/5] Detecting in image2: " << img2_path << std::endl;
    FaceDetectionResult results2[MAX_FACES];
    int n2 = 0;
    if (detect_faces_liveness(img2_path, results2, MAX_FACES, &n2) != 0 || n2 == 0) {
        std::cerr << "[ERROR] No face detected in image2." << std::endl;
        release_models(); return -1;
    }
    int face2_idx = -1;
    for (int i = 0; i < n2; i++) if (results2[i].is_real) { face2_idx = i; break; }
    if (face2_idx < 0) {
        std::cout << "[WARN] No real face in image2, using highest-confidence face." << std::endl;
        face2_idx = 0;
        for (int i = 1; i < n2; i++)
            if (results2[i].confidence > results2[face2_idx].confidence)
                face2_idx = i;
    }
    std::cout << "  Detected " << n2 << " face(s). Selected face " << face2_idx + 1
              << ": " << results2[face2_idx].label
              << " (conf=" << std::fixed << std::setprecision(3)
              << results2[face2_idx].confidence << ")" << std::endl;

    // ── Step 4: Extract embeddings ────────────────────────────────────────
    std::cout << "\n[4/5] Extracting embeddings..." << std::endl;
    float emb1[FACE_EMBEDDING_SIZE], emb2[FACE_EMBEDDING_SIZE];
    if (!extract_embedding(img1_path, results1[face1_idx], emb1)) {
        std::cerr << "[ERROR] Embedding extraction failed for image1." << std::endl;
        release_models(); return -1;
    }
    if (!extract_embedding(img2_path, results2[face2_idx], emb2)) {
        std::cerr << "[ERROR] Embedding extraction failed for image2." << std::endl;
        release_models(); return -1;
    }
    std::cout << "  Embeddings extracted (" << FACE_EMBEDDING_SIZE << "-dim)." << std::endl;

    // ── Step 5: Compare ───────────────────────────────────────────────────
    std::cout << "\n[5/5] Comparing..." << std::endl;

    // Cosine similarity (inline — no helper needed here)
    float dot = 0, n_a = 0, n_b = 0;
    for (int i = 0; i < FACE_EMBEDDING_SIZE; i++) {
        dot += emb1[i] * emb2[i];
        n_a += emb1[i] * emb1[i];
        n_b += emb2[i] * emb2[i];
    }
    float similarity = dot / (std::sqrt(n_a) * std::sqrt(n_b) + 1e-8f);
    bool  is_same    = similarity >= THRESHOLD;

    std::cout << "\n================================================" << std::endl;
    std::cout << "  Image 1   : " << img1_path << std::endl;
    std::cout << "             face=" << results1[face1_idx].label
              << (results1[face1_idx].is_real ? " [real]" : " [fake]") << std::endl;
    std::cout << "  Image 2   : " << img2_path << std::endl;
    std::cout << "             face=" << results2[face2_idx].label
              << (results2[face2_idx].is_real ? " [real]" : " [fake]") << std::endl;
    std::cout << "  Similarity: " << std::fixed << std::setprecision(4)
              << similarity << std::endl;
    std::cout << "  Threshold : " << THRESHOLD << std::endl;
    std::cout << "  Result    : "
              << (is_same ? "✓ Same person" : "✗ Different person") << std::endl;
    std::cout << "================================================\n" << std::endl;

    // ── Optional: Save annotated images ──────────────────────────────────
    if (out1_path) {
        cv::Mat img1 = cv::imread(img1_path);
        for (int i = 0; i < n1; i++) draw_result(img1, results1[i]);
        cv::imwrite(out1_path, img1);
        std::cout << "[INFO] Annotated image1 saved: " << out1_path << std::endl;
    }
    if (out2_path) {
        cv::Mat img2 = cv::imread(img2_path);
        for (int i = 0; i < n2; i++) draw_result(img2, results2[i]);
        cv::imwrite(out2_path, img2);
        std::cout << "[INFO] Annotated image2 saved: " << out2_path << std::endl;
    }

    release_models();
    return 0;
}
