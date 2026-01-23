#include "face_liveness.h"
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

// Function to draw detection results on the image
void draw_results(const char* image_path, const std::vector<FaceDetectionResult>& results, const char* save_path) {
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "[ERROR] Failed to load image for drawing: " << image_path << std::endl;
        return;
    }

    for (const auto& det : results) {
        // Draw bounding box
        cv::Rect bbox(det.bbox[0], det.bbox[1], det.bbox[2], det.bbox[3]);
        cv::Scalar color = (det.is_real) ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255); // Green for real, Red for fake
        cv::rectangle(img, bbox, color, 2);

        // Draw label (Liveness + Confidence)
        std::string label = std::string(det.label) + " (" + std::to_string(det.confidence).substr(0, 4) + ")";
        cv::putText(img, label, cv::Point(bbox.x, bbox.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);

        // Draw landmarks
        for (int j = 0; j < 5; ++j) {
            if (det.landmark_scores[j] > 0.5) { // Draw only confident landmarks
                cv::Point2f landmark(det.landmarks[j][0], det.landmarks[j][1]);
                cv::circle(img, landmark, 3, cv::Scalar(255, 255, 0), -1); // Cyan dots
            }
        }
    }

    // Save the result image
    cv::imwrite(save_path, img);
    std::cout << "[INFO] Result image saved to: " << save_path << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <image_path> <output_path>" << std::endl;
        return -1;
    }

    const char* yolo_model_path = "/home/sunrise/Code/weights/yolov8n-face.bin";
    const char* liveness_model_path = "/home/sunrise/Code/weights/anti-face.bin";
    const char* image_path = argv[1];
    const char* save_path = argv[2];

    // 1. Initialize models
    if (initialize_models(yolo_model_path, liveness_model_path) != 0) {
        std::cerr << "[ERROR] Failed to initialize models!" << std::endl;
        return -1;
    }
    std::cout << "[INFO] Models initialized successfully." << std::endl;

    // 2. Prepare for detection
    const int max_faces = 10;
    std::vector<FaceDetectionResult> results(max_faces);
    int num_faces = 0;

    // 3. Run detection
    std::cout << "[INFO] Starting face detection and liveness check..." << std::endl;
    int ret = detect_faces_liveness(image_path, results.data(), max_faces, &num_faces);
    if (ret != 0) {
        std::cerr << "[ERROR] Detection process failed!" << std::endl;
        release_models();
        return ret;
    }

    // 4. Print results
    std::cout << "[INFO] Detected " << num_faces << " face(s):" << std::endl;
    results.resize(num_faces); // Resize vector to actual number of faces
    for (int i = 0; i < num_faces; ++i) {
        const auto& det = results[i];
        std::cout << "  - Face " << i + 1 << ": "
                  << "Label=" << det.label
                  << ", Liveness Confidence=" << det.confidence
                  << ", BBox=[" << det.bbox[0] << ", " << det.bbox[1] << ", "
                  << det.bbox[2] << ", " << det.bbox[3] << "]" << std::endl;
        
        std::cout << "    Landmarks:" << std::endl;
        std::cout << "    [[";
        for (int j = 0; j < 5; ++j) {
            std::cout << det.landmarks[j][0] << " " << det.landmarks[j][1];
            if (j < 4) {
                std::cout << "] ["; // Add separator between points
            }
        }
        std::cout << "]]" << std::endl;
    }

    // 5. Draw results and save image
    if (num_faces > 0) {
        draw_results(image_path, results, save_path);
    } else {
        std::cout << "[INFO] No faces detected, nothing to draw." << std::endl;
    }

    // 6. Clean up
    release_models();
    std::cout << "[INFO] Resources released. Execution finished." << std::endl;

    return 0;
}
// Example command:
// ./main /home/sunrise/Code/Anti-Face-Recognition/images/face7.jpg output.jpg


