#include "aruco_board.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>

ArucoImageResult detectBoardTags(
    const std::string& imagePath)
{
    ArucoImageResult result;

    // 1. 读取灰度图
    cv::Mat gray = cv::imread(
        imagePath,
        cv::IMREAD_GRAYSCALE
    );

    if (gray.empty()) {
        return result;
    }

    result.read_ok = true;
    result.image_size = gray.size();

    // 2. 使用 FAST-Calib2 相同的字典
    cv::aruco::Dictionary dictionary =
        cv::aruco::getPredefinedDictionary(
            cv::aruco::DICT_6X6_250
        );

    // 3. 设置角点亚像素优化
    cv::aruco::DetectorParameters params;

    params.cornerRefinementMethod =
        cv::aruco::CORNER_REFINE_SUBPIX;

    // 4. 创建检测器
    cv::aruco::ArucoDetector detector(
        dictionary,
        params
    );

    // 5. 检测结果容器
    std::vector<int> ids;

    std::vector<std::vector<cv::Point2f>> corners;

    detector.detectMarkers(
        gray,
        corners,
        ids
    );

    // 6. 按 Tag ID 保存角点
    for (size_t i = 0; i < ids.size(); ++i) {

        int id = ids[i];

        // 只接受标定板上的四个 Tag
        if (id < 1 || id > 4) {
            continue;
        }

        if (corners[i].size() != 4) {
            continue;
        }

        result.tags[id] = corners[i];
    }

    return result;
}

std::vector<cv::Point3f> boardMarkerCorners(int id)
{
    float cx = 0.0f;
    float cy = 0.0f;

    // 四个 Tag 在标定板上的中心坐标
    switch (id) {

    case 1:
        cx = -550.0f;
        cy = 350.0f;
        break;

    case 2:
        cx = 550.0f;
        cy = 350.0f;
        break;

    case 3:
        cx = -550.0f;
        cy = -350.0f;
        break;

    case 4:
        cx = 550.0f;
        cy = -350.0f;
        break;

    default:
        return {};
    }

    // Tag 边长 200 mm，半边长 100 mm
    constexpr float half = 100.0f;

    // 顺序：左上、右上、右下、左下
    return {
        {cx - half, cy + half, 0.0f},
        {cx + half, cy + half, 0.0f},
        {cx + half, cy - half, 0.0f},
        {cx - half, cy - half, 0.0f}
    };
}