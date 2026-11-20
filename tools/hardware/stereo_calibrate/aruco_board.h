#pragma once

#include <opencv2/core.hpp>

#include <map>
#include <string>
#include <vector>

// 单张图像的 ArUco 检测结果
struct ArucoImageResult {
    bool read_ok = false;

    cv::Size image_size;

    // key：Tag ID
    // value：该 Tag 的四个二维角点
    std::map<int, std::vector<cv::Point2f>> tags;
};

// 检测一张图像中的 ArUco
ArucoImageResult detectBoardTags(
    const std::string& imagePath
);

// 根据 Tag ID 返回四个三维角点，单位 mm
std::vector<cv::Point3f> boardMarkerCorners(
    int id
);