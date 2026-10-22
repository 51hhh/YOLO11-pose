#include "volleyball/adapters/tensorrt/postprocess.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

namespace trt = volleyball::adapters::tensorrt;

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

bool near(float actual, float expected, float tolerance = 1.0e-3F) {
    return std::fabs(actual - expected) <= tolerance;
}

std::vector<float> transposedPreNmsOutput() {
    constexpr int channels = 6;
    constexpr int candidates = 7;
    std::vector<float> output(channels * candidates, 0.0F);
    const auto set = [&](int channel, int candidate, float value) {
        output[channel * candidates + candidate] = value;
    };

    set(0, 0, 20.0F); set(1, 0, 20.0F);
    set(2, 0, 10.0F); set(3, 0, 10.0F); set(4, 0, 0.9F);
    set(0, 1, 20.5F); set(1, 1, 20.5F);
    set(2, 1, 10.0F); set(3, 1, 10.0F);
    set(4, 1, 0.8F); set(5, 1, 0.81F);
    set(0, 2, 80.0F); set(1, 2, 40.0F);
    set(2, 2, 8.0F); set(3, 2, 8.0F); set(4, 2, 0.7F);
    return output;
}

std::vector<float> rowMajorFromTransposed(const std::vector<float>& input) {
    constexpr int output_channels = 5;
    constexpr int candidates = 7;
    std::vector<float> output(output_channels * candidates, 0.0F);
    for (int candidate = 0; candidate < candidates; ++candidate) {
        for (int channel = 0; channel < output_channels; ++channel) {
            output[candidate * output_channels + channel] =
                input[channel * candidates + candidate];
        }
    }
    return output;
}

void testPostNmsOutput() {
    const std::vector<float> output = {
        0.0F, 0.0F, 10.0F, 20.0F, 0.8F, 3.0F,
        1.0F, 2.0F, 3.0F, 4.0F, 0.1F, 2.0F,
    };
    const auto result = trt::decodeSingleOutput(
        {output.data(), output.size(), 2, 6, 0,
         trt::SingleOutputLayout::post_nms_xyxy_rows},
        2.0F,
        3.0F,
        {0.5F, 0.4F});
    check(result.success && result.detections.size() == 1,
          "post-NMS output should filter confidence without running NMS again");
    if (result.detections.size() == 1) {
        const auto& detection = result.detections.front();
        check(near(detection.cx, 10.0F) && near(detection.cy, 30.0F) &&
                  near(detection.width, 20.0F) &&
                  near(detection.height, 60.0F) &&
                  detection.class_id == 3,
              "post-NMS xyxy output should scale into center-size detections");
    }
}

void testPreNmsLayoutsAndNms() {
    const auto transposed = transposedPreNmsOutput();
    const auto column_result = trt::decodeSingleOutput(
        {transposed.data(), transposed.size(), 6, 7, 2,
         trt::SingleOutputLayout::pre_nms_channels_first},
        1.0F,
        1.0F,
        {0.5F, 0.4F});
    check(column_result.success && column_result.detections.size() == 2,
          "column-major pre-NMS output should decode and suppress overlap");
    if (column_result.detections.size() == 2) {
        check(near(column_result.detections[0].confidence, 0.9F) &&
                  near(column_result.detections[1].cx, 80.0F),
              "NMS should retain the highest-confidence overlap and distant box");
    }

    const auto row_major = rowMajorFromTransposed(transposed);
    const auto row_result = trt::decodeSingleOutput(
        {row_major.data(), row_major.size(), 7, 5, 1,
         trt::SingleOutputLayout::pre_nms_rows},
        1.0F,
        1.0F,
        {0.5F, 0.4F});
    check(row_result.success && row_result.detections.size() == 2,
          "row-major pre-NMS output should match column-major decoding");
}

void testRejectsInvalidSingleOutputView() {
    const auto output = transposedPreNmsOutput();
    const auto truncated = trt::decodeSingleOutput(
        {output.data(), output.size() - 1, 6, 7, 2,
         trt::SingleOutputLayout::pre_nms_channels_first},
        1.0F,
        1.0F,
        {0.5F, 0.4F});
    check(!truncated.success && truncated.detections.empty(),
          "single-output decoder must reject a truncated host buffer");

    const auto invalid_scale = trt::decodeSingleOutput(
        {output.data(), output.size(), 6, 7, 2,
         trt::SingleOutputLayout::pre_nms_channels_first},
        0.0F,
        1.0F,
        {0.5F, 0.4F});
    check(!invalid_scale.success,
          "single-output decoder must reject invalid coordinate scaling");

    auto nonfinite = output;
    nonfinite[0] = std::numeric_limits<float>::quiet_NaN();
    const auto invalid_detection = trt::decodeSingleOutput(
        {nonfinite.data(), nonfinite.size(), 6, 7, 2,
         trt::SingleOutputLayout::pre_nms_channels_first},
        1.0F,
        1.0F,
        {0.5F, 0.4F});
    check(!invalid_detection.success && invalid_detection.detections.empty(),
          "decoder must reject non-finite accepted detection geometry");

    const auto contradictory = trt::decodeSingleOutput(
        {output.data(), output.size(), 6, 7, 1,
         trt::SingleOutputLayout::pre_nms_channels_first},
        1.0F,
        1.0F,
        {0.5F, 0.4F});
    check(!contradictory.success && contradictory.detections.empty(),
          "decoder must reject metadata that contradicts the validated shape");
}

void testDflDecodeAndBounds() {
    const std::vector<float> classification = {10.0F};
    const std::vector<float> boxes = {
        -10.0F, 10.0F,
        -10.0F, 10.0F,
        -10.0F, 10.0F,
        -10.0F, 10.0F,
    };
    const trt::DflScaleView scale{
        classification.data(), classification.size(),
        boxes.data(), boxes.size(),
        1, 1, 1, 8,
    };
    const auto result = trt::decodeDflOutputs(
        {scale}, 2, 2.0F, 3.0F, {0.5F, 0.4F});
    check(result.success && result.detections.size() == 1,
          "DFL output should decode one confident grid cell");
    if (result.detections.size() == 1) {
        const auto& detection = result.detections.front();
        check(near(detection.cx, 8.0F) && near(detection.cy, 12.0F) &&
                  near(detection.width, 32.0F, 0.01F) &&
                  near(detection.height, 48.0F, 0.01F),
              "DFL softmax distances should produce scaled center-size output");
    }

    auto truncated_scale = scale;
    truncated_scale.box_elements -= 1;
    const auto truncated = trt::decodeDflOutputs(
        {truncated_scale}, 2, 1.0F, 1.0F, {0.5F, 0.4F});
    check(!truncated.success && truncated.detections.empty(),
          "DFL decoder must reject a truncated regression buffer");

    const auto empty = trt::decodeDflOutputs(
        {}, 2, 1.0F, 1.0F, {0.5F, 0.4F});
    check(!empty.success,
          "DFL decoder must reject a missing scale set");
}

}  // namespace

int main() {
    testPostNmsOutput();
    testPreNmsLayoutsAndNms();
    testRejectsInvalidSingleOutputView();
    testDflDecodeAndBounds();
    if (failures != 0) {
        std::cerr << failures << " check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "TensorRT postprocess tests passed\n";
    return EXIT_SUCCESS;
}
