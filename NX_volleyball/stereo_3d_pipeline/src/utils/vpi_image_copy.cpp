#include "vpi_image_copy.h"

#include "logger.h"

#include <cuda_runtime.h>

#include <cstddef>

namespace stereo3d {
namespace {

const char* operationName(const char* operation) {
    return operation ? operation : "VPI image copy";
}

bool finishCopy(VPIImage image,
                cv::Mat& output,
                bool copy_ok,
                const char* operation) {
    const VPIStatus unlock_status = vpiImageUnlock(image);
    if (unlock_status != VPI_SUCCESS) {
        LOG_ERROR("%s: VPI image unlock failed: status=%d",
                  operationName(operation), (int)unlock_status);
        copy_ok = false;
    }
    if (!copy_ok || output.empty()) {
        output.release();
        return false;
    }
    return true;
}

}  // namespace

bool copyVpiGray8ToHost(VPIImage image,
                        cv::Mat& output,
                        const char* operation) {
    output.release();
    if (!image) {
        LOG_ERROR("%s: cannot copy a null VPI image", operationName(operation));
        return false;
    }

    VPIImageData data{};
    const VPIStatus lock_status = vpiImageLockData(
        image, VPI_LOCK_READ, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &data);
    if (lock_status != VPI_SUCCESS) {
        LOG_ERROR("%s: VPI host read lock failed: status=%d",
                  operationName(operation), (int)lock_status);
        return false;
    }

    const auto& plane = data.buffer.pitch.planes[0];
    bool copy_ok = plane.width > 0 && plane.height > 0 &&
                   plane.pitchBytes >= plane.width && plane.data;
    if (!copy_ok) {
        LOG_ERROR("%s: invalid Gray8 VPI view: width=%d height=%d pitch=%d ptr=%p",
                  operationName(operation), plane.width, plane.height,
                  plane.pitchBytes, plane.data);
    } else {
        try {
            const cv::Mat view(plane.height, plane.width, CV_8UC1,
                               plane.data, plane.pitchBytes);
            view.copyTo(output);
        } catch (const cv::Exception& error) {
            LOG_ERROR("%s: OpenCV Gray8 copy failed: %s",
                      operationName(operation), error.what());
            copy_ok = false;
        }
    }
    return finishCopy(image, output, copy_ok, operation);
}

bool copyVpiBgr8ToHost(VPIImage image,
                       cv::Mat& output,
                       const char* operation) {
    output.release();
    if (!image) {
        LOG_ERROR("%s: cannot copy a null VPI image", operationName(operation));
        return false;
    }

    VPIImageData data{};
    const VPIStatus lock_status = vpiImageLockData(
        image, VPI_LOCK_READ, VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR, &data);
    if (lock_status != VPI_SUCCESS) {
        LOG_ERROR("%s: VPI CUDA read lock failed: status=%d",
                  operationName(operation), (int)lock_status);
        return false;
    }

    const auto& plane = data.buffer.pitch.planes[0];
    int pixel_width = plane.width;
    std::size_t row_bytes = pixel_width > 0
        ? static_cast<std::size_t>(pixel_width) * 3U
        : 0U;
    bool copy_ok = pixel_width > 0 && plane.height > 0 &&
                   plane.pitchBytes > 0 && plane.data;

    // VPI versions have reported packed-plane width either as pixels or bytes.
    // Preserve both representations while still rejecting an impossible pitch.
    if (copy_ok && row_bytes > static_cast<std::size_t>(plane.pitchBytes)) {
        if (plane.width % 3 == 0 &&
            static_cast<std::size_t>(plane.width) <=
                static_cast<std::size_t>(plane.pitchBytes)) {
            pixel_width = plane.width / 3;
            row_bytes = static_cast<std::size_t>(plane.width);
        } else {
            copy_ok = false;
        }
    }
    if (!copy_ok) {
        LOG_ERROR("%s: invalid BGR8 VPI view: width=%d height=%d pitch=%d ptr=%p",
                  operationName(operation), plane.width, plane.height,
                  plane.pitchBytes, plane.data);
    } else {
        try {
            output.create(plane.height, pixel_width, CV_8UC3);
        } catch (const cv::Exception& error) {
            LOG_ERROR("%s: OpenCV BGR8 allocation failed: %s",
                      operationName(operation), error.what());
            copy_ok = false;
        }
    }

    if (copy_ok) {
        const cudaError_t copy_status = cudaMemcpy2D(
            output.data, output.step[0], plane.data,
            static_cast<std::size_t>(plane.pitchBytes), row_bytes,
            static_cast<std::size_t>(plane.height), cudaMemcpyDeviceToHost);
        if (copy_status != cudaSuccess) {
            LOG_ERROR("%s: CUDA BGR8 copy failed: %s",
                      operationName(operation), cudaGetErrorString(copy_status));
            copy_ok = false;
        }
    }
    return finishCopy(image, output, copy_ok, operation);
}

}  // namespace stereo3d
