/**
 * @file vpi_image_copy.h
 * @brief Safe, synchronous copies from VPI-owned images to OpenCV host images.
 */

#ifndef STEREO_3D_PIPELINE_VPI_IMAGE_COPY_H_
#define STEREO_3D_PIPELINE_VPI_IMAGE_COPY_H_

#include <opencv2/core.hpp>
#include <vpi/Image.h>

namespace stereo3d {

/**
 * Copy a Gray8 host-pitch-linear VPI image while keeping its read lock alive.
 * The destination is released on every failure, including unlock failure.
 */
bool copyVpiGray8ToHost(VPIImage image,
                        cv::Mat& output,
                        const char* operation);

/**
 * Copy a BGR8 CUDA-pitch-linear VPI image synchronously to host memory while
 * keeping its read lock alive. The destination is released on every failure.
 */
bool copyVpiBgr8ToHost(VPIImage image,
                       cv::Mat& output,
                       const char* operation);

}  // namespace stereo3d

#endif  // STEREO_3D_PIPELINE_VPI_IMAGE_COPY_H_
