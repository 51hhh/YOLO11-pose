#ifndef STEREO_3D_PIPELINE_BAYER_TO_BGR_H_
#define STEREO_3D_PIPELINE_BAYER_TO_BGR_H_

#include <cuda_runtime_api.h>

extern "C" void launchBayerToBGR8(
    const unsigned char* bayer,
    unsigned char* bgr,
    int width,
    int height,
    int bayer_pitch,
    int bgr_pitch,
    cudaStream_t stream);

#endif  // STEREO_3D_PIPELINE_BAYER_TO_BGR_H_
