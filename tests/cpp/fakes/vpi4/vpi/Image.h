#ifndef TEST_FAKE_VPI4_IMAGE_H_
#define TEST_FAKE_VPI4_IMAGE_H_

#include <cstdint>

using VPIByte = std::uint8_t;

struct VPIImagePlanePitchLinear {
    int width = 0;
    int height = 0;
    int pitchBytes = 0;
    std::int64_t offsetBytes = 0;
    VPIByte* pBase = nullptr;
};

#endif  // TEST_FAKE_VPI4_IMAGE_H_
