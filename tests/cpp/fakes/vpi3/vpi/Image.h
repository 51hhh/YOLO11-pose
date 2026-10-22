#ifndef TEST_FAKE_VPI3_IMAGE_H_
#define TEST_FAKE_VPI3_IMAGE_H_

struct VPIImagePlanePitchLinear {
    int width = 0;
    int height = 0;
    int pitchBytes = 0;
    void* data = nullptr;
};

#endif  // TEST_FAKE_VPI3_IMAGE_H_
