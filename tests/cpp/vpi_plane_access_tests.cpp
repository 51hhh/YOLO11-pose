#include "utils/vpi_plane_access.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>

namespace {

void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

}  // namespace

int main() {
    std::uint8_t storage[16]{};
    VPIImagePlanePitchLinear plane{};

    stereo3d::setVpiPlaneData(plane, storage);
    check(stereo3d::vpiPlaneData(plane) == storage,
          "set/get must preserve the plane base address");

#if NV_VPI_VERSION_MAJOR >= 4
    plane.offsetBytes = 5;
    check(stereo3d::vpiPlaneData(plane) == storage + 5,
          "VPI4 getter must apply offsetBytes");

    plane.offsetBytes = -1;
    check(stereo3d::vpiPlaneData(plane) == nullptr,
          "negative VPI4 offsets must be rejected");
    plane.offsetBytes = 2;
#endif

    const VPIImagePlanePitchLinear& const_plane = plane;
    check(stereo3d::vpiPlaneData(const_plane) ==
              stereo3d::vpiPlaneData(plane),
          "const and mutable getters must address the same byte");

    std::cout << "VPI plane access compatibility test passed\n";
    return 0;
}
