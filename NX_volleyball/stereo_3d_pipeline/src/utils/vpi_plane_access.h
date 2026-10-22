/**
 * @file vpi_plane_access.h
 * @brief VPI 2/3 and VPI 4 pitch-linear plane address compatibility.
 *
 * VPI 4 replaced VPIImagePlanePitchLinear::data with pBase + offsetBytes.
 * Keep that SDK difference at one boundary instead of scattering version
 * conditionals through capture, pipeline and diagnostic code.
 */

#ifndef STEREO_3D_PIPELINE_VPI_PLANE_ACCESS_H_
#define STEREO_3D_PIPELINE_VPI_PLANE_ACCESS_H_

#include <vpi/Image.h>
#include <vpi/Version.h>

#include <cstdint>

namespace stereo3d {

inline std::uint8_t* vpiPlaneData(VPIImagePlanePitchLinear& plane) {
#if NV_VPI_VERSION_MAJOR >= 4
    if (!plane.pBase || plane.offsetBytes < 0) return nullptr;
    return reinterpret_cast<std::uint8_t*>(plane.pBase) + plane.offsetBytes;
#else
    return static_cast<std::uint8_t*>(plane.data);
#endif
}

inline const std::uint8_t* vpiPlaneData(
    const VPIImagePlanePitchLinear& plane) {
#if NV_VPI_VERSION_MAJOR >= 4
    if (!plane.pBase || plane.offsetBytes < 0) return nullptr;
    return reinterpret_cast<const std::uint8_t*>(plane.pBase) +
           plane.offsetBytes;
#else
    return static_cast<const std::uint8_t*>(plane.data);
#endif
}

inline void setVpiPlaneData(VPIImagePlanePitchLinear& plane, void* data) {
#if NV_VPI_VERSION_MAJOR >= 4
    plane.pBase = static_cast<VPIByte*>(data);
    plane.offsetBytes = 0;
#else
    plane.data = data;
#endif
}

}  // namespace stereo3d

#endif  // STEREO_3D_PIPELINE_VPI_PLANE_ACCESS_H_
