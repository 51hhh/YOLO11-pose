#pragma once

#include "args.h"

// 执行 ArUco 双目外参标定
int runArucoStereoCalibration(
    const StereoCalibrateArgs& args
);