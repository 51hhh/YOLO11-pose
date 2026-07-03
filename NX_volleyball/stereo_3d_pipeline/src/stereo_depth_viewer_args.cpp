/**
 * @file stereo_depth_viewer_args.cpp
 * @brief Command-line options for stereo_depth_viewer.
 */

#include "stereo_depth_viewer_args.h"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

bool hasValue(int i, int argc) {
    return i + 1 < argc;
}

void printHelp() {
    printf("Usage: stereo_depth_viewer -c <calib.yaml> [options]\n"
           "  -c, --calibration  标定文件路径\n"
           "  --config           pipeline.yaml 配置文件 (可选)\n"
           "  --max-disp N       最大视差 (默认: 256)\n"
           "  --win-size N       匹配窗口 (默认: 5)\n"
           "  --free-run         自由运行模式 (无外触发, 不启动PWM)\n"
           "  --pwm-freq Hz      PWM 触发频率 (默认: 15)\n"
           "  --no-pwm           禁用 PWM 自启动 (仅在外部已启动PWM时使用)\n"
           "  --diagnose         诊断模式\n"
           "  --swap-lr          左右相机互换\n"
           "  --headless         无头模式 (无窗口, 自动遍历所有模式)\n"
           "  --frames N         无头模式每种算法运行帧数 (默认: 30)\n"
           "  --crestereo PATH   CREStereo ONNX 模型路径\n"
           "  --hitnet PATH      HITNet ONNX 模型路径\n"
           "\n按键:\n"
           "  t   切换视图模式\n"
           "  q   退出\n");
}

}  // namespace

StereoDepthViewerArgs parseStereoDepthViewerArgs(int argc, char** argv) {
    StereoDepthViewerArgs args;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if ((arg == "-c" || arg == "--calibration") && hasValue(i, argc)) {
            args.calib_path = argv[++i];
        } else if (arg == "--config" && hasValue(i, argc)) {
            args.config_path = argv[++i];
        } else if (arg == "--max-disp" && hasValue(i, argc)) {
            args.max_disp = std::atoi(argv[++i]);
        } else if (arg == "--win-size" && hasValue(i, argc)) {
            args.win_size = std::atoi(argv[++i]);
        } else if (arg == "--free-run") {
            args.free_run = true;
        } else if (arg == "--diagnose") {
            args.diagnose = true;
        } else if (arg == "--swap-lr") {
            args.swap_lr = true;
        } else if (arg == "--headless") {
            args.headless = true;
        } else if (arg == "--frames" && hasValue(i, argc)) {
            args.headless_frames = std::atoi(argv[++i]);
        } else if (arg == "--pwm-freq" && hasValue(i, argc)) {
            args.pwm_freq = std::atof(argv[++i]);
        } else if (arg == "--no-pwm") {
            args.no_pwm = true;
        } else if (arg == "--crestereo" && hasValue(i, argc)) {
            args.crestereo_path = argv[++i];
        } else if (arg == "--hitnet" && hasValue(i, argc)) {
            args.hitnet_path = argv[++i];
        } else if (arg == "-h" || arg == "--help") {
            printHelp();
            args.should_exit = true;
            args.exit_code = 0;
            return args;
        }
    }

    if (args.calib_path.empty()) {
        fprintf(stderr, "Error: 必须指定标定文件 (-c <path>)\n");
        args.should_exit = true;
        args.exit_code = 1;
    }
    return args;
}
