/**
 * @file capture_stereo_images.cpp
 * @brief 海康双目相机棋盘格图像采集工具
 * 
 * 编译: g++ -o capture_stereo capture_stereo_images.cpp \
 *       -I/opt/MVS/include -L/opt/MVS/lib/aarch64 -lMvCameraControl \
 *       $(pkg-config --cflags --libs opencv4) -std=c++17
 * 
 * 运行: ./capture_stereo
 */

#include <iostream>
#include <opencv2/opencv.hpp>
#include <MvCameraControl.h>
#include <string>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <sys/stat.h>

class StereoCalibrationCapture {
public:
    StereoCalibrationCapture(int left_index = 0, int right_index = 1)
        : left_index_(left_index), right_index_(right_index),
          left_handle_(nullptr), right_handle_(nullptr),
          capture_count_(0) {
        memset(&device_list_, 0, sizeof(device_list_));
    }

    ~StereoCalibrationCapture() {
        close();
    }

    bool initialize() {
        // 初始化SDK
        int ret = MV_CC_Initialize();
        if (ret != MV_OK) {
            std::cerr << "❌ SDK初始化失败: 0x" << std::hex << ret << std::endl;
            return false;
        }
        std::cout << "✅ 海康 SDK 已初始化" << std::endl;

        // 枚举设备
        ret = MV_CC_EnumDevices(MV_USB_DEVICE | MV_GIGE_DEVICE, &device_list_);
        if (ret != MV_OK) {
            std::cerr << "❌ 枚举设备失败: 0x" << std::hex << ret << std::endl;
            return false;
        }

        if (device_list_.nDeviceNum < 2) {
            std::cerr << "❌ 需要至少2个相机，当前发现: " << device_list_.nDeviceNum << std::endl;
            return false;
        }

        std::cout << "✅ 发现 " << device_list_.nDeviceNum << " 个相机" << std::endl;

        // 打开左相机
        if (!openCamera(left_index_, left_handle_, "左相机")) {
            return false;
        }

        // 打开右相机
        if (!openCamera(right_index_, right_handle_, "右相机")) {
            return false;
        }

        // 配置相机参数
        configureCamera(left_handle_, "左相机");
        configureCamera(right_handle_, "右相机");

        // 创建输出目录
        createDirectory("calibration_images");
        createDirectory("calibration_images/left");
        createDirectory("calibration_images/right");

        std::cout << "\n📸 图像保存目录: calibration_images/" << std::endl;
        std::cout << "   左相机: calibration_images/left/" << std::endl;
        std::cout << "   右相机: calibration_images/right/" << std::endl;

        return true;
    }

    void run() {
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "🎯 双目标定图像采集工具" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        std::cout << "操作说明:" << std::endl;
        std::cout << "  空格键  - 采集当前帧对并保存" << std::endl;
        std::cout << "  q/ESC  - 退出程序" << std::endl;
        std::cout << "\n建议:" << std::endl;
        std::cout << "  • 采集至少 15-20 对图像" << std::endl;
        std::cout << "  • 棋盘格放置在不同位置、角度、深度" << std::endl;
        std::cout << "  • 确保棋盘格完整出现在两个相机视野中" << std::endl;
        std::cout << "  • 避免运动模糊（保持棋盘格静止）" << std::endl;
        std::cout << std::string(60, '=') << std::endl;

        cv::namedWindow("左相机", cv::WINDOW_NORMAL);
        cv::namedWindow("右相机", cv::WINDOW_NORMAL);

        while (true) {
            cv::Mat left_img = grabFrame(left_handle_);
            cv::Mat right_img = grabFrame(right_handle_);

            if (left_img.empty() || right_img.empty()) {
                std::cerr << "⚠️  采集失败，跳过" << std::endl;
                continue;
            }

            // 显示图像
            cv::imshow("左相机", left_img);
            cv::imshow("右相机", right_img);

            int key = cv::waitKey(30);

            if (key == ' ') {  // 空格键保存
                saveImagePair(left_img, right_img);
            } else if (key == 'q' || key == 27) {  // q或ESC退出
                break;
            }
        }

        std::cout << "\n✅ 采集完成！总共采集 " << capture_count_ << " 对图像" << std::endl;
        std::cout << "📁 图像保存在 calibration_images/ 目录" << std::endl;
        std::cout << "\n下一步: 运行 python3 stereo_calibration.py 进行标定\n" << std::endl;
    }

    void close() {
        if (left_handle_) {
            MV_CC_StopGrabbing(left_handle_);
            MV_CC_CloseDevice(left_handle_);
            MV_CC_DestroyHandle(left_handle_);
            left_handle_ = nullptr;
        }

        if (right_handle_) {
            MV_CC_StopGrabbing(right_handle_);
            MV_CC_CloseDevice(right_handle_);
            MV_CC_DestroyHandle(right_handle_);
            right_handle_ = nullptr;
        }

        MV_CC_Finalize();
    }

private:
    bool openCamera(int index, void*& handle, const std::string& name) {
        if (index >= static_cast<int>(device_list_.nDeviceNum)) {
            std::cerr << "❌ " << name << " 索引超出范围" << std::endl;
            return false;
        }

        int ret = MV_CC_CreateHandle(&handle, device_list_.pDeviceInfo[index]);
        if (ret != MV_OK) {
            std::cerr << "❌ " << name << " 创建句柄失败: 0x" << std::hex << ret << std::endl;
            return false;
        }

        ret = MV_CC_OpenDevice(handle);
        if (ret != MV_OK) {
            std::cerr << "❌ " << name << " 打开失败: 0x" << std::hex << ret << std::endl;
            MV_CC_DestroyHandle(handle);
            handle = nullptr;
            return false;
        }

        // 开始采集
        ret = MV_CC_StartGrabbing(handle);
        if (ret != MV_OK) {
            std::cerr << "❌ " << name << " 开始采集失败: 0x" << std::hex << ret << std::endl;
            return false;
        }

        std::cout << "✅ " << name << " (索引 " << index << ") 已打开" << std::endl;
        return true;
    }

    void configureCamera(void* handle, const std::string& name) {
        // 设置为自由运行模式（非触发）
        MV_CC_SetEnumValue(handle, "TriggerMode", 0);

        // 设置曝光时间为自动
        MV_CC_SetEnumValue(handle, "ExposureAuto", 2);  // 2=连续自动

        // 设置增益为自动
        MV_CC_SetEnumValue(handle, "GainAuto", 2);  // 2=连续自动

        // 获取当前分辨率
        MVCC_INTVALUE stParam;
        memset(&stParam, 0, sizeof(MVCC_INTVALUE));
        MV_CC_GetIntValue(handle, "Width", &stParam);
        int width = stParam.nCurValue;
        
        memset(&stParam, 0, sizeof(MVCC_INTVALUE));
        MV_CC_GetIntValue(handle, "Height", &stParam);
        int height = stParam.nCurValue;

        std::cout << "  " << name << " 分辨率: " << width << "x" << height << std::endl;
    }

    cv::Mat grabFrame(void* handle) {
        MV_FRAME_OUT stImageInfo;
        memset(&stImageInfo, 0, sizeof(MV_FRAME_OUT));

        int ret = MV_CC_GetImageBuffer(handle, &stImageInfo, 1000);
        if (ret != MV_OK) {
            return cv::Mat();
        }

        // 转换为OpenCV Mat
        cv::Mat image;
        MvGvspPixelType pixel_type = stImageInfo.stFrameInfo.enPixelType;
        
        if (pixel_type == PixelType_Gvsp_BayerRG8) {
            // Bayer RG8 → BGR (使用正确的Bayer模式)
            cv::Mat bayer(stImageInfo.stFrameInfo.nHeight, 
                         stImageInfo.stFrameInfo.nWidth, 
                         CV_8UC1, stImageInfo.pBufAddr);
            
            // 🔧 尝试正确的Bayer模式 (RG格式应该用BayerBG2BGR)
            // 海康相机Bayer RG8实际对应OpenCV的BG模式
            cv::cvtColor(bayer, image, cv::COLOR_BayerBG2BGR);
            
            std::cout << "  [Bayer RG8 → BGR 转换]" << std::endl;
        } else if (pixel_type == PixelType_Gvsp_RGB8_Packed) {
            // RGB → BGR
            cv::Mat rgb(stImageInfo.stFrameInfo.nHeight,
                       stImageInfo.stFrameInfo.nWidth,
                       CV_8UC3, stImageInfo.pBufAddr);
            cv::cvtColor(rgb, image, cv::COLOR_RGB2BGR);
        } else {
            // 其他格式使用SDK转换
            MV_CC_PIXEL_CONVERT_PARAM stConvertParam;
            memset(&stConvertParam, 0, sizeof(stConvertParam));
            
            unsigned int nBGRSize = stImageInfo.stFrameInfo.nWidth * 
                                   stImageInfo.stFrameInfo.nHeight * 3;
            unsigned char* pBGRBuf = new unsigned char[nBGRSize];
            
            stConvertParam.nWidth = stImageInfo.stFrameInfo.nWidth;
            stConvertParam.nHeight = stImageInfo.stFrameInfo.nHeight;
            stConvertParam.pSrcData = stImageInfo.pBufAddr;
            stConvertParam.nSrcDataLen = stImageInfo.stFrameInfo.nFrameLen;
            stConvertParam.enSrcPixelType = pixel_type;
            stConvertParam.enDstPixelType = PixelType_Gvsp_BGR8_Packed;
            stConvertParam.pDstBuffer = pBGRBuf;
            stConvertParam.nDstBufferSize = nBGRSize;
            
            ret = MV_CC_ConvertPixelType(handle, &stConvertParam);
            if (ret == MV_OK) {
                image = cv::Mat(stImageInfo.stFrameInfo.nHeight,
                              stImageInfo.stFrameInfo.nWidth,
                              CV_8UC3, pBGRBuf).clone();
            }
            
            delete[] pBGRBuf;
        }

        MV_CC_FreeImageBuffer(handle, &stImageInfo);
        return image;
    }

    void saveImagePair(const cv::Mat& left, const cv::Mat& right) {
        std::ostringstream ss;
        ss << std::setfill('0') << std::setw(4) << capture_count_;
        std::string filename = ss.str();

        std::string left_path = "calibration_images/left/left_" + filename + ".png";
        std::string right_path = "calibration_images/right/right_" + filename + ".png";

        cv::imwrite(left_path, left);
        cv::imwrite(right_path, right);

        capture_count_++;

        std::cout << "✅ 已保存第 " << capture_count_ << " 对图像: " 
                  << filename << ".png" << std::endl;
    }

    void createDirectory(const std::string& path) {
        mkdir(path.c_str(), 0755);
    }

private:
    int left_index_;
    int right_index_;
    void* left_handle_;
    void* right_handle_;
    MV_CC_DEVICE_INFO_LIST device_list_;
    int capture_count_;
};

int main(int argc, char** argv) {
    int left_index = 0;
    int right_index = 1;

    if (argc > 2) {
        left_index = std::atoi(argv[1]);
        right_index = std::atoi(argv[2]);
    }

    StereoCalibrationCapture capture(left_index, right_index);

    if (!capture.initialize()) {
        std::cerr << "❌ 初始化失败" << std::endl;
        return -1;
    }

    capture.run();
    capture.close();

    return 0;
}
