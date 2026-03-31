#!/usr/bin/env python3
"""
海康相机封装类
支持外部触发、图像采集和参数配置

依赖: MVS SDK (需先安装)
"""

import sys
import os
import numpy as np
import cv2
from ctypes import *

# 导入 MVS SDK
try:
    # 强制修正 MVCAM_COMMON_RUNENV (系统 profile 可能设置了错误路径)
    # SDK 内部通过此变量拼接 .so 路径: $MVCAM_COMMON_RUNENV/aarch64/libMvCameraControl.so
    _mvs_lib = "/opt/MVS/lib"
    _cur_env = os.getenv("MVCAM_COMMON_RUNENV", "")
    if not os.path.isfile(os.path.join(_cur_env, "aarch64", "libMvCameraControl.so")):
        os.environ["MVCAM_COMMON_RUNENV"] = _mvs_lib
    # aarch64 (Jetson NX 等 ARM 平台)
    sys.path.append("/opt/MVS/Samples/aarch64/Python/MvImport")
    # x86_64 备选路径
    sys.path.append("/opt/MVS/Samples/64/Python/MvImport")
    from MvCameraControl_class import *
except ImportError:
    print("❌ 错误: 无法导入 MVS SDK")
    print("请确保已安装海康 MVS SDK:")
    print("  sudo /opt/MVS/bin/setup.sh")
    sys.exit(1)


class HikCamera:
    """海康工业相机封装类"""
    
    def __init__(self, camera_index=0, serial_number=None):
        """
        初始化相机
        
        Args:
            camera_index: 相机索引 (0, 1, ...)
            serial_number: 相机序列号 (可选，用于指定特定相机)
        """
        self.camera_index = camera_index
        self.serial_number = serial_number
        self.cam = MvCamera()
        self.device_list = None
        self.is_opened = False
        self.is_grabbing = False
        
        # 图像缓存
        self.width = 0
        self.height = 0
        self.pixel_format = 0
        
    def list_devices(self):
        """列出所有可用相机"""
        device_list = MV_CC_DEVICE_INFO_LIST()
        ret = self.cam.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, device_list)
        
        if ret != 0:
            print(f"❌ 枚举设备失败: {hex(ret)}")
            return []
        
        if device_list.nDeviceNum == 0:
            print("⚠️  未找到相机")
            return []
        
        devices = []
        for i in range(device_list.nDeviceNum):
            device_info = cast(device_list.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
            
            if device_info.nTLayerType == MV_GIGE_DEVICE:
                # GigE 相机
                gige_info = device_info.SpecialInfo.stGigEInfo
                ip = f"{gige_info.nCurrentIp >> 24 & 0xFF}." \
                     f"{gige_info.nCurrentIp >> 16 & 0xFF}." \
                     f"{gige_info.nCurrentIp >> 8 & 0xFF}." \
                     f"{gige_info.nCurrentIp & 0xFF}"

                model = bytes(gige_info.chModelName).split(b'\x00', 1)[0].decode('utf-8', errors='ignore')
                serial = bytes(gige_info.chSerialNumber).split(b'\x00', 1)[0].decode('utf-8', errors='ignore')

                devices.append({
                    'index': i,
                    'type': 'GigE',
                    'model': model,
                    'serial': serial,
                    'ip': ip
                })
            
            elif device_info.nTLayerType == MV_USB_DEVICE:
                # USB 相机
                usb_info = device_info.SpecialInfo.stUsb3VInfo
                model = bytes(usb_info.chModelName).split(b'\x00', 1)[0].decode('utf-8', errors='ignore')
                serial = bytes(usb_info.chSerialNumber).split(b'\x00', 1)[0].decode('utf-8', errors='ignore')

                devices.append({
                    'index': i,
                    'type': 'USB',
                    'model': model,
                    'serial': serial,
                })
        
        return devices
    
    def open(self):
        """打开相机"""
        # 枚举设备
        self.device_list = MV_CC_DEVICE_INFO_LIST()
        ret = self.cam.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, self.device_list)
        
        if ret != 0:
            raise RuntimeError(f"枚举设备失败: {hex(ret)}")
        
        if self.device_list.nDeviceNum == 0:
            raise RuntimeError("未找到相机")
        
        # 选择相机
        if self.serial_number:
            # 根据序列号选择
            device_index = None
            for i in range(self.device_list.nDeviceNum):
                device_info = cast(self.device_list.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
                
                if device_info.nTLayerType == MV_GIGE_DEVICE:
                    gige_info = cast(device_info.SpecialInfo.stGigEInfo, MV_GIGE_DEVICE_INFO)
                    serial = gige_info.chSerialNumber.decode('utf-8')
                elif device_info.nTLayerType == MV_USB_DEVICE:
                    usb_info = cast(device_info.SpecialInfo.stUsb3VInfo, MV_USB3_DEVICE_INFO)
                    serial = usb_info.chSerialNumber.decode('utf-8')
                
                if serial == self.serial_number:
                    device_index = i
                    break
            
            if device_index is None:
                raise RuntimeError(f"未找到序列号为 {self.serial_number} 的相机")
            
            self.camera_index = device_index
        
        # 创建句柄
        ret = self.cam.MV_CC_CreateHandle(self.device_list.pDeviceInfo[self.camera_index])
        if ret != 0:
            raise RuntimeError(f"创建句柄失败: {hex(ret)}")
        
        # 打开设备
        ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            raise RuntimeError(f"打开设备失败: {hex(ret)}")
        
        self.is_opened = True
        
        # 获取图像参数
        stParam = MVCC_INTVALUE()
        ret = self.cam.MV_CC_GetIntValue("Width", stParam)
        if ret == 0:
            self.width = stParam.nCurValue
        
        ret = self.cam.MV_CC_GetIntValue("Height", stParam)
        if ret == 0:
            self.height = stParam.nCurValue
        
        ret = self.cam.MV_CC_GetIntValue("PixelFormat", stParam)
        if ret == 0:
            self.pixel_format = stParam.nCurValue
        
        print(f"✅ 相机已打开: {self.width}x{self.height}")
    
    def close(self):
        """关闭相机"""
        if self.is_grabbing:
            self.stop_grabbing()
        
        if self.is_opened:
            self.cam.MV_CC_CloseDevice()
            self.cam.MV_CC_DestroyHandle()
            self.is_opened = False
            print("✅ 相机已关闭")
    
    def set_trigger_mode(self, mode='On'):
        """
        设置触发模式
        
        Args:
            mode: 'On' 或 'Off'
        """
        if not self.is_opened:
            raise RuntimeError("相机未打开")
        
        value = 1 if mode == 'On' else 0
        ret = self.cam.MV_CC_SetEnumValue("TriggerMode", value)
        
        if ret != 0:
            raise RuntimeError(f"设置触发模式失败: {hex(ret)}")
        
        print(f"✅ 触发模式: {mode}")
    
    def set_trigger_source(self, source='Line0'):
        """
        设置触发源
        
        Args:
            source: 'Line0', 'Line1', 'Software', etc.
        """
        if not self.is_opened:
            raise RuntimeError("相机未打开")
        
        # 触发源映射
        source_map = {
            'Line0': 0,
            'Line1': 1,
            'Line2': 2,
            'Software': 7,
        }
        
        value = source_map.get(source, 0)
        ret = self.cam.MV_CC_SetEnumValue("TriggerSource", value)
        
        if ret != 0:
            raise RuntimeError(f"设置触发源失败: {hex(ret)}")
        
        print(f"✅ 触发源: {source}")
    
    def set_trigger_activation(self, activation='RisingEdge'):
        """
        设置触发激活方式
        
        Args:
            activation: 'RisingEdge', 'FallingEdge', 'LevelHigh', 'LevelLow'
        """
        if not self.is_opened:
            raise RuntimeError("相机未打开")
        
        activation_map = {
            'RisingEdge': 0,
            'FallingEdge': 1,
            'LevelHigh': 2,
            'LevelLow': 3,
        }
        
        value = activation_map.get(activation, 0)
        ret = self.cam.MV_CC_SetEnumValue("TriggerActivation", value)
        
        if ret != 0:
            raise RuntimeError(f"设置触发激活失败: {hex(ret)}")
        
        print(f"✅ 触发激活: {activation}")
    
    def set_exposure_time(self, exposure_us):
        """
        设置曝光时间
        
        Args:
            exposure_us: 曝光时间 (微秒)
        """
        if not self.is_opened:
            raise RuntimeError("相机未打开")
        
        ret = self.cam.MV_CC_SetFloatValue("ExposureTime", float(exposure_us))
        
        if ret != 0:
            raise RuntimeError(f"设置曝光时间失败: {hex(ret)}")
        
        print(f"✅ 曝光时间: {exposure_us} us")
    
    def set_gain(self, gain_db):
        """
        设置增益
        
        Args:
            gain_db: 增益 (dB)
        """
        if not self.is_opened:
            raise RuntimeError("相机未打开")
        
        ret = self.cam.MV_CC_SetFloatValue("Gain", float(gain_db))
        
        if ret != 0:
            raise RuntimeError(f"设置增益失败: {hex(ret)}")
        
        print(f"✅ 增益: {gain_db} dB")
    
    def start_grabbing(self):
        """开始采集"""
        if not self.is_opened:
            raise RuntimeError("相机未打开")
        
        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            raise RuntimeError(f"开始采集失败: {hex(ret)}")
        
        self.is_grabbing = True
        print("✅ 开始采集")
    
    def stop_grabbing(self):
        """停止采集"""
        if self.is_grabbing:
            ret = self.cam.MV_CC_StopGrabbing()
            if ret != 0:
                print(f"⚠️  停止采集失败: {hex(ret)}")
            else:
                self.is_grabbing = False
                print("✅ 停止采集")
    
    def grab_image(self, timeout_ms=1000):
        """
        采集一帧图像
        
        Args:
            timeout_ms: 超时时间 (毫秒)
        
        Returns:
            numpy array (BGR 格式) 或 None
        """
        if not self.is_grabbing:
            raise RuntimeError("未开始采集")
        
        # 创建图像缓存
        stFrameInfo = MV_FRAME_OUT_INFO_EX()
        memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))
        
        # 分配内存
        data_size = self.width * self.height * 3
        pData = (c_ubyte * data_size)()
        
        # 获取图像
        ret = self.cam.MV_CC_GetOneFrameTimeout(pData, data_size, stFrameInfo, timeout_ms)
        
        if ret != 0:
            return None
        
        # 转换为 numpy array
        if stFrameInfo.enPixelType == PixelType_Gvsp_Mono8:
            # 灰度图
            image = np.frombuffer(pData, dtype=np.uint8).reshape(stFrameInfo.nHeight, stFrameInfo.nWidth)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        elif stFrameInfo.enPixelType == PixelType_Gvsp_RGB8_Packed:
            # RGB
            image = np.frombuffer(pData, dtype=np.uint8).reshape(stFrameInfo.nHeight, stFrameInfo.nWidth, 3)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        elif stFrameInfo.enPixelType == PixelType_Gvsp_BGR8_Packed:
            # BGR
            image = np.frombuffer(pData, dtype=np.uint8).reshape(stFrameInfo.nHeight, stFrameInfo.nWidth, 3)
        
        else:
            # 其他格式，尝试转换
            stConvertParam = MV_CC_PIXEL_CONVERT_PARAM()
            memset(byref(stConvertParam), 0, sizeof(stConvertParam))
            
            stConvertParam.nWidth = stFrameInfo.nWidth
            stConvertParam.nHeight = stFrameInfo.nHeight
            stConvertParam.pSrcData = pData
            stConvertParam.nSrcDataLen = stFrameInfo.nFrameLen
            stConvertParam.enSrcPixelType = stFrameInfo.enPixelType
            stConvertParam.enDstPixelType = PixelType_Gvsp_BGR8_Packed
            
            nConvertSize = stFrameInfo.nWidth * stFrameInfo.nHeight * 3
            pConvertData = (c_ubyte * nConvertSize)()
            stConvertParam.pDstBuffer = pConvertData
            stConvertParam.nDstBufferSize = nConvertSize
            
            ret = self.cam.MV_CC_ConvertPixelType(stConvertParam)
            if ret != 0:
                print(f"⚠️  像素格式转换失败: {hex(ret)}")
                return None
            
            image = np.frombuffer(pConvertData, dtype=np.uint8).reshape(stFrameInfo.nHeight, stFrameInfo.nWidth, 3)
        
        return image
    
    def __enter__(self):
        """上下文管理器"""
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器"""
        self.close()


# ==================== 测试代码 ====================
if __name__ == "__main__":
    print("="*60)
    print("海康相机测试")
    print("="*60)
    
    # 列出所有相机
    cam = HikCamera()
    devices = cam.list_devices()
    
    print(f"\n找到 {len(devices)} 个相机:")
    for dev in devices:
        print(f"  [{dev['index']}] {dev['type']} - {dev['model']}")
        print(f"      序列号: {dev['serial']}")
        if 'ip' in dev:
            print(f"      IP: {dev['ip']}")
    
    if len(devices) == 0:
        print("\n❌ 未找到相机，退出")
        sys.exit(1)
    
    # 打开第一个相机
    print(f"\n打开相机 0...")
    with HikCamera(0) as cam:
        # 配置触发
        cam.set_trigger_mode('On')
        cam.set_trigger_source('Line0')
        cam.set_trigger_activation('RisingEdge')
        cam.set_exposure_time(800)  # 800us
        
        # 开始采集
        cam.start_grabbing()
        
        print("\n等待触发信号...")
        print("(请确保 PWM 触发正在运行)")
        print("按 Ctrl+C 退出\n")
        
        frame_count = 0
        try:
            while True:
                image = cam.grab_image(timeout_ms=2000)
                
                if image is not None:
                    frame_count += 1
                    print(f"\r✅ 已采集 {frame_count} 帧", end='', flush=True)
                    
                    # 显示图像 (可选)
                    # cv2.imshow('Camera', image)
                    # cv2.waitKey(1)
                else:
                    print("\r⚠️  超时，未收到触发", end='', flush=True)
        
        except KeyboardInterrupt:
            print(f"\n\n✅ 测试完成，共采集 {frame_count} 帧")
