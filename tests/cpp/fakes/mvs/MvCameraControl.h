#pragma once

// 主机测试共用的最小海康 MVS SDK 桩；不进入生产构建。

#include <cstdint>

#define MV_OK 0
#define MV_GIGE_DEVICE 0x00000001U
#define MV_USB_DEVICE 0x00000004U
#define MV_GrabStrategy_OneByOne 0U
#define PixelType_Gvsp_BayerRG8 0x01080009U

struct MV_USB3_DEVICE_INFO {
    unsigned char chSerialNumber[64];
    unsigned char chModelName[64];
};

struct MV_GIGE_DEVICE_INFO {
    unsigned char chSerialNumber[64];
    unsigned char chModelName[64];
};

union MV_SPECIAL_INFO {
    MV_USB3_DEVICE_INFO stUsb3VInfo;
    MV_GIGE_DEVICE_INFO stGigEInfo;
};

struct MV_CC_DEVICE_INFO {
    unsigned int nTLayerType;
    MV_SPECIAL_INFO SpecialInfo;
};

struct MV_CC_DEVICE_INFO_LIST {
    unsigned int nDeviceNum;
    MV_CC_DEVICE_INFO** pDeviceInfo;
};

struct MVCC_ENUMVALUE {
    unsigned int nCurValue;
};

struct MVCC_FLOATVALUE {
    float fCurValue;
};

struct MVCC_INTVALUE {
    std::uint64_t nCurValue;
};

struct MV_CHUNK_DATA_CONTENT {
    unsigned int nChunkID;
    unsigned int nChunkLen;
    unsigned char* pChunkData;
};

struct MV_UNPARSED_CHUNK_LIST {
    MV_CHUNK_DATA_CONTENT* pUnparsedChunkContent;
};

struct MV_FRAME_OUT_INFO_EX {
    unsigned int nWidth;
    unsigned int nHeight;
    unsigned int nDevTimeStampHigh;
    unsigned int nDevTimeStampLow;
    std::uint64_t nHostTimeStamp;
    unsigned int nFrameNum;
    unsigned int nFrameCounter;
    unsigned int nTriggerIndex;
    unsigned int nSecondCount;
    unsigned int nCycleCount;
    unsigned int nCycleOffset;
    float fGain;
    float fExposureTime;
    unsigned int nInput;
    unsigned int nOutput;
    unsigned int nUnparsedChunkNum;
    MV_UNPARSED_CHUNK_LIST UnparsedChunkList;
    unsigned int nExtendWidth;
    unsigned int nExtendHeight;
    std::uint64_t nFrameLenEx;
    unsigned int nFrameLen;
};

struct MV_FRAME_OUT {
    unsigned char* pBufAddr;
    MV_FRAME_OUT_INFO_EX stFrameInfo;
};

extern "C" {
int MV_CC_EnumDevices(unsigned int transport_layer_type,
                      MV_CC_DEVICE_INFO_LIST* device_list);
int MV_CC_CreateHandle(void** handle, MV_CC_DEVICE_INFO* device_info);
int MV_CC_OpenDevice(void* handle);
int MV_CC_CloseDevice(void* handle);
int MV_CC_DestroyHandle(void* handle);
int MV_CC_SetEnumValue(void* handle, const char* node, unsigned int value);
int MV_CC_SetEnumValueByString(void* handle, const char* node, const char* value);
int MV_CC_SetFloatValue(void* handle, const char* node, float value);
int MV_CC_SetBoolValue(void* handle, const char* node, bool value);
int MV_CC_GetEnumValue(void* handle, const char* node, MVCC_ENUMVALUE* value);
int MV_CC_GetFloatValue(void* handle, const char* node, MVCC_FLOATVALUE* value);
int MV_CC_GetBoolValue(void* handle, const char* node, bool* value);
int MV_CC_GetIntValue(void* handle, const char* node, MVCC_INTVALUE* value);
int MV_CC_SetGrabStrategy(void* handle, unsigned int strategy);
int MV_CC_SetImageNodeNum(void* handle, unsigned int count);
int MV_CC_StartGrabbing(void* handle);
int MV_CC_StopGrabbing(void* handle);
int MV_CC_GetImageBuffer(void* handle, MV_FRAME_OUT* frame,
                         unsigned int timeout_ms);
int MV_CC_FreeImageBuffer(void* handle, MV_FRAME_OUT* frame);
}
