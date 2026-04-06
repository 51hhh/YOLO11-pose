#!/usr/bin/env python3
import sys
sys.path.insert(0, "/opt/MVS/Samples/64/Python/MvImport")
from MvCameraControl_class import *
from ctypes import cast, POINTER

dl = MV_CC_DEVICE_INFO_LIST()
ret = MvCamera.MV_CC_EnumDevices(MV_USB_DEVICE, dl)
print("Found %d cameras" % dl.nDeviceNum)
for i in range(dl.nDeviceNum):
    info = cast(dl.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
    usb = info.SpecialInfo.stUsb3VInfo
    sn = usb.chSerialNumber.decode("utf-8")
    model = usb.chModelName.decode("utf-8")
    print("  [%d] Model: %s, SN: %s" % (i, model, sn))
