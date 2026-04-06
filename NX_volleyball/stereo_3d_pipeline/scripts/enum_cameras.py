#!/usr/bin/env python3
import os, sys
os.environ.setdefault("MVCAM_COMMON_RUNENV", "/opt/MVS/lib")
sys.path.insert(0, "/opt/MVS/Samples/aarch64/Python/MvImport")
from MvCameraControl_class import *
from ctypes import cast, POINTER

dl = MV_CC_DEVICE_INFO_LIST()
ret = MvCamera.MV_CC_EnumDevices(MV_USB_DEVICE, dl)
print("Found %d cameras" % dl.nDeviceNum)
for i in range(dl.nDeviceNum):
    info = cast(dl.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
    usb = info.SpecialInfo.stUsb3VInfo
    sn = bytes(usb.chSerialNumber).split(b"\x00")[0].decode()
    model = bytes(usb.chModelName).split(b"\x00")[0].decode()
    uid = bytes(usb.chUserDefinedName).split(b"\x00")[0].decode()
    print("  [%d] Model: %s, SN: %s, UserID: %s" % (i, model, sn, uid))
