# 标定配置

`vehicle_stereo_main.yaml` 是 profile 引用的资产描述，固定标定 ID、创建时间、
源文件和坐标系名称。`stereo_calib.yaml` 是当前 OpenCV 双目标定资产，保存
原始内参、畸变、左右外参以及校正投影矩阵；它不是 profile 配置层。

正式采集时，`python/volleyball_vision/calibration.py` 会从校正投影矩阵生成
符合 `contracts/capture/v2.1.0/calibration.schema.json` 的
`calibrations/vehicle-stereo-main.json`。当前 JSON 只描述以左目校正光心为原点
的双目 rig：右目平移来自实测基线，畸变在校正模型中为零。仓库尚未保存
rig 到车体 `base_link` 的实测变换；完成独立外参测量前不得猜测或伪造它。

