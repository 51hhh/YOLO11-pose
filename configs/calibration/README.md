# 标定配置

`*.yaml` 只选择标定 ID、文件和坐标系；真实内外参使用 `contracts/capture/v2.1.0/calibration.schema.json` 的 JSON 格式保存。

`vehicle-stereo-main.json` 必须由标定流程产生，不得从旧 YAML 猜测 `T_body_camera`。在实机重新测量并完成坐标系验证前，本仓库不伪造该文件。

