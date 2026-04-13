## sync_to_nx.ps1 -- Sync NX_volleyball to Jetson Orin NX via pscp/plink
## Usage: powershell -ExecutionPolicy Bypass -File sync_to_nx.ps1

$NX_HOST = "192.168.31.56"
$NX_USER = "nvidia"
$NX_PASS = "nvidia"
$NX_DIR  = "/home/nvidia/NX_volleyball"

$LOCAL_ROOT = Split-Path -Parent $PSScriptRoot   # NX_volleyball/

Write-Host "=== Sync NX_volleyball to $NX_USER@${NX_HOST}:$NX_DIR ===" -ForegroundColor Cyan

# 1. Create remote directory structure
Write-Host "[1/3] Creating remote directories..."
$dirs = @(
    "$NX_DIR",
    "$NX_DIR/model",
    "$NX_DIR/stereo_3d_pipeline",
    "$NX_DIR/stereo_3d_pipeline/src",
    "$NX_DIR/stereo_3d_pipeline/src/capture",
    "$NX_DIR/stereo_3d_pipeline/src/detect",
    "$NX_DIR/stereo_3d_pipeline/src/rectify",
    "$NX_DIR/stereo_3d_pipeline/src/stereo",
    "$NX_DIR/stereo_3d_pipeline/src/fusion",
    "$NX_DIR/stereo_3d_pipeline/src/pipeline",
    "$NX_DIR/stereo_3d_pipeline/src/utils",
    "$NX_DIR/stereo_3d_pipeline/src/track",
    "$NX_DIR/stereo_3d_pipeline/src/calibration",
    "$NX_DIR/stereo_3d_pipeline/config",
    "$NX_DIR/stereo_3d_pipeline/scripts",
    "$NX_DIR/stereo_3d_pipeline/docs",
    "$NX_DIR/scripts",
    "$NX_DIR/calibration",
    "$NX_DIR/ros2_ws/src/volleyball_stereo_driver/model",
    "$NX_DIR/ros2_ws/src/volleyball_stereo_driver/calibration",
    "$NX_DIR/ros2_ws/src/volleyball_stereo_driver/config"
)
$mkdirCmd = "mkdir -p " + ($dirs -join " ")
& plink -ssh "$NX_USER@$NX_HOST" -pw $NX_PASS -batch $mkdirCmd

# 2. Sync files using pscp (filtered)
Write-Host "[2/3] Uploading files (exclude calibration images)..."

function Upload-TreeFiltered {
    param(
        [string]$LocalDir,
        [string]$RemoteDir,
        [string[]]$ExcludeDirPatterns = @(),
        [string[]]$ExcludeExt = @()
    )

    if (!(Test-Path $LocalDir)) {
        Write-Host "  SKIP (not found): $LocalDir" -ForegroundColor Yellow
        return
    }

    $files = Get-ChildItem -Path $LocalDir -File -Recurse -ErrorAction SilentlyContinue | Where-Object {
        $path = $_.FullName.ToLower()
        $ext  = $_.Extension.ToLower()

        foreach ($pat in $ExcludeDirPatterns) {
            if ($path -like "*$($pat.ToLower())*") { return $false }
        }
        if ($ExcludeExt -contains $ext) { return $false }
        return $true
    }

    foreach ($f in $files) {
        $rel = $f.FullName.Substring($LocalDir.Length).TrimStart('\\')
        $relUnix = $rel -replace '\\', '/'
        $remoteSubDir = [System.IO.Path]::GetDirectoryName($relUnix)
        if ([string]::IsNullOrWhiteSpace($remoteSubDir)) {
            $remoteSubDir = $RemoteDir
        } else {
            $remoteSubDir = "$RemoteDir/$remoteSubDir"
        }
        & plink -ssh "$NX_USER@$NX_HOST" -pw $NX_PASS -batch "mkdir -p '$remoteSubDir'" 2>&1 | Out-Null
        & pscp -pw $NX_PASS -p $f.FullName "$NX_USER@$NX_HOST`:$remoteSubDir/" 2>&1 | Out-Null
    }

    Write-Host "  DONE: $LocalDir -> $RemoteDir ($($files.Count) files)"
}

$ExcludeCalibDirs = @("\\calib_500\\", "\\calibration_images\\")
$ExcludeImageExts = @('.jpg','.jpeg','.png','.bmp')

# Core project files
Upload-TreeFiltered "$LOCAL_ROOT\stereo_3d_pipeline\src"    "$NX_DIR/stereo_3d_pipeline/src" $ExcludeCalibDirs $ExcludeImageExts
Upload-TreeFiltered "$LOCAL_ROOT\stereo_3d_pipeline\config" "$NX_DIR/stereo_3d_pipeline/config" $ExcludeCalibDirs $ExcludeImageExts
Upload-TreeFiltered "$LOCAL_ROOT\stereo_3d_pipeline\scripts" "$NX_DIR/stereo_3d_pipeline/scripts" $ExcludeCalibDirs $ExcludeImageExts

# Docs and build files
if (Test-Path "$LOCAL_ROOT\stereo_3d_pipeline\CMakeLists.txt") {
    & pscp -pw $NX_PASS -p "$LOCAL_ROOT\stereo_3d_pipeline\CMakeLists.txt" "$NX_USER@$NX_HOST`:$NX_DIR/stereo_3d_pipeline/" 2>&1 | Out-Null
}
if (Test-Path "$LOCAL_ROOT\stereo_3d_pipeline\README.md") {
    & pscp -pw $NX_PASS -p "$LOCAL_ROOT\stereo_3d_pipeline\README.md" "$NX_USER@$NX_HOST`:$NX_DIR/stereo_3d_pipeline/" 2>&1 | Out-Null
}
if (Test-Path "$LOCAL_ROOT\stereo_3d_pipeline\docs") {
    Upload-TreeFiltered "$LOCAL_ROOT\stereo_3d_pipeline\docs" "$NX_DIR/stereo_3d_pipeline/docs" $ExcludeCalibDirs $ExcludeImageExts
}

# Models (keep all engine/onnx artifacts)
Upload-TreeFiltered "$LOCAL_ROOT\model" "$NX_DIR/model" $ExcludeCalibDirs $ExcludeImageExts

# ROS2 driver config/model, calibration only yaml (exclude calibration images)
Upload-TreeFiltered "$LOCAL_ROOT\ros2_ws\src\volleyball_stereo_driver\config" "$NX_DIR/ros2_ws/src/volleyball_stereo_driver/config" $ExcludeCalibDirs $ExcludeImageExts
Upload-TreeFiltered "$LOCAL_ROOT\ros2_ws\src\volleyball_stereo_driver\model" "$NX_DIR/ros2_ws/src/volleyball_stereo_driver/model" $ExcludeCalibDirs $ExcludeImageExts
Upload-TreeFiltered "$LOCAL_ROOT\ros2_ws\src\volleyball_stereo_driver\calibration" "$NX_DIR/ros2_ws/src/volleyball_stereo_driver/calibration" $ExcludeCalibDirs $ExcludeImageExts

# Utility script
if (Test-Path "$LOCAL_ROOT\scripts\convert_yolo_to_tensorrt.py") {
    & pscp -pw $NX_PASS -p "$LOCAL_ROOT\scripts\convert_yolo_to_tensorrt.py" "$NX_USER@$NX_HOST`:$NX_DIR/scripts/" 2>&1 | Out-Null
}

# 3. Summary
Write-Host "[3/3] Sync summary"
Write-Host "  - Calibration image directories excluded: calib_500, calibration_images"
Write-Host "  - Calibration image extensions excluded: .jpg .jpeg .png .bmp"

Write-Host ""
Write-Host "=== Sync complete ===" -ForegroundColor Green
Write-Host "Next: ssh nvidia@$NX_HOST and run:"
Write-Host "  cd ~/NX_volleyball/stereo_3d_pipeline/scripts"
Write-Host "  bash setup_nx_env.sh"
Write-Host "  bash benchmark.sh"
