"""
deploy_to_nx.py - Deploy stereo_3d_pipeline to Jetson NX via SSH
Usage: python scripts/deploy_to_nx.py [sync|build|test|all]
"""
import paramiko
import os
import sys
import stat
import time

NX_HOST = "192.168.31.56"
NX_USER = "nvidia"
NX_PASS = "nvidia"
NX_DIR  = "/home/nvidia/NX_volleyball/stereo_3d_pipeline"
LOCAL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_ssh():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(NX_HOST, username=NX_USER, password=NX_PASS, timeout=10)
    return ssh

def run_cmd(ssh, cmd, print_output=True):
    print(f"[CMD] {cmd}")
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=120)
    out = stdout.read().decode('utf-8', errors='replace')
    err = stderr.read().decode('utf-8', errors='replace')
    if print_output:
        if out.strip(): print(out)
        if err.strip(): print(f"[STDERR] {err}")
    return out, err, stdout.channel.recv_exit_status()

def setup_ssh_key(ssh):
    """Deploy SSH public key for future passwordless access"""
    key_path = os.path.expanduser("~/.ssh/id_rsa.pub")
    if not os.path.exists(key_path):
        print("No SSH public key found, skipping key deployment")
        return
    with open(key_path) as f:
        pubkey = f.read().strip()
    run_cmd(ssh, f'mkdir -p ~/.ssh && chmod 700 ~/.ssh')
    # Check if key already deployed
    out, _, _ = run_cmd(ssh, f'grep -c "{pubkey[:50]}" ~/.ssh/authorized_keys 2>/dev/null || echo 0', False)
    if out.strip() == '0':
        run_cmd(ssh, f'echo "{pubkey}" >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys')
        print("[OK] SSH key deployed")
    else:
        print("[OK] SSH key already deployed")

def sync_files(ssh):
    """Upload source files to NX via tar archive"""
    import tarfile
    import io

    print("=== Syncing files ===")

    # Create tar in memory
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode='w:gz') as tar:
        sync_list = [
            "src/pipeline/pipeline.h",
            "src/pipeline/pipeline.cpp",
            "src/pipeline/frame_slot.h",
            "src/pipeline/sync.h",
            "src/stereo/vpi_stereo.h",
            "src/stereo/vpi_stereo.cpp",
            "src/stereo/roi_stereo_matcher.h",
            "src/stereo/roi_stereo_matcher.cpp",
            "src/stereo/roi_stereo_match.cu",
            "src/stereo/onnx_stereo.h",
            "src/stereo/onnx_stereo.cpp",
            "src/detect/trt_detector.h",
            "src/detect/trt_detector.cpp",
            "src/detect/detect_preprocess.cu",
            "src/fusion/coordinate_3d.h",
            "src/fusion/coordinate_3d.cpp",
            "src/fusion/depth_extract.cu",
            "src/calibration/stereo_calibration.h",
            "src/calibration/stereo_calibration.cpp",
            "src/calibration/stereo_calibrate.cpp",
            "src/calibration/capture_chessboard.cpp",
            "src/calibration/pwm_trigger.h",
            "src/rectify/vpi_rectifier.h",
            "src/rectify/vpi_rectifier.cpp",
            "src/capture/hikvision_camera.h",
            "src/capture/hikvision_camera.cpp",
            "src/utils/profiler.h",
            "src/utils/logger.h",
            "src/utils/zero_copy_alloc.h",
            "src/main.cpp",
            "src/stereo_depth_viewer.cpp",
            "CMakeLists.txt",
            "config/pipeline.yaml",
            "config/pipeline_roi.yaml",
            "config/pipeline_dla.yaml",
        ]
        count = 0
        for rel in sync_list:
            full = os.path.join(LOCAL_DIR, rel)
            if os.path.exists(full):
                tar.add(full, arcname=rel)
                count += 1
        print(f"  Packed {count} files")

    tar_buffer.seek(0)
    tar_data = tar_buffer.read()
    print(f"  Archive size: {len(tar_data)//1024} KB")

    # Upload via SFTP
    run_cmd(ssh, f"mkdir -p {NX_DIR}", False)
    sftp = ssh.open_sftp()
    remote_tar = f"{NX_DIR}/_deploy.tar.gz"
    with sftp.file(remote_tar, 'wb') as f:
        f.write(tar_data)
    sftp.close()

    # Extract on NX
    run_cmd(ssh, f"cd {NX_DIR} && tar xzf _deploy.tar.gz && rm _deploy.tar.gz")
    print(f"[SYNC] {count} files uploaded and extracted")

def build(ssh):
    """Build on NX"""
    print("\n=== Building on NX ===")
    run_cmd(ssh, f"cd {NX_DIR} && mkdir -p build")

    # CMake configure
    out, err, rc = run_cmd(ssh,
        f"cd {NX_DIR}/build && cmake .. -DCMAKE_BUILD_TYPE=Release -DCUDA_ARCH=87 2>&1")
    if rc != 0:
        print(f"[ERROR] CMake failed (rc={rc})")
        return False

    # Make
    out, err, rc = run_cmd(ssh, f"cd {NX_DIR}/build && make -j6 2>&1")
    if rc != 0:
        print(f"[ERROR] Build failed (rc={rc})")
        return False

    # Check binary
    out, _, _ = run_cmd(ssh, f"ls -la {NX_DIR}/build/stereo_pipeline 2>/dev/null")
    if "stereo_pipeline" in out:
        print("[OK] Binary built successfully")
        return True
    else:
        print("[ERROR] Binary not found after build")
        return False

def test_dry_run(ssh):
    """Test with dry-run (no camera)"""
    print("\n=== Testing (dry-run, ROI mode) ===")

    # Check if engine file exists, if not create a dummy test
    out, _, _ = run_cmd(ssh, f"ls /home/nvidia/NX_volleyball/model/yolo26_fp16.engine 2>/dev/null", False)
    if "yolo26" not in out:
        print("[WARN] Engine file not found. Testing pipeline init only.")

    out, err, rc = run_cmd(ssh,
        f"cd {NX_DIR} && timeout 8 build/stereo_pipeline --config config/pipeline_roi.yaml 2>&1 || true")
    return True

def perf_test(ssh):
    """Performance benchmark"""
    print("\n=== Performance Test ===")
    run_cmd(ssh, "sudo nvpmodel -m 0 2>/dev/null; sudo jetson_clocks 2>/dev/null || true", False)
    out, err, rc = run_cmd(ssh,
        f"cd {NX_DIR} && timeout 10 build/stereo_pipeline --config config/pipeline_roi.yaml 2>&1 || true")
    return True

def main():
    action = sys.argv[1] if len(sys.argv) > 1 else "all"

    print(f"=== Deploy stereo_3d_pipeline to NX ({NX_HOST}) ===")
    print(f"Action: {action}")
    print(f"Local dir: {LOCAL_DIR}")

    ssh = get_ssh()
    print("[OK] SSH connected")

    # Always setup SSH key for future passwordless access
    setup_ssh_key(ssh)

    if action in ("sync", "all"):
        sync_files(ssh)
    if action in ("build", "all"):
        if not build(ssh):
            ssh.close()
            sys.exit(1)
    if action in ("test", "all"):
        test_dry_run(ssh)
    if action == "perf":
        perf_test(ssh)

    ssh.close()
    print("\n=== Done ===")

if __name__ == "__main__":
    main()
