"""
deploy_to_nx.py - Deploy stereo_3d_pipeline to Jetson NX via SSH
Usage: python scripts/deploy_to_nx.py [sync|build|test|all]
"""
import paramiko
import os
import shlex
import sys
import stat
import time

NX_HOST = os.environ.get("NX_HOST", "10.42.0.149")
NX_USER = "nvidia"
NX_PASS = "nvidia"
NX_DIR  = "/home/nvidia/NX_volleyball/stereo_3d_pipeline"
LOCAL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYNC_ALL_CONFIGS = os.environ.get("DEPLOY_SYNC_ALL_CONFIGS", "0") == "1"
CONFIG_SYNC_ALLOWLIST = {
    "disparity_offset_fit_20260709.json",
    "pipeline.yaml",
    "pipeline_dual_yolo_roi.yaml",
    "pipeline_rdk_joint.yaml",
    "pipeline_record_p0p1.yaml",
}
REMOVED_CONFIGS = {
    "pipeline_dla.yaml",
    "pipeline_roi.yaml",
    "pipeline_roi_freerun.yaml",
    "pipeline_yolo26_gpu.yaml",
    "pipeline_yolo26_mixed.yaml",
    "pipeline_splitA.yaml",
    "pipeline_yolo11s_960.yaml",
    "pipeline_yolo11s_960_lighttrack.yaml",
    "pipeline_yolo11s_960_mixformer.yaml",
    "pipeline_yolo11s_960_nanotrack.yaml",
    "pipeline_yolo11s_960_siamfc.yaml",
    "pipeline_roi_nanotrack.yaml",
    "pipeline_roi_mixformer.yaml",
    "pipeline_yolo8m_960.yaml",
    "pipeline_zed.yaml",
}

ROS_SETUP = """\
. /opt/ros/humble/setup.bash
if [ -f /home/nvidia/volleyball_ros2_ws/install/setup.bash ]; then
    . /home/nvidia/volleyball_ros2_ws/install/setup.bash
elif [ -f /home/nvidia/ros2_ws/install/setup.bash ]; then
    . /home/nvidia/ros2_ws/install/setup.bash
else
    echo 'volleyball_interfaces workspace setup.bash not found' >&2
    exit 2
fi
"""


def with_ros(command):
    return "bash -lc " + shlex.quote(ROS_SETUP + "\n" + command)

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
        sync_list = ["CMakeLists.txt", "package.xml"]
        for root in ("src", "config", "tests"):
            root_dir = os.path.join(LOCAL_DIR, root)
            if not os.path.isdir(root_dir):
                continue
            for dirpath, dirnames, filenames in os.walk(root_dir):
                dirnames[:] = sorted(
                    d for d in dirnames
                    if d not in ("__pycache__", ".pytest_cache", "build"))
                for name in sorted(filenames):
                    if root == "config" and not SYNC_ALL_CONFIGS:
                        if os.path.relpath(dirpath, root_dir) != ".":
                            continue
                        if name not in CONFIG_SYNC_ALLOWLIST:
                            continue
                    rel = os.path.relpath(os.path.join(dirpath, name), LOCAL_DIR)
                    sync_list.append(rel)
        sync_list = sorted(set(sync_list))
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
    removed = " ".join(f"config/{name}" for name in sorted(REMOVED_CONFIGS))
    run_cmd(ssh, f"cd {NX_DIR} && rm -f {removed}")
    run_cmd(ssh, "mkdir -p /home/nvidia/NX_volleyball/calibration")
    _, _, runtime_rc = run_cmd(
        ssh, "sudo install -d -o nvidia -g nvidia /run/volleyball")
    if runtime_rc != 0:
        raise RuntimeError("Cannot prepare writable /run/volleyball on NX")
    calibration_file = os.path.abspath(
        os.path.join(LOCAL_DIR, "..", "calibration", "stereo_calib.yaml"))
    if not os.path.isfile(calibration_file):
        raise FileNotFoundError(f"Calibration file not found: {calibration_file}")
    sftp = ssh.open_sftp()
    sftp.put(
        calibration_file,
        "/home/nvidia/NX_volleyball/calibration/stereo_calib.yaml")
    sftp.close()
    print(f"[SYNC] {count} files uploaded and extracted")

def build(ssh):
    """Build on NX"""
    print("\n=== Building on NX ===")
    run_cmd(ssh, f"cd {NX_DIR} && mkdir -p build")

    # CMake configure
    out, err, rc = run_cmd(ssh, with_ros(
        f"cd {NX_DIR}/build && cmake .. -DCMAKE_BUILD_TYPE=Release "
        "-DCUDA_ARCH=87 -DREQUIRE_ROS2=ON 2>&1"))
    if rc != 0:
        print(f"[ERROR] CMake failed (rc={rc})")
        return False

    # Make
    out, err, rc = run_cmd(
        ssh, with_ros(f"cd {NX_DIR}/build && make -j6 2>&1"))
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
    out, _, _ = run_cmd(
        ssh,
        "ls /home/nvidia/NX_volleyball/model/yolo26_gpu_fp16.engine 2>/dev/null",
        False)
    if "yolo26" not in out:
        print("[WARN] Engine file not found. Testing pipeline init only.")

    out, err, rc = run_cmd(ssh, with_ros(
        f"cd {NX_DIR} && timeout 8 build/stereo_pipeline "
        "--config config/pipeline_rdk_joint.yaml 2>&1"))
    if rc not in (0, 124):
        print(f"[ERROR] Pipeline startup test failed (rc={rc})")
        return False
    return True

def perf_test(ssh):
    """Performance benchmark"""
    print("\n=== Performance Test ===")
    run_cmd(ssh, "sudo nvpmodel -m 0 2>/dev/null; sudo jetson_clocks 2>/dev/null || true", False)
    out, err, rc = run_cmd(ssh, with_ros(
        f"cd {NX_DIR} && timeout 10 build/stereo_pipeline "
        "--config config/pipeline_rdk_joint.yaml 2>&1"))
    if rc not in (0, 124):
        print(f"[ERROR] Performance test failed (rc={rc})")
        return False
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
        if not test_dry_run(ssh):
            ssh.close()
            sys.exit(1)
    if action == "perf":
        if not perf_test(ssh):
            ssh.close()
            sys.exit(1)

    ssh.close()
    print("\n=== Done ===")

if __name__ == "__main__":
    main()
