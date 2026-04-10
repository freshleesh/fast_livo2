#!/usr/bin/env python3
"""Integration test for FAST-LIVO2 offline mapping and relocalization pipelines."""

import math
import os
import signal
import subprocess
import sys
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from nav_msgs.msg import Odometry

# ── Configuration ────────────────────────────────────────────────────────────
ROSBAG_PATH = "/media/leesh/T7/bag/rosbag2_2026_03_17-01_53_58"
NODE_STARTUP_DELAY = 3.0
POST_BAG_WAIT = 5.0
MIN_ODOM_MSGS = 10
MAX_CONSECUTIVE_JUMP = 5.0  # meters, for mapping mode
TRAJECTORY_TOLERANCE_RATIO = 0.10  # 10% of total path length, for reloc mode

LAUNCH_CMD = {
    "mapping": ["ros2", "launch", "fast_livo", "offline_mapping.launch.py", "use_rviz:=False"],
    "reloc": ["ros2", "launch", "fast_livo", "relocalization.launch.py", "use_rviz:=False"],
}
KILL_TARGETS = ["fastlivo_mapping"]
# ─────────────────────────────────────────────────────────────────────────────


class TopicMonitor(Node):
    def __init__(self):
        super().__init__("test_monitor")
        self.odom_msgs = []
        self.lock = threading.Lock()
        self.create_subscription(Odometry, "/aft_mapped_to_init", self._odom_cb, 10)

    def _odom_cb(self, msg):
        with self.lock:
            self.odom_msgs.append(msg)


def cleanup_nodes():
    for target in KILL_TARGETS:
        subprocess.run(["pkill", "-f", target], capture_output=True)
    time.sleep(1.0)
    # Force kill if still alive
    for target in KILL_TARGETS:
        subprocess.run(["pkill", "-9", "-f", target], capture_output=True)
    time.sleep(1.0)


def parse_trajectory_pcd(path):
    """Parse ASCII PCD file, return list of (x, y, z) tuples."""
    points = []
    in_data = False
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line == "DATA ascii":
                in_data = True
                continue
            if in_data and line:
                parts = line.split()
                points.append((float(parts[0]), float(parts[1]), float(parts[2])))
    return points


def path_length(points):
    total = 0.0
    for i in range(1, len(points)):
        dx = points[i][0] - points[i - 1][0]
        dy = points[i][1] - points[i - 1][1]
        dz = points[i][2] - points[i - 1][2]
        total += math.sqrt(dx * dx + dy * dy + dz * dz)
    return total


def nearest_distance(point, ref_points):
    min_d = float("inf")
    px, py, pz = point
    for rx, ry, rz in ref_points:
        d = math.sqrt((px - rx) ** 2 + (py - ry) ** 2 + (pz - rz) ** 2)
        if d < min_d:
            min_d = d
    return min_d


def get_map_path(mode):
    """Read the prior_map_dir from the relocalization config."""
    from ament_index_python.packages import get_package_share_directory
    pkg_dir = get_package_share_directory("fast_livo")
    config_name = {
        "mapping": "mid360_offline_mapping.yaml",
        "reloc": "mid360_relocalization.yaml",
    }[mode]
    search_key = "prior_map_dir" if mode == "reloc" else "map_save_path"
    config_path = os.path.join(pkg_dir, "config", config_name)
    with open(config_path, "r") as f:
        for line in f:
            if search_key in line:
                val = line.split('"')[1] if '"' in line else line.split("'")[1]
                return val
    return None


def odom_position(msg):
    p = msg.pose.pose.position
    return (p.x, p.y, p.z)


def is_nan_odom(msg):
    p = msg.pose.pose.position
    o = msg.pose.pose.orientation
    return any(math.isnan(v) for v in [p.x, p.y, p.z, o.x, o.y, o.z, o.w])


def quat_norm(msg):
    o = msg.pose.pose.orientation
    return math.sqrt(o.x ** 2 + o.y ** 2 + o.z ** 2 + o.w ** 2)


def dist(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def run_test(mode):
    print(f"\n{'='*60}")
    print(f"  FAST-LIVO2 Integration Test — {mode} mode")
    print(f"{'='*60}\n")

    # ── Pre-checks ────────────────────────────────────────────────
    if not os.path.isdir(ROSBAG_PATH):
        print(f"FAIL: rosbag not found at {ROSBAG_PATH}")
        return 1

    ref_trajectory = None
    ref_threshold = None
    if mode == "reloc":
        map_path = get_map_path(mode)
        if map_path is None:
            print("FAIL: could not read prior_map_dir from reloc config")
            return 1
        traj_pcd = os.path.join(map_path, "trajectory.pcd")
        if not os.path.isfile(traj_pcd):
            print(f"FAIL: reference trajectory not found at {traj_pcd}")
            return 1
        ref_trajectory = parse_trajectory_pcd(traj_pcd)
        total_length = path_length(ref_trajectory)
        ref_threshold = total_length * TRAJECTORY_TOLERANCE_RATIO
        print(f"Reference trajectory: {len(ref_trajectory)} points, "
              f"path length: {total_length:.3f}m, threshold: {ref_threshold:.3f}m")

    # ── Cleanup ───────────────────────────────────────────────────
    print("Cleaning up residual nodes...")
    cleanup_nodes()

    # ── Launch ────────────────────────────────────────────────────
    rclpy.init()
    monitor = TopicMonitor()
    executor = SingleThreadedExecutor()
    executor.add_node(monitor)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    print(f"Starting SLAM node ({mode})...")
    slam_proc = subprocess.Popen(
        LAUNCH_CMD[mode],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    time.sleep(NODE_STARTUP_DELAY)

    print("Playing rosbag...")
    bag_proc = subprocess.Popen(
        ["ros2", "bag", "play", ROSBAG_PATH],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    bag_proc.wait()
    print(f"Rosbag finished. Waiting {POST_BAG_WAIT}s for processing...")
    time.sleep(POST_BAG_WAIT)

    # ── Shutdown ──────────────────────────────────────────────────
    print("Sending SIGINT to SLAM node...")
    slam_proc.send_signal(signal.SIGINT)
    try:
        slam_stdout, _ = slam_proc.communicate(timeout=15)
    except subprocess.TimeoutExpired:
        slam_proc.kill()
        slam_stdout, _ = slam_proc.communicate()

    slam_output = slam_stdout.decode("utf-8", errors="replace")
    exit_code = slam_proc.returncode

    executor.shutdown()
    rclpy.shutdown()

    cleanup_nodes()

    # ── Evaluate ──────────────────────────────────────────────────
    results = {}
    details = {}

    # Check 1: No crash
    crash_codes = {-11: "SIGSEGV", -6: "SIGABRT", -4: "SIGILL"}
    if exit_code in crash_codes:
        results["no_crash"] = False
        details["no_crash"] = f"exit code {exit_code} ({crash_codes[exit_code]})"
    else:
        results["no_crash"] = True
        details["no_crash"] = f"exit code {exit_code}"

    # Check 2: Gravity alignment
    gravity_ok = "Gravity Alignment Finished" in slam_output
    results["gravity_aligned"] = gravity_ok
    details["gravity_aligned"] = "found" if gravity_ok else "NOT found in stdout"

    # Check 3: Odometry count
    odom_count = len(monitor.odom_msgs)
    results["odom_published"] = odom_count >= MIN_ODOM_MSGS
    details["odom_published"] = f"{odom_count} messages (min: {MIN_ODOM_MSGS})"

    # Check 4: Odometry sanity (NaN, quaternion)
    nan_indices = [i for i, m in enumerate(monitor.odom_msgs) if is_nan_odom(m)]
    bad_quat = [(i, quat_norm(m)) for i, m in enumerate(monitor.odom_msgs)
                if abs(quat_norm(m) - 1.0) > 0.01]
    sanity_ok = len(nan_indices) == 0 and len(bad_quat) == 0
    results["odom_sanity"] = sanity_ok
    if nan_indices:
        details["odom_sanity"] = f"NaN at frames: {nan_indices[:5]}"
    elif bad_quat:
        details["odom_sanity"] = f"bad quat norm at frames: {[(i, f'{n:.4f}') for i, n in bad_quat[:5]]}"
    else:
        details["odom_sanity"] = "OK"

    # Check 5: Mode-specific
    if mode == "mapping":
        # Consecutive frame jump check
        jumps = []
        for i in range(1, len(monitor.odom_msgs)):
            d = dist(odom_position(monitor.odom_msgs[i]),
                     odom_position(monitor.odom_msgs[i - 1]))
            if d >= MAX_CONSECUTIVE_JUMP:
                jumps.append((i, d))
        results["no_jumps"] = len(jumps) == 0
        if jumps:
            details["no_jumps"] = f"{len(jumps)} jumps >= {MAX_CONSECUTIVE_JUMP}m: {[(i, f'{d:.2f}m') for i, d in jumps[:5]]}"
        else:
            details["no_jumps"] = "OK"

    elif mode == "reloc":
        # Reference trajectory comparison
        max_dev = 0.0
        worst_idx = 0
        deviations = []
        for i, msg in enumerate(monitor.odom_msgs):
            pos = odom_position(msg)
            d = nearest_distance(pos, ref_trajectory)
            deviations.append(d)
            if d > max_dev:
                max_dev = d
                worst_idx = i
        traj_ok = max_dev <= ref_threshold if ref_threshold else False
        results["trajectory_accuracy"] = traj_ok
        avg_dev = sum(deviations) / len(deviations) if deviations else 0
        details["trajectory_accuracy"] = (
            f"max deviation: {max_dev:.3f}m (frame {worst_idx}), "
            f"avg: {avg_dev:.3f}m, threshold: {ref_threshold:.3f}m"
        )

    # ── Report ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Results")
    print(f"{'='*60}")
    all_passed = True
    for key, passed in results.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"  [{status}] {key}: {details[key]}")

    print(f"\n{'='*60}")
    if all_passed:
        print("  ALL CHECKS PASSED")
    else:
        print("  SOME CHECKS FAILED")
    print(f"{'='*60}\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "mapping"
    if mode not in ("mapping", "reloc"):
        print(f"Usage: {sys.argv[0]} [mapping|reloc]")
        sys.exit(1)
    sys.exit(run_test(mode))
