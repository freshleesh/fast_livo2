#!/usr/bin/env python3
"""
Convert iPhone SensorRecorder data to ROS2 bag compatible with FAST-LIVO2.

Topics created:
    /livox/lidar    sensor_msgs/PointCloud2  (depth → 3D point cloud, mid360-compatible)
    /livox/imu      sensor_msgs/Imu
    /image_raw      sensor_msgs/Image        (RGB, bgr8)

Usage:
    python convert_for_fastlivo.py slam.zip -o /path/to/output_bag
"""

import argparse
import struct
import sys
import io
from pathlib import Path

import numpy as np

try:
    from rosbags.rosbag2 import Writer
    from rosbags.typesys import get_typestore, Stores
    _store = get_typestore(Stores.ROS2_HUMBLE)
    serialize_cdr = _store.serialize_cdr

    Time = _store.types['builtin_interfaces/msg/Time']
    Header = _store.types['std_msgs/msg/Header']
    Imu = _store.types['sensor_msgs/msg/Imu']
    Image = _store.types['sensor_msgs/msg/Image']
    PointCloud2 = _store.types['sensor_msgs/msg/PointCloud2']
    PointField = _store.types['sensor_msgs/msg/PointField']
    Quaternion = _store.types['geometry_msgs/msg/Quaternion']
    Vector3 = _store.types['geometry_msgs/msg/Vector3']
except ImportError:
    print("Error: pip install rosbags numpy")
    sys.exit(1)

import os as _os
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
try:
    from rc_dataset import SensorRecording
except ImportError:
    from rc_dataset_reader import SensorRecording


def make_header(timestamp_ns: int, frame_id: str = "iphone") -> Header:
    sec = int(timestamp_ns // 1_000_000_000)
    nsec = int(timestamp_ns % 1_000_000_000)
    return Header(stamp=Time(sec=sec, nanosec=nsec), frame_id=frame_id)


def depth_to_pointcloud2(depth: np.ndarray, confidence: np.ndarray,
                         fx: float, fy: float, cx: float, cy: float,
                         depth_w: int, depth_h: int,
                         img_w: int, img_h: int,
                         timestamp_ns: int, conf_thresh: int = 1) -> PointCloud2:
    """Convert iPhone depth map to PointCloud2 with mid360-compatible fields.

    iPhone depth intrinsics differ from RGB intrinsics due to resolution difference.
    Depth: 256x192, RGB: 1920x1440. We scale the intrinsics accordingly.
    """
    # Scale intrinsics from RGB resolution to depth resolution
    scale_x = depth_w / img_w
    scale_y = depth_h / img_h
    dfx = fx * scale_x
    dfy = fy * scale_y
    dcx = cx * scale_x
    dcy = cy * scale_y

    # Create pixel grid
    v, u = np.mgrid[0:depth_h, 0:depth_w]  # v=row, u=col

    # Filter by confidence and valid depth
    mask = (confidence >= conf_thresh) & (depth > 0.05) & (depth < 20.0)

    u_valid = u[mask].astype(np.float32)
    v_valid = v[mask].astype(np.float32)
    d_valid = depth[mask].astype(np.float32)

    # Unproject to 3D (camera frame)
    # ARKit camera: image plane convention
    x_cam = (u_valid - dcx) * d_valid / dfx
    y_cam = (v_valid - dcy) * d_valid / dfy
    z_cam = d_valid

    # ARKit camera frame to LiDAR-like frame for FAST-LIVO2
    # ARKit: +X right, +Y down (in image), +Z forward (depth direction)
    # We want: +X forward, +Y left, +Z up (ROS LiDAR convention)
    x_lidar = z_cam          # forward
    y_lidar = -x_cam         # left
    z_lidar = -y_cam         # up

    n_points = len(x_lidar)

    # Build mid360-compatible point cloud: x, y, z, intensity, tag, line, timestamp
    # intensity: float32, tag: uint8, line: uint8, timestamp: float64
    # Field offsets: x(0), y(4), z(8), pad(12), intensity(16), tag(20), line(21), pad(22), timestamp(24)
    # Total point size: 32 bytes (to match mid360_ros::Point with alignment)

    # Actually, let's use standard PCL-compatible layout:
    # x(0,f32), y(4,f32), z(8,f32), intensity(12,f32) padding(16,f32),
    # tag(20,u8), line(21,u8), padding(22-23), timestamp(24,f64) = 32 bytes
    point_step = 32
    data = np.zeros(n_points * point_step, dtype=np.uint8)

    # Pack points
    for i in range(n_points):
        offset = i * point_step
        struct.pack_into('<fff', data, offset, x_lidar[i], y_lidar[i], z_lidar[i])
        struct.pack_into('<f', data, offset + 16, 100.0)  # intensity
        data[offset + 20] = 0  # tag
        data[offset + 21] = 0  # line
        struct.pack_into('<d', data, offset + 24, timestamp_ns / 1e9)  # timestamp as seconds

    fields = [
        PointField(name='x', offset=0, datatype=7, count=1),       # FLOAT32
        PointField(name='y', offset=4, datatype=7, count=1),
        PointField(name='z', offset=8, datatype=7, count=1),
        PointField(name='intensity', offset=16, datatype=7, count=1),
        PointField(name='tag', offset=20, datatype=2, count=1),     # UINT8
        PointField(name='line', offset=21, datatype=2, count=1),
        PointField(name='timestamp', offset=24, datatype=8, count=1), # FLOAT64
    ]

    header = make_header(timestamp_ns, "iphone_lidar")
    return PointCloud2(
        header=header,
        height=1,
        width=n_points,
        fields=fields,
        is_bigendian=False,
        point_step=point_step,
        row_step=point_step * n_points,
        data=data,
        is_dense=True,
    )


def depth_to_pointcloud2_fast(depth: np.ndarray, confidence: np.ndarray,
                              fx: float, fy: float, cx: float, cy: float,
                              depth_w: int, depth_h: int,
                              img_w: int, img_h: int,
                              timestamp_ns: int, conf_thresh: int = 1) -> PointCloud2:
    """Vectorized version - much faster than per-point packing."""
    scale_x = depth_w / img_w
    scale_y = depth_h / img_h
    dfx = fx * scale_x
    dfy = fy * scale_y
    dcx = cx * scale_x
    dcy = cy * scale_y

    v, u = np.mgrid[0:depth_h, 0:depth_w]
    mask = (confidence >= conf_thresh) & (depth > 0.05) & (depth < 20.0)

    u_valid = u[mask].astype(np.float32)
    v_valid = v[mask].astype(np.float32)
    d_valid = depth[mask].astype(np.float32)

    x_cam = (u_valid - dcx) * d_valid / dfx
    y_cam = (v_valid - dcy) * d_valid / dfy
    z_cam = d_valid

    # Keep points in camera frame (X-right, Y-down, Z-forward)
    # iPhone depth is already aligned to the wide camera by Apple's ML pipeline
    # Extrinsics in FAST-LIVO2 config handle the frame conversions

    n_points = len(x_cam)
    if n_points == 0:
        header = make_header(timestamp_ns, "iphone_lidar")
        return PointCloud2(
            header=header, height=1, width=0,
            fields=[], is_bigendian=False,
            point_step=32, row_step=0,
            data=np.array([], dtype=np.uint8),
            is_dense=True,
        )

    # L515-compatible format: PointXYZRGB (x,y,z,rgb as packed uint32)
    point_step = 16  # x(4) + y(4) + z(4) + rgb(4)

    # PCL PointXYZRGB stores rgb as float32 (reinterpreted from uint32)
    # Pack as uint32 then view as float32
    rgb_uint = np.full(n_points, 0x00C8C8C8, dtype=np.uint32)
    rgb_float = rgb_uint.view(np.float32)

    structured = np.zeros(n_points, dtype=[
        ('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('rgb', '<f4')
    ])
    structured['x'] = x_cam
    structured['y'] = y_cam
    structured['z'] = z_cam
    structured['rgb'] = rgb_float

    data = np.frombuffer(structured.tobytes(), dtype=np.uint8)

    fields = [
        PointField(name='x', offset=0, datatype=7, count=1),  # FLOAT32
        PointField(name='y', offset=4, datatype=7, count=1),
        PointField(name='z', offset=8, datatype=7, count=1),
        PointField(name='rgb', offset=12, datatype=7, count=1),  # FLOAT32
    ]

    header = make_header(timestamp_ns, "iphone_lidar")
    return PointCloud2(
        header=header,
        height=1,
        width=n_points,
        fields=fields,
        is_bigendian=False,
        point_step=point_step,
        row_step=point_step * n_points,
        data=data,
        is_dense=True,
    )


def read_imu_fixed(rec: SensorRecording):
    """Read IMU with correct record_size=60 (not 56 as in original reader)."""
    path = rec.path / 'imu.bin'
    if not path.exists():
        return
    data = path.read_bytes()
    record_size = 60  # 8(ts) + 12(raw_acc) + 12(user_acc) + 12(gyro) + 16(quat)
    for i in range(0, len(data), record_size):
        if i + record_size > len(data):
            break
        ts = struct.unpack_from('<d', data, i)[0]
        raw_acc = np.array(struct.unpack_from('<3f', data, i + 8))
        user_acc = np.array(struct.unpack_from('<3f', data, i + 20))
        gyro = np.array(struct.unpack_from('<3f', data, i + 32))
        quat = np.array(struct.unpack_from('<4f', data, i + 44))  # w, x, y, z
        yield ts, raw_acc, user_acc, gyro, quat


def convert(recording_path: str, output_path: str):
    rec = SensorRecording(recording_path)
    print(f"Recording: {rec.path.name}")
    print(f"Mode: {rec.metadata.get('mode', 'unknown')}, "
          f"Duration: {rec.metadata.get('duration_seconds', 0):.1f}s")

    intr = rec.intrinsics
    if not intr:
        print("ERROR: No camera intrinsics found!")
        sys.exit(1)

    fx, fy = intr['fx'], intr['fy']
    cx_img, cy_img = intr['cx'], intr['cy']
    img_w, img_h = intr['width'], intr['height']
    depth_res = rec.metadata.get('depth_resolution', {'width': 256, 'height': 192})
    depth_w, depth_h = depth_res['width'], depth_res['height']

    print(f"RGB: {img_w}x{img_h}, Depth: {depth_w}x{depth_h}")
    print(f"Intrinsics: fx={fx:.1f}, fy={fy:.1f}, cx={cx_img:.1f}, cy={cy_img:.1f}")

    # Pre-load confidence frames indexed by timestamp for matching with depth
    print("Loading confidence frames...")
    conf_map = {}
    for ts, conf in rec.confidence_frames():
        conf_map[f"{ts:.6f}"] = conf

    with Writer(Path(output_path), version=9) as writer:
        conns = {}
        conns['lidar'] = writer.add_connection('/livox/lidar', 'sensor_msgs/msg/PointCloud2', typestore=_store)
        conns['imu'] = writer.add_connection('/livox/imu', 'sensor_msgs/msg/Imu', typestore=_store)
        conns['image'] = writer.add_connection('/image_raw', 'sensor_msgs/msg/Image', typestore=_store)
        counts = {k: 0 for k in conns}

        # IMU (with fixed record size)
        # Keep IMU in CoreMotion frame: X-right, Y-up, Z-toward-user
        # Extrinsic in FAST-LIVO2 config handles the rotation to "lidar" (=camera) frame
        print("Converting IMU...", end=" ", flush=True)
        zero_cov = np.zeros(9, dtype=np.float64)
        for ts, raw_acc, user_acc, gyro, quat in read_imu_fixed(rec):
            ts_ns = rec.to_unix_ns(ts)
            acc_ms2 = raw_acc * 9.81  # g → m/s²

            msg = Imu(
                header=make_header(ts_ns, "iphone_imu"),
                orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
                orientation_covariance=np.full(9, -1.0, dtype=np.float64),
                angular_velocity=Vector3(
                    x=float(gyro[0]),
                    y=float(gyro[1]),
                    z=float(gyro[2])),
                angular_velocity_covariance=zero_cov.copy(),
                linear_acceleration=Vector3(
                    x=float(acc_ms2[0]),
                    y=float(acc_ms2[1]),
                    z=float(acc_ms2[2])),
                linear_acceleration_covariance=zero_cov.copy())
            writer.write(conns['imu'], ts_ns, serialize_cdr(msg, type(msg).__msgtype__))
            counts['imu'] += 1
        print(f"{counts['imu']} msgs")

        # Depth → PointCloud2
        print("Converting depth → PointCloud2...", end=" ", flush=True)
        for ts, depth_arr in rec.depth_frames():
            ts_ns = rec.to_unix_ns(ts)
            key = f"{ts:.6f}"
            conf = conf_map.get(key, np.full_like(depth_arr, 2, dtype=np.uint8))

            pc2_msg = depth_to_pointcloud2_fast(
                depth_arr, conf, fx, fy, cx_img, cy_img,
                depth_w, depth_h, img_w, img_h, ts_ns, conf_thresh=1)

            if pc2_msg.width > 0:
                writer.write(conns['lidar'], ts_ns, serialize_cdr(pc2_msg, type(pc2_msg).__msgtype__))
                counts['lidar'] += 1
        print(f"{counts['lidar']} msgs")

        # RGB rear → raw Image (bgr8)
        print("Converting RGB images...", end=" ", flush=True)
        from PIL import Image as PILImage
        for ts, jpeg_bytes in rec.rgb_rear():
            ts_ns = rec.to_unix_ns(ts)
            pil_img = PILImage.open(io.BytesIO(jpeg_bytes))
            # Resize to a reasonable resolution for FAST-LIVO2 (960x720)
            target_w, target_h = 960, 720
            pil_img = pil_img.resize((target_w, target_h), PILImage.LANCZOS)
            rgb_arr = np.array(pil_img)
            # Convert RGB to BGR for ROS
            bgr_arr = rgb_arr[:, :, ::-1].copy()

            msg = Image(
                header=make_header(ts_ns, "iphone_camera"),
                height=target_h,
                width=target_w,
                encoding='bgr8',
                is_bigendian=0,
                step=target_w * 3,
                data=np.frombuffer(bgr_arr.tobytes(), dtype=np.uint8))
            writer.write(conns['image'], ts_ns, serialize_cdr(msg, type(msg).__msgtype__))
            counts['image'] += 1
            if counts['image'] % 100 == 0:
                print(f"{counts['image']}...", end=" ", flush=True)
        print(f"{counts['image']} msgs")

    print(f"\nDone! Bag saved to: {output_path}")
    print(f"Total messages: {sum(counts.values())}")
    for k, v in counts.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert iPhone data to FAST-LIVO2 ROS2 bag")
    parser.add_argument("input", help="Recording directory or .zip file")
    parser.add_argument("-o", "--output", default="iphone_fastlivo_bag", help="Output bag directory")
    args = parser.parse_args()

    convert(args.input, args.output)
