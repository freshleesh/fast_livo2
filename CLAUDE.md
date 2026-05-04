# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build

This is a ROS2 (Jazzy on macOS arm64) colcon package. The package name is `fast_livo` (not `fast_livo2`).

### Dependencies (one-time setup)

```bash
# Sophus + GTSAM via conda-forge
conda install -n ros_env -c conda-forge sophus "gtsam=4.2.0"

# vikit_common — build with plain cmake (colcon doesn't propagate conda includes correctly)
mkdir -p /tmp/build_vikit_common
cmake -S ~/ros2_ws/src/rpg_vikit/vikit_common -B /tmp/build_vikit_common \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=~/ros2_ws/install/vikit_common \
  -DCMAKE_PREFIX_PATH=/Users/mini/miniconda3/envs/ros_env \
  -DSophus_DIR=/Users/mini/miniconda3/envs/ros_env/share/sophus/cmake
make -C /tmp/build_vikit_common -j$(sysctl -n hw.ncpu) install

# vikit_ros — build with colcon using the build script
bash ~/ros2_ws/build_vikit_ros.sh
```

### Building fast_livo

Use the build script at `~/ros2_ws/build_fastlivo.sh`:

```bash
conda run -n ros_env bash ~/ros2_ws/build_fastlivo.sh
```

The script passes the required cmake args (Python path, Sophus dir, GTSAM dir, conda include path). Build type is hardcoded to `Release` with `-O3`. To switch to Debug, edit [CMakeLists.txt](CMakeLists.txt) lines 7–9.

### macOS-specific patches applied to this repo

The following changes were made to support macOS arm64 (Apple Silicon) with conda ROS Jazzy:

- **[CMakeLists.txt](CMakeLists.txt)**: `arm64` added to the 64-bit ARM branch (was only matching `aarch64`); removed x86 SSE flags; replaced hardcoded `.so` vikit path with `${CMAKE_SHARED_LIBRARY_SUFFIX}`; `stdc++fs` made GCC-conditional; added `-undefined dynamic_lookup` for all intermediate dylibs
- **[include/common_lib.h](include/common_lib.h)**: `<experimental/filesystem>` guarded with `#ifndef __APPLE__`
- **[include/utils/types.h](include/utils/types.h)**: `EIGEN_ALIGN16` moved to prefix position (Clang doesn't support it as struct suffix)
- **[include/multi-session/Incremental_mapping.hpp](include/multi-session/Incremental_mapping.hpp)** and `.cpp`: `std::experimental::optional/nullopt` → `std::optional/nullopt`

## Launch

Each launch takes two YAML files: a main param file and a camera param file. The `use_rviz` arg defaults to `False`.

```bash
# Offline mapping (replay rosbag)
ros2 launch fast_livo offline_mapping.launch.py use_rviz:=True

# Online LIVO (live sensor stream)
ros2 launch fast_livo online_livo.launch.py use_rviz:=True

# Relocalization against a saved prior map
ros2 launch fast_livo relocalization.launch.py use_rviz:=True

# Multi-session map merging
ros2 launch fast_livo multi_session.launch.py
```

Save the map after mapping:
```bash
ros2 service call /fast_livo/save_map fast_livo/srv/SaveMap
```

## Integration Test

```bash
# Requires a rosbag at the path set in test/test_integration.py (ROSBAG_PATH)
python3 test/test_integration.py mapping
python3 test/test_integration.py reloc
```

The test launches the node, plays the bag, then checks: no crash, gravity alignment completed, odometry published without NaN/bad quaternions, and mode-specific trajectory checks.

## Architecture

The system is a tightly-coupled LiDAR-Inertial-Visual odometry and mapping pipeline. Three SLAM modes are selected via YAML: `ONLY_LO=0`, `ONLY_LIO=1`, `LIVO=2` (controlled by `img_en` and `lidar_en` params).

### Main Components

**`LIVMapper`** ([include/LIVMapper.h](include/LIVMapper.h), [src/LIVMapper.cpp](src/LIVMapper.cpp))
The top-level orchestrator. Owns all subsystems and drives the main processing loop in `LIVMapper::run()`. Handles:
- Sensor synchronization (`sync_packages()`) — merges LiDAR + IMU + image into `LidarMeasureGroup`
- State estimation dispatch: `handleLIO()` for LiDAR-Inertial, `handleVIO()` for Visual-Inertial
- Pose graph optimization (GTSAM iSAM2), loop closure via `loopClosureThread()`
- Map saving, PCD publishing, TF broadcasting

**`VIOManager`** ([include/vio.h](include/vio.h), [src/vio.cpp](src/vio.cpp))
Sparse-direct photometric visual odometry. Maintains a `feat_map` (hash map of `VOXEL_LOCATION → VOXEL_POINTS*`) of 3D `VisualPoint`s. For each image, retrieves nearby points from the voxel map into a `SubSparseMap`, computes affine-warped patches, and runs ESIKF updates (`updateState()` / `updateStateInverse()`).

**`VoxelMapManager`** ([include/voxel_map.h](include/voxel_map.h), [src/voxel_map.cpp](src/voxel_map.cpp))
Incremental probabilistic voxel map for LIO. Voxels contain octree-structured `VoxelOctoTree` planes; new LiDAR points are matched to local planes via an ESIKF (`PointToPlane` residuals). Supports sliding window map management.

**`ImuProcess`** ([include/IMU_Processing.h](include/IMU_Processing.h), [src/IMU_Processing.cpp](src/IMU_Processing.cpp))
IMU pre-integration between LiDAR scans, point-cloud undistortion, and ESIKF forward-propagation. State vector is 19-dimensional (`DIM_STATE`): rotation (SO3), position, velocity, angular velocity, gyro bias, acc bias, inverse exposure time.

**`Preprocess`** ([include/preprocess.h](include/preprocess.h), [src/preprocess.cpp](src/preprocess.cpp))
LiDAR input adapter. Handles both Livox `CustomMsg` (types AVIA=1, MID360=7) and standard `PointCloud2` (Velodyne, Ouster, etc.). Selects lidar type via `lidar_type` YAML param.

**Auxiliary subsystems:**
- `ZUPT` ([src/zupt.cpp](src/zupt.cpp)) — Zero Velocity Update pseudo-measurement
- `WheelOdometryConstraint` ([include/wheel_odometry.h](include/wheel_odometry.h)) — optional wheel odometry factor
- `SCManager` ([include/sc-relo/Scancontext.h](include/sc-relo/Scancontext.h)) — Scan Context descriptor for loop closure candidate detection
- `ERASOR` ([include/ground_detection.h](include/ground_detection.h)) — dynamic object removal / ground detection
- `FRICP-toolkit` ([include/FRICP-toolkit/](include/FRICP-toolkit/)) — Fast Robust ICP for loop closure alignment

### Threading Model

- **Main thread**: `LIVMapper::run()` — blocks on `sig_buffer` condition variable waiting for new sensor data
- **Callback groups**: separate `rclcpp::CallbackGroup` for LiDAR, IMU, image, and odometry subscriptions (parallel dispatch)
- **Loop closure thread**: `loopClosureThread()` runs continuously in background
- **IMU propagation timer**: publishes high-rate IMU-propagated odometry between LiDAR updates
- **Visualization timer** (`viz_timer_`): drains `viz_queue_` to decouple heavy publish work from the estimation thread

### Configuration

Each sensor/platform combination has its own YAML pair in [config/](config/):
- Main params file (e.g., `mid360_offline_mapping.yaml`) — covers `common`, `extrin_calib`, `preprocess`, `vio`, `imu`, `lio`, `pgo`, `localization` sections
- Camera params file (e.g., `camera_see3cam.yaml`) — camera intrinsics loaded by `vikit_ros`

Key extrinsics:
- `extrinsic_T/R`: IMU-to-LiDAR
- `Rcl/Pcl`: LiDAR-to-Camera (rotation matrix + translation)

### Code Conventions

- Eigen matrix/vector typedefs (`M3D`, `V3D`, `MD`, etc.) are defined in [include/utils/types.h](include/utils/types.h)
- `using namespace Sophus` and `using namespace std` are active in most translation units
- `PRE_ROS_IRON` compile-time macro selects the correct `cv_bridge` header (`<cv_bridge/cv_bridge.h>` vs `.hpp`)
- `ARM_ARCH` / `X86_ARCH` macros are set automatically by CMake based on detected CPU
- OpenMP parallelism is gated on `MP_EN` (auto-set when >1 CPU core detected, capped at 8)
- Clang-format config is at [clang-format](clang-format) (no leading dot — pass explicitly: `clang-format -style=file:clang-format`)
