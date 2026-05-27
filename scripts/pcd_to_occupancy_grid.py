#!/usr/bin/env python3
"""Flatten a PCD map into a 2D occupancy-grid PGM (ROS map_server compatible).

Edit the CONFIG block below, then run:
    python3 pcd_to_occupancy_grid.py
"""

import os

import numpy as np
import open3d as o3d


# ============================== CONFIG ==============================
MAP_ROOT    = "/Users/mini/ros2_ws/src/IFAC2026_SH/src/slam/fast_livo2/map"  # parent of named map dirs
MAP_NAME    = "hall_0521_1"     # reads <MAP_ROOT>/<MAP_NAME>/cloudGlobal.pcd,
                           # writes  <MAP_ROOT>/<MAP_NAME>_2D/<MAP_NAME>.pgm
PCD_FILE    = "cloudGlobal.pcd"  # input PCD file name inside the map dir
Z_MIN       = -0.5          # meters (inclusive)
Z_MAX       = 2.0          # meters (inclusive)
RESOLUTION  = 0.05         # cell size in meters
MIN_POINTS  = 2            # min points per cell to mark occupied
# ====================================================================


def pcd_to_grid(points_xy, resolution, min_points):
    x_min, y_min = points_xy.min(axis=0) - resolution
    x_max, y_max = points_xy.max(axis=0) + resolution

    width = int(np.ceil((x_max - x_min) / resolution))
    height = int(np.ceil((y_max - y_min) / resolution))

    ix = np.floor((points_xy[:, 0] - x_min) / resolution).astype(np.int64)
    iy = np.floor((points_xy[:, 1] - y_min) / resolution).astype(np.int64)
    mask = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
    ix, iy = ix[mask], iy[mask]

    counts = np.zeros((height, width), dtype=np.int32)
    np.add.at(counts, (iy, ix), 1)

    # ROS map_server PGM convention: 0 = occupied (black), 254 = free, 205 = unknown.
    grid = np.full((height, width), 205, dtype=np.uint8)
    grid[counts >= min_points] = 0
    return grid, (x_min, y_min)


def write_pgm(path, grid):
    # PGM is stored with row 0 at the top; map_server expects the bottom-left
    # corner to correspond to (origin_x, origin_y), so flip vertically.
    flipped = np.flipud(grid)
    h, w = flipped.shape
    with open(path, "wb") as f:
        f.write(f"P5\n{w} {h}\n255\n".encode("ascii"))
        f.write(flipped.tobytes())


def write_yaml(path, pgm_name, resolution, origin):
    with open(path, "w") as f:
        f.write(f"image: {pgm_name}\n")
        f.write(f"resolution: {resolution}\n")
        f.write(f"origin: [{origin[0]:.6f}, {origin[1]:.6f}, 0.0]\n")
        f.write("negate: 0\n")
        f.write("occupied_thresh: 0.65\n")
        f.write("free_thresh: 0.196\n")


def main():
    if Z_MIN >= Z_MAX:
        raise SystemExit("Z_MIN must be < Z_MAX")
    if RESOLUTION <= 0:
        raise SystemExit("RESOLUTION must be > 0")
    if not MAP_NAME:
        raise SystemExit("MAP_NAME must be set")

    pcd_path = os.path.join(MAP_ROOT, MAP_NAME, PCD_FILE)
    out_dir = os.path.join(MAP_ROOT, f"{MAP_NAME}_2D")

    pcd = o3d.io.read_point_cloud(pcd_path)
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        raise SystemExit(f"No points in {pcd_path}")

    z_mask = (pts[:, 2] >= Z_MIN) & (pts[:, 2] <= Z_MAX)
    pts_in = pts[z_mask]
    if pts_in.size == 0:
        raise SystemExit(
            f"No points in z range [{Z_MIN}, {Z_MAX}] "
            f"(pcd z range: [{pts[:,2].min():.3f}, {pts[:,2].max():.3f}])")

    grid, origin = pcd_to_grid(pts_in[:, :2], RESOLUTION, MIN_POINTS)
    h, w = grid.shape
    occupied = int((grid == 0).sum())
    print(f"Loaded {len(pts)} points from {pcd_path}")
    print(f"{len(pts_in)} points in z-slice [{Z_MIN}, {Z_MAX}]")
    print(f"Grid: {w} x {h} cells @ {RESOLUTION} m, "
          f"origin = ({origin[0]:.3f}, {origin[1]:.3f})")
    print(f"Occupied cells: {occupied} ({100.0*occupied/(w*h):.2f}%)")

    os.makedirs(out_dir, exist_ok=True)
    pgm_path = os.path.join(out_dir, f"{MAP_NAME}.pgm")
    yaml_path = os.path.join(out_dir, f"{MAP_NAME}.yaml")
    write_pgm(pgm_path, grid)
    write_yaml(yaml_path, os.path.basename(pgm_path), RESOLUTION, origin)
    print(f"Wrote {pgm_path}\nWrote {yaml_path}")


if __name__ == "__main__":
    main()
