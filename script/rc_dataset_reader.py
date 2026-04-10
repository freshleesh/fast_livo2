"""
Parses iPhone SensorRecorder recording directories.
"""
import json
import struct
import zipfile
import tempfile
from pathlib import Path
from typing import Iterator, Tuple, Optional

import numpy as np

# Register HEIF/HEIC plugin for Pillow
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass


class SensorRecording:
    """Reads a recording directory produced by the iOS SensorRecorder app."""

    def __init__(self, path: str):
        self.path = Path(path)
        if self.path.suffix == '.zip':
            self._tmpdir = tempfile.mkdtemp()
            with zipfile.ZipFile(self.path) as z:
                z.extractall(self._tmpdir)
            # Find the recording directory inside
            dirs = [d for d in Path(self._tmpdir).iterdir() if d.is_dir()]
            self.path = dirs[0] if dirs else Path(self._tmpdir)

        meta_path = self.path / 'metadata.json'
        if meta_path.exists():
            self.metadata = json.loads(meta_path.read_text())
        else:
            self.metadata = {}

    @property
    def intrinsics(self) -> dict:
        return self.metadata.get('intrinsics', {})

    @property
    def boot_to_unix_offset(self) -> float:
        return self.metadata.get('boot_to_unix_offset', 0.0)

    def to_unix(self, boot_ts: float) -> float:
        return boot_ts + self.boot_to_unix_offset

    def to_unix_ns(self, boot_ts: float) -> int:
        return int(self.to_unix(boot_ts) * 1e9)

    # ── Poses ──────────────────────────────────────────

    def poses(self) -> Iterator[Tuple[float, np.ndarray]]:
        """Yield (timestamp, 4x4_matrix) from poses.bin"""
        path = self.path / 'poses.bin'
        if not path.exists():
            return
        data = path.read_bytes()
        record_size = 72  # 8 + 64
        for i in range(0, len(data), record_size):
            if i + record_size > len(data):
                break
            ts = struct.unpack_from('<d', data, i)[0]
            values = struct.unpack_from('<16f', data, i + 8)
            matrix = np.array(values, dtype=np.float32).reshape(4, 4, order='F')
            yield ts, matrix

    # ── IMU ────────────────────────────────────────────

    def imu(self) -> Iterator[Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Yield (ts, raw_acc[3], user_acc[3], gyro[3], quaternion[4]) from imu.bin"""
        path = self.path / 'imu.bin'
        if not path.exists():
            return
        data = path.read_bytes()
        record_size = 56  # 8 + 12 + 12 + 12 + 16
        for i in range(0, len(data), record_size):
            if i + record_size > len(data):
                break
            ts = struct.unpack_from('<d', data, i)[0]
            raw_acc = np.array(struct.unpack_from('<3f', data, i + 8))
            user_acc = np.array(struct.unpack_from('<3f', data, i + 20))
            gyro = np.array(struct.unpack_from('<3f', data, i + 32))
            quat = np.array(struct.unpack_from('<4f', data, i + 44))  # w, x, y, z
            yield ts, raw_acc, user_acc, gyro, quat

    # ── RGB ────────────────────────────────────────────

    def rgb_rear(self) -> Iterator[Tuple[float, bytes]]:
        """Yield (timestamp, jpeg_bytes) for rear camera. HEIC auto-converted to JPEG."""
        yield from self._read_images('rgb_rear')

    def rgb_front(self) -> Iterator[Tuple[float, bytes]]:
        """Yield (timestamp, jpeg_bytes) for front camera. HEIC auto-converted to JPEG."""
        yield from self._read_images('rgb_front')

    def _read_images(self, subfolder: str) -> Iterator[Tuple[float, bytes]]:
        img_dir = self.path / subfolder
        index_path = self.path / f'{subfolder}_index.bin'
        if not img_dir.exists() or not index_path.exists():
            return
        index_data = index_path.read_bytes()
        entry_size = 12  # 8 (ts) + 4 (frame_no)
        for i in range(0, len(index_data), entry_size):
            if i + entry_size > len(index_data):
                break
            ts = struct.unpack_from('<d', index_data, i)[0]
            frame_no = struct.unpack_from('<I', index_data, i + 8)[0]

            # Try HEIC first, then JPEG fallback
            heic_path = img_dir / f'{frame_no:05d}.heic'
            jpg_path = img_dir / f'{frame_no:05d}.jpg'

            if heic_path.exists():
                jpeg_bytes = self._heic_to_jpeg(heic_path)
                if jpeg_bytes:
                    yield ts, jpeg_bytes
            elif jpg_path.exists():
                yield ts, jpg_path.read_bytes()

    @staticmethod
    def _heic_to_jpeg(heic_path) -> Optional[bytes]:
        """Convert HEIC to JPEG bytes using Pillow."""
        try:
            from PIL import Image as PILImage
            import io
            img = PILImage.open(heic_path)
            buf = io.BytesIO()
            img.save(buf, format='JPEG', quality=90)
            return buf.getvalue()
        except ImportError:
            # If Pillow not available, try reading raw bytes
            # (won't work for ROS2 CompressedImage but won't crash)
            return heic_path.read_bytes()
        except Exception:
            return None

    # ── Depth ──────────────────────────────────────────

    def depth_frames(self) -> Iterator[Tuple[float, np.ndarray]]:
        """Yield (timestamp, float32_array) from depth/*.bin"""
        yield from self._read_depth_files('depth', np.float32, 4)

    def confidence_frames(self) -> Iterator[Tuple[float, np.ndarray]]:
        """Yield (timestamp, uint8_array) from confidence/*.bin"""
        yield from self._read_depth_files('confidence', np.uint8, 1)

    def _read_depth_files(self, subfolder, dtype, bytes_per_pixel):
        depth_dir = self.path / subfolder
        if not depth_dir.exists():
            return
        for bin_path in sorted(depth_dir.glob('*.bin')):
            data = bin_path.read_bytes()
            w, h = struct.unpack_from('<2I', data, 0)
            ts = struct.unpack_from('<d', data, 8)[0]
            pixels = np.frombuffer(data[16:], dtype=dtype).reshape(h, w)
            yield ts, pixels

    # ── Feature Points ─────────────────────────────────

    def features(self) -> Iterator[Tuple[float, np.ndarray, np.ndarray]]:
        """Yield (timestamp, ids[N], points[N,3]) from features.bin"""
        path = self.path / 'features.bin'
        if not path.exists():
            return
        data = path.read_bytes()
        offset = 0
        while offset < len(data):
            if offset + 12 > len(data):
                break
            ts = struct.unpack_from('<d', data, offset)[0]
            count = struct.unpack_from('<I', data, offset + 8)[0]
            offset += 12
            entry_size = 20  # 8 (id) + 12 (xyz)
            if offset + count * entry_size > len(data):
                break
            ids = np.zeros(count, dtype=np.uint64)
            points = np.zeros((count, 3), dtype=np.float32)
            for j in range(count):
                ids[j] = struct.unpack_from('<Q', data, offset)[0]
                points[j] = struct.unpack_from('<3f', data, offset + 8)
                offset += entry_size
            yield ts, ids, points

    # ── Planes ─────────────────────────────────────────

    def planes(self) -> Iterator[dict]:
        """Yield plane dicts from planes.jsonl"""
        path = self.path / 'planes.jsonl'
        if not path.exists():
            return
        for line in path.read_text().strip().split('\n'):
            if line:
                yield json.loads(line)

    # ── Mesh ───────────────────────────────────────────

    def mesh_files(self) -> list:
        """Return list of PLY file paths."""
        mesh_dir = self.path / 'mesh'
        if not mesh_dir.exists():
            return []
        return sorted(mesh_dir.glob('*.ply'))

    # ── Magnetometer ───────────────────────────────────

    def magnetometer(self) -> Iterator[Tuple[float, np.ndarray]]:
        """Yield (timestamp, xyz[3]) from magnetometer.bin"""
        path = self.path / 'magnetometer.bin'
        if not path.exists():
            return
        data = path.read_bytes()
        record_size = 20  # 8 + 12
        for i in range(0, len(data), record_size):
            if i + record_size > len(data):
                break
            ts = struct.unpack_from('<d', data, i)[0]
            xyz = np.array(struct.unpack_from('<3f', data, i + 8))
            yield ts, xyz

    # ── GPS ─────────────────────────────────────────────

    def gps(self) -> Iterator[Tuple[float, float, float, float, float, float]]:
        """Yield (timestamp, lat, lon, alt, h_acc, v_acc) from gps.bin"""
        path = self.path / 'gps.bin'
        if not path.exists():
            return
        data = path.read_bytes()
        record_size = 40  # 8 + 8 + 8 + 8 + 4 + 4
        for i in range(0, len(data), record_size):
            if i + record_size > len(data):
                break
            ts = struct.unpack_from('<d', data, i)[0]
            lat, lon, alt = struct.unpack_from('<3d', data, i + 8)
            hacc, vacc = struct.unpack_from('<2f', data, i + 32)
            yield ts, lat, lon, alt, hacc, vacc
