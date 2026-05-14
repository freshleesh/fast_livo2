// Binary container for serialized FAST-LIVO maps.
//
// One file holds an ordered sequence of independently-typed chunks (LIO
// voxel map, VIO feat map, ...). Each chunk is length-prefixed so an
// unknown type can be skipped, which keeps the format forward-compatible
// when we add new data later.
//
// Endianness is host (little-endian on the target platforms). Files are
// not portable across architectures with different endianness — fine for
// the closed loop "save and load on the same machine" usage; if we ever
// need to ship maps cross-platform, swap to a fixed byte order here.

#pragma once

#include <cstdint>
#include <fstream>
#include <string>

namespace fmap_io {

constexpr uint32_t kMagic = 0x4D324C46;  // "FL2M" in little-endian
// Version 2: VIO chunk embeds a per-Feature grayscale crop so warpAffine can
// produce real per-pose patches instead of the v1 degenerate path.
// Version 3: crop bumped 32 -> 64 (the 32x32 size truncated samples at the
// deepest pyramid levels and dominated the photometric residual, dragging
// pose into a wrong direction during localization).
constexpr uint32_t kVersion = 3;

enum ChunkType : uint32_t {
  kChunkLioVoxelMap = 0x01,
  kChunkVioFeatMap = 0x02,
};

struct GlobalHeader {
  uint32_t magic;
  uint32_t version;
  uint32_t chunk_count;
  uint32_t reserved;  // pad to 16 B
};

struct ChunkHeader {
  uint32_t type;
  uint32_t reserved;
  uint64_t length;  // payload bytes, excluding this header
};

// Write a chunk: header + payload (caller-supplied via the callable). Returns
// the offset of the chunk header so the caller can rewind to patch
// chunk_count later if needed. The callable must write exactly `length` bytes
// to the stream and return that count.
template <typename PayloadWriter>
inline bool writeChunk(std::ostream &os, uint32_t type, PayloadWriter w) {
  const auto header_pos = os.tellp();
  ChunkHeader h{type, 0, 0};
  os.write(reinterpret_cast<const char *>(&h), sizeof(h));
  if (!os) return false;
  const auto payload_begin = os.tellp();
  if (!w(os)) return false;
  const auto payload_end = os.tellp();
  h.length = static_cast<uint64_t>(payload_end - payload_begin);
  os.seekp(header_pos);
  os.write(reinterpret_cast<const char *>(&h), sizeof(h));
  os.seekp(payload_end);
  return static_cast<bool>(os);
}

inline bool writeGlobalHeader(std::ostream &os, uint32_t chunk_count) {
  GlobalHeader h{kMagic, kVersion, chunk_count, 0};
  os.write(reinterpret_cast<const char *>(&h), sizeof(h));
  return static_cast<bool>(os);
}

inline bool readGlobalHeader(std::istream &is, GlobalHeader &out) {
  is.read(reinterpret_cast<char *>(&out), sizeof(out));
  if (!is) return false;
  return out.magic == kMagic && out.version == kVersion;
}

inline bool readChunkHeader(std::istream &is, ChunkHeader &out) {
  is.read(reinterpret_cast<char *>(&out), sizeof(out));
  return static_cast<bool>(is);
}

}  // namespace fmap_io
