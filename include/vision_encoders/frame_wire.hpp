#pragma once

/// Self-describing multi-plane camera wire — one parser for every consumer.
///
/// The outer framing (u32 LE length + payload) is unchanged, so a producer
/// without depth stays byte-identical to the legacy stream. See readme.md §wire.
///
///   payload := <JPEG|PNG bytes>                 legacy, rgb only
///            | "VD1" u8 nplanes plane[nplanes]
///   plane   := u8 kind | u8 fmt | u16 w | u16 h | f32 fx,fy,cx,cy
///            | u32 nbytes | bytes
///
/// Intrinsics are per-plane, in THAT plane's own pixels, so a consumer never
/// needs a camera config and a downscaled depth plane stays exact.

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace vision_encoders
{
namespace wire
{

inline constexpr char MAGIC[3] = {'V', 'D', '1'};
inline constexpr size_t HEADER_BYTES = 4;    // magic + nplanes
inline constexpr size_t PLANE_BYTES = 26;    // kind..nbytes

enum class Kind : uint8_t { COLOR = 0, DEPTH = 1 };
enum class Fmt : uint8_t { RAW_U16_MM = 0, JPEG = 1, PNG = 2 };

/// One image plane. `data` points into the caller's payload — no copy, no owner.
struct Plane
{
  Kind kind = Kind::COLOR;
  Fmt fmt = Fmt::JPEG;
  uint16_t w = 0, h = 0;
  float fx = 0.f, fy = 0.f, cx = 0.f, cy = 0.f;  // 0 = producer sent none
  const uint8_t * data = nullptr;
  uint32_t nbytes = 0;

  bool has_intrinsics() const { return fx > 0.f && fy > 0.f; }
};

struct Frame
{
  std::vector<Plane> planes;

  const Plane * find(Kind k) const
  {
    for (const auto & p : planes) {
      if (p.kind == k) return &p;
    }
    return nullptr;
  }
};

namespace detail
{
template<typename T>
inline T rd(const uint8_t * p, size_t & off)
{
  T v;
  std::memcpy(&v, p + off, sizeof(T));   // fields are unaligned; hosts are LE
  off += sizeof(T);
  return v;
}

template<typename T>
inline void wr(std::vector<uint8_t> & out, T v)
{
  const auto * p = reinterpret_cast<const uint8_t *>(&v);
  out.insert(out.end(), p, p + sizeof(T));
}
}  // namespace detail

/// Parse one payload. A payload without the magic is a legacy codec blob and
/// yields a single COLOR plane with no intrinsics. False = malformed.
inline bool parse(const uint8_t * p, size_t n, Frame & out)
{
  out.planes.clear();
  if (n == 0) return false;
  if (n < HEADER_BYTES || std::memcmp(p, MAGIC, sizeof(MAGIC)) != 0) {
    out.planes.push_back(Plane{Kind::COLOR, Fmt::JPEG, 0, 0, 0.f, 0.f, 0.f, 0.f,
        p, static_cast<uint32_t>(n)});
    return true;
  }
  size_t off = sizeof(MAGIC);
  const auto count = detail::rd<uint8_t>(p, off);
  out.planes.reserve(count);
  for (uint8_t i = 0; i < count; ++i) {
    if (off + PLANE_BYTES > n) return false;
    Plane pl;
    pl.kind = static_cast<Kind>(detail::rd<uint8_t>(p, off));
    pl.fmt = static_cast<Fmt>(detail::rd<uint8_t>(p, off));
    pl.w = detail::rd<uint16_t>(p, off);
    pl.h = detail::rd<uint16_t>(p, off);
    pl.fx = detail::rd<float>(p, off);
    pl.fy = detail::rd<float>(p, off);
    pl.cx = detail::rd<float>(p, off);
    pl.cy = detail::rd<float>(p, off);
    pl.nbytes = detail::rd<uint32_t>(p, off);
    if (off + pl.nbytes > n) return false;
    pl.data = p + off;
    off += pl.nbytes;
    out.planes.push_back(pl);
  }
  return true;
}

inline bool parse(const std::vector<uint8_t> & payload, Frame & out)
{
  return parse(payload.data(), payload.size(), out);
}

/// Producer side: start a multi-plane payload, then append_plane() per plane.
inline std::vector<uint8_t> begin_payload(uint8_t nplanes)
{
  std::vector<uint8_t> out(MAGIC, MAGIC + sizeof(MAGIC));
  out.push_back(nplanes);
  return out;
}

inline void append_plane(
  std::vector<uint8_t> & out, Kind kind, Fmt fmt, uint16_t w, uint16_t h,
  float fx, float fy, float cx, float cy, const void * data, uint32_t nbytes)
{
  detail::wr<uint8_t>(out, static_cast<uint8_t>(kind));
  detail::wr<uint8_t>(out, static_cast<uint8_t>(fmt));
  detail::wr<uint16_t>(out, w);
  detail::wr<uint16_t>(out, h);
  detail::wr<float>(out, fx);
  detail::wr<float>(out, fy);
  detail::wr<float>(out, cx);
  detail::wr<float>(out, cy);
  detail::wr<uint32_t>(out, nbytes);
  const auto * b = static_cast<const uint8_t *>(data);
  out.insert(out.end(), b, b + nbytes);
}

}  // namespace wire
}  // namespace vision_encoders
