#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace nn2prog::mlperf_kws {

inline constexpr std::size_t kInputElements = 49 * 10;
inline constexpr std::size_t kFrames = 1024;
inline constexpr std::uint32_t kSeed = 0x4d4c5046u;
inline constexpr std::uint64_t kExpectedDecisionChecksum = 675379874863121749u;
using Frame = std::array<std::int8_t, kInputElements>;

class FrameGenerator {
 public:
  void next(std::size_t index, Frame& frame) {
    for (auto& value : frame) {
      state_ ^= state_ << 13;
      state_ ^= state_ >> 17;
      state_ ^= state_ << 5;
      value = static_cast<std::int8_t>(state_ & 0xffu);
    }
    if (index == 0) frame.fill(-128);
    if (index == 1) frame.fill(127);
    if (index == 2) frame.fill(83);
    if (index == 3) frame.fill(0);
  }

 private:
  std::uint32_t state_ = kSeed;
};

inline std::uint64_t append_decision(std::uint64_t checksum, std::uint8_t decision) {
  return checksum * 131u + decision;
}

}  // namespace nn2prog::mlperf_kws
