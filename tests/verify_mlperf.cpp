#include "model.h"
#include "input_protocol.h"

#include <cstdint>
#include <iostream>

int main() {
  nn2prog::generated::Model model;
  nn2prog::mlperf_kws::FrameGenerator generator;
  nn2prog::mlperf_kws::Frame frame;
  std::uint64_t checksum = 0;
  for (std::size_t i = 0; i < nn2prog::mlperf_kws::kFrames; ++i) {
    generator.next(i, frame);
    checksum = nn2prog::mlperf_kws::append_decision(checksum, model.invoke(frame));
  }
  const bool passed = checksum == nn2prog::mlperf_kws::kExpectedDecisionChecksum;
  std::cout << "frames=" << nn2prog::mlperf_kws::kFrames
            << " checksum=" << checksum << " status=" << (passed ? "PASS" : "FAIL") << '\n';
  return passed ? 0 : 1;
}
