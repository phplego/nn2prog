#include "model.h"

#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>

#ifndef NN2PROG_FEATURE_COUNT
#error "NN2PROG_FEATURE_COUNT is required"
#endif

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: verify FEATURES.bin EXPECTED.bin\n";
    return 2;
  }
  std::ifstream feature_stream(argv[1], std::ios::binary);
  std::ifstream expected_stream(argv[2], std::ios::binary);
  std::vector<std::uint8_t> features((std::istreambuf_iterator<char>(feature_stream)), {});
  std::vector<std::uint8_t> expected((std::istreambuf_iterator<char>(expected_stream)), {});
  if (features.empty() || features.size() % NN2PROG_FEATURE_COUNT != 0 ||
      expected.size() != features.size() / NN2PROG_FEATURE_COUNT) {
    std::cerr << "invalid golden test data\n";
    return 1;
  }

  nn2prog::generated::Model model;
  std::array<std::int8_t, NN2PROG_FEATURE_COUNT> input{};
  std::size_t mismatches = 0;
  for (std::size_t frame = 0; frame < expected.size(); ++frame) {
    for (std::size_t index = 0; index < input.size(); ++index)
      input[index] = static_cast<std::int8_t>(features[frame * input.size() + index]);
    const auto actual = model.invoke(input);
    if (actual != expected[frame]) {
      if (mismatches < 5)
        std::cerr << "frame " << frame << ": expected " << +expected[frame]
                  << ", got " << +actual << '\n';
      ++mismatches;
    }
  }
  std::cout << "frames=" << expected.size() << " mismatches=" << mismatches << '\n';
  return mismatches == 0 ? 0 : 1;
}
