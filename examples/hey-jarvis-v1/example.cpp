#include "model.h"

#include <array>
#include <cstdint>
#include <iostream>

int main() {
  nn2prog::generated::Model model;
  std::array<std::int8_t, 40> features{};

  // A streaming model retains state, so feed several feature frames.
  // Zeroes are only an API demonstration, not silence PCM or an accuracy test.
  std::uint8_t score = 0;
  for (int frame = 0; frame < 10; ++frame) score = model.invoke(features);
  std::cout << "Hey Jarvis v1 score byte: " << static_cast<unsigned>(score) << '\n';
}
