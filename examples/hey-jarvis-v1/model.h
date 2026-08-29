#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
namespace nn2prog::generated {
class Model {
 public:
  Model() { reset(); }
  void reset();
  std::uint8_t invoke(const std::array<std::int8_t,40>& input);
  static std::size_t working_memory_bytes();
 private:
  std::array<std::int8_t,200> state_stream_11_states{};
  std::array<std::int8_t,48> state_stream_12_states{};
  std::array<std::int8_t,48> state_stream_13_states{};
  std::array<std::int8_t,48> state_stream_14_states{};
  std::array<std::int8_t,96> state_stream_15_states{};
  std::array<std::int8_t,96> state_stream_16_states{};
  std::array<std::int8_t,96> state_stream_17_states{};
  std::array<std::int8_t,96> state_stream_18_states{};
  std::array<std::int8_t,96> state_stream_19_states{};
  std::array<std::int8_t,96> state_stream_20_states{};
  std::array<std::int8_t,7104> state_stream_21_states{};
};
}
