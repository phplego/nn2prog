#include "input_protocol.h"

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>

#include "esp_cpu.h"
#include "esp_heap_caps.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#if defined(NN2PROG_ENGINE_TFLM)
#include "model_data.h"
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/schema/schema_generated.h>
#elif defined(NN2PROG_ENGINE_GENERATED)
#include "model.h"
#else
#error Select exactly one benchmark engine
#endif

namespace {
constexpr int kTrials = 3;
constexpr int kWarmupFrames = 8;
constexpr std::size_t kBenchmarkFrames = 64;
constexpr std::uint64_t kExpectedBenchmarkChecksum = 17774897695124712843u;
std::array<std::uint32_t, kBenchmarkFrames> cycle_samples;

#if defined(NN2PROG_ENGINE_TFLM)
constexpr std::size_t kClasses = 12;
std::uint8_t first_argmax(const std::int8_t* values) {
  std::uint8_t result = 0;
  for (std::size_t i = 1; i < kClasses; ++i)
    if (values[i] > values[result]) result = static_cast<std::uint8_t>(i);
  return result;
}
#endif

#if defined(NN2PROG_ENGINE_TFLM)
class Engine {
 public:
  bool initialize() {
    if (resolver_.AddConv2D() != kTfLiteOk || resolver_.AddDepthwiseConv2D() != kTfLiteOk ||
        resolver_.AddAveragePool2D() != kTfLiteOk || resolver_.AddReshape() != kTfLiteOk ||
        resolver_.AddFullyConnected() != kTfLiteOk || resolver_.AddSoftmax() != kTfLiteOk)
      return false;
    interpreter_ = new (interpreter_storage_.data()) tflite::MicroInterpreter(
        tflite::GetModel(g_mlperf_kws_model), resolver_, arena_.data(), arena_.size());
    return interpreter_->AllocateTensors() == kTfLiteOk && interpreter_->input(0)->type == kTfLiteInt8 &&
           interpreter_->output(0)->type == kTfLiteInt8;
  }

  std::uint8_t invoke(const nn2prog::mlperf_kws::Frame& frame) {
    std::memcpy(interpreter_->input(0)->data.int8, frame.data(), frame.size());
    if (interpreter_->Invoke() != kTfLiteOk) std::abort();
    return first_argmax(interpreter_->output(0)->data.int8);
  }

  std::size_t working_bytes() const { return interpreter_->arena_used_bytes(); }
  static constexpr const char* name() { return "tflite_micro_esp_nn"; }

 private:
  tflite::MicroMutableOpResolver<6> resolver_;
  alignas(16) std::array<std::uint8_t, 32 * 1024> arena_{};
  alignas(tflite::MicroInterpreter)
      std::array<std::uint8_t, sizeof(tflite::MicroInterpreter)> interpreter_storage_{};
  tflite::MicroInterpreter* interpreter_ = nullptr;
};
#else
class Engine {
 public:
  bool initialize() { return true; }
  std::uint8_t invoke(const nn2prog::mlperf_kws::Frame& frame) { return model_.invoke(frame); }
  static std::size_t working_bytes() { return nn2prog::generated::Model::working_memory_bytes(); }
  static constexpr const char* name() { return "nn2prog"; }

 private:
  nn2prog::generated::Model model_;
};
#endif

bool run_trial(Engine& engine, int trial) {
  nn2prog::mlperf_kws::FrameGenerator generator;
  nn2prog::mlperf_kws::Frame frame;
  std::uint64_t checksum = 0;
  std::uint64_t total = 0;
  for (std::size_t i = 0; i < kBenchmarkFrames; ++i) {
    generator.next(i, frame);
    const std::uint32_t begin = esp_cpu_get_cycle_count();
    const std::uint8_t decision = engine.invoke(frame);
    const std::uint32_t elapsed = esp_cpu_get_cycle_count() - begin;
    cycle_samples[i] = elapsed;
    total += elapsed;
    checksum = nn2prog::mlperf_kws::append_decision(checksum, decision);
    // Keep the watchdog and idle task healthy. This yield is outside the timed
    // region and is identical for both engines.
    vTaskDelay(1);
  }
  std::sort(cycle_samples.begin(), cycle_samples.end());
  const auto percentile = [](std::size_t numerator, std::size_t denominator) {
    return cycle_samples[(cycle_samples.size() - 1) * numerator / denominator];
  };
  const bool passed = checksum == kExpectedBenchmarkChecksum;
  std::printf(
      "result,%s,%d,%zu,%" PRIu64 ",%" PRIu64 ",%" PRIu32 ",%" PRIu32 ",%" PRIu32
      ",%" PRIu32 ",%" PRIu32 ",%" PRIu64 ",%s\n",
      Engine::name(), trial, cycle_samples.size(), total, total / cycle_samples.size(),
      percentile(50, 100), percentile(95, 100), percentile(99, 100), cycle_samples.front(),
      cycle_samples.back(), checksum,
      passed ? "PASS" : "FAIL");
  return passed;
}
}  // namespace

extern "C" void app_main() {
  static Engine engine;
  std::printf("nn2prog_mlperf_kws_esp32,1\n");
  std::printf("engine,%s\n", Engine::name());
  if (!engine.initialize()) {
    std::printf("initialization,FAIL\n");
    return;
  }
  nn2prog::mlperf_kws::FrameGenerator warmup_generator;
  nn2prog::mlperf_kws::Frame warmup_frame;
  for (int i = 0; i < kWarmupFrames; ++i) {
    warmup_generator.next(i, warmup_frame);
    engine.invoke(warmup_frame);
  }
  std::printf("initialization,PASS\n");
  std::printf("working_bytes,%zu\n", engine.working_bytes());
  std::printf("free_internal_heap_bytes,%zu\n",
              heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
  std::printf("columns,engine,trial,frames,total_cycles,mean_cycles,p50_cycles,p95_cycles,"
              "p99_cycles,min_cycles,max_cycles,checksum,status\n");
  bool passed = true;
  for (int trial = 0; trial < kTrials; ++trial) passed = run_trial(engine, trial) && passed;
  std::printf("stack_high_water_bytes,%zu\n",
              static_cast<std::size_t>(uxTaskGetStackHighWaterMark(nullptr)));
  std::printf("benchmark_complete,%s\n", passed ? "PASS" : "FAIL");
}
