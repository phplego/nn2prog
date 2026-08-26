idf_component_register(
  SRCS "benchmark_main.cpp" "../../../generated/mlperf-kws/esp32/model.cpp"
  INCLUDE_DIRS "." "../../../generated/mlperf-kws/esp32" "../../../tests/mlperf-kws")
target_compile_definitions(${COMPONENT_LIB} PRIVATE NN2PROG_ENGINE_GENERATED=1)
target_compile_features(${COMPONENT_LIB} PRIVATE cxx_std_17)
