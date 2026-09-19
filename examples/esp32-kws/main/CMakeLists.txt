idf_component_register(
  SRCS "benchmark_main.cpp" "../../../generated/mlperf-kws/${IDF_TARGET}/model.cpp"
  INCLUDE_DIRS "." "../../../generated/mlperf-kws/${IDF_TARGET}" "../../../tests/mlperf-kws")
target_compile_definitions(${COMPONENT_LIB} PRIVATE NN2PROG_ENGINE_GENERATED=1)
target_compile_features(${COMPONENT_LIB} PRIVATE cxx_std_17)
if(IDF_TARGET STREQUAL "esp32s3")
  target_compile_options(${COMPONENT_LIB} PRIVATE -O3)
endif()
