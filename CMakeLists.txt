cmake_minimum_required(VERSION 3.20)
project(opensoda LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# TensorFlow Lite runtime. The punctuation model's two seq_flow_lite custom ops
# are implemented in src/punctuator.cc.
set(TFLITE_SOURCE_DIR "${CMAKE_SOURCE_DIR}/third_party/tensorflow/tensorflow/lite"
    CACHE PATH "Path to tensorflow/lite in a TensorFlow checkout")
# Several TFLite dependencies still declare cmake_minimum_required < 3.5.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "" FORCE)
set(ENV{CMAKE_POLICY_VERSION_MINIMUM} 3.5)
set(TFLITE_ENABLE_XNNPACK OFF CACHE BOOL "" FORCE)
# libsoda's TFLite kernels round every multiply and add separately; clang's
# default contraction into fused multiply-adds changes the encoder's outputs.
add_compile_options(-ffp-contract=off)
set(TFLITE_ENABLE_GPU OFF CACHE BOOL "" FORCE)
# libsoda's float matrix multiplies go through ruy, as do ours with this: the
# language identifier's outputs are then bit-identical to libsoda's.
set(TFLITE_ENABLE_RUY ON CACHE BOOL "" FORCE)
add_subdirectory("${TFLITE_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/tensorflow-lite" EXCLUDE_FROM_ALL)
# TFLite's CMake build leaves out the NEON kernels for 4-bit fully connected
# layers (the Bazel build has them), so en-US's models would run the scalar
# reference kernel, which is slower and rounds differently from libsoda's.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64" OR CMAKE_OSX_ARCHITECTURES MATCHES "arm64")
  set(TFLITE_4BIT_DIR "${TFLITE_SOURCE_DIR}/kernels/internal/optimized/4bit")
  target_sources(tensorflow-lite PRIVATE
    "${TFLITE_4BIT_DIR}/neon_fully_connected.cc"
    "${TFLITE_4BIT_DIR}/neon_fully_connected_aarch64_nosdot.cc"
    "${TFLITE_4BIT_DIR}/neon_fully_connected_aarch64_sdot.cc")
  target_compile_definitions(tensorflow-lite PRIVATE FC_4BIT_NEON)
  # They include "include/cpuinfo.h" from the cpuinfo checkout's root.
  target_include_directories(tensorflow-lite PRIVATE "${cpuinfo_SOURCE_DIR}")
endif()
# TFLite detects the Arm dot product instructions only on Linux; this
# replacement also detects them on macOS (see cmake/tflite_cpu_check.cc).
get_target_property(TFLITE_SOURCES tensorflow-lite SOURCES)
list(FILTER TFLITE_SOURCES EXCLUDE REGEX "kernels/internal/optimized/cpu_check\\.cc$")
set_property(TARGET tensorflow-lite PROPERTY SOURCES ${TFLITE_SOURCES})
target_sources(tensorflow-lite PRIVATE "${CMAKE_SOURCE_DIR}/cmake/tflite_cpu_check.cc")
# libsoda's TFLite uses a newer Eigen, whose float tanh rounds differently;
# the speech detectors' LSTM gates only match libsoda's bit for bit with it.
# Replace ptanh_float in the Eigen copy TFLite fetched (the newer Eigen as a
# whole does not build with this TFLite).
include(FetchContent)
FetchContent_GetProperties(eigen SOURCE_DIR EIGEN_DIR)
if(NOT EIGEN_DIR)
  set(EIGEN_DIR "${CMAKE_BINARY_DIR}/eigen")
endif()
set(EIGEN_MATH "${EIGEN_DIR}/Eigen/src/Core/arch/Default/GenericPacketMathFunctions.h")
file(READ "${EIGEN_MATH}" EIGEN_MATH_TEXT)
file(READ "${CMAKE_SOURCE_DIR}/cmake/eigen_ptanh_float.h" PTANH_FLOAT)
set(PTANH_START "template <typename T>\nEIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS T ptanh_float(const T& a_x) {")
string(FIND "${EIGEN_MATH_TEXT}" "${PTANH_START}" PTANH_BEGIN)
string(FIND "${EIGEN_MATH_TEXT}" "${PTANH_FLOAT}" PTANH_DONE)
if(PTANH_DONE EQUAL -1)
  if(PTANH_BEGIN EQUAL -1)
    message(FATAL_ERROR "ptanh_float not found in ${EIGEN_MATH}")
  endif()
  string(SUBSTRING "${EIGEN_MATH_TEXT}" ${PTANH_BEGIN} -1 PTANH_REST)
  string(FIND "${PTANH_REST}" "\n}\n" PTANH_LENGTH)
  math(EXPR PTANH_LENGTH "${PTANH_LENGTH} + 3")
  string(SUBSTRING "${EIGEN_MATH_TEXT}" 0 ${PTANH_BEGIN} PTANH_BEFORE)
  string(SUBSTRING "${PTANH_REST}" ${PTANH_LENGTH} -1 PTANH_AFTER)
  file(WRITE "${EIGEN_MATH}" "${PTANH_BEFORE}${PTANH_FLOAT}${PTANH_AFTER}")
endif()

# MARISA trie, used by SODA's compact symbol tables.
set(MARISA_SOURCE_DIR "${CMAKE_SOURCE_DIR}/third_party/marisa-trie"
    CACHE PATH "Path to a marisa-trie checkout")
set(ENABLE_TOOLS OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
add_subdirectory("${MARISA_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/marisa" EXCLUDE_FROM_ALL)

add_library(opensoda_core STATIC
  src/biasing.cc
  src/config.cc
  src/endpointer.cc
  src/frontend.cc
  src/fst_normalizer.cc
  src/language_packs.cc
  src/punctuator.cc
  src/roadrunner_capitalizer.cc
  src/spoken_punctuation.cc
  src/pqrnn_spoken_punctuation.cc
  src/unicode_case.cc
  src/truecaser.cc
  src/langid.cc
  src/recognizer.cc
  src/symbol_table.cc
  src/transducer.cc
  src/vad_frontend.cc
)
target_include_directories(opensoda_core PUBLIC src)
target_link_libraries(opensoda_core PUBLIC tensorflow-lite marisa)

# Drop-in replacement for libsoda's C API.
add_library(soda SHARED src/soda_api.cc)
target_include_directories(soda PUBLIC include)
target_link_libraries(soda PRIVATE opensoda_core)
set_target_properties(soda PROPERTIES CXX_VISIBILITY_PRESET hidden)

add_executable(soda_transcribe tools/soda_transcribe.cc)
target_link_libraries(soda_transcribe PRIVATE opensoda_core)

add_executable(soda_api_client tools/soda_api_client.cc)
target_include_directories(soda_api_client PRIVATE src)

add_executable(soda_punctuate tools/soda_punctuate.cc)
target_link_libraries(soda_punctuate PRIVATE opensoda_core)

add_executable(soda_capitalize tools/soda_capitalize.cc)
target_link_libraries(soda_capitalize PRIVATE opensoda_core)
