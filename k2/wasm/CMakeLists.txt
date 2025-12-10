cmake_minimum_required(VERSION 3.10)
project(K2WASM)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CMakeLists.txt for K2 WASM

if(NOT EMSCRIPTEN)
  message(FATAL_ERROR "This directory is intended for Emscripten build only")
endif()

# Add stubs to include path (before k2 source to override)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/stubs)

# Add k2 root to include path
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../)

# Define K2 macros
# We do not define K2_WITH_CUDA so that it is undefined in C++.
# This triggers #ifndef K2_WITH_CUDA (for fake_cuda.h) and skips #ifdef K2_WITH_CUDA.

# Include transform.cmake from k2 root
include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/transform.cmake)

# Sources
set(K2_CUDA_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/../../k2/csrc/intersect_dense_pruned.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/../../k2/csrc/fsa_algo.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/../../k2/csrc/fsa_utils.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/../../k2/csrc/ragged_ops.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/../../k2/csrc/ragged_utils.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/overrides/array_ops.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/../../k2/csrc/algorithms.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/../../k2/csrc/context.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/../../k2/csrc/ragged.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/../../k2/csrc/utils.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/overrides/thread_pool.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/../../k2/csrc/math.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/../../k2/csrc/host_shim.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/../../k2/csrc/log.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/overrides/default_context.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/../../k2/csrc/fsa.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/../../k2/csrc/hash.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/../../k2/csrc/dtype.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/../../k2/csrc/tensor.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/../../k2/csrc/tensor_ops.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/../../k2/csrc/timer.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/../../k2/csrc/moderngpu_allocator.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/../../k2/csrc/array_of_ragged.cu
)

# Transform .cu to .cc
transform(OUTPUT_VARIABLE K2_CPU_SOURCES SRCS ${K2_CUDA_SOURCES})

set(K2_WASM_SOURCES
  ${K2_CPU_SOURCES}
  # Add other sources as needed by dependencies
  ${CMAKE_CURRENT_SOURCE_DIR}/intersect_dense_pruned_wasm.cc
)

# No need for set_source_files_properties LANGUAGE CXX since they are now .cc files

add_executable(k2-wasm ${K2_WASM_SOURCES})

# Link flags for Emscripten
set_target_properties(k2-wasm PROPERTIES LINK_FLAGS "-s WASM=1 -s ALLOW_MEMORY_GROWTH=1 -s DISABLE_EXCEPTION_CATCHING=0 -s EXPORTED_RUNTIME_METHODS='[\"ccall\",\"cwrap\",\"getValue\",\"setValue\",\"HEAP8\",\"HEAP16\",\"HEAP32\",\"HEAPF32\",\"HEAPF64\",\"getExceptionMessage\"]' -s EXPORTED_FUNCTIONS='[\"_malloc\",\"_free\"]' -lembind -s MODULARIZE=1")
