cmake_minimum_required(VERSION 3.24)

# Pre-1.0: siblings vendor this repo via add_subdirectory and build it from
# source, so the version is a pin point for a tag, not an ABI promise.
project(broimage VERSION 0.1.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Coverage. Scoped to a top-level build: a consumer that add_subdirectory()s
# broimage has no business having its own build instrumented from in here.
option(BROIMAGE_COVERAGE "Instrument the build for gcov coverage (GCC/Clang)" OFF)
if(BROIMAGE_COVERAGE AND PROJECT_IS_TOP_LEVEL)
    if(MSVC)
        message(FATAL_ERROR
            "BROIMAGE_COVERAGE requires GCC or Clang (gcov instrumentation); MSVC is unsupported.")
    endif()
    add_compile_options(--coverage -O0 -g)
    add_link_options(--coverage)
    message(STATUS "broimage: coverage instrumentation ON (-O0)")
endif()

if(MSVC)
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:/MP>
        $<$<COMPILE_LANGUAGE:CXX>:/permissive->
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
    )
    if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    endif()
endif()

# --- brotensor dependency (optional) ---
#
# broimage's only use of brotensor is the tensor_adapter TU (image_normalize
# and the u8-NHWC -> f32-NCHW preprocessor), whose header merely forward-
# declares brotensor::Tensor. The other 11 source files operate on host
# pointers and never touch brotensor. So the whole brotensor dependency is
# gated: with BROIMAGE_WITH_TENSOR=OFF broimage builds CPU-only with no Tensor
# type at all, which is what a minimal `bro` build wants. Default ON preserves
# the full image-preprocessing surface for standalone and AI-enabled builds.
option(BROIMAGE_WITH_TENSOR "Build the brotensor adapter (image preprocessing to/from tensors)" ON)

# --- GPU backend selection ---
#
# broimage ships no GPU kernels of its own. GPU image preprocessing
# (image_normalize, u8_nhwc -> f32_nchw, future GPU resize) lives in
# brotensor — broimage forwards the backend choice so a standalone GPU build
# resolves brotensor's CUDA, Metal or Vulkan backend. Built inside bro,
# brotensor is already a target (added transitively) and the sibling lookup
# below reuses it, backend and all.
option(BROTENSOR_WITH_CUDA   "Build the CUDA GPU backend"   OFF)
option(BROTENSOR_WITH_METAL  "Build the Metal GPU backend"  OFF)
option(BROTENSOR_WITH_VULKAN "Build the Vulkan GPU backend" OFF)

# --- Dependencies ---
#
# Resolved by cmake/bro_deps.cmake, like every repo in the bro ecosystem: an
# existing target (a superbuild already added it, and its pins win) first;
# otherwise the ../<name> working tree beside the top-level project; otherwise
# the commit pinned here, fetched at configure. Leaves first, so these pins
# are the ones a standalone build uses.
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/bro_deps.cmake)

option(BROIMAGE_ENABLE_API "Build Bronze JavaScript API" ON)
# OFF builds the plain C++ kernels only, even when brass is present: what a
# consumer that wants decode/encode/resize (a thumbnailer) and not a JIT backend
# in its build passes.
option(BROIMAGE_WITH_JIT "Accelerate kernels with the brass JIT when brass is available" ON)

# bromath, header-only math.
bro_dependency(bromath GITHUB wlejon/bromath TARGET bromath::bromath)

# brass, the JIT backend for the kernels.
find_package(OpenMP QUIET)
if(BROIMAGE_WITH_JIT)
    bro_dependency(brass GITHUB wlejon/brass)
endif()

# bronze, the JavaScript runtime the api binds (it brings brass too). The api
# is bound across bronze's C++ embed boundary, so it has to build against the
# bronze the surrounding tree builds against.
if(BROIMAGE_ENABLE_API)
    bro_dependency(bronze GITHUB wlejon/bronze TARGET bronze_runtime_shared
                   OPTIONS BRONZE_BUILD_SHARED_RUNTIME=ON BRONZE_BUILD_TESTS=OFF)
endif()

# brotensor, for the tensor adapter.
if(BROIMAGE_WITH_TENSOR)
    bro_dependency(brotensor GITHUB wlejon/brotensor)
endif()


# --- stb_image (vendored: decode + encode) ---
#
# Bundled because broimage is the home for image decode/encode in the stack;
# the rest of the bro tree should depend on broimage for these, not roll its
# own stb wrapper. Guarded so a parent build that already declared an
# stb_image target (bro/third_party/CMakeLists.txt does) reuses it.
if(NOT TARGET stb_image)
    add_library(stb_image STATIC
        third_party/stb/stb_image.cpp
        third_party/stb/stb_image_write.cpp
    )
    target_include_directories(stb_image PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/stb
    )
endif()

# --- basis_universal transcoder (vendored: KTX2 -> GPU formats) ---
#
# Transcode-only slice of BinomialLLC/basis_universal (transcoder/ plus the
# decompression-only zstd single file for Zstd-supercompressed KTX2). The
# encoder is deliberately absent — authoring .ktx2 belongs to offline tools.
if(NOT TARGET basisu_transcoder AND
   EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/basis_universal/transcoder/basisu_transcoder.cpp")
    # zstddeclib.c is C, and this project() is CXX-only: without the language
    # enabled, Ninja SILENTLY drops the file from the target and every ZSTD_*
    # symbol goes undefined at link. The bro superbuild masks it (bro's own
    # project() enables C), so this line is what keeps the standalone build
    # honest.
    enable_language(C)
    add_library(basisu_transcoder STATIC
        third_party/basis_universal/transcoder/basisu_transcoder.cpp
        third_party/basis_universal/zstd/zstddeclib.c
    )
    target_include_directories(basisu_transcoder PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/basis_universal/transcoder
    )
    if(MSVC)
        target_compile_options(basisu_transcoder PRIVATE /W0)
    else()
        target_compile_options(basisu_transcoder PRIVATE -w)
    endif()
endif()
if(NOT TARGET basisu_transcoder)
    message(STATUS "broimage: basis_universal not found, KTX2 transcode disabled")
endif()

# --- broimage library ---

add_library(broimage STATIC)
add_library(broimage::broimage ALIAS broimage)

target_include_directories(broimage PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(broimage PUBLIC cxx_std_20)
target_link_libraries(broimage
    PUBLIC
        bromath::bromath
    PRIVATE
        stb_image
)
if(BROIMAGE_WITH_TENSOR)
    target_link_libraries(broimage PUBLIC brotensor::brotensor)
endif()

if(MSVC)
    target_compile_options(broimage PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4>)
else()
    target_compile_options(broimage PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall>
        $<$<COMPILE_LANGUAGE:CXX>:-Wextra>)
endif()

target_sources(broimage PRIVATE
    src/version.cpp
    src/decode.cpp
    src/decode_bounded.cpp
    src/decode_tiff.cpp
    src/encode.cpp
    src/png_text.cpp
    src/color.cpp
    src/geometric.cpp
    src/alpha.cpp
    src/normalize.cpp
    src/preproc.cpp
    src/presets.cpp
    src/kernels.cpp
    src/tiling.cpp
)
if(BROIMAGE_WITH_TENSOR)
    target_sources(broimage PRIVATE src/tensor_adapter.cpp)
endif()
if(TARGET basisu_transcoder)
    target_sources(broimage PRIVATE src/ktx2.cpp)
    target_link_libraries(broimage PRIVATE basisu_transcoder)
    target_compile_definitions(broimage PUBLIC BROIMAGE_HAS_KTX2=1)
endif()
if(BROIMAGE_WITH_JIT AND TARGET brass)
    target_link_libraries(broimage PUBLIC brass)
    target_compile_definitions(broimage PUBLIC BROIMAGE_HAS_BRASS_JIT=1)
    target_sources(broimage PRIVATE
        src/jit/image_pipeline.cpp
        src/jit/jit_image_compiler.cpp
    )
    if(OpenMP_CXX_FOUND)
        target_link_libraries(broimage PUBLIC OpenMP::OpenMP_CXX)
    endif()
endif()
# --- Standalone Bronze JavaScript API ---
# (BROIMAGE_ENABLE_API is declared with the dependencies above, which add bronze.)
if(BROIMAGE_ENABLE_API)
    add_subdirectory(src/api)
endif()

# --- Install ---
#
# Off by default: broimage is consumed as a sibling repo via the same
# add_subdirectory pattern bro uses for its other libraries, not via
# find_package. See brolm/CMakeLists.txt for the longer explanation of why
# no CMake package config is generated.
option(BROIMAGE_INSTALL "Install the broimage static library and public headers" OFF)
if(BROIMAGE_INSTALL)
    include(GNUInstallDirs)
    install(TARGETS broimage
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
    install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
endif()

# --- Tests ---
option(BROIMAGE_TESTS "Build broimage tests" ON)
if(BROIMAGE_TESTS AND CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    enable_testing()
    add_subdirectory(tests)
endif()
