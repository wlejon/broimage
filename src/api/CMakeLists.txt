add_library(broimage_api STATIC
    api.cpp
    host_class.cpp
    native_image_codecs.cpp
    native_image_decode.cpp
    native_image_geometry.cpp
    native_image_ops.cpp
    native_image_preproc.cpp
)

target_include_directories(broimage_api PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
)

target_link_libraries(broimage_api PUBLIC
    broimage
    bronze_runtime_shared
)

target_compile_features(broimage_api PUBLIC cxx_std_20)

if(MSVC)
    target_compile_options(broimage_api PRIVATE /W4)
else()
    target_compile_options(broimage_api PRIVATE -Wall -Wextra)
endif()
