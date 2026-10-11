# libjpeg-turbo, broimage's JPEG decoder (BSD-3 / IJG).
#
# stb_image decodes a JPEG scalar, single-threaded and always at full size;
# libjpeg-turbo decodes it with SIMD and, through the IDCT, at 1/8..8/8 of its
# size, which is what a thumbnail or a screen-sized view of a 24 MP photo wants.
#
# Resolved like bro's other heavy dependencies: an installed copy first (the
# system package on Linux, Homebrew on macOS, vcpkg on Windows when bro's
# toolchain is active), otherwise the pinned release built from source.
# BROIMAGE_FETCH_JPEG_TURBO=ON skips the search and builds the pinned source.
#
# Defines broimage_jpeg_turbo (an INTERFACE target to link) or, with
# BROIMAGE_WITH_JPEG_TURBO=OFF, nothing: JPEG then stays with stb_image and a
# scaled decode is a full decode and a resize.

option(BROIMAGE_WITH_JPEG_TURBO "Decode JPEG with libjpeg-turbo (SIMD, DCT-scaled decode); OFF leaves it to stb_image" ON)
option(BROIMAGE_FETCH_JPEG_TURBO "Build libjpeg-turbo from its pinned source instead of using an installed one" OFF)

# 3.2.0. Moving it: the tag's commit here, and in bro's cmake/bro_pins.cmake.
set(BROIMAGE_JPEG_TURBO_REF c85e6b905bf237038faa936dab160ebfc5da0344)

if(NOT BROIMAGE_WITH_JPEG_TURBO OR TARGET broimage_jpeg_turbo)
    return()
endif()

add_library(broimage_jpeg_turbo INTERFACE)
set(_jt_how "")

if(NOT BROIMAGE_FETCH_JPEG_TURBO)
    # Homebrew's prefix is not on CMake's search path unless its bin is on PATH.
    set(_jt_hints "")
    if(APPLE)
        list(APPEND _jt_hints /opt/homebrew /usr/local /opt/homebrew/opt/jpeg-turbo /usr/local/opt/jpeg-turbo)
    endif()
    find_package(libjpeg-turbo CONFIG QUIET HINTS ${_jt_hints})
    if(TARGET libjpeg-turbo::jpeg)
        target_link_libraries(broimage_jpeg_turbo INTERFACE libjpeg-turbo::jpeg)
        set(_jt_how "installed ${libjpeg-turbo_VERSION} (${libjpeg-turbo_DIR})")
    elseif(TARGET libjpeg-turbo::jpeg-static)
        target_link_libraries(broimage_jpeg_turbo INTERFACE libjpeg-turbo::jpeg-static)
        set(_jt_how "installed ${libjpeg-turbo_VERSION}, static (${libjpeg-turbo_DIR})")
    else()
        # A libjpeg without the package config (an older distro package): it is
        # libjpeg-turbo when its jpeglib.h has the RGBA output colour spaces.
        find_package(JPEG QUIET)
        if(JPEG_FOUND)
            include(CheckSymbolExists)
            include(CMakePushCheckState)
            cmake_push_check_state(RESET)
            set(CMAKE_REQUIRED_INCLUDES ${JPEG_INCLUDE_DIRS})
            check_symbol_exists(JCS_EXTENSIONS "stdio.h;jpeglib.h" BROIMAGE_JPEG_IS_TURBO)
            cmake_pop_check_state()
            if(BROIMAGE_JPEG_IS_TURBO)
                target_link_libraries(broimage_jpeg_turbo INTERFACE JPEG::JPEG)
                set(_jt_how "installed (${JPEG_LIBRARIES})")
            endif()
        endif()
    endif()
endif()

if(NOT _jt_how)
    # The pinned release, built as a static library beside this build. It is a
    # standalone CMake project that does not expect to be add_subdirectory()'d,
    # so it builds as an external project; its headers are its source tree's
    # plus the jconfig.h its configure step writes.
    include(ExternalProject)
    enable_language(C)
    bro_dependency(libjpeg-turbo GITHUB libjpeg-turbo/libjpeg-turbo REF ${BROIMAGE_JPEG_TURBO_REF}
                   THIRD_PARTY SOURCE_ONLY)
    set(_jt_src "${libjpeg-turbo_SOURCE_DIR}")
    set(_jt_bin "${CMAKE_BINARY_DIR}/_deps/libjpeg-turbo-build")
    get_property(_jt_multi GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
    if(MSVC)
        set(_jt_name "${CMAKE_STATIC_LIBRARY_PREFIX}jpeg-static${CMAKE_STATIC_LIBRARY_SUFFIX}")
    else()
        set(_jt_name "${CMAKE_STATIC_LIBRARY_PREFIX}jpeg${CMAKE_STATIC_LIBRARY_SUFFIX}")
    endif()
    set(_jt_args
        -DENABLE_SHARED=OFF
        -DENABLE_STATIC=ON
        -DWITH_TURBOJPEG=OFF
        -DWITH_TOOLS=OFF
        -DWITH_TESTS=OFF
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    )
    if(CMAKE_OSX_ARCHITECTURES)
        list(APPEND _jt_args -DCMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES})
    endif()
    if(CMAKE_OSX_DEPLOYMENT_TARGET)
        list(APPEND _jt_args -DCMAKE_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET})
    endif()
    if(MSVC)
        # The C runtime the libraries around it use: the DLL one unless this
        # build asked for the static one.
        if(DEFINED CMAKE_MSVC_RUNTIME_LIBRARY AND NOT CMAKE_MSVC_RUNTIME_LIBRARY MATCHES "DLL")
            list(APPEND _jt_args -DWITH_CRT_DLL=OFF)
        else()
            list(APPEND _jt_args -DWITH_CRT_DLL=ON)
        endif()
    endif()
    if(_jt_multi)
        set(_jt_configs Debug Release RelWithDebInfo MinSizeRel)
        set(_jt_byproducts "")
        foreach(_c IN LISTS _jt_configs)
            list(APPEND _jt_byproducts "${_jt_bin}/${_c}/${_jt_name}")
        endforeach()
    else()
        # A C library has no business being slow in a Debug build of its caller.
        list(APPEND _jt_args -DCMAKE_BUILD_TYPE=Release)
        set(_jt_byproducts "${_jt_bin}/${_jt_name}")
    endif()
    ExternalProject_Add(broimage_jpeg_turbo_build
        SOURCE_DIR "${_jt_src}"
        BINARY_DIR "${_jt_bin}"
        CMAKE_ARGS ${_jt_args}
        BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --config $<CONFIG> --target jpeg-static
        INSTALL_COMMAND ""
        BUILD_BYPRODUCTS ${_jt_byproducts}
        USES_TERMINAL_BUILD TRUE
    )
    # The imported target's include directory has to exist at generate time.
    file(MAKE_DIRECTORY "${_jt_bin}")
    add_library(broimage_jpeg_turbo_lib STATIC IMPORTED GLOBAL)
    set_target_properties(broimage_jpeg_turbo_lib PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${_jt_src}/src;${_jt_bin}")
    if(_jt_multi)
        foreach(_c IN LISTS _jt_configs)
            string(TOUPPER "${_c}" _cu)
            set_target_properties(broimage_jpeg_turbo_lib PROPERTIES
                IMPORTED_LOCATION_${_cu} "${_jt_bin}/${_c}/${_jt_name}")
        endforeach()
        set_property(TARGET broimage_jpeg_turbo_lib PROPERTY IMPORTED_CONFIGURATIONS ${_jt_configs})
    else()
        set_target_properties(broimage_jpeg_turbo_lib PROPERTIES IMPORTED_LOCATION "${_jt_bin}/${_jt_name}")
    endif()
    target_link_libraries(broimage_jpeg_turbo INTERFACE broimage_jpeg_turbo_lib)
    add_dependencies(broimage_jpeg_turbo broimage_jpeg_turbo_build)
    set(_jt_how "pinned source ${BROIMAGE_JPEG_TURBO_REF}, built static")
endif()

target_compile_definitions(broimage_jpeg_turbo INTERFACE BROIMAGE_HAS_JPEG_TURBO=1)
message(STATUS "broimage: libjpeg-turbo ${_jt_how}")
