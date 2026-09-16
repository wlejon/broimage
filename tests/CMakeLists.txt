function(broimage_add_test name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE broimage)
    if(MSVC)
        target_compile_options(${name} PRIVATE /W4)
    else()
        target_compile_options(${name} PRIVATE -Wall -Wextra)
    endif()
    add_test(NAME ${name} COMMAND ${name})
endfunction()

broimage_add_test(broimage_test_smoke           test_smoke.cpp)
broimage_add_test(broimage_test_decode_encode   test_decode_encode.cpp)
broimage_add_test(broimage_test_color           test_color.cpp)
broimage_add_test(broimage_test_geometric       test_geometric.cpp)
broimage_add_test(broimage_test_alpha           test_alpha.cpp)
broimage_add_test(broimage_test_normalize       test_normalize.cpp)
broimage_add_test(broimage_test_preproc         test_preproc.cpp)
broimage_add_test(broimage_test_kernels         test_kernels.cpp)
broimage_add_test(broimage_test_tiling          test_tiling.cpp)
if(TARGET basisu_transcoder)
    broimage_add_test(broimage_test_ktx2        test_ktx2.cpp)
endif()
if(BROIMAGE_WITH_TENSOR)
    broimage_add_test(broimage_test_tensor_adapter  test_tensor_adapter.cpp)
endif()

if(BROIMAGE_ENABLE_API)
    add_executable(broimage_test_api test_image_api.cpp)
    target_include_directories(broimage_test_api PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/api
    )
    target_link_libraries(broimage_test_api PRIVATE
        broimage_api
        bronze_runtime_shared
    )
    if(MSVC)
        target_compile_options(broimage_test_api PRIVATE /W4)
    else()
        target_compile_options(broimage_test_api PRIVATE -Wall -Wextra)
    endif()
    # bronze's shared runtime is built into BRONZE_SHARED_RUNTIME_DIR, not
    # beside this executable, and the PE loader only looks beside the .exe
    # (0xc0000135 + a modal "dll was not found" otherwise). Stage it next to
    # the test as a post-build step and put both dirs on ctest's PATH.
    if(WIN32)
        add_custom_command(TARGET broimage_test_api POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:bronze_runtime_shared>
                $<TARGET_FILE_DIR:broimage_test_api>
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_LINKER_FILE:bronze_runtime_shared>
                $<TARGET_FILE_DIR:broimage_test_api>
        )
    endif()
    add_test(NAME broimage_test_api COMMAND broimage_test_api)
    if(WIN32)
        set_tests_properties(broimage_test_api PROPERTIES
            ENVIRONMENT "PATH=$<TARGET_FILE_DIR:bronze_runtime_shared>;$<TARGET_FILE_DIR:broimage_test_api>;$ENV{PATH}")
    endif()
endif()
