cmake_minimum_required(VERSION 3.14)
project(NeuralCore VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler warnings
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Collect source files
file(GLOB_RECURSE NEURALCORE_SOURCES "src/*.cpp")

# Library target
add_library(neuralcore ${NEURALCORE_SOURCES})
target_include_directories(neuralcore PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Tests
enable_testing()

file(GLOB TEST_SOURCES "tests/*.cpp")
foreach(test_file ${TEST_SOURCES})
    get_filename_component(test_name ${test_file} NAME_WE)
    add_executable(${test_name} ${test_file})
    target_link_libraries(${test_name} PRIVATE neuralcore)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

# Examples
file(GLOB EXAMPLE_SOURCES "examples/*.cpp")
foreach(example_file ${EXAMPLE_SOURCES})
    get_filename_component(example_name ${example_file} NAME_WE)
    add_executable(${example_name} ${example_file})
    target_link_libraries(${example_name} PRIVATE neuralcore)
endforeach()
