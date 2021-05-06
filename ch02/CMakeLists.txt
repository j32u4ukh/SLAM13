cmake_minimum_required(VERSION 3.0.0)

# project name
project(ch02 VERSION 0.1.0)

set(CMAKE_BUILD_TYPE "Debug")

# set(HEADER_FILES ${PROJECT_SOURCE_DIR}/include/*.hpp)

# automatically creates a BUILD_TESTING option
# After including the module, use code like:
# >>> if(BUILD_TESTING)
# >>> ... CMake code to create tests ...
# >>> endif()
# to creating tests when testing is enabled.
# The definition will write into CTestConfig.cmake
include(CTest)

# Enables testing for this directory and below. See also the add_test() 
# command. Note that ctest expects to find a test file in the build
# directory root. Therefore, this command should be in the source
# directory root. 
enable_testing()

# file(GLOB TARGET_SRC "./src/*.cpp" )

add_executable(hello_slam ./src/main.cpp)

# 添加函數庫(library)
add_library(hello ./lib/hello.cpp)

# 將'可執行程式'與'要使用的函式庫'鍊結起來
target_link_libraries(hello_slam hello)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

target_include_directories(hello PUBLIC ${PROJECT_SOURCE_DIR}/include)
