cmake_minimum_required(VERSION 3.0.0)

project(ch05 VERSION 0.1.0)

# 添加c++ 11標準支持
set(CMAKE_CXX_FLAGS "-std=c++11 -O2")

include(CTest)
enable_testing()

# file(GLOB TARGET_SRC "./src/*.cpp" )
# add_executable(main ${TARGET_SRC})

add_executable(main ./src/main.cpp)

# 添加函數庫(library)
add_library(slam ./lib/slam.cpp)

# Eigen
include_directories("/usr/include/eigen3")

# 尋找OpenCV庫
find_package(OpenCV REQUIRED)

# 添加 header
include_directories(${OpenCV_INCLUDE_DIRS})

add_subdirectory(imageBasics)
add_subdirectory(stereo)
add_subdirectory(rgbd)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

target_include_directories(slam PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(slam ${Pangolin_LIBRARIES})
