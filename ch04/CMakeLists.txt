cmake_minimum_required(VERSION 3.0.0)
project(new_project VERSION 0.1.0)

set(CMAKE_BUILD_TYPE "Release")

include(CTest)
enable_testing()

# file(GLOB TARGET_SRC "./src/*.cpp" )
# add_executable(main ${TARGET_SRC})

add_executable(main ./src/useSophus.cpp)

# 添加函數庫(library)
add_library(slam ./lib/slam.cpp)

# Eigen
include_directories("/usr/include/eigen3")

# 為使用 sophus，需要使用find_package命令找到它
find_package(Sophus REQUIRED)
target_link_libraries(slam Sophus::Sophus)

# add Pangolin include
find_package( Pangolin )
include_directories( ${Pangolin_INCLUDE_DIRS} )

add_subdirectory(example)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

target_include_directories(slam PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(slam ${Pangolin_LIBRARIES})
