cmake_minimum_required(VERSION 3.0.0)

project(ch03 VERSION 0.1.0)

set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS "-O3")

include(CTest)
enable_testing()

# 添加 Eigen（由於皆為 header 文件，因此不需要利用 target_link_libraries 去鏈結函式庫）
include_directories("/usr/include/eigen3")

# add Pangolin include
find_package( Pangolin )
include_directories( ${Pangolin_INCLUDE_DIRS} )

# file(GLOB TARGET_SRC "./src/*.cpp" )

add_executable(main ./src/main.cpp)

# 添加函數庫(library)
add_library(slam ./lib/slam.cpp)

# 將'可執行程式'與'要使用的函式庫'鍊結起來
target_link_libraries(main slam)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

target_include_directories(slam PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(slam ${Pangolin_LIBRARIES})