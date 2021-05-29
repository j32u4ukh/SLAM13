cmake_minimum_required(VERSION 3.0.0)
project(ch10 VERSION 0.1.0)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11 -O2")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

# Eigen
include_directories("/usr/include/eigen3")

# sophus 
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

# g2o 
find_package(G2O REQUIRED)
include_directories(${G2O_INCLUDE_DIRS})

add_executable(pose_graph_g2o_SE3 ./src/pose_graph_g2o_SE3.cpp)
target_link_libraries(pose_graph_g2o_SE3
        g2o_core g2o_stuff g2o_types_slam3d ${CHOLMOD_LIBRARIES}
        )

add_executable(pose_graph_g2o_lie ./src/pose_graph_g2o_lie_algebra.cpp)
target_link_libraries(pose_graph_g2o_lie
        g2o_core g2o_stuff
        ${CHOLMOD_LIBRARIES}
        ${Sophus_LIBRARIES}
        )

include(CTest)
enable_testing()

# file(GLOB TARGET_SRC "./src/*.cpp" )
# add_executable(main ${TARGET_SRC})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
