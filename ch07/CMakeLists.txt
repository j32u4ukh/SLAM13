cmake_minimum_required(VERSION 3.0.0)

project(ch07 VERSION 0.1.0)

set(CMAKE_BUILD_TYPE "Release")

# -D ENABLE_SSE
add_definitions("-DENABLE_SSE")

set(CMAKE_CXX_FLAGS "-std=c++11 -O2 ${SSE_FLAGS} -msse4")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# g2o
list(APPEND CMAKE_MODULE_PATH /home/j32u4ukh/Documents/slambook2/3rdparty/g2o/cmake_modules) 

find_package(OpenCV 4 REQUIRED)
find_package(G2O REQUIRED)
find_package(Sophus REQUIRED)

include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${G2O_INCLUDE_DIRS}
        ${Sophus_INCLUDE_DIRS}
        "/usr/include/eigen3"
)

# orb_cv
add_executable(orb_cv src/orb_cv.cpp)
target_link_libraries(orb_cv ${OpenCV_LIBS})

# orb_self
add_executable(orb_self src/orb_self.cpp)
target_link_libraries(orb_self ${OpenCV_LIBS})

# pose_estimation_2d2d
# add_executable( pose_estimation_2d2d pose_estimation_2d2d.cpp extra.cpp ) # use this if in OpenCV2 
add_executable(pose_estimation_2d2d src/pose_estimation_2d2d.cpp)
target_link_libraries(pose_estimation_2d2d ${OpenCV_LIBS})

# triangulation
# # add_executable( triangulation triangulation.cpp extra.cpp) # use this if in opencv2
add_executable(triangulation src/triangulation.cpp)
target_link_libraries(triangulation ${OpenCV_LIBS})

# pose_estimation_3d2d
add_executable(pose_estimation_3d2d src/pose_estimation_3d2d.cpp)
target_link_libraries(pose_estimation_3d2d
        g2o_core g2o_stuff
        ${OpenCV_LIBS})

# pose_estimation_3d3d
add_executable(pose_estimation_3d3d src/pose_estimation_3d3d.cpp)
target_link_libraries(pose_estimation_3d3d
        g2o_core g2o_stuff
        ${OpenCV_LIBS})

include(CTest)
enable_testing()

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
