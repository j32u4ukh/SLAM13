cmake_minimum_required(VERSION 3.0.0)

project(ch09 VERSION 0.1.0)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-O3 -std=c++11")

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# g2o
list(APPEND CMAKE_MODULE_PATH /home/j32u4ukh/Documents/CppProjects/SLAM13/submodule/g2o/cmake_modules) 

Find_Package(G2O REQUIRED)
Find_Package(Eigen3 REQUIRED)
Find_Package(Ceres REQUIRED)
Find_Package(Sophus REQUIRED)
Find_Package(CSparse REQUIRED)

SET(G2O_LIBS g2o_csparse_extension g2o_stuff g2o_core cxsparse)

include_directories(${PROJECT_SOURCE_DIR} ${EIGEN3_INCLUDE_DIR} ${CSPARSE_INCLUDE_DIR})

# add_executable(main src/main.cpp)
add_library(bal_common src/common.cpp)
add_executable(bundle_adjustment_g2o src/bundle_adjustment_g2o.cpp)
add_executable(bundle_adjustment_ceres src/bundle_adjustment_ceres.cpp)

target_link_libraries(bundle_adjustment_ceres ${CERES_LIBRARIES} bal_common)
target_link_libraries(bundle_adjustment_g2o ${G2O_LIBS} bal_common)

include(CTest)
enable_testing()

# file(GLOB TARGET_SRC "./src/*.cpp" )
# add_executable(main ${TARGET_SRC})


set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
