cmake_minimum_required(VERSION 3.0.0)

project(ch11 VERSION 0.1.0)

set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

include(CTest)
enable_testing()

# file(GLOB TARGET_SRC "./src/*.cpp" )
# add_executable(main ${TARGET_SRC})

# opencv 
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# dbow3 
# dbow3 is a simple lib so I assume you installed it in default directory 
set( DBoW3_INCLUDE_DIRS "/usr/local/include" )
set( DBoW3_LIBS "/usr/local/lib/libDBoW3.so" )

add_executable( feature_training ./src/feature_training.cpp )
target_link_libraries( feature_training ${OpenCV_LIBS} ${DBoW3_LIBS} )

add_executable( loop_closure ./src/loop_closure.cpp )
target_link_libraries( loop_closure ${OpenCV_LIBS} ${DBoW3_LIBS} )

add_executable( gen_vocab ./src/gen_vocab_large.cpp )
target_link_libraries( gen_vocab ${OpenCV_LIBS} ${DBoW3_LIBS} )

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
