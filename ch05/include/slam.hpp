#ifndef INCLUDE_SLAM_HPP
#define INCLUDE_SLAM_HPP

#include <iostream>
#include <vector>
#include <unistd.h>
#include <sophus/se3.hpp>
#include <pangolin/pangolin.h>

typedef Eigen::Matrix<double, 6, 1> Vector6d;

void displayPointCloud(const std::vector<Vector6d, Eigen::aligned_allocator<Vector6d>> &pointcloud);

#endif // !