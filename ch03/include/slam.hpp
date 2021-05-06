#ifndef INCLUDE_SLAM_HPP
#define INCLUDE_SLAM_HPP

#include <iostream>
#include <iomanip>
#include <ctime>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <pangolin/pangolin.h>

#define MATRIX_SIZE 50

struct RotationMatrix {
    Eigen::Matrix3d matrix = Eigen::Matrix3d::Identity();
};

struct TranslationVector {
    Eigen::Vector3d trans = Eigen::Vector3d(0, 0, 0);
};

struct QuaternionDraw {
    Eigen::Quaterniond q;
};

std::ostream &operator<<(std::ostream &out, const RotationMatrix &r);
std::istream &operator>>(std::istream &in, RotationMatrix &r);

std::ostream &operator<<(std::ostream &out, const TranslationVector &t);
std::istream &operator>>(std::istream &in, TranslationVector &t);

std::ostream &operator<<(std::ostream &out, const QuaternionDraw quat);
std::istream &operator>>(std::istream &in, const QuaternionDraw quat);


void eigenMatrix();
void eigenGeometry();
void visualizeGeometry();

#endif // !