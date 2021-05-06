#include <iostream>
#include <Eigen/Core>

int main() {
    std::cout << "Hello from Fenix!" << std::endl;
    Eigen::Vector3d vector(1, 3, 9);
    std::cout << "vector:" << vector << std::endl;
    vector /= 3.0;
    std::cout << "normalized vector:" << vector << std::endl;
}