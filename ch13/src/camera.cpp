#include "myslam/camera.h"

namespace myslam {

Camera::Camera() {
}

Vec3 Camera::world2camera(const Vec3 &p_w, const SE3 &T_c_w) {
    // 世界座標的點先透過 T_c_w 轉換到左右相機共同的座標系下
    // 又左右相機之間存在平移，因此透過 pose_ 轉換到左或右相機的座標系下
    // 又此專案為雙目相機，因此兩相機間只有平移，沒有旋轉！！
    return pose_ * T_c_w * p_w;
}

Vec3 Camera::camera2world(const Vec3 &p_c, const SE3 &T_c_w) {
    return T_c_w.inverse() * pose_inv_ * p_c;
}

Vec2 Camera::camera2pixel(const Vec3 &p_c) {
    // p_c = (x, y, z) x/z or y/z 皆是為了歸一化到成像平面
    return Vec2(
            fx_ * p_c(0, 0) / p_c(2, 0) + cx_,
            fy_ * p_c(1, 0) / p_c(2, 0) + cy_
    );
}

Vec3 Camera::pixel2camera(const Vec2 &p_p, double depth) {
    // x, y 也乘了 depth，是因為 p_p 上的數值在從空間中歸一化到成像平面上時，也除以了 z 值
    return Vec3(
            (p_p(0, 0) - cx_) * depth / fx_,
            (p_p(1, 0) - cy_) * depth / fy_,
            depth
    );
}

Vec2 Camera::world2pixel(const Vec3 &p_w, const SE3 &T_c_w) {
    return camera2pixel(world2camera(p_w, T_c_w));
}

Vec3 Camera::pixel2world(const Vec2 &p_p, const SE3 &T_c_w, double depth) {
    return camera2world(pixel2camera(p_p, depth), T_c_w);
}

}
