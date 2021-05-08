#ifndef SnavelyReprojection_H
#define SnavelyReprojection_H

#include <iostream>
#include "ceres/ceres.h"
#include "rotation.h"

class SnavelyReprojectionError {
public:
    SnavelyReprojectionError(double observation_x, double observation_y) : observed_x(observation_x),
                                                                           observed_y(observation_y) {}

    template<typename T>
    bool operator()(const T *const camera,
                    const T *const point,
                    T *residuals) const {
        // camera[0,1,2] are the angle-axis rotation
        T predictions[2];
        CamProjectionWithDistortion(camera, point, predictions);
        residuals[0] = predictions[0] - T(observed_x);
        residuals[1] = predictions[1] - T(observed_y);

        return true;
    }

    // camera : 9 dims array
    // [0-2] : angle-axis rotation
    // [3-5] : translateion
    // [6-8] : camera parameter, [6] focal length, [7-8] second and forth order radial distortion(k1, k2)
    // point : 3D location.
    // predictions : 2D predictions with center of the image plane.
    template<typename T>
    static inline bool CamProjectionWithDistortion(const T *camera, const T *point, T *predictions) {
        // 1. p' = RP + t
        T p[3];

        // [RP] using Rodrigues' formula to compute
        AngleAxisRotatePoint(camera, point, p);

        // [+ t]
        // camera[3, 4, 5] are the translation
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];

        // 2. P_normalize  = [X/Z, Y/Z, 1]
        // Compute the center fo distortion
        // BAL 資料在投影時，假設投影平面在相機光心之後，因此在投影後應乘以係數 -1
        T xp = -p[0] / p[2];
        T yp = -p[1] / p[2];

        // Apply second and fourth order radial distortion
        const T &l1 = camera[7];
        const T &l2 = camera[8];

        T r2 = xp * xp + yp * yp;

        // 3. u' = u * (1 + k1 * rc^2 + k2 * rc^4) 
        T distortion = T(1.0) + r2 * (l1 + l2 * r2);

        // 4. us = fx * u' (+ cx) BAL 數據集已將 cx/cy 的偏移去除，因此這裡無須再添加
        // 原本應乘以 fx/fy，但像素基本上是正方形，可用 focal 取代兩者
        // 以圖像中心為原點的（預測）座標
        const T &focal = camera[6];
        predictions[0] = focal * distortion * xp;
        predictions[1] = focal * distortion * yp;

        return true;
    }

    static ceres::CostFunction *Create(const double observed_x, const double observed_y) {
        // (預測座標, 相機參數, 路標空間點)
        return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 9, 3>(
            new SnavelyReprojectionError(observed_x, observed_y)));
    }

private:
    double observed_x;
    double observed_y;
};

#endif // SnavelyReprojection.h

