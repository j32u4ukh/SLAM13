//
// Created by gaoxiang on 19-5-2.
//
#pragma once

#ifndef MYSLAM_FEATURE_H
#define MYSLAM_FEATURE_H

#include <memory>
#include <opencv2/features2d.hpp>
#include "myslam/common_include.h"

namespace myslam {

struct Frame;
struct MapPoint;

/**
 * 2D 特征點
 * 在三角化之後會被關聯一個地圖點
 * 
 * Frame with cv::KeyPoint
 */
struct Feature {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Feature> Ptr;

    /* weak_ptr，基本上是一個需要搭配 shared_ptr 來一起使用的特例
     * 可以使用 shared_ptr 所指向的資源，但當 shared_ptr 釋放資源後，不會要求系統把資源留下來使用，
     * 而是會隨著 shared_ptr 的消失、把相關的資源釋放掉。
     * 
     * Frame 持有 Feature 的 shared_ptr，因此要避免 Feature 再持有 Frame 的 shared_ptr，
     * 且 Frame 的實際所有權歸'地圖'所有，
     * 否則會造成兩者相互參考，因此這裡使用 weak_ptr */
    
    // 持有該 feature 的 frame
    std::weak_ptr<Frame> frame_;    
    
    // 2D 提取位置
    cv::KeyPoint position_;              
    
    // 關聯地圖點
    std::weak_ptr<MapPoint> map_point_;  

    // 是否為異常點
    bool is_outlier_ = false;     
    
    // 標識是否提在左圖，false 為右圖
    bool is_on_left_image_ = true;  

   public:
    Feature() {}

    Feature(std::shared_ptr<Frame> frame, const cv::KeyPoint &kp) : frame_(frame), position_(kp) {}
};
}  // namespace myslam

#endif  // MYSLAM_FEATURE_H
