#pragma once
#ifndef MYSLAM_MAPPOINT_H
#define MYSLAM_MAPPOINT_H

#include "myslam/common_include.h"

namespace myslam {

struct Frame;

struct Feature;

/**
 * 路標點類
 * 特征點在三角化之後形成路標點
 */
struct MapPoint {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    
    typedef std::shared_ptr<MapPoint> Ptr;
    
    // ID
    unsigned long id_ = 0;  
        
    bool is_outlier_ = false;
    
    // Position in world
    Vec3 pos_ = Vec3::Zero();  
    
    std::mutex data_mutex_;
    
    // being observed by feature matching algo.
    int observed_times_ = 0;  
    
    // 觀測到這個 MapPoint 的所有 Feature（當要優化這些特徵時，可以進一步修正所對應的 MapPoint）
    std::list<std::weak_ptr<Feature>> observations_;

    MapPoint() {}

    MapPoint(long id, Vec3 position);

    Vec3 Pos() {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return pos_;
    }

    void SetPos(const Vec3 &pos) {
        std::unique_lock<std::mutex> lck(data_mutex_);
        pos_ = pos;
    };

    void AddObservation(std::shared_ptr<Feature> feature) {
        std::unique_lock<std::mutex> lck(data_mutex_);
        observations_.push_back(feature);
        observed_times_++;
    }

    void RemoveObservation(std::shared_ptr<Feature> feat);

    std::list<std::weak_ptr<Feature>> GetObs() {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return observations_;
    }

    // factory function
    static MapPoint::Ptr CreateNewMappoint();
};
}  // namespace myslam

#endif  // MYSLAM_MAPPOINT_H
