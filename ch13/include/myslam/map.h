#pragma once
#ifndef MAP_H
#define MAP_H

#include "myslam/common_include.h"
#include "myslam/frame.h"
#include "myslam/mappoint.h"

namespace myslam {

/**
 * @brief 地圖
 * 和地圖的交互：前端調用 InsertKeyframe 和 InsertMapPoint 插入新幀和地圖點，後端維護地圖的結構，'判定 outlier'/'剔除'等等
 */
class Map {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Map> Ptr;

    // LandmarksType 為字典(key: unsigned long; value: MapPoint::Ptr)
    typedef std::unordered_map<unsigned long, MapPoint::Ptr> LandmarksType;

    // KeyframesType 為字典(key: unsigned long; value: Frame::Ptr)
    typedef std::unordered_map<unsigned long, Frame::Ptr> KeyframesType;

    Map() {}

    /// 增加一個關鍵幀
    void InsertKeyFrame(Frame::Ptr frame);
    
    /// 增加一個地圖頂點
    void InsertMapPoint(MapPoint::Ptr map_point);

    /// 獲取所有地圖點
    // LandmarksType 為字典(key: unsigned long; value: MapPoint::Ptr)
    LandmarksType GetAllMapPoints() {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return landmarks_;
    }
    
    /// 獲取所有關鍵幀
    // KeyframesType 為字典(key: unsigned long; value: Frame::Ptr)
    KeyframesType GetAllKeyFrames() {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return keyframes_;
    }

    /// 獲取激活地圖點
    // LandmarksType 為字典(key: unsigned long; value: MapPoint::Ptr)
    LandmarksType GetActiveMapPoints() {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return active_landmarks_;
    }

    /// 獲取激活關鍵幀
    // KeyframesType 為字典(key: unsigned long; value: Frame::Ptr)
    KeyframesType GetActiveKeyFrames() {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return active_keyframes_;
    }

    /// 清理 map 中觀測數量為零的點
    void CleanMap();

   private:
    // 將舊的關鍵幀置為不活躍狀態
    void RemoveOldKeyframe();

    std::mutex data_mutex_;
    
    // all landmarks
    // LandmarksType 為字典(key: unsigned long; value: MapPoint::Ptr)
    LandmarksType landmarks_; 
    
    // active landmarks
    // LandmarksType 為字典(key: unsigned long; value: MapPoint::Ptr)
    LandmarksType active_landmarks_;  
    
    // all key-frames
    // KeyframesType 為字典(key: unsigned long; value: Frame::Ptr)
    KeyframesType keyframes_;  
    
    // active key-frames
    // KeyframesType 為字典(key: unsigned long; value: Frame::Ptr)
    KeyframesType active_keyframes_;  

    Frame::Ptr current_frame_ = nullptr;

    // settings
    // 激活的關鍵幀數量
    int num_active_keyframes_ = 7;  
};
}  // namespace myslam

#endif  // MAP_H
