//
// Created by gaoxiang on 19-5-2.
//

#ifndef MYSLAM_BACKEND_H
#define MYSLAM_BACKEND_H

#include "myslam/common_include.h"
#include "myslam/frame.h"
#include "myslam/map.h"

namespace myslam {
class Map;

/**
 * 後端
 * 有單獨優化線程，在Map更新時啟動優化
 * Map更新由前端觸发
 */ 
class Backend {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Backend> Ptr;

    /// 構造函數中啟動優化線程並掛起
    Backend();

    // 設置左右目的相機，用於獲得內外參
    void SetCameras(Camera::Ptr left, Camera::Ptr right) {
        cam_left_ = left;
        cam_right_ = right;
    }

    /// 設置地圖
    void SetMap(std::shared_ptr<Map> map) { map_ = map; }

    /// 觸发地圖更新，啟動優化
    void UpdateMap();

    /// 關閉後端線程
    void Stop();

   private:
    /// 後端線程
    void BackendLoop();

    /// 對給定關鍵幀和路標點進行優化
    void Optimize(Map::KeyframesType& keyframes, Map::LandmarksType& landmarks);

    std::shared_ptr<Map> map_;
    std::thread backend_thread_;
    std::mutex data_mutex_;

    /*使用 std::condition_variable 的 wait 會把目前的執行緒 thread 停下來並且等候事件通知，而在另外一個執行緒裡
     * 我們可以使用 std::condition_variable 的 notify_one 或 notify_all 去發送通知那些正在等待的事件
     * wait: 阻塞當前執行緒直到條件變量被喚醒
     * notify_one: 通知一個正在等待的執行緒
     * notify_all: 通知所有正在等待的執行緒 
     * 使用 std::condition_variable 的 wait 必須要搭配 std::unique_lock<std::mutex> 一起使用。 
     * 參考：https://shengyu7697.github.io/blog/2020/02/20/std-condition-variable/
     */
    std::condition_variable map_update_;
    
    /* 多個執行緒同時存取 backend_running_ 會造成資料不正確，一般應透過 std::mutex 來上鎖，以確保存取順序。
     * 而使用 std::atomic 可以不上鎖就達到相同的效果，而且效率更加。
     * 參考：https://shengyu7697.github.io/blog/2020/04/13/std-atomic/
     */
    std::atomic<bool> backend_running_;

    Camera::Ptr cam_left_ = nullptr, cam_right_ = nullptr;
};

}  // namespace myslam

#endif  // MYSLAM_BACKEND_H
