#pragma once

#ifndef MYSLAM_FRAME_H
#define MYSLAM_FRAME_H

#include "myslam/camera.h"
#include "myslam/common_include.h"

namespace myslam {

/* forward declare
若你要創造一個名為 B 的類別時，它有個成員為 A 類別的指標，這種情況我們通常會在 B.h 的上端寫一句 #include "A.h"
這並沒有不對的地方，但當我們更動了 A.h 檔時，Compiler 在編譯的時後會重編所有 include A.h 的檔案，這當然包括 Class B
這時我們可以使用 froward declaration 的技巧取代 #include "A.h" 這行指令，
移除該行 include 命令，並在 Class B 的前面事先宣告 Class A;

class A;

class B
{
private:
  A* fPtrA ;
  // 或是連類別 A 一起宣告 class A* fPtrA; 就可以連上面的 class A; 也省略
public:
  void mymethod(const& A) const ;
};

如此可以縮短專案建置的時間，但這有個需要注意的重點，
此處我們只有宣告 Class A 讓編譯 B 時不發生錯誤，A 類別的實際大小並不知道，
所以我們只能在宣告成員變數為 reference 或 pointer 才可以使用這個技巧，
就如同上例 fPtrA 是一個位址，mymethod 所用的參數是一個 reference
因為指標的容量是視你的作業系統為 x86 或 x64 而固定的，沒有例外，
當我們使用其他類別是必須採用繼承或宣告為一個實體時，無可選擇還是要用 include。

參考網頁：https://ascii-iicsa.blogspot.com/2010/12/forward-declaration.html
*/
struct MapPoint;
struct Feature;

/**
 * 幀
 * 每一幀分配獨立id，關鍵幀分配關鍵幀 ID
 * 考慮到這些資料可能被多個執行續存取和修改，在關鍵部份會加上執行續鎖
 */
struct Frame {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Frame> Ptr;

    unsigned long id_ = 0;           // id of this frame
    unsigned long keyframe_id_ = 0;  // id of key frame
    bool is_keyframe_ = false;       // 是否為關鍵幀
    double time_stamp_;              // 時間戳，暫不使用
    SE3 pose_;                       // Tcw 形式Pose
    
    /* std::mutex 算是多執行緒中常用到的基本功能，mutex 用來上鎖一段多執行緒會交互存取的程式區塊，
     * 確保同一時間內只有一個執行緒能夠存取這段程式區塊，避免程式發生不預期的意外狀況
     * 參考：https://shengyu7697.github.io/blog/2020/02/05/std-mutex/
     */
    // Pose數據鎖
    std::mutex pose_mutex_;    

    // stereo images      
    cv::Mat left_img_, right_img_;   

    // extracted features in left image
    std::vector<std::shared_ptr<Feature>> features_left_;
    
    // corresponding features in right image, set to nullptr if no corresponding
    std::vector<std::shared_ptr<Feature>> features_right_;

// data members
public:  
    Frame() {}

    Frame(long id, double time_stamp, const SE3 &pose, const Mat &left, const Mat &right);

    // set and get pose, thread safe
    SE3 Pose() {
        /*std::unique_lock 是一種可以轉移所有權 (move constructor and move assignment) 的智慧指標 (smart pointer)。
         * 應該是避免重複呼叫 std::mutex （同一條 thread 鎖住自己兩次會造成 deadlock）
         */
        std::unique_lock<std::mutex> lck(pose_mutex_);
        return pose_;
    }

    void SetPose(const SE3 &pose) {
        std::unique_lock<std::mutex> lck(pose_mutex_);
        pose_ = pose;
    }

    // 設置『關鍵幀』並分配關鍵幀的 id（factory_id）
    void SetKeyFrame();

    /// 工廠構建模式，分配id 
    static std::shared_ptr<Frame> CreateFrame();
};

}  // namespace myslam

#endif  // MYSLAM_FRAME_H
