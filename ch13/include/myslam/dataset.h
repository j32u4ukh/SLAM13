#ifndef MYSLAM_DATASET_H
#define MYSLAM_DATASET_H
#include "myslam/camera.h"
#include "myslam/common_include.h"
#include "myslam/frame.h"

namespace myslam {

/**
 * 數據集讀取
 * 構造時傳入配置文件路徑，配置文件的dataset_dir為數據集路徑
 * Init之後可獲得相機和下一幀圖像
 */
class Dataset {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Dataset> Ptr;
    Dataset(const std::string& dataset_path);

    /// 初始化，返回是否成功
    bool Init();

    /// create and return the next frame containing the stereo images
    Frame::Ptr NextFrame();

    /// get camera by id
    Camera::Ptr GetCamera(int camera_id) const {
        return cameras_.at(camera_id);
    }

   private:
    std::string dataset_path_;
    int current_image_index_ = 0;

    std::vector<Camera::Ptr> cameras_;
};
}  // namespace myslam

#endif
