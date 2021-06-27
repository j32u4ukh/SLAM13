//
// Created by gaoxiang on 19-5-2.
//

#include <opencv2/opencv.hpp>

#include "myslam/algorithm.h"
#include "myslam/backend.h"
#include "myslam/config.h"
#include "myslam/feature.h"
#include "myslam/frontend.h"
#include "myslam/g2o_types.h"
#include "myslam/map.h"
#include "myslam/viewer.h"

namespace myslam {

Frontend::Frontend() {
    gftt_ =
        cv::GFTTDetector::create(Config::Get<int>("num_features"), 0.01, 20);
    num_features_init_ = Config::Get<int>("num_features_init");
    num_features_ = Config::Get<int>("num_features");
}

bool Frontend::AddFrame(myslam::Frame::Ptr frame) {
    current_frame_ = frame;

    switch (status_) {
        case FrontendStatus::INITING:
            StereoInit();
            break;
        case FrontendStatus::TRACKING_GOOD:
        case FrontendStatus::TRACKING_BAD:
            Track();
            break;
        case FrontendStatus::LOST:
            Reset();
            break;
    }

    last_frame_ = current_frame_;
    return true;
}

bool Frontend::StereoInit() {
    // 抽取出當前頁框的 Feature，並返回 Feature 數量
    int num_features_left = DetectFeatures();
    int num_coor_features = FindFeaturesInRight();
    
    if (num_coor_features < num_features_init_) {
        return false;
    }

    // 特徵數量大於設定的數量，才會進入 BuildInitMap，進而將 current_frame_ 設為 key frame
    // 目前看起來 BuildInitMap 永遠返回 true，沒有 false 或是跳出例外等情況。
    bool build_map_success = BuildInitMap();
    
    if (build_map_success) {
        status_ = FrontendStatus::TRACKING_GOOD;
        
        if (viewer_) {
            viewer_->AddCurrentFrame(current_frame_);
            viewer_->UpdateMap();
        }
        
        return true;
    }

    return false;
}

// 抽取出當前頁框的 Feature，並返回 Feature 數量
int Frontend::DetectFeatures() {
    cv::Mat mask(current_frame_->left_img_.size(), CV_8UC1, 255);

    // 標注出特徵點位置區域，協助 gftt_->detect 尋找 cv::KeyPoint
    // 以特徵點為中心，形成長寬各 20 像素的矩形
    // line type: cv::FILLED 填滿矩形
    // 參考：https://blog.gtwang.org/programming/opencv-drawing-functions-tutorial/
    for (auto &feat : current_frame_->features_left_) {
        cv::rectangle(mask, feat->position_.pt - cv::Point2f(10, 10),
                      feat->position_.pt + cv::Point2f(10, 10), 0, cv::FILLED);
    }

    std::vector<cv::KeyPoint> keypoints;

    // 尋找 cv::KeyPoint 儲存至 keypoints
    // mask – Mask specifying where to look for keypoints (optional). 
    // It must be a 8-bit integer(CV_8UC1) matrix with non-zero values in the region of interest.
    // keypoints: cv::vector<cv::KeyPoint>
    gftt_->detect(current_frame_->left_img_, keypoints, mask);
    int cnt_detected = 0;

    for (auto &kp : keypoints) {
        // 利用『當前頁框 current_frame_』和『關鍵點 kp』形成 Feature
        // 由『當前頁框 current_frame_』的『特徵陣列 features_left_』進行管理
        current_frame_->features_left_.push_back(Feature::Ptr(new Feature(current_frame_, kp)));
        cnt_detected++;
    }

    LOG(INFO) << "Detect " << cnt_detected << " new features";
    return cnt_detected;
}

int Frontend::FindFeaturesInRight() {
    // use LK flow to estimate points in the right image
    std::vector<cv::Point2f> kps_left, kps_right;
    
    for (auto &kp : current_frame_->features_left_) {
        // 取出前一步驟抽取出的關鍵點的位置，存入左圖關鍵點陣列 kps_left
        kps_left.push_back(kp->position_.pt);

        /* 
        weak_ptr 取出 shared_ptr 的時候，shared_ptr 可能將要被刪除。weak_ptr 如何保證 thread safety？
        因此 weak_ptr 取出 shared_ptr 的 API 是 weak_ptr::lock()。

        kp->map_point_ 為 weak_ptr 要取得 MapPoint，而 MapPoint 為 shared_ptr，因此這裡要使用 lock()

        參考：https://medium.com/fcamels-notes/gcc-4-8-4-weak-ptr-lock-%E7%9A%84%E5%AF%A6%E4%BD%9C
        -a37fd284dc8
        */
        auto mp = kp->map_point_.lock();
        
        if (mp) {
            // use projected points as initial guess
            auto px = camera_right_->world2pixel(mp->pos_, current_frame_->Pose());
            kps_right.push_back(cv::Point2f(px[0], px[1]));
            
        } else {
            // use same pixel in left iamge
            kps_right.push_back(kp->position_.pt);
        }
    }

    std::vector<uchar> status;
    Mat error;
    
    // kps_right：輸入的標定圖像的特徵點（可以是其他特徵點檢測方法找到的點） 
    cv::calcOpticalFlowPyrLK(
        current_frame_->left_img_, current_frame_->right_img_, kps_left,
        kps_right, status, error, cv::Size(11, 11), 3,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
        cv::OPTFLOW_USE_INITIAL_FLOW);

    int num_good_pts = 0;
    
    for (size_t i = 0; i < status.size(); ++i) {
        if (status[i]) {
            cv::KeyPoint kp(kps_right[i], 7);
            Feature::Ptr feat(new Feature(current_frame_, kp));
            feat->is_on_left_image_ = false;
            current_frame_->features_right_.push_back(feat);
            num_good_pts++;
            
        } else {
            current_frame_->features_right_.push_back(nullptr);
        }
    }
    
    LOG(INFO) << "Find " << num_good_pts << " in the right image.";
    return num_good_pts;
}

bool Frontend::BuildInitMap() {
    std::vector<SE3> poses{camera_left_->pose(), camera_right_->pose()};
    size_t cnt_init_landmarks = 0;
    
    for (size_t i = 0; i < current_frame_->features_left_.size(); ++i) {
        if (current_frame_->features_right_[i] == nullptr){
            continue;
        }
        
        // create map point from triangulation
        std::vector<Vec3> points{
            camera_left_->pixel2camera(
                Vec2(current_frame_->features_left_[i]->position_.pt.x,
                     current_frame_->features_left_[i]->position_.pt.y)),
            camera_right_->pixel2camera(
                Vec2(current_frame_->features_right_[i]->position_.pt.x,
                     current_frame_->features_right_[i]->position_.pt.y))};
        Vec3 pworld = Vec3::Zero();

        if (triangulation(poses, points, pworld) && pworld[2] > 0) {
            auto new_map_point = MapPoint::CreateNewMappoint();

            new_map_point->SetPos(pworld);
            new_map_point->AddObservation(current_frame_->features_left_[i]);
            new_map_point->AddObservation(current_frame_->features_right_[i]);

            current_frame_->features_left_[i]->map_point_ = new_map_point;
            current_frame_->features_right_[i]->map_point_ = new_map_point;
            cnt_init_landmarks++;

            map_->InsertMapPoint(new_map_point);
        }
    }
    
    // 特徵點數量超過設定的數量，才會進入此函式，並在此被設為 KeyFrame
    current_frame_->SetKeyFrame();
    map_->InsertKeyFrame(current_frame_);
    backend_->UpdateMap();

    LOG(INFO) << "Initial map created with " << cnt_init_landmarks
              << " map points";

    return true;
}

bool Frontend::Track() {
    // 若有 last_frame_
    if (last_frame_) {
        // current_frame_ 被新增時沒有 pose 的數值，若有前一幀的數據，則利用它估計 current_frame_ 的初始值
        // 若沒有前一幀的第一幀，會在 FrontendStatus::INITING 階段被初始化
        // relative_motion_ = current_frame_->Pose() * last_frame_->Pose().inverse()
        // 又，這裡的 last_frame_ 為計算 relative_motion_ 時的 current_frame_
        // 因此，實際上相當於 current_frame_->Pose() * last_frame_->Pose().inverse() * current_frame_->Pose()
        current_frame_->SetPose(relative_motion_ * last_frame_->Pose());
    }

    int num_track_last = TrackLastFrame();

    // 扣除離群點後的特徵數量
    tracking_inliers_ = EstimateCurrentPose();

    // 追蹤到的特徵大於 num_features_tracking_
    if (tracking_inliers_ > num_features_tracking_) {
        // tracking good
        status_ = FrontendStatus::TRACKING_GOOD;
    } 
    
    // 追蹤到的特徵小於 num_features_tracking_，但大於 num_features_tracking_bad_
    else if (tracking_inliers_ > num_features_tracking_bad_) {
        // tracking bad
        status_ = FrontendStatus::TRACKING_BAD;
    } 
    
    // 追蹤到的特徵小於 num_features_tracking_bad_
    else {
        // lost
        status_ = FrontendStatus::LOST;
    }

    InsertKeyframe();

    // last_frame_ 到 current_frame_ 的轉換矩陣
    relative_motion_ = current_frame_->Pose() * last_frame_->Pose().inverse();

    if (viewer_){
        viewer_->AddCurrentFrame(current_frame_);
    }
    
    return true;
}

int Frontend::TrackLastFrame() {
    // use LK flow to estimate points in the right image
    std::vector<cv::Point2f> kps_last, kps_current;
    
    // 遍歷前一幀左圖像的特徵點
    for (auto &kp : last_frame_->features_left_) {
        // 前一幀左圖像的特徵點的位置
        kps_last.push_back(kp->position_.pt);

        if (kp->map_point_.lock()) {
            // use project point
            auto mp = kp->map_point_.lock();
            auto px = camera_left_->world2pixel(mp->pos_, current_frame_->Pose());
            kps_current.push_back(cv::Point2f(px[0], px[1]));
            
        } else {
            kps_current.push_back(kp->position_.pt);            
        }
    }

    std::vector<uchar> status;
    Mat error;
    cv::calcOpticalFlowPyrLK(
        last_frame_->left_img_, current_frame_->left_img_, kps_last,
        kps_current, status, error, cv::Size(11, 11), 3,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
        cv::OPTFLOW_USE_INITIAL_FLOW);

    int num_good_pts = 0;

    // 更新 current_frame_ 左圖向上找到的特徵點
    for (size_t i = 0; i < status.size(); ++i) {

        // 若光流法有追蹤到新圖像中的特徵點
        if (status[i]) {
            cv::KeyPoint kp(kps_current[i], 7);
            Feature::Ptr feature(new Feature(current_frame_, kp));

            // 有追蹤到的同一個空間點
            // 因此 current_frame_ 的第 i 個 feature 的空間點 和前一幀的第 i 個 feature 的空間點
            feature->map_point_ = last_frame_->features_left_[i]->map_point_;

            current_frame_->features_left_.push_back(feature);
            num_good_pts++;
        }
    }

    LOG(INFO) << "Find " << num_good_pts << " in the last image.";
    return num_good_pts;
}

int Frontend::EstimateCurrentPose() {
    // setup g2o
    typedef g2o::BlockSolver_6_3 BlockSolverType;
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;
    
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    // vertex
    // camera vertex_pose
    VertexPose *vertex_pose = new VertexPose();  
    vertex_pose->setId(0);
    vertex_pose->setEstimate(current_frame_->Pose());
    optimizer.addVertex(vertex_pose);

    // K
    Mat33 K = camera_left_->K();

    // edges
    int index = 1;
    std::vector<EdgeProjectionPoseOnly *> edges;
    std::vector<Feature::Ptr> features;

    std::vector<std::shared_ptr<myslam::Feature>> left_current_features;
    left_current_features = current_frame_->features_left_;
    
    for (size_t i = 0; i < left_current_features.size(); ++i) {
        auto mp = left_current_features[i]->map_point_.lock();
        
        if (mp) {
            features.push_back(left_current_features[i]);

            EdgeProjectionPoseOnly *edge = new EdgeProjectionPoseOnly(mp->pos_, K);
            edge->setId(index);
            edge->setVertex(0, vertex_pose);
            edge->setMeasurement(toVec2(left_current_features[i]->position_.pt));
            edge->setInformation(Eigen::Matrix2d::Identity());

            // 穩健核心函數：避免在發生誤比對時，誤差值增長過快，導致該誤差對估計影響過大
            edge->setRobustKernel(new g2o::RobustKernelHuber);

            edges.push_back(edge);
            optimizer.addEdge(edge);

            index++;
        }
    }

    // estimate the Pose the determine the outliers
    const double chi2_th = 5.991;
    int cnt_outlier = 0;
    
    for (int iteration = 0; iteration < 4; ++iteration) {
        vertex_pose->setEstimate(current_frame_->Pose());
        optimizer.initializeOptimization();

        // 優化 10 次
        optimizer.optimize(10);

        // 計算 outlier 個數
        cnt_outlier = 0;

        // count the outliers
        // 優化完成後，對每一條邊都進行檢查
        for (size_t i = 0; i < edges.size(); ++i) {
            auto e = edges[i];
            
            // features[i]->is_outlier_ 預設為 false，因此不會呼叫到 computeError
            // if (features[i]->is_outlier_) {
            //     e->computeError();
            // }
            // 這裡應該所有 edge 都計算誤差，已修改為自己認為的版本
            e->computeError();
            
            /*
            computeError 若未被呼叫，也無法利用 e->chi2() 來計算誤差分配的卡方值，
            因為 e->chi2() 僅在 e->computeError() 後才有效。

            同樣的，也無法將 features[i]->is_outlier_ 設置為 true

            當誤差分配的卡方值大於設定的門檻，剔除誤差較大的邊（認爲是錯誤的邊），並設置 setLevel 爲 1，
            即下次不再對該邊進行優化 
             */ 
            if (e->chi2() > chi2_th) {
                features[i]->is_outlier_ = true;
                e->setLevel(1);
                cnt_outlier++;
            }             
            else {
                features[i]->is_outlier_ = false;                
                e->setLevel(0);                
            };

            // 前面幾次 iteration 已經離群點給移除，因此可不再需要使用 RobustKernel
            if (iteration == 2) {
                e->setRobustKernel(nullptr);
            }
        }
    }

    LOG(INFO) << "Outlier/Inlier in pose estimating: " << cnt_outlier << "/"
              << features.size() - cnt_outlier;
              
    // Set pose and outlier
    current_frame_->SetPose(vertex_pose->estimate());

    LOG(INFO) << "Current Pose = \n" << current_frame_->Pose().matrix();

    for (auto &feat : features) {
        if (feat->is_outlier_) {
            // reset()：手動釋放記憶體
            feat->map_point_.reset();
            
            // maybe we can still use it in future
            feat->is_outlier_ = false;  
        }
    }

    return features.size() - cnt_outlier;
}

bool Frontend::InsertKeyframe() {
    if (tracking_inliers_ >= num_features_needed_for_keyframe_) {
        // still have enough features, but won't insert keyframe
        return false;
    }
    
    // 有效特徵點多於門檻 num_features_needed_for_keyframe_，將此頁框設為 KeyFrame
    // current frame is a new keyframe
    current_frame_->SetKeyFrame();
    map_->InsertKeyFrame(current_frame_);

    LOG(INFO) << "Set frame " << current_frame_->id_ << " as keyframe "
              << current_frame_->keyframe_id_;

    // 更新觀測到 MapPoint 的 Feature
    SetObservationsForKeyFrame();

    // detect new features
    DetectFeatures();  

    // track in right image
    FindFeaturesInRight();

    // triangulate map points
    TriangulateNewPoints();

    // update backend because we have a new keyframe
    backend_->UpdateMap();

    if (viewer_){
        // 更新關鍵頁框與路標點的字典（用於呈現）
        viewer_->UpdateMap();
    }

    return true;
}

void Frontend::SetObservationsForKeyFrame() {
    for (auto &feat : current_frame_->features_left_) {
        auto mp = feat->map_point_.lock();
        
        if (mp){
            mp->AddObservation(feat);
        }
    }
}

int Frontend::TriangulateNewPoints() {
    std::vector<SE3> poses{camera_left_->pose(), camera_right_->pose()};
    SE3 current_pose_Twc = current_frame_->Pose().inverse();
    int cnt_triangulated_pts = 0;

    for (size_t i = 0; i < current_frame_->features_left_.size(); ++i) {

        // .expired() 查看 weak_ptr 使否可用
        if (current_frame_->features_left_[i]->map_point_.expired() &&
            current_frame_->features_right_[i] != nullptr) {
                
            // 左圖的特征點未關聯地圖點且存在右圖匹配點，嘗試三角化
            // 根據相機內參，將像素點轉換成左右相機在成像平面上的點（深度為 1）
            std::vector<Vec3> points{
                camera_left_->pixel2camera(
                    Vec2(current_frame_->features_left_[i]->position_.pt.x,
                         current_frame_->features_left_[i]->position_.pt.y)),
                camera_right_->pixel2camera(
                    Vec2(current_frame_->features_right_[i]->position_.pt.x,
                         current_frame_->features_right_[i]->position_.pt.y))};

            Vec3 pworld = Vec3::Zero();

            // 利用三角測量，計算空間點 pworld
            if (triangulation(poses, points, pworld) && pworld[2] > 0) {
                auto new_map_point = MapPoint::CreateNewMappoint();

                // current_pose_Tw：從『相機座標系』到『世界座標系』
                pworld = current_pose_Twc * pworld;

                new_map_point->SetPos(pworld);

                // MapPoint 會紀錄觀測到自己的 Feature
                new_map_point->AddObservation(current_frame_->features_left_[i]);
                new_map_point->AddObservation(current_frame_->features_right_[i]);

                // Feature 也會紀錄自己觀測到的 MapPoint
                current_frame_->features_left_[i]->map_point_ = new_map_point;
                current_frame_->features_right_[i]->map_point_ = new_map_point;

                map_->InsertMapPoint(new_map_point);

                cnt_triangulated_pts++;
            }
        }
    }

    LOG(INFO) << "new landmarks: " << cnt_triangulated_pts;
    return cnt_triangulated_pts;
}

bool Frontend::Reset() {
    LOG(INFO) << "Reset is not implemented. ";
    return true;
}

}  // namespace myslam
