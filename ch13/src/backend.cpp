//
// Created by gaoxiang on 19-5-2.
//

#include "myslam/backend.h"
#include "myslam/algorithm.h"
#include "myslam/feature.h"
#include "myslam/g2o_types.h"
#include "myslam/map.h"
#include "myslam/mappoint.h"

namespace myslam {

Backend::Backend() {
    backend_running_.store(true);

    // std::thread 參考：https://shengyu7697.github.io/std-thread/
    /* 在一個 class 的 member function 裡面，事實上都會比原本的參數再多一個隱藏變數 this,
    this 是一個 constant pointer 指向 call 這個 function 的 object 

    class Simple
    {
        private:
            int m_id;

        public:
            Simple(int id)
            {
                setID(id);
            }

            void setID(int id) { m_id = id; }
            int getID() { return m_id; }
    };

    compiler 會把一個 class 的 member function 多一個變數 實際上的 setID() 的 signature 變成這樣

    void setID(Simple* const this, int id) { this->m_id = id; }

    Q1: Simple* const this?? const 不是不可以變的意思嗎?
    A1: const 擺在這裡代表 this 是一個 const pointer 代表說這個 pointer 只能指到一個固定的 address，
    指的地方不能改。那如果 const 擺前面 const Simple* this 那就是 this 的值不能改。
    
    再具體一點 如果 this 是一個房子的地址，前者是地址不能變，後者是房子裡面住的人不能變。

    參考：https://www.jyt0532.com/2017/01/08/bind/
    */
    backend_thread_ = std::thread(std::bind(&Backend::BackendLoop, this));
}

void Backend::UpdateMap() {
    std::unique_lock<std::mutex> lock(data_mutex_);
    map_update_.notify_one();
}

void Backend::Stop() {
    backend_running_.store(false);
    map_update_.notify_one();
    backend_thread_.join();
}

void Backend::BackendLoop() {
    while (backend_running_.load()) {
        std::unique_lock<std::mutex> lock(data_mutex_);
        map_update_.wait(lock);

        /// 後端僅優化激活的 Frames 和 Landmarks
        // KeyframesType 為字典(key: unsigned long; value: Frame::Ptr)
        Map::KeyframesType active_kfs = map_->GetActiveKeyFrames();

        // LandmarksType 為字典(key: unsigned long; value: MapPoint::Ptr)
        Map::LandmarksType active_landmarks = map_->GetActiveMapPoints();

        // 根據 KeyframesType 和 LandmarksType 對 keyframe 們的位姿估計進行優化
        Optimize(active_kfs, active_landmarks);
    }
}

// KeyframesType 為字典(key: unsigned long; value: Frame::Ptr)
// LandmarksType 為字典(key: unsigned long; value: MapPoint::Ptr)
// 批次對『Frame 的位姿估計 以及 路標點的位置』進行優化
void Backend::Optimize(Map::KeyframesType &keyframes, Map::LandmarksType &landmarks) {
    // setup g2o
    typedef g2o::BlockSolver_6_3 BlockSolverType;
    typedef g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType> LinearSolverType;
    
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    // pose 頂點，使用 Keyframe id
    std::map<unsigned long, VertexPose *> vertices;
    unsigned long max_kf_id = 0;

    // 遍歷 Frame 字典
    for (auto &keyframe : keyframes) {
        auto kf = keyframe.second;

        // camera vertex_pose
        VertexPose *vertex_pose = new VertexPose();  
        
        vertex_pose->setId(kf->keyframe_id_);

        // 優化關鍵 Frame 的位姿
        vertex_pose->setEstimate(kf->Pose());

        optimizer.addVertex(vertex_pose);
        
        if (kf->keyframe_id_ > max_kf_id) {
            max_kf_id = kf->keyframe_id_;
        }

        vertices.insert({kf->keyframe_id_, vertex_pose});
    }

    // 路標頂點，使用路標 id 索引
    std::map<unsigned long, VertexXYZ *> vertices_landmarks;

    // K 和左右外參
    Mat33 K = cam_left_->K();
    SE3 left_ext = cam_left_->pose();
    SE3 right_ext = cam_right_->pose();

    // edges
    int index = 1;
    
    // robust kernel 閾值
    double chi2_th = 5.991;  
    
    std::map<EdgeProjection *, Feature::Ptr> edges_and_features;

    // 遍歷 MapPoint 字典
    for (auto &landmark : landmarks) {
        if (landmark.second->is_outlier_){
            continue;
        }
        
        unsigned long landmark_id = landmark.second->id_;

        // 利用 MapPoint 取得 Feature 陣列
        auto observations = landmark.second->GetObs();
        
        for (auto &obs : observations) {
            // 檢查是否可以取用該 Feature
            if (obs.lock() == nullptr){
                continue;
            }
            
            // 取得 Feature
            auto feat = obs.lock();
            
            // 檢查是否可以取用含有該 Feature 的 Frame，以及該 Feature 是否可以取用
            if (feat->is_outlier_ || feat->frame_.lock() == nullptr){
                continue;
            }

            // 取得含有該 Feature 的 Frame
            auto frame = feat->frame_.lock();
            EdgeProjection *edge = nullptr;
            
            if (feat->is_on_left_image_) {
                edge = new EdgeProjection(K, left_ext);                
            } else {
                edge = new EdgeProjection(K, right_ext);                
            }

            // 如果 landmark 還沒有被加入優化，則新加一個頂點
            if (vertices_landmarks.find(landmark_id) == vertices_landmarks.end()) {
                VertexXYZ *v = new VertexXYZ;

                // 取得路標的 MapPoint 的位置點(Vec3)
                v->setEstimate(landmark.second->Pos());
                v->setId(landmark_id + max_kf_id + 1);
                v->setMarginalized(true);

                // std::map<unsigned long, VertexXYZ *> vertices_landmarks
                vertices_landmarks.insert({landmark_id, v});
                optimizer.addVertex(v);
            }

            edge->setId(index);
            
            // edge->setVertex(int, Vertex*)
            // pose VertexPose*
            edge->setVertex(0, vertices.at(frame->keyframe_id_));    
            
            // landmark VertexXYZ*
            edge->setVertex(1, vertices_landmarks.at(landmark_id)); 
            
            edge->setMeasurement(toVec2(feat->position_.pt));
            edge->setInformation(Mat22::Identity());

            auto rk = new g2o::RobustKernelHuber();
            rk->setDelta(chi2_th);

            // 
            edge->setRobustKernel(rk);

            // 
            edges_and_features.insert({edge, feat});

            optimizer.addEdge(edge);

            index++;
        }
    }

    // do optimization and eliminate the outliers
    optimizer.initializeOptimization();

    // 進行 10 次優化
    optimizer.optimize(10);

    int cnt_outlier = 0, cnt_inlier = 0;
    int iteration = 0;
    
    // 尋找適當的卡方門檻值
    while (iteration < 5) {
        cnt_outlier = 0;
        cnt_inlier = 0;
        
        // determine if we want to adjust the outlier threshold
        for (auto &ef : edges_and_features) {
            if (ef.first->chi2() > chi2_th) {
                cnt_outlier++;
                
            } else {
                cnt_inlier++;
                
            }
        }
        
        double inlier_ratio = cnt_inlier / double(cnt_inlier + cnt_outlier);
        
        // inlier 比例足夠多時，表示卡方門檻值足夠大了
        if (inlier_ratio > 0.5) {
            break;
            
        } 
        
        // inlier 數量不足，擴大卡方門檻值
        else {
            chi2_th *= 2;
            iteration++;
            
        }
    }

    // 移除 Feature 的離群值
    for (auto &ef : edges_and_features) {
        if (ef.first->chi2() > chi2_th) {
            ef.second->is_outlier_ = true;
            // remove the observation
            ef.second->map_point_.lock()->RemoveObservation(ef.second);
            
        } else {
            ef.second->is_outlier_ = false;
        }
    }

    LOG(INFO) << "Outlier/Inlier in optimization: " << cnt_outlier << "/"
              << cnt_inlier;

    // 更新關鍵頁框的位姿估計
    for (auto &v : vertices) {
        keyframes.at(v.first)->SetPose(v.second->estimate());
    }
    
    // 更新路標點的 MapPoint 的位置
    for (auto &v : vertices_landmarks) {
        landmarks.at(v.first)->SetPos(v.second->estimate());
    }
}

}  // namespace myslam
