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
        Map::KeyframesType active_kfs = map_->GetActiveKeyFrames();
        Map::LandmarksType active_landmarks = map_->GetActiveMapPoints();
        Optimize(active_kfs, active_landmarks);
    }
}

void Backend::Optimize(Map::KeyframesType &keyframes, Map::LandmarksType &landmarks) {
    // setup g2o
    typedef g2o::BlockSolver_6_3 BlockSolverType;
    typedef g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType> LinearSolverType;
    
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(
            g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    // pose 頂點，使用Keyframe id
    std::map<unsigned long, VertexPose *> vertices;
    unsigned long max_kf_id = 0;
    for (auto &keyframe : keyframes) {
        auto kf = keyframe.second;
        // camera vertex_pose
        VertexPose *vertex_pose = new VertexPose();  
        
        vertex_pose->setId(kf->keyframe_id_);
        vertex_pose->setEstimate(kf->Pose());
        optimizer.addVertex(vertex_pose);
        
        if (kf->keyframe_id_ > max_kf_id) {
            max_kf_id = kf->keyframe_id_;
        }

        vertices.insert({kf->keyframe_id_, vertex_pose});
    }

    // 路標頂點，使用路標id索引
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

    for (auto &landmark : landmarks) {
        if (landmark.second->is_outlier_){
            continue;
        }
        
        unsigned long landmark_id = landmark.second->id_;
        auto observations = landmark.second->GetObs();
        
        for (auto &obs : observations) {
            if (obs.lock() == nullptr){
                continue;
            }
            
            auto feat = obs.lock();
            
            if (feat->is_outlier_ || feat->frame_.lock() == nullptr){
                continue;
            }

            auto frame = feat->frame_.lock();
            EdgeProjection *edge = nullptr;
            
            if (feat->is_on_left_image_) {
                edge = new EdgeProjection(K, left_ext);
                
            } else {
                edge = new EdgeProjection(K, right_ext);
                
            }

            // 如果landmark還沒有被加入優化，則新加一個頂點
            if (vertices_landmarks.find(landmark_id) ==
                vertices_landmarks.end()) {
                VertexXYZ *v = new VertexXYZ;
                v->setEstimate(landmark.second->Pos());
                v->setId(landmark_id + max_kf_id + 1);
                v->setMarginalized(true);
                vertices_landmarks.insert({landmark_id, v});
                optimizer.addVertex(v);
            }

            edge->setId(index);
            
            // pose
            edge->setVertex(0, vertices.at(frame->keyframe_id_));    
            
            // landmark
            edge->setVertex(1, vertices_landmarks.at(landmark_id)); 
            
            edge->setMeasurement(toVec2(feat->position_.pt));
            edge->setInformation(Mat22::Identity());
            auto rk = new g2o::RobustKernelHuber();
            rk->setDelta(chi2_th);
            edge->setRobustKernel(rk);
            edges_and_features.insert({edge, feat});

            optimizer.addEdge(edge);

            index++;
        }
    }

    // do optimization and eliminate the outliers
    optimizer.initializeOptimization();
    optimizer.optimize(10);

    int cnt_outlier = 0, cnt_inlier = 0;
    int iteration = 0;
    
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
        
        if (inlier_ratio > 0.5) {
            break;
            
        } else {
            chi2_th *= 2;
            iteration++;
            
        }
    }

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

    // Set pose and lanrmark position
    for (auto &v : vertices) {
        keyframes.at(v.first)->SetPose(v.second->estimate());
    }
    
    for (auto &v : vertices_landmarks) {
        landmarks.at(v.first)->SetPos(v.second->estimate());
    }
}

}  // namespace myslam
