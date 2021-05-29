/*
 * <one line to give the program's name and a brief idea of what it does.>
 * Copyright (C) 2016  <copyright holder> <email>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include "myslam/map.h"
#include "myslam/feature.h"

namespace myslam {

void Map::InsertKeyFrame(Frame::Ptr frame) {
    current_frame_ = frame;
    
    if (keyframes_.find(frame->keyframe_id_) == keyframes_.end()) {
        keyframes_.insert(make_pair(frame->keyframe_id_, frame));
        active_keyframes_.insert(make_pair(frame->keyframe_id_, frame));
        
    } else {
        keyframes_[frame->keyframe_id_] = frame;
        active_keyframes_[frame->keyframe_id_] = frame;
        
    }

    // 關鍵幀僅保留 num_active_keyframes_ 個，數量超出時則刪除舊的關鍵幀
    if (active_keyframes_.size() > num_active_keyframes_) {
        RemoveOldKeyframe();
    }
}

void Map::InsertMapPoint(MapPoint::Ptr map_point) {
    if (landmarks_.find(map_point->id_) == landmarks_.end()) {
        landmarks_.insert(make_pair(map_point->id_, map_point));
        active_landmarks_.insert(make_pair(map_point->id_, map_point));
        
    } else {
        landmarks_[map_point->id_] = map_point;
        active_landmarks_[map_point->id_] = map_point;
    }
}

void Map::RemoveOldKeyframe() {
    if (current_frame_ == nullptr){
        return;
    }
    
    // 尋找與當前幀最近與最遠的兩個關鍵幀
    double max_dis = 0, min_dis = 9999;
    double max_kf_id = 0, min_kf_id = 0;
    auto Twc = current_frame_->Pose().inverse();
    
    for (auto& kf : active_keyframes_) {

        if (kf.second == current_frame_){
            continue;
        }
        
        auto dis = (kf.second->Pose() * Twc).log().norm();
        
        if (dis > max_dis) {
            max_dis = dis;
            max_kf_id = kf.first;
        }
        
        if (dis < min_dis) {
            min_dis = dis;
            min_kf_id = kf.first;
        }
    }

    // 最近閾值
    const double min_dis_th = 0.2;  
    Frame::Ptr frame_to_remove = nullptr;
    
    if (min_dis < min_dis_th) {
        // 如果存在很近的幀，優先刪掉最近的
        frame_to_remove = keyframes_.at(min_kf_id);
        
    } else {
        // 刪掉最遠的
        frame_to_remove = keyframes_.at(max_kf_id);
        
    }

    LOG(INFO) << "remove keyframe " << frame_to_remove->keyframe_id_;
    
    // remove keyframe and landmark observation
    active_keyframes_.erase(frame_to_remove->keyframe_id_);
    
    for (auto feat : frame_to_remove->features_left_) {

        auto mp = feat->map_point_.lock();
        
        if (mp) {
            mp->RemoveObservation(feat);
        }
    }
    
    for (auto feat : frame_to_remove->features_right_) {

        if (feat == nullptr){
            continue;
        }
        
        auto mp = feat->map_point_.lock();
        
        if (mp) {
            mp->RemoveObservation(feat);
        }
    }

    CleanMap();
}

void Map::CleanMap() {
    int cnt_landmark_removed = 0;
    
    for (auto iter = active_landmarks_.begin(); iter != active_landmarks_.end();) {

        // 指標指著同一個位置，但前一次將原本的內容移除了，因此下一次近來會指向下一筆數據
        if (iter->second->observed_times_ == 0) {
            iter = active_landmarks_.erase(iter);
            cnt_landmark_removed++;
            
        } else {
            ++iter;
        }
    }
    
    LOG(INFO) << "Removed " << cnt_landmark_removed << " active landmarks";
}

}  // namespace myslam
