#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <boost/format.hpp>  // for formating strings
#include <sophus/se3.hpp>
#include <pangolin/pangolin.h>

#include "slam.hpp"

using namespace std;
typedef vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> TrajectoryType;
typedef Eigen::Matrix<double, 6, 1> Vector6d;

// 在pangolin中畫圖，已寫好，無需調整
void showPointCloud(
    const vector<Vector6d, Eigen::aligned_allocator<Vector6d>> &pointcloud);

int main(int argc, char **argv) {
    // 彩色圖和深度圖
    vector<cv::Mat> colorImgs, depthImgs;    
    
    // 相機位姿
    TrajectoryType poses;         

    ifstream fin("./pose.txt");
    
    if (!fin) {
        cerr << "請在有pose.txt的目錄下運行此程序" << endl;
        return 1;
    }

    for (int i = 0; i < 5; i++) {
        boost::format fmt("./%s/%d.%s"); //圖像文件格式
        colorImgs.push_back(cv::imread((fmt % "color" % (i + 1) % "png").str()));
        depthImgs.push_back(cv::imread((fmt % "depth" % (i + 1) % "pgm").str(), -1)); // 使用-1讀取原始圖像

        double data[7] = {0};
        
        for (auto &d:data){
            fin >> d;
        }
        
        // 計算轉換矩陣（相機的位姿）
        Sophus::SE3d pose(Eigen::Quaterniond(data[6], data[3], data[4], data[5]), 
                          Eigen::Vector3d(data[0], data[1], data[2]));
        poses.push_back(pose);
    }

    // 相機內參 
    double cx = 325.5;
    double cy = 253.5;
    double fx = 518.0;
    double fy = 519.0;
    double depthScale = 1000.0;
    
    vector<Vector6d, Eigen::aligned_allocator<Vector6d>> pointcloud;

    // reserve: 用於指定長度，避免反覆配置記憶體
    // 參考：http://pingyeh.blogspot.com/2011/09/stl-vector.html
    pointcloud.reserve(1000000);

    // 計算點雲並拼接
    for (int i = 0; i < 5; i++) {
        cout << "轉換圖像中: " << i + 1 << endl;
        
        // 顏色圖（原始圖片）
        cv::Mat color = colorImgs[i];
        
        //  深度圖（事先估計好的深度）
        cv::Mat depth = depthImgs[i];
        
        // 轉換矩陣
        Sophus::SE3d T = poses[i];
        
        for (int v = 0; v < color.rows; v++){
            for (int u = 0; u < color.cols; u++) {
                // 深度值
                unsigned int d = depth.ptr<unsigned short>(v)[u]; 
                
                // 為0表示沒有測量到
                if (d == 0){
                    continue; 
                }
                
                Eigen::Vector3d point;
                
                // Z
                point[2] = double(d) / depthScale;
                
                // X
                point[0] = (u - cx) * point[2] / fx;
                
                // Y
                point[1] = (v - cy) * point[2] / fy;
                
                // 相機座標系下的點，轉換到世界座標
                Eigen::Vector3d pointWorld = T * point;

                Vector6d p;

                // 參考：https://eigen.tuxfamily.org/dox/group__TutorialBlockOperations.html
                p.head<3>() = pointWorld;
                
                // blue
                p[5] = color.data[v * color.step + u * color.channels()] / 255.0;   
                
                // green
                p[4] = color.data[v * color.step + u * color.channels() + 1] / 255.0; 
                
                // red
                p[3] = color.data[v * color.step + u * color.channels() + 2] / 255.0; 
                
                pointcloud.push_back(p);
            }
        }
    }

    cout << "點雲共有" << pointcloud.size() << "個點。。" << endl;
    
    // pointcloud: vector<Vector6d, Eigen::aligned_allocator<Vector6d>> 
    // showPointCloud(pointcloud);
    displayPointCloud(pointcloud);
    return 0;
}

void showPointCloud(const vector<Vector6d, Eigen::aligned_allocator<Vector6d>> &pointcloud) {

    if (pointcloud.empty()) {
        cerr << "Point cloud is empty!" << endl;
        return;
    }

    pangolin::CreateWindowAndBind("Point Cloud Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
        pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
    );

    pangolin::View &d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
        .SetHandler(new pangolin::Handler3D(s_cam));

    while (pangolin::ShouldQuit() == false) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        glPointSize(2);
        glBegin(GL_POINTS);
        
        for (auto &p: pointcloud) {
            glColor3d(p[3] / 255.0, p[4] / 255.0, p[5] / 255.0);
            glVertex3d(p[0], p[1], p[2]);
        }
        
        glEnd();
        pangolin::FinishFrame();
        
        // sleep 5 ms
        usleep(5000);   
    }
    return;
}
