#include <iostream>
#include <fstream>

using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <octomap/octomap.h>    // for octomap 

#include <Eigen/Geometry>
#include <boost/format.hpp>  // for formating strings

int main(int argc, char **argv) {
    // 彩色圖和深度圖
    vector<cv::Mat> colorImgs, depthImgs;    

    // 相機位姿
    vector<Eigen::Isometry3d> poses;         

    ifstream fin("./data/pose.txt");

    if (!fin) {
        cerr << "cannot find pose file" << endl;
        return 1;
    }

    for (int i = 0; i < 5; i++) {
        //圖像文件格式
        boost::format fmt("./data/%s/%d.%s"); 

        colorImgs.push_back(cv::imread((fmt % "color" % (i + 1) % "png").str()));

        // 使用 -1 讀取原始深度圖
        depthImgs.push_back(cv::imread((fmt % "depth" % (i + 1) % "png").str(), -1)); 

        double data[7] = {0};

        for (int i = 0; i < 7; i++) {
            fin >> data[i];
        }

        Eigen::Quaterniond q(data[6], data[3], data[4], data[5]);
        Eigen::Isometry3d T(q);
        T.pretranslate(Eigen::Vector3d(data[0], data[1], data[2]));
        poses.push_back(T);
    }

    // 計算點雲並拼接
    // 相機內參 
    double cx = 319.5;
    double cy = 239.5;
    double fx = 481.2;
    double fy = -480.0;

    // 控制模型規模
    double depthScale = 5000.0;

    cout << "正在將圖像轉換為 Octomap ..." << endl;

    // octomap tree 
    octomap::OcTree tree(0.01); // 參數為分辨率

    for (int i = 0; i < 5; i++) {
        cout << "轉換圖像中: " << i + 1 << endl;
        cv::Mat color = colorImgs[i];
        cv::Mat depth = depthImgs[i];

        // 從數據集讀取相機位姿
        Eigen::Isometry3d T = poses[i];

        // the point cloud in octomap 
        octomap::Pointcloud cloud;  

        for (int v = 0; v < color.rows; v++)
            for (int u = 0; u < color.cols; u++) {

                // 深度值
                unsigned int d = depth.ptr<unsigned short>(v)[u]; 

                // 為 0 表示沒有測量到
                if (d == 0) {
                    continue; 
                }

                Eigen::Vector3d point;
                point[2] = double(d) / depthScale;
                point[0] = (u - cx) * point[2] / fx;
                point[1] = (v - cy) * point[2] / fy;

                // 同一張圖像內的所有點都共用同一個相機位姿 T
                Eigen::Vector3d pointWorld = T * point;

                // 將世界坐標系的點放入點雲
                cloud.push_back(pointWorld[0], pointWorld[1], pointWorld[2]);
            }

        // 將點雲存入八叉樹地圖，給定原點，這樣可以計算投射線
        tree.insertPointCloud(cloud, octomap::point3d(T(0, 3), T(1, 3), T(2, 3)));
    }

    // 更新中間節點的占據信息並寫入磁盤
    tree.updateInnerOccupancy();

    // 儲存八叉樹
    cout << "saving octomap ... " << endl;
    tree.writeBinary("octomap.bt");

    return 0;
}
