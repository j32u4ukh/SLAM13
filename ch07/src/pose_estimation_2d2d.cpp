#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
// #include "extra.h" // use this if in OpenCV2

using namespace std;
using namespace cv;

/****************************************************
 * 本程序演示了如何使用2D-2D的特征匹配估計相機運動
 * **************************************************/

// Use ORB to detect, compute and match
void find_feature_matches(
  const Mat &img_1, const Mat &img_2,
  std::vector<KeyPoint> &keypoints_1,
  std::vector<KeyPoint> &keypoints_2,
  std::vector<DMatch> &matches);

void pose_estimation_2d2d(
  std::vector<KeyPoint> keypoints_1,
  std::vector<KeyPoint> keypoints_2,
  std::vector<DMatch> matches,
  Mat &R, Mat &t);

// 像素坐標轉相機歸一化坐標
Point2d pixel2cam(const Point2d &p, const Mat &K);

int main(int argc, char **argv) {
    if (argc != 3) {
        cout << "usage: pose_estimation_2d2d img1 img2" << endl;
        return 1;
    }
    
    //-- 讀取圖像
    Mat img_1 = imread(argv[1], cv::IMREAD_COLOR);
    Mat img_2 = imread(argv[2], cv::IMREAD_COLOR);
    assert(img_1.data && img_2.data && "Can not load images!");

    vector<KeyPoint> keypoints_1, keypoints_2;
    vector<DMatch> matches;
    find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
    cout << "一共找到了" << matches.size() << "組匹配點" << endl;

    //-- 估計兩張圖像間運動
    Mat R, t;
    pose_estimation_2d2d(keypoints_1, keypoints_2, matches, R, t);
    
    //-- 驗證 E = t^ * R * scale
    // t -> t^ 反對稱矩陣
    Mat t_x = (Mat_<double>(3, 3) <<                  0, -t.at<double>(2, 0),  t.at<double>(1, 0),
                                     t.at<double>(2, 0),                   0, -t.at<double>(0, 0),
                                    -t.at<double>(1, 0),  t.at<double>(0, 0),                   0);

    Mat essential_matrix = t_x * R;
    cout << "essential_matrix = t^R =" << endl << essential_matrix << endl;

    //-- 驗證對極約束
    // K: 相機內參
    Mat K = (Mat_<double>(3, 3) <<  520.9,      0,  325.1, 
                                        0,  521.0,  249.7, 
                                        0,      0,      1);
    Mat fundamental_matrix = K.inv().t() * essential_matrix * K.inv();
    cout << "fundamental_matrix = K_-T * essential_matrix * K_-1 =" << endl << fundamental_matrix << endl;
    
    for (DMatch m: matches) {
        // pixel2cam: 像素坐標轉相機歸一化坐標
        Point2d pt1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);

        // 歸一化坐標
        Mat y1 = (Mat_<double>(3, 1) << pt1.x, pt1.y, 1);

        Point2d pt2 = pixel2cam(keypoints_2[m.trainIdx].pt, K);
        Mat y2 = (Mat_<double>(3, 1) << pt2.x, pt2.y, 1);

        // 對極約束1（理論值應為 0）
        // Mat d1 = y2.t() * t_x * R * y1;
        Mat d1 = y2.t() * fundamental_matrix * y1;
        cout << "epipolar constraint1 = " << d1 << endl;

        // 對極約束2（理論值應為 0） p1, p2 為空間點座標，根據尺度相似關係，Z 值設為 1 應該也是可以的
        Mat p1 = (Mat_<double>(3, 1) << keypoints_1[m.queryIdx].pt.x, keypoints_1[m.queryIdx].pt.y, 1);
        Mat p2 = (Mat_<double>(3, 1) << keypoints_2[m.queryIdx].pt.x, keypoints_2[m.queryIdx].pt.y, 1);
        Mat d2 = p2.t() * essential_matrix * p1;
        cout << "epipolar constraint2 = " << d2 << endl;
    }
    
    return 0;
}

void find_feature_matches(const Mat &img_1, const Mat &img_2,
                          std::vector<KeyPoint> &keypoints_1,
                          std::vector<KeyPoint> &keypoints_2,
                          std::vector<DMatch> &matches) {
    //-- 初始化
    Mat descriptors_1, descriptors_2;
    
    // used in OpenCV3
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    
    // use this if you are in OpenCV2
    // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
    // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    
    //-- 第一步:檢測 Oriented FAST 角點位置
    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);

    //-- 第二步:根據角點位置計算 BRIEF 描述子
    descriptor->compute(img_1, keypoints_1, descriptors_1);
    descriptor->compute(img_2, keypoints_2, descriptors_2);

    //-- 第三步:對兩幅圖像中的BRIEF描述子進行匹配，使用 Hamming 距離
    vector<DMatch> match;
    
    //BFMatcher matcher ( NORM_HAMMING );
    matcher->match(descriptors_1, descriptors_2, match);

    //-- 第四步:匹配點對篩選
    double min_dist = 10000, max_dist = 0;

    //找出所有匹配之間的最小距離和最大距離, 即是最相似的和最不相似的兩組點之間的距離
    for (int i = 0; i < descriptors_1.rows; i++) {
        double dist = match[i].distance;
        
        if (dist < min_dist){
            min_dist = dist;
        }
        
        if (dist > max_dist) {            
            max_dist = dist;
        }
    }

    printf("-- Max dist : %f \n", max_dist);
    printf("-- Min dist : %f \n", min_dist);

    //當描述子之間的距離大於兩倍的最小距離時,即認為匹配有誤.但有時候最小距離會非常小,設置一個經驗值30作為下限.
    for (int i = 0; i < descriptors_1.rows; i++) {
        if (match[i].distance <= max(2 * min_dist, 30.0)) {
            matches.push_back(match[i]);
        }
    }
}

// 像素坐標轉相機歸一化坐標
Point2d pixel2cam(const Point2d &p, const Mat &K) {
  return Point2d
    (
      (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
      (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
    );
}

void pose_estimation_2d2d(std::vector<KeyPoint> keypoints_1,
                          std::vector<KeyPoint> keypoints_2,
                          std::vector<DMatch> matches,
                          Mat &R, Mat &t) {
    // 相機內參,TUM Freiburg2
    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

    //-- 把匹配點轉換為vector<Point2f>的形式
    vector<Point2f> points1;
    vector<Point2f> points2;

    for (int i = 0; i < (int) matches.size(); i++) {
        points1.push_back(keypoints_1[matches[i].queryIdx].pt);
        points2.push_back(keypoints_2[matches[i].trainIdx].pt);
    }

    //-- 計算基礎矩陣
    Mat fundamental_matrix;

    // cv::findFundamentalMat：計算基礎矩陣(fundamental_matrix)
    // FM_8POINT： 使用八點法來計算
    fundamental_matrix = findFundamentalMat(points1, points2, FM_8POINT);
    cout << "fundamental_matrix is " << endl << fundamental_matrix << endl;

    //-- 計算本質矩陣
    //相機光心, TUM dataset 標定值
    Point2d principal_point(325.1, 249.7);  
    
    //相機焦距, TUM dataset標定值
    double focal_length = 521;      
    
    Mat essential_matrix;
    essential_matrix = findEssentialMat(points1, points2, focal_length, principal_point);
    cout << "essential_matrix is " << endl << essential_matrix << endl;

    //-- 計算單應矩陣
    //-- 但是本例中場景不是平面，單應矩陣意義不大
    Mat homography_matrix;

    // cv::findHomography
    homography_matrix = findHomography(points1, points2, RANSAC, 3);
    cout << "homography_matrix is " << endl << homography_matrix << endl;

    //-- 從本質矩陣中恢覆旋轉和平移信息.
    // 此函數僅在Opencv3中提供 cv::recoverPose
    recoverPose(essential_matrix, points1, points2, R, t, focal_length, principal_point);
    cout << "R is " << endl << R << endl;
    cout << "t is " << endl << t << endl;

    // 利用公式計算的 fundamental_matrix
    fundamental_matrix = K.inv().t() * essential_matrix * K.inv();
    cout << "fundamental_matrix, compute from essential_matrix" << endl << fundamental_matrix << endl;
}
