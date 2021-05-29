#include <iostream>
#include <opencv2/opencv.hpp>
// #include "extra.h" // used in opencv2
using namespace std;
using namespace cv;

// Use ORB to detect, compute and match
void find_feature_matches(
  const Mat &img_1, const Mat &img_2,
  std::vector<KeyPoint> &keypoints_1,
  std::vector<KeyPoint> &keypoints_2,
  std::vector<DMatch> &matches);

// essential matrix -> R, t
void pose_estimation_2d2d(
  const std::vector<KeyPoint> &keypoints_1,
  const std::vector<KeyPoint> &keypoints_2,
  const std::vector<DMatch> &matches,
  Mat &R, Mat &t);

void triangulation(
  const vector<KeyPoint> &keypoint_1,
  const vector<KeyPoint> &keypoint_2,
  const std::vector<DMatch> &matches,
  const Mat &R, const Mat &t,
  vector<Point3d> &points
);

/// 作圖用
inline cv::Scalar get_color(float depth) {
    float up_th = 50, low_th = 10, th_range = up_th - low_th;
    
    if (depth > up_th){
        depth = up_th;
    } 
    
    if (depth < low_th){
        depth = low_th;
    }
    
    return cv::Scalar(255 * depth / th_range, 0, 255 * (1 - depth / th_range));
}

// 像素坐標轉相機歸一化坐標
Point2f pixel2cam(const Point2d &p, const Mat &K);

int main(int argc, char **argv) {
    if (argc != 3) {
        cout << "usage: triangulation img1 img2" << endl;
        return 1;
    }
    
    //-- 讀取圖像
    Mat img_1 = imread(argv[1], cv::IMREAD_COLOR);
    Mat img_2 = imread(argv[2], cv::IMREAD_COLOR);

    vector<KeyPoint> keypoints_1, keypoints_2;
    vector<DMatch> matches;
    find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
    cout << "一共找到了" << matches.size() << "組匹配點" << endl;

    //-- 估計兩張圖像間運動
    Mat R, t;
    pose_estimation_2d2d(keypoints_1, keypoints_2, matches, R, t);

    //-- 三角化
    vector<Point3d> points;

    // 利用投影點，取得空間點
    triangulation(keypoints_1, keypoints_2, matches, R, t, points);

    //-- 驗證三角化點與特征點的重投影關系
    Mat K = (Mat_<double>(3, 3) <<  520.9,      0,  325.1, 
                                        0,  521.0,  249.7, 
                                        0,      0,      1);
    Mat img1_plot = img_1.clone();
    Mat img2_plot = img_2.clone();
    
    for (int i = 0; i < matches.size(); i++) {
        // 第一個圖
        float depth1 = points[i].z;
        cout << "depth: " << depth1 << endl;
        Point2d pt1_cam = pixel2cam(keypoints_1[matches[i].queryIdx].pt, K);
        cv::circle(img1_plot, keypoints_1[matches[i].queryIdx].pt, 2, get_color(depth1), 2);

        // 第二個圖
        Mat pt2_trans = R * (Mat_<double>(3, 1) << points[i].x, points[i].y, points[i].z) + t;
        float depth2 = pt2_trans.at<double>(2, 0);
        cv::circle(img2_plot, keypoints_2[matches[i].trainIdx].pt, 2, get_color(depth2), 2);
    }
    
    cv::imshow("img 1", img1_plot);
    cv::imshow("img 2", img2_plot);
    cv::waitKey();

    return 0;
}

// Use ORB to detect, compute and match
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
    
    // BFMatcher matcher ( NORM_HAMMING );
    matcher->match(descriptors_1, descriptors_2, match);

    //-- 第四步:匹配點對篩選
    double min_dist = 10000, max_dist = 0;

    //找出所有匹配之間的最小距離和最大距離, 即是最相似的和最不相似的兩組點之間的距離
    for (int i = 0; i < descriptors_1.rows; i++) {
        double dist = match[i].distance;
        
        if (dist < min_dist){            
            min_dist = dist;
        }
        
        if (dist > max_dist){
            max_dist = dist;
        }
    }

    printf("-- Max dist : %f \n", max_dist);
    printf("-- Min dist : %f \n", min_dist);

    // 當描述子之間的距離大於兩倍的最小距離時,即認為匹配有誤.但有時候最小距離會非常小,
    // 設置一個經驗值30作為下限.
    for (int i = 0; i < descriptors_1.rows; i++) {
        if (match[i].distance <= max(2 * min_dist, 30.0)) {
            matches.push_back(match[i]);
        }
    }
}

// essential matrix -> R, t
void pose_estimation_2d2d(
  const std::vector<KeyPoint> &keypoints_1,
  const std::vector<KeyPoint> &keypoints_2,
  const std::vector<DMatch> &matches,
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

    //-- 計算本質矩陣
    //相機主點, TUM dataset標定值
    Point2d principal_point(325.1, 249.7);    

    //相機焦距, TUM dataset標定值
    int focal_length = 521;      
    
    Mat essential_matrix;
    essential_matrix = findEssentialMat(points1, points2, focal_length, principal_point);

    //-- 從本質矩陣中恢覆旋轉和平移信息.
    recoverPose(essential_matrix, points1, points2, R, t, focal_length, principal_point);
}

void triangulation(
  const vector<KeyPoint> &keypoint_1,
  const vector<KeyPoint> &keypoint_2,
  const std::vector<DMatch> &matches,
  const Mat &R, const Mat &t,
  vector<Point3d> &points) {      
    // Camera1 -> 未旋轉或平移
    Mat T1 = (Mat_<float>(3, 4) <<
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0);

    // Camera2 -> 相對於 Camera1 旋轉 & 平移
    Mat T2 = (Mat_<float>(3, 4) <<
        R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0, 0),
        R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
        R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0)
    );

    // 相機內參
    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    vector<Point2f> pts_1, pts_2;

    for (DMatch m:matches) {
        // 將像素坐標轉換至相機坐標
        pts_1.push_back(pixel2cam(keypoint_1[m.queryIdx].pt, K));
        pts_2.push_back(pixel2cam(keypoint_2[m.trainIdx].pt, K));
    }

    // 利用三角測量，恢復匹配點的深度（pts_4d：以齊次座標形式呈現）
    Mat pts_4d;
    cv::triangulatePoints(T1, T2, pts_1, pts_2, pts_4d);

    // 轉換成非齊次坐標(x, y, z)
    for (int i = 0; i < pts_4d.cols; i++) {
        Mat x = pts_4d.col(i);
        
        // 歸一化
        x /= x.at<float>(3, 0); 
        
        Point3d p(
        x.at<float>(0, 0),
        x.at<float>(1, 0),
        x.at<float>(2, 0)
        );
        
        points.push_back(p);
    }
}

Point2f pixel2cam(const Point2d &p, const Mat &K) {
  return Point2f
    (
      (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
      (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
    );
}

