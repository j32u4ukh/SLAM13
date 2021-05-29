#include <iostream>
#include <vector>
#include <fstream>

using namespace std;

#include <boost/timer.hpp>

// for sophus（若修改了 include 和 using 順序，Sophus 所指涉的對象就會不同導致無法找到 SE3d）
#include <sophus/se3.hpp>

using Sophus::SE3d;

// for eigen
#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace Eigen;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;


/**********************************************
* 本程序演示了單目相機在已知軌跡下的稠密深度估計
* 使用極線搜索 + NCC 匹配的方式，與書本的 12.2 節對應
* 請注意本程序並不完美，你完全可以改進它——我其實在故意暴露一些問題(這是借口)。
***********************************************/

// ------------------------------------------------------------------
// parameters
// 邊緣寬度
const int boarder = 20;   

// 圖像寬度
const int width = 640;    

// 圖像高度
const int height = 480;   

// 相機內參
const double fx =  481.2f;       
const double fy = -480.0f;
const double cx =  319.5f;
const double cy =  239.5f;

// NCC 取的窗口半寬度
const int ncc_window_size = 3; 

// NCC窗口面積
const int ncc_area = (2 * ncc_window_size + 1) * (2 * ncc_window_size + 1); 

// 收斂判定：最小方差
const double min_cov = 0.1;     

// 發散判定：最大方差
const double max_cov = 10;      

// ------------------------------------------------------------------
// 重要的函數
/**
 * @brief 從 REMODE 數據集讀取數據
 * 
 * @param path 數據資料夾
 * @param color_image_files 儲存圖片路徑
 * @param poses 儲存位姿資訊
 * @param ref_depth 儲存深度圖
 * @return true 
 * @return false 
 */
bool readDatasetFiles(
    const string &path,
    vector<string> &color_image_files,
    vector<SE3d> &poses,
    cv::Mat &ref_depth
);

/**
 * 根據新的圖像更新深度估計
 * @param ref           參考圖像
 * @param curr          當前圖像
 * @param T_C_R         參考圖像到當前圖像的位姿
 * @param depth         深度
 * @param depth_cov     深度方差
 * @return              是否成功
 */
bool update(
    const Mat &ref,
    const Mat &curr,
    const SE3d &T_C_R,
    Mat &depth,
    Mat &depth_cov2
);

/**
 * 極線搜索
 * @param ref                   參考圖像
 * @param curr                  當前圖像
 * @param T_C_R                 參考圖像到當前圖像的位姿
 * @param pt_ref                參考圖像中點的位置
 * @param depth_mu              深度均值
 * @param depth_cov             深度方差
 * @param pt_curr               當前點
 * @param epipolar_direction    極線方向
 * @return                      是否成功
 */
bool epipolarSearch(
    const Mat &ref,
    const Mat &curr,
    const SE3d &T_C_R,
    const Vector2d &pt_ref,
    const double &depth_mu,
    const double &depth_cov,
    Vector2d &pt_curr,
    Vector2d &epipolar_direction
);

/**
 * 更新深度濾波器
 * @param pt_ref                參考圖像點
 * @param pt_curr               當前圖像點
 * @param T_C_R                 參考圖像到當前圖像的位姿
 * @param epipolar_direction    極線方向
 * @param depth                 深度均值
 * @param depth_cov2            深度方向
 * @return                      是否成功
 */
bool updateDepthFilter(
    const Vector2d &pt_ref,
    const Vector2d &pt_curr,
    const SE3d &T_C_R,
    const Vector2d &epipolar_direction,
    Mat &depth,
    Mat &depth_cov2
);

/**
 * 計算 NCC 評分
 * @param ref       參考圖像
 * @param curr      當前圖像
 * @param pt_ref    參考點
 * @param pt_curr   當前點
 * @return          NCC 評分
 */
double NCC(const Mat &ref, const Mat &curr, const Vector2d &pt_ref, const Vector2d &pt_curr);

// 雙線性灰度插值
inline double getBilinearInterpolatedValue(const Mat &img, const Vector2d &pt) {
    uchar *d = &img.data[int(pt(1, 0)) * img.step + int(pt(0, 0))];
    double xx = pt(0, 0) - floor(pt(0, 0));
    double yy = pt(1, 0) - floor(pt(1, 0));

    // d[0] 為 (pt.x, pt.y); d[img.step] 為 (pt.x, pt.y + 1)，d[0] 往下一個像素的位置，指標位置差一個 img.step（相當於圖片寬度）
    return ((1 - xx) * (1 - yy) * double(d[0]) +
            xx * (1 - yy) * double(d[1]) +
            (1 - xx) * yy * double(d[img.step]) +
            xx * yy * double(d[img.step + 1])) / 255.0;
}

// ------------------------------------------------------------------
// 一些小工具
// 顯示估計的深度圖
void plotDepth(const Mat &depth_truth, const Mat &depth_estimate);

// 像素到相機坐標系
inline Vector3d px2cam(const Vector2d px) {
    return Vector3d(
        (px(0, 0) - cx) / fx,
        (px(1, 0) - cy) / fy,
        1
    );
}

// 相機坐標系到像素
inline Vector2d cam2px(const Vector3d p_cam) {
    // p_cam(2, 0)：z 值（深度） 除以深度來歸一化座標（z = 1）
    return Vector2d(
        p_cam(0, 0) * fx / p_cam(2, 0) + cx,
        p_cam(1, 0) * fy / p_cam(2, 0) + cy
    );
}

// 檢測一個點是否在圖像邊框內
inline bool inside(const Vector2d &pt) {
    return pt(0, 0) >= boarder && pt(1, 0) >= boarder
           && pt(0, 0) + boarder < width && pt(1, 0) + boarder <= height;
}

/**
 * @brief 顯示極線匹配
 * 
 * @param ref 
 * @param curr 
 * @param px_ref 
 * @param px_curr 
 */
void showEpipolarMatch(const Mat &ref, const Mat &curr, const Vector2d &px_ref, const Vector2d &px_curr);

/**
 * @brief 顯示極線
 * 
 * @param ref 
 * @param curr 
 * @param px_ref 
 * @param px_min_curr 
 * @param px_max_curr 
 */
void showEpipolarLine(const Mat &ref, const Mat &curr, const Vector2d &px_ref, const Vector2d &px_min_curr,
                      const Vector2d &px_max_curr);

/**
 * @brief 評測深度估計
 * 
 * @param depth_truth 
 * @param depth_estimate 
 */
void evaludateDepth(const Mat &depth_truth, const Mat &depth_estimate);
// ------------------------------------------------------------------


int main(int argc, char **argv) {
    if (argc != 2) {
        cout << "Usage: dense_mapping path_to_test_dataset" << endl;
        return -1;
    }

    // 從數據集讀取數據
    vector<string> color_image_files;
    vector<SE3d> poses_TWC;
    Mat ref_depth;
    bool ret = readDatasetFiles(argv[1], color_image_files, poses_TWC, ref_depth);
    
    if (ret == false) {
        cout << "Reading image files failed!" << endl;
        return -1;
    }

    cout << "read total " << color_image_files.size() << " files." << endl;

    // 第一張圖（gray-scale image）
    Mat ref = imread(color_image_files[0], 0);
    cout << "imread: " << color_image_files[0] << endl;

    SE3d pose_ref_TWC = poses_TWC[0];
    cout << "poses_TWC[0]: " << pose_ref_TWC.matrix() << endl;
    
    // 深度初始值
    double init_depth = 3.0;    
    
    // 方差初始值
    double init_cov2 = 3.0;    
    
    // 深度圖
    Mat depth(height, width, CV_64F, init_depth);   
    
    // 深度圖方差
    Mat depth_cov2(height, width, CV_64F, init_cov2);         

    // 剩下的圖片依序和第 0 張圖片進行深度估計，修正對 '深度圖' 和 '深度圖方差' 的估計
    for (int index = 1; index < color_image_files.size(); index++) {
        cout << "*** loop " << index << " ***" << endl;
        Mat curr = imread(color_image_files[index], 0);
        cout << "imread: " << color_image_files[index] << endl;
        
        if (curr.data == nullptr){
            continue;
        }
        
        SE3d pose_curr_TWC = poses_TWC[index];
        cout << "poses_TWC[" << index << "]: " << pose_curr_TWC.matrix() << endl;
        
        // 坐標轉換關系： T_C_W * T_W_R = T_C_R
        SE3d pose_T_C_R = pose_curr_TWC.inverse() * pose_ref_TWC;
        cout << "pose_T_C_R: " << pose_T_C_R.matrix() << endl;
        
        // 根據新的圖像更新深度估計
        update(ref, curr, pose_T_C_R, depth, depth_cov2);

        // （利用數據集讀取而來的實際深度）評測深度估計（，但並未使用該數據來校正估計）
        evaludateDepth(ref_depth, depth);

        // 顯示估計的深度圖
        plotDepth(ref_depth, depth);

        // 呈現圖片 curr
        imshow("image", curr);
        waitKey(1);
    }

    cout << "estimation returns, saving depth map ..." << endl;

    // 將估計的深度圖寫出
    imwrite("depth.png", depth);
    
    cout << "done." << endl;

    return 0;
}

bool readDatasetFiles(
    const string &path,
    vector<string> &color_image_files,
    std::vector<SE3d> &poses,
    cv::Mat &ref_depth) {
    
    ifstream fin(path + "/first_200_frames_traj_over_table_input_sequence.txt");
    
    if (!fin){
        return false;
    }

    string image;
    // while (!fin.eof()) {
    while (fin >> image) {
        // 數據格式：圖像文件名 tx, ty, tz, qx, qy, qz, qw ，注意是 TWC 而非 TCW        
        // fin >> image;
        
        cout << "image:" << image << endl;
        
        double data[7];
        
        for (double &d:data){
            fin >> d;
        }

        color_image_files.push_back(path + string("/images/") + image);
        poses.push_back(
            SE3d(Quaterniond(data[6], data[3], data[4], data[5]),
                 Vector3d(data[0], data[1], data[2]))
        );
        
        if (!fin.good()){
            break;
        }
    }
    
    cout << "Close first_200_frames_traj_over_table_input_sequence.txt" << endl;
    fin.close();

    // load reference depth
    fin.open(path + "/depthmaps/scene_000.depth");
    ref_depth = cv::Mat(height, width, CV_64F);
    
    if (!fin){
        return false;
    }
    
    for (int y = 0; y < height; y++){
        for (int x = 0; x < width; x++) {
            double depth = 0;
            fin >> depth;
            ref_depth.ptr<double>(y)[x] = depth / 100.0;
        }
    }        

    cout << "Close depthmaps/scene_000.depth" << endl;
    fin.close();

    return true;
}

// 對整個深度圖進行更新
bool update(const Mat &ref, const Mat &curr, const SE3d &T_C_R, Mat &depth, Mat &depth_cov2) {
    // 遍歷每個像素
    for (int x = boarder; x < width - boarder; x++)
        for (int y = boarder; y < height - boarder; y++) {            
            
            // 深度已收斂或发散
            if (depth_cov2.ptr<double>(y)[x] < min_cov || depth_cov2.ptr<double>(y)[x] > max_cov){
                continue;
            }
                
            // 在極線上搜索 (x,y) 的匹配
            Vector2d pt_curr;
            Vector2d epipolar_direction;

            // 極線搜索
            bool ret = epipolarSearch(
                ref,
                curr,
                T_C_R,
                Vector2d(x, y),
                depth.ptr<double>(y)[x],
                sqrt(depth_cov2.ptr<double>(y)[x]),
                pt_curr,
                epipolar_direction
            );

            // 匹配失敗
            if (ret == false) {
                continue;
            }                

            // 取消該注釋以顯示匹配
            // showEpipolarMatch(ref, curr, Vector2d(x, y), pt_curr);

            // 匹配成功，更新深度圖
            updateDepthFilter(Vector2d(x, y), pt_curr, T_C_R, epipolar_direction, depth, depth_cov2);
        }
}

// 極線搜索
// 方法見書 12.2 12.3 兩節
bool epipolarSearch(
    const Mat &ref, const Mat &curr,
    const SE3d &T_C_R, const Vector2d &pt_ref,
    const double &depth_mu, const double &depth_cov,
    Vector2d &pt_curr, Vector2d &epipolar_direction) {
    
    // 取得 pt_ref 在相機座標系下的位置
    Vector3d f_ref = px2cam(pt_ref);

    // 標準化為單位向量
    f_ref.normalize();
    
    // 參考幀的 P 向量
    // depth_mu：深度均值
    Vector3d P_ref = f_ref * depth_mu;    

    // 按深度均值投影的像素
    // T_C_R：參考圖像到當前圖像的位姿
    Vector2d px_mean_curr = cam2px(T_C_R * P_ref); 
    
    // 取正負 3 個標準差，作為深度估計的可能範圍
    double d_min = depth_mu - 3 * depth_cov, d_max = depth_mu + 3 * depth_cov;
    
    if (d_min < 0.1){
        d_min = 0.1;
    }
    
    // 按最小深度投影的像素
    Vector2d px_min_curr = cam2px(T_C_R * (f_ref * d_min));  
    
    // 按最大深度投影的像素
    Vector2d px_max_curr = cam2px(T_C_R * (f_ref * d_max));    

    // 極線（線段形式）
    Vector2d epipolar_line = px_max_curr - px_min_curr;    
    
    // 極線方向
    epipolar_direction = epipolar_line;        
    epipolar_direction.normalize();
    
    // 極線線段的半長度
    double half_length = 0.5 * epipolar_line.norm();    
    
    // 我們不希望搜索太多東西
    if (half_length > 100){
        half_length = 100;   
    }

    // 取消此句注釋以顯示極線（線段）
    // showEpipolarLine( ref, curr, pt_ref, px_min_curr, px_max_curr );

    // 在極線上搜索，以深度均值點為中心，左右各取半長度
    double best_ncc = -1.0;
    Vector2d best_px_curr;
    
    // l+=sqrt(2)
    for (double l = -half_length; l <= half_length; l += 0.7) { 
        
        // 待匹配點
        Vector2d px_curr = px_mean_curr + l * epipolar_direction;  
        
        // 檢查 px_curr 是否在圖像邊框內
        if (!inside(px_curr)){
            continue;
        }
            
        // 計算待匹配點與參考幀的 NCC
        double ncc = NCC(ref, curr, pt_ref, px_curr);

        if (ncc > best_ncc) {
            best_ncc = ncc;
            best_px_curr = px_curr;
        }
    }
    
     // 只相信 NCC 很高的匹配
    if (best_ncc < 0.85f){
        return false;
    }
        
    pt_curr = best_px_curr;

    return true;
}

double NCC(
    const Mat &ref, const Mat &curr,
    const Vector2d &pt_ref, const Vector2d &pt_curr) {
    // 零均值-歸一化互相關
    // 先算均值
    double mean_ref = 0, mean_curr = 0;
    
    // 參考幀和當前幀的均值
    vector<double> values_ref, values_curr; 
    
    for (int x = -ncc_window_size; x <= ncc_window_size; x++)
        for (int y = -ncc_window_size; y <= ncc_window_size; y++) {
            // pt_ref(1, 0): pt_ref.y; pt_ref(0, 0): pt_ref.x
            double value_ref = double(ref.ptr<uchar>(int(y + pt_ref(1, 0)))[int(x + pt_ref(0, 0))]) / 255.0;
            mean_ref += value_ref;

            double value_curr = getBilinearInterpolatedValue(curr, pt_curr + Vector2d(x, y));
            mean_curr += value_curr;

            values_ref.push_back(value_ref);
            values_curr.push_back(value_curr);
        }

    mean_ref /= ncc_area;
    mean_curr /= ncc_area;

    // 計算 Zero mean NCC
    double numerator = 0, demoniator1 = 0, demoniator2 = 0;

    for (int i = 0; i < values_ref.size(); i++) {
        double n = (values_ref[i] - mean_ref) * (values_curr[i] - mean_curr);
        numerator += n;
        demoniator1 += (values_ref[i] - mean_ref) * (values_ref[i] - mean_ref);
        demoniator2 += (values_curr[i] - mean_curr) * (values_curr[i] - mean_curr);
    }
    
    // 防止分母出現零
    return numerator / sqrt(demoniator1 * demoniator2 + 1e-10);   
}

// 更新深度濾波器
bool updateDepthFilter(
    const Vector2d &pt_ref,
    const Vector2d &pt_curr,
    const SE3d &T_C_R,
    const Vector2d &epipolar_direction,
    Mat &depth,
    Mat &depth_cov2) {
    // ===== 這段還看不懂 =====
    // 用三角化計算深度
    SE3d T_R_C = T_C_R.inverse();

    Vector3d f_ref = px2cam(pt_ref);
    f_ref.normalize();
    
    Vector3d f_curr = px2cam(pt_curr);
    f_curr.normalize();

    // 方程
    // d_ref * f_ref = d_cur * ( R_RC * f_cur ) + t_RC
    // f2 = R_RC * f_cur
    // 轉化成下面這個矩陣方程組
    // => [ f_ref^T f_ref, -f_ref^T f2 ] [d_ref]   [f_ref^T t]
    //    [ f_2^T f_ref, -f2^T f2      ] [d_cur] = [f2^T t   ]
    Vector3d t = T_R_C.translation();
    Vector3d f2 = T_R_C.so3() * f_curr;
    Vector2d b = Vector2d(t.dot(f_ref), t.dot(f2));
    Matrix2d A;
    A(0, 0) = f_ref.dot(f_ref);
    A(0, 1) = -f_ref.dot(f2);
    A(1, 0) = -A(0, 1);
    A(1, 1) = -f2.dot(f2);

    Vector2d ans = A.inverse() * b;
    
    // ref 側的結果
    Vector3d xm = ans[0] * f_ref;   
    
    // cur 結果
    Vector3d xn = t + ans[1] * f2;       
    
    // P的位置，取兩者的平均
    Vector3d p_esti = (xm + xn) / 2.0;      
    
    // 深度值
    double depth_estimation = p_esti.norm();   

    // ==========

    // 計算不確定性（以一個像素為誤差）
    Vector3d p = f_ref * depth_estimation;
    Vector3d a = p - t;
    double t_norm = t.norm();
    double a_norm = a.norm();
    double alpha = acos(f_ref.dot(t) / t_norm);
    double beta = acos(-a.dot(t) / (a_norm * t_norm));

    // p2 to p2'
    Vector3d f_curr_prime = px2cam(pt_curr + epipolar_direction);
    f_curr_prime.normalize();

    double beta_prime = acos(f_curr_prime.dot(-t) / t_norm);
    double gamma = M_PI - alpha - beta_prime;

    double p_prime = t_norm * sin(beta_prime) / sin(gamma);

    // 兩個深度估計的差值，作為深度估計的標準差（誤差範圍）
    double d_cov = p_prime - depth_estimation;
    double d_cov2 = d_cov * d_cov;

    // 高斯融合
    double mu = depth.ptr<double>(int(pt_ref(1, 0)))[int(pt_ref(0, 0))];
    double sigma2 = depth_cov2.ptr<double>(int(pt_ref(1, 0)))[int(pt_ref(0, 0))];

    double mu_fuse = (d_cov2 * mu + sigma2 * depth_estimation) / (sigma2 + d_cov2);
    double sigma_fuse2 = (sigma2 * d_cov2) / (sigma2 + d_cov2);

    // 更新對 '深度圖' 和 '深度圖方差' 的估計
    depth.ptr<double>(int(pt_ref(1, 0)))[int(pt_ref(0, 0))] = mu_fuse;
    depth_cov2.ptr<double>(int(pt_ref(1, 0)))[int(pt_ref(0, 0))] = sigma_fuse2;

    return true;
}

// 後面這些太簡單我就不注釋了（其實是因為懶）
void plotDepth(const Mat &depth_truth, const Mat &depth_estimate) {
    imshow("depth_truth", depth_truth * 0.4);
    imshow("depth_estimate", depth_estimate * 0.4);
    imshow("depth_error", depth_truth - depth_estimate);
    waitKey(1);
}

// 評測深度估計(實際深度為從數據集讀取而來)
void evaludateDepth(const Mat &depth_truth, const Mat &depth_estimate) {
    // 平均誤差
    double ave_depth_error = 0;  
    
    // 平方誤差
    double ave_depth_error_sq = 0;   
    
    int cnt_depth_data = 0;

    for (int y = boarder; y < depth_truth.rows - boarder; y++){
        for (int x = boarder; x < depth_truth.cols - boarder; x++) {

            double error = depth_truth.ptr<double>(y)[x] - depth_estimate.ptr<double>(y)[x];
            ave_depth_error += error;
            ave_depth_error_sq += error * error;
            cnt_depth_data++;
        }
    }
        
    ave_depth_error /= cnt_depth_data;
    ave_depth_error_sq /= cnt_depth_data;

    cout << "Average squared error = " << ave_depth_error_sq << ", average error: " << ave_depth_error << endl;
}

void showEpipolarMatch(const Mat &ref, const Mat &curr, const Vector2d &px_ref, const Vector2d &px_curr) {
    Mat ref_show, curr_show;
    cv::cvtColor(ref, ref_show, ColorConversionCodes::COLOR_GRAY2BGR);
    cv::cvtColor(curr, curr_show, ColorConversionCodes::COLOR_GRAY2BGR);

    cv::circle(ref_show, cv::Point2f(px_ref(0, 0), px_ref(1, 0)), 5, cv::Scalar(0, 0, 250), 2);
    cv::circle(curr_show, cv::Point2f(px_curr(0, 0), px_curr(1, 0)), 5, cv::Scalar(0, 0, 250), 2);

    imshow("ref", ref_show);
    imshow("curr", curr_show);
    waitKey(1);
}

void showEpipolarLine(const Mat &ref, const Mat &curr, const Vector2d &px_ref, const Vector2d &px_min_curr, const Vector2d &px_max_curr) {

    Mat ref_show, curr_show;
    cv::cvtColor(ref, ref_show, ColorConversionCodes::COLOR_GRAY2BGR);
    cv::cvtColor(curr, curr_show, ColorConversionCodes::COLOR_GRAY2BGR);

    cv::circle(ref_show, cv::Point2f(px_ref(0, 0), px_ref(1, 0)), 5, cv::Scalar(0, 255, 0), 2);
    cv::circle(curr_show, cv::Point2f(px_min_curr(0, 0), px_min_curr(1, 0)), 5, cv::Scalar(0, 255, 0), 2);
    cv::circle(curr_show, cv::Point2f(px_max_curr(0, 0), px_max_curr(1, 0)), 5, cv::Scalar(0, 255, 0), 2);
    cv::line(curr_show, Point2f(px_min_curr(0, 0), px_min_curr(1, 0)), Point2f(px_max_curr(0, 0), px_max_curr(1, 0)),
             Scalar(0, 255, 0), 1);

    imshow("ref", ref_show);
    imshow("curr", curr_show);
    waitKey(1);
}
