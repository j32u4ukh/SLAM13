//
// Created by xiang on 18-11-19.
//

#include <iostream>
#include <opencv2/core/core.hpp>
#include <ceres/ceres.h>
#include <chrono>

using namespace std;

// 代價函數的計算模型
struct CURVE_FITTING_COST {
    // x, y 數據
    const double _x, _y;    
    
    CURVE_FITTING_COST(double x, double y) : _x(x), _y(y) {}

    // residual：剩餘的
    // 殘差的計算 模型參數，有3維        
    template<typename T>
    bool operator()(const T *const abc,  T *residual) const {
        // y-exp(ax^2+bx+c)
        residual[0] = T(_y) - ceres::exp(abc[0] * T(_x) * T(_x) + abc[1] * T(_x) + abc[2]); 
        return true;
    }
};

int main(int argc, char **argv) {
    // 真實參數值
    double ar = 1.0, br = 2.0, cr = 1.0;   
    
    // 估計參數值
    double ae = 2.0, be = -1.0, ce = 5.0;       
    
    // 數據點
    int N = 100;     
    
    // 噪聲Sigma值
    double w_sigma = 1.0;                        
    double inv_sigma = 1.0 / w_sigma;
    
     // OpenCV隨機數產生器
    cv::RNG rng;                                

    // 數據
    vector<double> x_data, y_data;      
    
    for (int i = 0; i < N; i++) {
        double x = i / 100.0;
        x_data.push_back(x);
        y_data.push_back(exp(ar * x * x + br * x + cr) + rng.gaussian(w_sigma * w_sigma));
    }

    double abc[3] = {ae, be, ce};

    // 構建最小二乘問題
    ceres::Problem problem;

    for (int i = 0; i < N; i++) {
        // 向問題中添加誤差項
        problem.AddResidualBlock(            
            // 使用自動求導，模板參數：誤差類型，輸出維度，輸入維度，維數要與前面 struct 中一致
            // CURVE_FITTING_COST: 代價函數的計算模型
            new ceres::AutoDiffCostFunction<CURVE_FITTING_COST, 1, 3>(
                new CURVE_FITTING_COST(x_data[i], y_data[i])
            ),
            
            // 核函數，這裡不使用，為空
            nullptr,            
            
            // 參數塊：待估計參數
            abc                 
        );
    }

    // 配置求解器
    ceres::Solver::Options options;
    
    // 增量方程如何求解
    options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;  
    
    // 輸出到 cout
    options.minimizer_progress_to_stdout = true;   

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    
    // 優化信息
    ceres::Solver::Summary summary;                
    
    // 開始優化
    ceres::Solve(options, &problem, &summary);  
    
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "solve time cost = " << time_used.count() << " seconds. " << endl;

    // 輸出結果
    cout << summary.BriefReport() << endl;
    cout << "estimated a,b,c = ";
    
    for (auto a : abc){        
        cout << a << " ";
    }
    cout << endl;

    return 0;
}
