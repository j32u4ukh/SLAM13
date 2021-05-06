#include "slam.hpp"

std::ostream &operator<<(std::ostream &out, const RotationMatrix &r) {
    out.setf(std::ios::fixed);
    Eigen::Matrix3d matrix = r.matrix;
    out << '=';
    out << "[" << std::setprecision(2) << matrix(0, 0) << "," << matrix(0, 1) << "," << matrix(0, 2) << "],"
        << "[" << matrix(1, 0) << "," << matrix(1, 1) << "," << matrix(1, 2) << "],"
        << "[" << matrix(2, 0) << "," << matrix(2, 1) << "," << matrix(2, 2) << "]";
    return out;
}

std::istream &operator>>(std::istream &in, RotationMatrix &r) {
    return in;
}

std::ostream &operator<<(std::ostream &out, const TranslationVector &t) {
    out << "=[" << t.trans(0) << ',' << t.trans(1) << ',' << t.trans(2) << "]";
    return out;
}

std::istream &operator>>(std::istream &in, TranslationVector &t) {
    return in;
}

std::ostream &operator<<(std::ostream &out, const QuaternionDraw quat) {
    auto c = quat.q.coeffs();
    out << "=[" << c[0] << "," << c[1] << "," << c[2] << "," << c[3] << "]";
    return out;
}

std::istream &operator>>(std::istream &in, const QuaternionDraw quat) {
    return in;
}

void eigenMatrix(){
    // Matrix3d 實質上是 Eigen::Matrix<double, 3, 3>
    // 初始化為全零的矩陣
    Eigen::Matrix3d matrix_33 = Eigen::Matrix3d::Zero();
    
    // 如果不確定矩陣大小，可以使用動態大小的矩陣（本例中未使用）
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> matrix_dynamic;
    
    // 更簡單的版本（本例中未使用）
    Eigen::MatrixXd matrix_x;
    
    // ＝＝＝＝＝ 對 Eigen 進行操作 ＝＝＝＝＝
    // 輸入資料（初始化）
    Eigen::Matrix<float, 2, 3> matrix_23;
    matrix_23 << 1, 2, 3, 4, 5, 6;
    
    // 輸出矩陣
    std::cout << "matrix_23 from 1 to 6:\n" << matrix_23 << std::endl;
    std::cout << matrix_23.size() << std::endl;
    
    // 利用（）存取矩陣中的元素
    std::cout << "print matrix_23:" << std::endl;
    for(int i = 0; i < 2; i++){
        
        for(int j = 0; j < 3; j++){            
            std::cout << matrix_23(i, j) << "\t";
        }
        
        std::cout << std::endl;
    }
    
    // Eigen::Vector3d 歐拉角
    Eigen::Vector3d v_3d;
    Eigen::Matrix<float, 3, 1> vd_3d;

    // 旋轉向量
    Eigen::AngleAxisd angle_axis;
    
    v_3d << 3, 2, 1;
    vd_3d << 4, 5, 6;
    
    // 矩陣與向量相乘（實際上仍是矩陣和矩陣）
    // 利用 cast<double> 做顯性轉換是必須的
    // Eigen 的 * 已重載為矩陣乘法
    Eigen::Matrix<double, 2, 1> result = matrix_23.cast<double>() * v_3d;
    std::cout << "[1,2,3;4,5,6]*[3,2,1] = " << result.transpose() << std::endl;

    // 矩陣和矩陣相乘
    Eigen::Matrix<float, 2, 1> result2 = matrix_23 * vd_3d;
    std::cout << "[1,2,3;4,5,6]*[4,5,6]: " << result2.transpose() << std::endl;
    
    // 一些矩陣運算
    // 產生一個隨機數矩陣(3 X 3)
    matrix_33 = Eigen::Matrix3d::Random();     
    std::cout << "random matrix: \n" << matrix_33 << std::endl;
    std::cout << "轉置 transpose: \n" << matrix_33.transpose() << std::endl;
    
    std::cout << "各元素和 sum: " << matrix_33.sum() << std::endl;

    // tr(M)
    std::cout << "跡 trace: " << matrix_33.trace() << std::endl;
    std::cout << "數乘 times 10: \n" << 10 * matrix_33 << std::endl;

    // TODO: 驗證旋轉矩陣為"正交矩陣"（逆與轉置相同）且行列式為 +/- 1
    std::cout << "逆 inverse: \n" << matrix_33.inverse() << std::endl;
    std::cout << "行列式 det: " << matrix_33.determinant() << std::endl;

    // 特征值
    // 實對稱矩陣可以保證對角化成功
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen_solver(matrix_33.transpose() * matrix_33);
    std::cout << "特征值 Eigen values = \n" << eigen_solver.eigenvalues() << std::endl;
    std::cout << "特征向量 Eigen vectors = \n" << eigen_solver.eigenvectors() << std::endl;

    // 解方程
    // 我們求解 matrix_NN * x = v_Nd 這個方程(求解 x)
    // N的大小在前邊的宏里定義，矩陣數值由隨機數生成
    // 直接求逆自然是最直接的，但是求逆運算量大

    Eigen::Matrix<double, MATRIX_SIZE, MATRIX_SIZE> matrix_NN;
    matrix_NN = Eigen::MatrixXd::Random(MATRIX_SIZE, MATRIX_SIZE);
    
    // 保證半正定
    matrix_NN = matrix_NN * matrix_NN.transpose();  
    
    Eigen::Matrix<double, MATRIX_SIZE, 1> v_Nd = Eigen::MatrixXd::Random(MATRIX_SIZE, 1);

    // 計時
    clock_t time_stt = clock();
    
    // 直接求逆
    Eigen::Matrix<double, MATRIX_SIZE, 1> x = matrix_NN.inverse() * v_Nd;
    
    std::cout << "time of normal inverse is " 
    << 1000 * (clock() - time_stt) / (double) CLOCKS_PER_SEC << "ms" << std::endl;
        
    std::cout << "x = " << x.transpose() << std::endl;

    
    time_stt = clock();
    
    // 通常用矩陣分解來求，例如 QR 分解，速度會快很多
    x = matrix_NN.colPivHouseholderQr().solve(v_Nd);
    
    std::cout << "time of QR decomposition is " 
    << 1000 * (clock() - time_stt) / (double) CLOCKS_PER_SEC << "ms" << std::endl;
    std::cout << "x = " << x.transpose() << std::endl;

    
    time_stt = clock();
    
    // 對於正定矩陣，還可以用 cholesky 分解來解方程
    x = matrix_NN.ldlt().solve(v_Nd);
    
    std::cout << "time of ldlt decomposition is " 
    << 1000 * (clock() - time_stt) / (double) CLOCKS_PER_SEC << "ms" << std::endl;
    std::cout << "x = " << x.transpose() << std::endl;
}

void eigenGeometry(){
    // Eigen/Geometry 提供了各種旋轉和平移的表示
    // 3D 旋轉矩陣直接使用 Matrix3d 或 Matrix3f
    Eigen::Matrix3d rotation_matrix = Eigen::Matrix3d::Identity();
    
    // 旋轉向量使用 AngleAxis, 它底層不直接是Matrix，但運算可以當作矩陣（因為重載了運算符）
    // 沿 Z 軸旋轉 45 度
    Eigen::AngleAxisd rotation_vector(M_PI / 4, Eigen::Vector3d(0, 0, 1));    

    // 設置 std::cout 輸出數值位數
    std::cout.precision(3);
    
    // 用 matrix()轉換成矩陣
    // vector.matrix() 將回傳 vector.toRotationMatrix() 的結果
    std::cout << "rotation matrix =\n" << rotation_vector.matrix() << std::endl;   
    
    // 也可以直接賦值
    rotation_matrix = rotation_vector.toRotationMatrix();
    
    // 用 AngleAxis 可以進行坐標變換
    Eigen::Vector3d v(1, 0, 0);
    Eigen::Vector3d v_rotated = rotation_vector * v;    
    std::cout << "(1,0,0) after rotation (by angle axis) = " << v_rotated.transpose() << std::endl;
    
    // 或者用旋轉矩陣
    v_rotated = rotation_matrix * v;
    std::cout << "(1,0,0) after rotation (by matrix) = " << v_rotated.transpose() << std::endl;
    
    // 歐拉角: 可以將旋轉矩陣直接轉換成歐拉角
    // eulerAngles: 指定旋轉軸的順序, 形成 ZYX 順序，即 yaw-pitch-roll 順序
    Eigen::Vector3d euler_angles = rotation_matrix.eulerAngles(2, 1, 0);
    std::cout << "yaw pitch roll = " << euler_angles.transpose() << std::endl;
    
    // 歐氏變換矩陣使用 Eigen::Isometry
    // 雖然稱為3d，實質上是4＊4 的矩陣(第4列為 (0, 0, 0, 1) <- 形成齊次座標形式)
    // 實際宣告為 Transform<double, 3, 1> 應該是針對旋轉矩陣，平移的部份透過 pretranslate 另外設置
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();    
    
    // 按照rotation_vector進行旋轉
    T.rotate(rotation_vector);
    
    // 把平移向量設成(1,3,4)
    T.pretranslate(Eigen::Vector3d(1, 3, 4));
    std::cout << "Transform matrix = \n" << T.matrix() << std::endl;
    
    // 用變換矩陣進行坐標變換
    // 相當於 R * v + t
    Eigen::Vector3d v_transformed = T * v;
    std::cout << "v tranformed = " << v_transformed.transpose() << std::endl;
    
    // 對於仿射和射影變換，使用 Eigen::Affine3d 和 Eigen::Projective3d 即可，略
    
    // 四元數
    // 可以直接把 AngleAxis 賦值給四元數，反之亦然
    Eigen::Quaterniond q = Eigen::Quaterniond(rotation_vector);
    
    // 請注意 coeffs 的順序是 (x,y,z,w), w為實部，前三者為虛部
    std::cout << "quaternion from rotation vector = " << q.coeffs().transpose() << std::endl; 
    
    // 也可以把旋轉矩陣賦給它
    q = Eigen::Quaterniond(rotation_matrix);
    std::cout << "quaternion from rotation matrix = " << q.coeffs().transpose() << std::endl;
    
    // 使用四元數旋轉一個向量，使用重載的乘法即可
    // 注意數學上是 qvq^{-1}
    v_rotated = q * v;    
    std::cout << "(1,0,0) after rotation = " << v_rotated.transpose() << std::endl;
    
    // 用常規向量乘法表示，則應該如下計算
    std::cout << "should be equal to " 
    << (q * Eigen::Quaterniond(0, 1, 0, 0) * q.inverse()).coeffs().transpose() << std::endl;
    
    // ＝＝＝＝＝3.6.2 demo =====
    Eigen::Quaterniond q1(0.35, 0.2, 0.3, 0.1), q2(-0.5, 0.4, -0.1, 0.2);
    q1.normalize();
    q2.normalize();
    
    Eigen::Vector3d t1(0.3, 0.1, 0.1), t2(-0.1, 0.5, 0.3), p1(0.5, 0, 0.2);
    Eigen::Isometry3d T1w(q1), T2w(q2);
    T1w.pretranslate(t1);
    T2w.pretranslate(t2);
    
    // T1w: world to coordinate1; T1w.inverse(): coordinate1 to world
    // p1 是 coordinate1 下的座標，因此 T1w.inverse() * p1 為 p1 的世界座標
    // 再透過 T2w 由世界座標轉換成 coordinate2
    Eigen::Vector3d p2 = T2w * T1w.inverse() * p1;
    std::cout.precision(10);
    std::cout << "p2:\n" << p2.transpose() << std::endl;
}

void visualizeGeometry(){
    /* 使用 Pangolin 時，主要步驟為：
       創建窗口 -> 定義視角矩陣（觀測點） -> 創建交互式視圖（可以用鼠標進行視角旋轉）-> 循環刷新

       當我們需要定義 UI 工具欄，首先定義寬度，接著依次定義各個 bouton 的功能，然後在循環中對接相應的操作

       參考：http://www.biexiaoyu1994.com/%E5%BA%93%E5%87%BD%E6%95%B0%E5%AD%A6%E4%B9%A0/2019/02/20/pangolin/
     */
    // create display window -> title, width, height
    pangolin::CreateWindowAndBind("visualize geometry", 1000, 600);

    // If enabled, do depth comparisons and update the depth buffer. Note that even if 
    // the depth buffer exists and the depth mask is non-zero, the depth buffer is not updated if 
    // the depth test is disabled. 
    glEnable(GL_DEPTH_TEST);
    
    // Define Projection and initial ModelView matrix
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1000, 600, 420, 420, 500, 300, 0.1, 1000),

        // 對應的是 glLookAt, 攝像機位置, 參考點位置, up vector(上向量)
        pangolin::ModelViewLookAt(3, 3, 3, 0, 0, 0, pangolin::AxisY)
    );
    
    // 工具欄的長度
    const int UI_WIDTH = 500;

    /* Create Interactive View in window
    setBounds 跟 opengl 的 viewport 有關
    SetBounds 中分別為下、上、左、右的範圍，最後為 aspect（Camera 的寬高比例值）

    [沒設置 UI 的 View]
    pangolin::Handler3D handler(s_cam);
    pangolin::View& d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, 0.0, 1.0, -640.0f/480.0f)
        .SetHandler(&handler);
    */
    pangolin::View &d_cam = pangolin::CreateDisplay().
    SetBounds(0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH), 1.0, -1000.0f / 600.0f).    
    SetHandler(new pangolin::Handler3D(s_cam));
    
    // ui
    /* 工具欄設置
    "ui.name" 用於定義名字
    對於 bool 型的初始化，第一個 true/false 代表默認值；第二個決定是按鈕（false）還是勾選框（true）
    對於 double/int 型的初始化，數字分別為默認值，最小值和最大值，最後一個 bool 值為 true 時範圍成 log 分布
    也可以自定義類型，如 RotationMatrix, TranslationVector, QuaternionDraw
    */
    pangolin::Var<RotationMatrix> rotation_matrix("ui.R", RotationMatrix());
    pangolin::Var<TranslationVector> translation_vector("ui.t", TranslationVector());
    pangolin::Var<TranslationVector> euler_angles("ui.rpy", TranslationVector());
    pangolin::Var<QuaternionDraw> quaternion("ui.q", QuaternionDraw());

    // 創建工具欄
    pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(UI_WIDTH));

    /* 本例中未使用到
    // 定義鍵盤快捷鍵；以及執行的函數
    pangolin::RegisterKeyPressCallback(pangolin::PANGO_CTRL + 'b',                           			pangolin::SetVarFunctor<double>("ui.A Double", 3.5)); 
    pangolin::RegisterKeyPressCallback(pangolin::PANGO_CTRL + 'r', SampleMethod);
    */

    while (!pangolin::ShouldQuit()) {
        // Clear screen and activate view to render into
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        pangolin::OpenGlMatrix matrix = s_cam.GetModelViewMatrix();
        Eigen::Matrix<double, 4, 4> m = matrix;
        RotationMatrix R; 
        
        for (int i = 0; i < 3; i++){
            for (int j = 0; j < 3; j++){
                R.matrix(i, j) = m(j, i);                
            }            
        }
        
        // 呈現旋轉矩陣
        rotation_matrix = R;

        TranslationVector t;
        t.trans = Eigen::Vector3d(m(0, 3), m(1, 3), m(2, 3));
        t.trans = -R.matrix * t.trans;
        
        // 呈現平移向量
        translation_vector = t;

        TranslationVector euler;
        
        // 旋轉矩陣便換成歐拉角（須指定旋轉順序）
        euler.trans = R.matrix.eulerAngles(2, 1, 0);
        
        // 呈現歐拉角
        euler_angles = euler;

        QuaternionDraw quat;
        quat.q = Eigen::Quaterniond(R.matrix);
        
        // 呈現四元數
        quaternion = quat;

        // 清空畫面(白色)
        glColor3f(1.0, 1.0, 1.0);

        // 畫出有色立方體
        pangolin::glDrawColouredCube();
        
        // draw the original axis
        // 指定描繪形式： 線
        glBegin(GL_LINES);   
        glLineWidth(3);
        
        // X 軸
        glColor3f(0.8f, 0.f, 0.f);
        glVertex3f(0, 0, 0);
        glVertex3f(10, 0, 0);
        
        // Y 軸
        glColor3f(0.f, 0.8f, 0.f);
        glVertex3f(0, 0, 0);
        glVertex3f(0, 10, 0);
        
        // Z 軸
        glColor3f(0.2f, 0.2f, 1.f);
        glVertex3f(0, 0, 0);
        glVertex3f(0, 0, 10);
        
        // 結束：指定描繪形式
        glEnd();

        pangolin::FinishFrame();
    }
}
