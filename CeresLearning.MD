## 使用方式

1. 定義『參數塊』。通常為向量，在 SLAM 當中也可以定義成四元數、李代數等特殊的結構。
2. 定義『殘差塊』的計算方式，回傳殘差值。Ceres 對它們求平方和後，作為目標函數的值。
3. 殘差塊通常也需要定義雅可比的計算方式。可以手動指定計算方式，也可以使用『自動求導』功能，但殘差塊需要按照特定的寫法來撰寫：殘差的計算過程應該是一個帶有範本的括號算子。
4. 將所有參數塊和殘差塊加入 Ceres 定義的 Problem 物件中，呼叫 Solve 函數求解即可。求解之前還可以設定：反覆運算次數、中止條件等。

### LinearSolverType Solver::Options::linear_solver_type

Default: SPARSE_NORMAL_CHOLESKY / DENSE_QR

Type of linear solver used to compute the solution to the linear least squares problem in each iteration of the Levenberg-Marquardt algorithm. 

If Ceres is built with support for SuiteSparse or CXSparse or Eigen’s sparse Cholesky factorization, the default is SPARSE_NORMAL_CHOLESKY, it is DENSE_QR otherwise.

### bool Solver::Options::minimizer_progress_to_stdout

Default: false

By default the Minimizer progress is logged to STDERR depending on the vlog level. 

If this flag is set to true, and Solver::Options::logging_type is not SILENT, the logging output is sent to STDOUT.

### class Solver::Summary

Summary of the various stages of the solver after termination.

### string Solver::Summary::BriefReport() const

A brief one line description of the state of the solver after termination.


## 參考

[參考網站1](http://ceres-solver.org/nnls_solving.html)