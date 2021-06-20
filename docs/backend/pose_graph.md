# 位姿圖(Pose Graph)

多次觀測後，特徵點的位置已經收斂，變化很小，繼續進行收斂的意義不大。因此，通常在最佳化幾次後就把特徵點固定住，只把他們看作位姿估計的約束，而**不再實際的最佳化它們的位置估計**。

可以建置一個只有軌跡的圖最佳化，而位姿節點之間的邊，可以由兩個關鍵頁框之間透過特徵比對之後獲得的運動估計來指定初值。

只關心所有的相機位姿之間的聯繫，**省去大量的特徵點最佳化的計算**，只保留了關鍵頁框的軌跡，進一步建置了所謂的位姿圖(特徵點還保留著，只是省去對它的最佳化計算)。