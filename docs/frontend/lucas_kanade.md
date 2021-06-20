# Lucas-Kanade 光流

![Lucas-Kanade 光流](image/LucasKanade.png)

LK 光流認為來自相機的影像是隨時間變化的，影像灰階可看作時間和位置的函數。在 t 時刻，位於 (x, y) 處的像素，它的灰階值可以寫成 I(x, y, t)。

隨著相機的運動，它的影像座標將發生變化，估計這個空間點的在其他時刻影像中的位置，需要使用『灰階不變假設』。

**灰階不變假設**：同一個空間點的像素灰階值，在各個影像中是固定不變的。

對於 t 時刻位於 (x, y) 處的像素，令 t + dt 時刻運動到 (x + dx, y + dy) 處，由於灰階不變假設，有： I(x + dx, y + dy, t + dt) = I(x, y, t)。

```
『灰階不變假設』是一個很強的假設，實際中很有可能不成立。
```

對左式進行泰勒展開，保留一階項,得

又因為假設了灰階不變，所以灰階變化量為 0

同除以 dt 得

其中 dx/dt 為像素在 x 軸上的速度，而 dy/dt 為 y 軸上的速度，把它們記作 u, v。
同時，dI/dx 為影像在該點處 x 方向的梯度，另一項則是在 y 方向的梯度，記作 Ix, Iy。把影像灰階對時間的變化量記為 It，寫成矩陣形式有：

該式為帶有兩個變數 (u, v) 的一次方程，僅憑它無法計算出 u, v。因此需要使用額外的約束來計算 u, v，假設『某一個視窗內的像素具有相同的運動』。

考慮一個大小為 w*w 的視窗，它含有 w^2 數量的像素。該視窗內像素具有同樣的運動，因此共有 w^2 個方程式：

記：

於是整個方程式為：

這是一個關於 u, v 的超定線性方程，通常會求最小平方解：

這樣就獲得像素在影像間的運動速度 u, v。當 t 取離散的時刻而非連續時，我們可以估計某區塊像素在許多個影像中出現的位置。

由於像素梯度僅在局部有效，所以如果一次反覆運算不夠好，會反覆運算多次這個方程式。

##### 多層光流

將光流寫成最佳化問題，就必須假設最佳化的初值接近最佳值，才能在某種程度上保障演算法的收斂。

如果相機運動較快，兩張影像差異較明顯，那麼『單層影像光流法』容易達到一個『局部極小值』，這種情況可以利用影像金字塔來改善。

**影像金字塔**：對同一影像進行縮放，獲得不同解析度下的影像。

以原始影像作為金字塔底層，每往上一層，就對下層影像進行一定倍率的縮放。

計算光流時，先從頂層開始計算，然後將上層的追蹤結果，作為下層光流的初值。

上層的影像相對粗糙，所以這個過程也稱為**由粗至精(Coarse-to-fine)**的光流。

![Coarse-to-fine](CoarseToFine.png)