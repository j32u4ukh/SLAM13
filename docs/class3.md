# 後端

相機運動的過程中，會產生 "位姿 X" 和 "路標 Y"，而兩者的估計都會受到雜訊影響，因此會將它們看成**服從某種機率分佈的隨機變數**，而非單獨的數值。

因此，問題變成：

*Q: 當我們擁有某些運動資料 u 和觀測資料 z 時，如何確定狀態量 x, y 的分佈？*

我們可以假設狀態量和雜訊項服從高斯分佈，這表示程式中只須儲存他們的『平均值(對變數最佳值的估計)』和『變異數矩陣(度量的不確定性)』即可。

因此，問題變成：

*Q: 當我們擁有某些運動資料 u 和觀測資料 z 時，如何估計狀態量 x, y 的高斯分佈？*

隨著時間，估計誤差也不斷累積著(因為下一次的估計，是基於前一次估計結果來進行的)，若能觀察到實際的位置，就能降低估計的不確定性。

## Bundle Adjustment(BA)

指從視覺影像中提煉出最佳的 3D 模型和相機模型(內外參數)。

考慮從任意特徵點發射出來的技術光線(bundles of light rays)，它們會在幾個相機的成像平面上變成像素或是檢測到的特徵點。

如果調整(adjustment)各相機姿態和各特徵點的空間位置，使得這些光線最後收束到相機的光心，稱為 BA。

## 位姿圖(Pose Graph)

多次觀測後，特徵點的位置已經收斂，變化很小，繼續進行收斂的意義不大。因此，通常在最佳化幾次後就把特徵點固定住，只把他們看作位姿估計的約束，而**不再實際的最佳化它們的位置估計**。

可以建置一個只有軌跡的圖最佳化，而位姿節點之間的邊，可以由兩個關鍵頁框之間透過特徵比對之後獲得的運動估計來指定初值。

只關心所有的相機位姿之間的聯繫，**省去大量的特徵點最佳化的計算**，只保留了關鍵頁框的軌跡，進一步建置了所謂的位姿圖(特徵點還保留著，只是省去對它的最佳化計算)。

# 回路檢測

當相機實際上經過同一地點第二次時，估計的軌跡也應該經過同一點。回到同一點這件事稱為"回路"。

*回路檢測的關鍵，就是如何有效的檢測出**相機經過同一地點**這件事。*

可能方法：

1. 『假定任兩張圖像之間都可能存在回路，並兩兩影像做一遍特徵比對。』隨著影像增加，計算量增加的過於快速，且該假設並不合理。

至少應預計『哪裡可能出現回路』，才不至於盲目的檢測。

2. 以『里程計為基礎(Odometry based)』的幾何關係。當發現相機運動到了之前的某個位置附近時，檢測他們有沒有回路關係。但由於誤差累積的關係，經常無法發現這個狀況，也無從談起回路檢測。

3. 以『外觀為基礎(Appearance based)』的幾何關係。與前後端的估計都無關，僅根據兩幅影像的相似性確定回路檢測關係。此種作法擺脫了誤差累積，使迴路檢測模組成為 SLAM 系統中一個相對獨立的模組(當然前端可為它提供特徵點)。

## 外觀為基礎的回路檢測

核心問題為：**如何計算影像間的相似性？**

可能方法：

1. 『兩圖轉為灰階後直接相減』，灰階為不穩定的測量值，嚴重受到環境光源和相機曝光影響。且當相機發生微小變化，即使像素值不變，由於像素位置發生改變，兩者差距也會變大。不是一個好的評估方式。

2. 利用特徵點搭配詞袋模型，不直接使用特徵比對，而是用『影像上有哪幾種特徵』來描述一幅影像。檢測到兩幅影像滿足回路時，進行特徵比對，估計兩者的相對運動。將估計後的運動，放入位姿圖來修正之前估計的誤差。

## Key frame 的處理

1. Key frame 不能選得太近，否則彼此之間都太相似，反而無法檢測出回路。最好是足夠稀疏，彼此不太相同，但又能涵蓋整個環境。

2. 在檢測回路時，當發現第一幀和第 n 幀形成回路時，和第 n + 1 幀、第 n + 2 幀應該也會形成回路。第一次形成回路時，便將先前的誤差給校正了，後面幾幀所形成的相同回路,無法提供更多資訊，通常會將這些相近的回路聚集為一種，使演算法不要反覆檢測同一種回路。

<table>
  <tr>
    <td><a href="https://j32u4ukh.github.io/SLAM13/class2.html">上一篇</a></td>
    <td><a href="https://j32u4ukh.github.io/SLAM13/">首頁</a></td>
    <td><a href="https://j32u4ukh.github.io/SLAM13/class4.html">下一篇</a></td>
  </tr>
</table>