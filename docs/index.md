# SLAM Learning

我將在這裡把從 2021/02/27 開始學習的 SLAM 內容，嘗試以一般人都聽得懂的方式記錄下來，日文版和英文版將分別放在 <a herf="https://j32u4ukh.github.io/SLAM13/jp">JP</a> 和 <a herf="https://j32u4ukh.github.io/SLAM13/en">EN</a>，點下去便可看到翻譯後的版本。

自分が 2021/02/27 から勉強している、SLAM というの内容をこれから一般人さえも理解できる方法で、ここに記録して。日本語バージョンと英語バージョンは下の画像の JP と EN　のなかに、それを入れば翻訳したバージョンが見える。

I'll try to record the content of SLAM that I started to learn from February 27 2021, with the way which make average person can also understand. The Japanese version and the English version will put into EN and JP in the image below, click to read to the translated version.

* <a href="https://j32u4ukh.github.io/SLAM13/class1.html">Class 1: SLAM</a>
* <a href="https://j32u4ukh.github.io/SLAM13/class2.html">Class 2: 前端(視覺里程計)</a>
* <a href="https://j32u4ukh.github.io/SLAM13/class3.html">Class 3: 後端 & 回路檢測</a>
* <a href="https://j32u4ukh.github.io/SLAM13/class4.html">Class 4: 建圖(概述) & 整體流程</a>

* <a href="https://j32u4ukh.github.io/SLAM13/class5.html">Class 5: 影像間匹配點</a>
* <a href="https://j32u4ukh.github.io/SLAM13/class6.html">Class 6: (相機)運動</a>
* <a href="https://j32u4ukh.github.io/SLAM13/class7.html">Class 7: 估計空間點的位置</a>
* <a href="https://j32u4ukh.github.io/SLAM13/class8.html">Class 8: 後端</a>
* <a href="https://j32u4ukh.github.io/SLAM13/class9.html">Class 9: 回路檢測</a>
* <a href="https://j32u4ukh.github.io/SLAM13/class10.html">Class 10: 建圖</a>

# 主題式章節

* 前端(主題類以外)
* 前端：特徵點法
* 前端：SIFT
* 前端：FAST
* 前端：ORB
* 前端：特徵比對
* 前端：對極幾何
* 前端：八點法
* 前端：單應矩陣
* 前端：三角測量
* 前端：PnP
* 前端：ICP
* 前端：直接法
* 前端：光流法
* 前端：針孔相機模型
* 前端：位姿
* 前端：旋轉
* 前端：李群
* 前端：李代數
* 前端：轉換
* 前端：左右目視差
* 前端：極線搜索
* 前端：區塊比對

---
* 後端(主題類以外)
* 後端：Bundle Adjustment(BA)
* 後端：位姿圖
* 後端：SLAM 模型
* 後端：最小平方法
* 後端：梯度法
* 後端：高斯牛頓法
* 後端：列文伯格-馬夸特方法
* 後端：擴充卡爾曼濾波(EKF)
* 後端：非線性最佳化

* 後端：逆深度
* 後端：馬哈拉諾比斯距離(Mahalanobis)
* 後端：滑動視窗濾波(SWF)、Schur 、邊緣

---
* 回路檢測(主題類以外)
* 回路檢測：詞袋模型
* 回路檢測：字典

---
* 建圖(主題類以外)
* 建圖：一元稠密重建
* 建圖：八叉樹地圖(Octo-map)

---
* 像素梯度問題
* 流形
* 實對稱矩陣
* 均勻-高斯混合分布
* 矩陣微分
* 單應性變換
* KD Tree
* 奇異矩陣
* 病態矩陣 
* 零空間維數
* 吳消去法
* 鄰接矩陣(adjacency matrix)
* 位元元姿

後端優化將狀態量視為分佈來估計，實際上估計與更新是怎麼運作的呢？
