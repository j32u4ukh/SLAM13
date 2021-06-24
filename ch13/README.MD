bin 當中有兩個執行檔，run_kitti_stereo(source: /app/run_kitti_stereo.cpp) & test_triangulation(source: /test/test_triangulation.cpp)。

* 利用 LK 光流法估計特徵點在右圖中的位置。

### cv::calcOpticalFlowPyrLK

光流描述的是圖像上每個像素點的灰度的位置（速度）變化情況，光流的研究是利用圖像序列中的像素強度數據的時域變化和相關性來確定各自像素位置的“運動”。研究光流場的目的就是為了從圖片序列中近似得到不能直接得到的運動場。
光流法的前提假設：

1. 相鄰幀之間的亮度恒定；
2. 相鄰視頻幀的取幀時間連續，或者，相鄰幀之間物體的運動比較“微小”；
3. 保持空間一致性；即，同一子圖像的像素點具有相同的運動；

```
CV_EXPORTS_W void calcOpticalFlowPyrLK( InputArray prevImg, InputArray nextImg,
                           InputArray prevPts, CV_OUT InputOutputArray nextPts,
                           OutputArray status, OutputArray err,
                           Size winSize=Size(21,21), int maxLevel=3,
                           TermCriteria criteria=TermCriteria(
                            TermCriteria::COUNT+TermCriteria::EPS,
                            30, 0.01),
                           double derivLambda=0.5,
                           int flags=0,
                           double minEigThreshold=1e-4);
```

各個參數代表的含義如下：

- prevImg：你的標定圖像的灰度圖
- nextImg：你想搜尋的圖像的灰度圖
- prevPts：輸入的標定圖像的特征點（可以是其他特征點檢測方法找到的點）
- nextPts：輸出場景的特征點
- status：輸出狀態向量（無符號 char），如果在當前圖像中能夠光流得到標定的特征點位置改變，則設置 status 的對應位置為 1，否則設置為 0
- err：輸出錯誤向量；向量的每個元素被設為相應特征的一個錯誤，誤差測量的類型可以在flags參數中設置；如果流不被发現然後錯誤未被定義（使用status（狀態）參數找到此情形）。
- winSize：在每個金字塔水平搜尋窗口的尺寸
- maxLevel：金字塔的高度，初始為 3 層

當使用 calcOpticalFlowPyrLK 作為光流金字塔的算法時候，我們只需要知道以下的幾點：

> calcOpticalFlowPyrLK 必須和其他的角點識別算法進行搭配使用，比如使用 goodFeaturesToTrack，將其他的角點識別算法中獲得的角點作為光流算法的 prevPts
> 
> status 的大小和當前需要識別的光流移動的特征點大小一樣，所以我們可以判定當前的圖像是否還能與標定圖像進行光流的依據