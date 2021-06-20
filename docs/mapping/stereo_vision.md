# 立體視覺(Stereo Vision)

## 一元稠密重建

在稠密重建中，我們需要知道每一個像素(或大部分像素)的距離，對此有以下幾個解決方案：

1. 一元相機：估計相機運動，並利用『三角測量』或『極線搜索』計算像素的距離。

2. 二元相機：利用左右目的視差計算像素的距離，多目相機的原理相同。

3. RGB-D 相機：直接獲取像素距離。

前兩種方式稱為『立體視覺(Stereo Vision)』,其中移動一元相機的又稱為『移動角度的立體視覺(Moving View Stereo, MVS)』。

<table>
  <tr>
    <td></td>
    <td><b>一元、二元</b></td>
    <td><b>RGB-D</b></td>
  </tr>
  <tr>
    <td><b>優點</b></td>
    <td>在 RGB-D 無法被極佳應用的室外、大場景場合中，仍可透過立體視覺估計深度資訊</td>
    <td>直接測量距離,計算量較小</td>
  </tr>
  <tr>
    <td><b>缺點</b></td>
    <td>深度估計的計算量極大，且不比 RGB-D 可靠</td>
    <td>受到量程、應用範圍和光源等限制，在室外、大場景場合中無法有效測量距離</td>
  </tr>
</table>

* 使用**極線搜索**和**塊比對技術**，來得知第一幅圖的某像素在其他圖的位置。
* 和一般三角測量不同的是，會進行多次估計，使深度估計收斂。