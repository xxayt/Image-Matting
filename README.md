参考[A Python library for alpha matting (github.com)](https://github.com/pymatting/pymatting)

# Alpha Matting

- 效果展示：第一列第二行开始，分别为Closed Formed Matting、KNN Matting和Learning Based Digital Matting三种算法得到的Alpha矩阵展示

  <img src="result\lemur_at_SCUdoor2.png" alt="lemur_at_SCUdoor2" style="zoom:25%;" />

- 配置说明：此次我在AutoDL的线上环境平台的实验环境中，配置了PyTorch1.11.10、Python3.8、Cuda 11.3的镜像环境。

  - 在CPU环境下，需要安装的环境如下

    ```python
    numpy>=1.16.0
    pillow>=5.2.0 
    numba>=0.47.0 
    scipy>=1.1.0 
    ```

  - 在GPU环境下，还需要安装的环境如下（其实不用GPU也慢不了多少）

    ```python
    cupy-cuda90>=6.5.0 or similar 
    pyopencl>=2019.1.2 
    ```

  - 在安装此环境后，可直接运行 `cutout.py` 和 `grid.py` ，即可得到前景图片和换背景图片