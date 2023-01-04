# 三种方式生成图片的组合图
from core.util import *
from alpha.estimate_alpha_cf import *
from alpha.estimate_alpha_knn import *
from alpha.estimate_alpha_lbdm import *
from foreground.estimate_foreground_ml import estimate_foreground_ml

image_path = "data/lemur.png"
trimap_path = "data/lemur_trimap.png"
background_path = "data/SCUdoor.png"

grid_path = "result/lemur_at_SCUdoor2.png"

alpha_list = ["cf", "knn", "lbdm"]

def grid():
    scale = 1.0
    # 加载原图
    image = load_image(image_path, "RGB", scale, "box")
    # 加载trimap图片
    trimap = load_image(trimap_path, "GRAY", scale, "nearest")
    # 加载背景图
    new_background = load_image(background_path, "RGB", scale, "box")
    print("xxx")
    # 利用image和trimap估计alpha透明度矩阵
    alpha_cf = estimate_alpha_cf(image, trimap)
    alpha_knn = estimate_alpha_knn(image, trimap)
    alpha_lbdm = estimate_alpha_lbdm(image, trimap)
    print("yyy")
    # 根据alpha获取前景图和后景图
    foreground_cf = estimate_foreground_ml(image, alpha_cf)
    foreground_knn = estimate_foreground_ml(image, alpha_knn)
    foreground_lbdm = estimate_foreground_ml(image, alpha_lbdm)
    print("zzz")
    # 利用alpha矩阵对前后背景按照公式进行合并
    new_image_cf = blend(foreground_cf, new_background, alpha_cf)
    new_image_knn = blend(foreground_knn, new_background, alpha_knn)
    new_image_lbdm = blend(foreground_lbdm, new_background, alpha_lbdm)
    # 将换背景图保存在四方格中
    images = [image, trimap, alpha_cf, new_image_cf, alpha_knn, new_image_knn, alpha_lbdm, new_image_lbdm]
    grid = make_grid(images, 2, 4)
    save_image(grid_path, grid)

if __name__ == '__main__':
    grid()