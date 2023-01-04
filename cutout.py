from core.util import *
from alpha.estimate_alpha_cf import *
from alpha.estimate_alpha_knn import *
from alpha.estimate_alpha_lbdm import *
from foreground.estimate_foreground_ml import estimate_foreground_ml

image_path = "data/lemur.png"
trimap_path = "data/lemur_trimap.png"
background_path = "data/SCUdoor.png"

cutout_path = "result/"

alpha_list = ["cf", "knn", "lbdm"]
# 裁剪出前景图
def cutout(title):
    scale = 1.0
    image = load_image(image_path, "RGB", scale)  # 加载原图
    trimap = load_image(trimap_path, "GRAY", scale)  # 加载trimap图片
    print("xxx")
    # 利用image和trimap估计alpha透明度矩阵
    if title=="cf":
        alpha = estimate_alpha_cf(image, trimap)
    elif title=="knn":
        alpha = estimate_alpha_knn(image, trimap)
    elif title=="lbdm":
        alpha = estimate_alpha_lbdm(image, trimap)
    print("yyy")
    # 根据alpha获取前景图
    foreground = estimate_foreground_ml(image, alpha)
    # 根据前景图和alpha图进行裁剪7
    print("zzz")
    cutout = stack_images(foreground, alpha)
    # 保存图片
    save_image(cutout_path+title+"_cutout.png", cutout)

if __name__ == '__main__':
    for title in alpha_list:
        cutout(title)