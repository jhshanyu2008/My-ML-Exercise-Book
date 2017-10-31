"""
直接用sklearn机器学习包测试图片内容聚类
我要把测试图片的外圈背景给删了
"""

import numpy as np
import PIL.Image as image
from sklearn.cluster import KMeans


def load_image(file_name):
    # 二进制打开图片
    fr = open(file_name, 'rb')
    dataMatrix = []
    # 以列表形式返回图片像素值
    img = image.open(fr)
    # 图片宽和长
    m, n = img.size
    # 将每个像素点RGB颜色处理到 0-1范围内并二维列表存放
    for r in range(m):
        for c in range(n):
            dataMatrix.append(np.array(img.getpixel((r, c))) / 256)
    fr.close()
    # 返回需要的信息
    return np.mat(dataMatrix), m, n


def main_func(img_name="test.png", k=15):
    fr = open(img_name, 'rb')
    img_file = image.open(fr)
    img_data, row, col = load_image(img_name)
    # k个中心
    raw_label = KMeans(n_clusters=k).fit_predict(img_data)
    # 聚类获得每个像素所属的类别
    label_matrix = raw_label.reshape([row, col])
    # 创建新的PNG图保各种结果
    pic_new = image.new("RGBA", (row, col))
    # 生成各个分类后的图片
    deletNum = 0
    for n in range(k):
        for r in range(row):
            for c in range(col):
                if label_matrix[r][c] == n:
                    pic_new.putpixel((r, c), img_file.getpixel((r, c)))
                else:
                    pic_new.putpixel((r, c), (255, 255, 255, 0))
        # 等会在组合图片时排除不满足这个条件的类别
        if pic_new.getpixel((160, 10)) != (255, 255, 255, 0):
            deletNum = n
        pic_new.save('{0}.png'.format(n), 'PNG')
    # 生成需要的图片
    for r in range(row):
        for c in range(col):
            if label_matrix[r][c] != deletNum:
                pic_new.putpixel((r, c), img_file.getpixel((r, c)))
            else:
                pic_new.putpixel((r, c), (255, 255, 255, 0))
    pic_new.save('result.png', 'PNG')


main_func("shigure.png")
