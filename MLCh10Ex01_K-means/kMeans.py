"""
k均值算法 目的就是最小化平方误差：
E = sum(i=1 to k)(sum(X∈Ci)||X-μi||^2) 其中 μi是各个簇分类元素和其中心的均差
这个算法的一个难点是中心的查找，k均值选择了贪心算法，通过迭代来找到近似解：
西瓜书是这么记述的：
1.首先随机选择 k个样本作初始的均值 μ向量
2.计算各个点到各个 μ的距离，选择最小的那个作为该点的分类
3.每个点都被分类后，原始的簇已经形成，可以重新计算各个 μ
4.重复 2、3的过程，直到前后两次迭代没有变化为止，结束。

机器学习实战里的逻辑基本完全一样，除了开始的时候它没有直接取样本，而是做了个随机数计算
"""
from numpy import *
import matplotlib.pyplot as plt


# 读取数据集
def load_dataset(fileName='testSet.txt'):
    dataMatrix = []
    fr = open(fileName)
    for sample in fr.readlines():
        # map函数对指定列表做固定函数操作，等价于 [float(x) for x in sample.strip().split()]
        # 【注意】python3中，map对结果做了封装，正常显示需要附加 list操作
        featValues = map(float, sample.strip().split())
        dataMatrix.append(list(featValues))
    return dataMatrix


# 两个向量的距离
def euclidean_distance(vecA, vecB):
    # power函数，就像字面意思：前一个参数的所有元素作后一个参数次的方
    # sum、sqrt也是，分别是求和和开放，numpy真是实用
    # 这句话就是求vecA和vecB的欧氏距离
    distance = sqrt(sum(power(vecA - vecB, 2)))
    return distance


# 随机查找质心
def random_centroid(A_mat, k):
    """
    k是质心的数量
    :param A_mat:
    :param k:
    :return:
    """
    featSum = shape(A_mat)[1]
    centroid = mat(zeros((k, featSum)))
    # 我还是习惯 i指代特征序列，j指代样本序列
    for i in range(featSum):
        # 取最值的函数我试了试 list状态下不能用
        minI = min(A_mat[:, i])
        maxI = max(A_mat[:, i])
        rangeI = float(maxI - minI)
        # random.rand(k,1)生成 k行 1列的 0-1的随机数
        centroid[:, i] = minI + rangeI * random.rand(k, 1)
    return centroid


# k均值迭代逻辑函数
def k_means(A_mat, k, distFunc=euclidean_distance, centroidFunc=random_centroid):
    """
     这里有两个参数是函数指针，类似于 C#里的事件，感觉比那个随意，python啥都很随意
    :param A_mat:
    :param k:
    :param distFunc:
    :param centroidFunc:
    :return:
    """
    sampleSum = len(A_mat)
    # 这里存放样本分类结果，两列，第一列记录类别，第二列记录和质心的欧式距离平方
    clusterAssess = mat(zeros((sampleSum, 2)))
    # 随机选取质心，即 μ
    centroids = centroidFunc(A_mat, k)
    # 判断前后两次迭代是否有变化的标志位
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        # 遍历各个样本
        for j in range(sampleSum):
            # 初始化最小距离及其序号，inf即 infinite“无穷大”
            minDist = inf
            minIndex = -1
            # 遍历各个质心
            for n in range(k):
                # 计算第 n个质心和第 j个样本的距离
                distNJ = distFunc(centroids[n, :], A_mat[j, :])
                # 记录当前样本的最小距离和这个距离所属的质心序号
                if distNJ < minDist:
                    minDist = distNJ
                    minIndex = n
            # 检查这个样本是否和上一次迭代后的分类相同，不同的话就必须进行下一次迭代
            if clusterAssess[j, 0] != minIndex:
                clusterChanged = True
            # 更新样本分类，顺便存下和质心的欧式距离的平方
            clusterAssess[j, :] = minIndex, minDist ** 2
        print("******centroids updated******\n{}".format(centroids))
        # 计算新的质心，即 μ
        for cent in range(k):
            # 选出各个cent分类下的样本，[nonzero(clusterAssess[:, 0].A == cent)[0]]
            # clusterAssess[:, 0].A == cent 得到 True的才会被选取
            # nonzero函数返回两个元素，第一个元素记录各个满足条件数所在的行，第二个是相应的列
            # 【注意】这条语句必须在矩阵状态下才能使用。其实 nonzero内部这个转 array不是必须的
            centSample_mat = A_mat[nonzero(clusterAssess[:, 0].A == cent)[0]]
            # 求取平均值得到新的centroid 即μ，axis=0 即是按照第一维度(列方向)求均值
            centroids[cent, :] = mean(centSample_mat, axis=0)
    return centroids, clusterAssess


# 绘图辅助函数
def plot_fitline(A_mat, centroids, clusterAssess, titleTxt):
    k = len(centroids)
    sampleSum = len(A_mat)
    color = {0: 'red', 1: 'blue', 2: 'green', 3: 'black', 4: 'yellow'}
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # 绘制点坐标
    for n in range(k):
        xcord = []
        ycord = []
        for j in range(sampleSum):
            if clusterAssess[j, 0] == n:
                xcord.append(A_mat[j, 0])
                ycord.append(A_mat[j, 1])
        ax.scatter(xcord, ycord, s=30, c=color[n], marker='s')

    # 绘制 centroid坐标
    for n in range(k):
        xcord = centroids[n, 0]
        ycord = centroids[n, 1]
        ax.scatter(xcord, ycord, s=50, c=color[n], marker='*')

    ax.set_title(titleTxt)
    ax.set_xlabel('First feature')
    ax.set_ylabel('Second feature')
    plt.show()


def main_func(fileName='testSet.txt', k=4):
    dataMatrix = load_dataset(fileName)
    A_mat = mat(dataMatrix)
    centroids, clusterAssess = k_means(A_mat, k)
    plot_fitline(A_mat, centroids, clusterAssess, 'testSet')


main_func(k=4)
