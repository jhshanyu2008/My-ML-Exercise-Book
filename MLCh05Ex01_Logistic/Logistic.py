"""
sigmoid函数：代替阶跃函数处理瞬间跳跃难以处理的问题
   σ(z) = 1/(1+exp(-z)) 这个函数性能很利于计算:
   z<<0时σ(z)趋于0 z=0时σ(z)趋于0.5 z>>0时σ(z)趋于 1

# 今天对 Logistic的概念没有理清，干了一天无用功，找到篇转载的博文，原作者已经自己删了，恕我没法写出处，文章写的很不错。
# 明天边看边重写 ，机器学习实战在这章省略的有点过分了，这样根本看不懂的好吧大哥
# 今天把绘制 3D图熟悉了下，也不算白干了，心好累。
"""
from math import *
from numpy import *
# sympy有名称冲突，不能同时导入
import sympy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

Begin = ("I just want it be lighted".format(Axes3D.__name__))


# 读取测试数据
def read_dataset(fileName=''):
    """
    读取测试数据并转化为浮点数/整数
    :param fileName:
    :return:
    """
    dataMatrix = []
    labelMatrix = []
    fr = open(fileName)
    for frLine in fr.readlines():
        lineList = frLine.strip().split()
        # Xi = (x1;x2;...;xn;1) A = (X1';X2';...;Xn')
        dataMatrix.append([float(x) for x in lineList][:-1] + [1.0])
        # 标签行向量 W'，有需要时再转置
        labelMatrix.append(int(lineList[-1]))
    return dataMatrix, labelMatrix


# sigmoid的公式专门写个函数
def sigmoid(z):
    sigemaZ = 1.0 / (1 + exp(-z))
    return sigemaZ


# 梯度上升函数
def gradient_Ascent(dataMatrix, labelMatrix, maxCycles=500):
    """
    梯度上身函数，步长默认1,迭代次数默认300次
    :param maxCycles:
    :param dataMatrix:
    :param labelMatrix:
    :param alpha:
    :return:
    """
    # A阵
    dataMat = mat(dataMatrix)
    # Y'向量
    labelMat = mat(labelMatrix)
    # 获取属性矩阵的行(样本数)、列数(属性数)
    rNum, cNum = shape(dataMat)
    # 载入步长和最大迭代次数
    # alpha = alpha
    maxCycles = maxCycles
    # 初始化 W列向量，它和列长度和属性数相同
    weightsMat = zeros((cNum, 1))
    # 开始迭代计算 W:= W - α*(2/n) * A'.* ((σ(Z)-Y)*(dσ(zi)/dzi))
    for loopNum in range(maxCycles):
        # Z' = (A.*W)'
        Z_trans = (dataMat * weightsMat).transpose()
        h = sigmoid(Z_trans)
        Z_trans_list = list(array(Z_trans)[0])
        h_list = list(array(h)[0])
        diffFxList = []
        # df =  A'.*((σ(Z) - Y) * (dσ(zi) / dzi))
        for i in range(rNum):
            z = sympy.Symbol("z")
            # 这一步的求导非常耗费时间，我先试着把精度调小
            diffZ = sympy.diff(1.0 / (1 + sympy.exp(-z)), z).subs('z', Z_trans_list[i])
            # (σ(Z) - Y) * (dσ(zi) / dzi)
            diffFxList.append(float((h_list[0] - labelMatrix[i]) * diffZ))
        # A'.*((σ(Z) - Y) * (dσ(zi) / dzi))
        diffFxMat = dataMat.transpose() * mat(diffFxList).transpose()
        # W:= W - α*(2/n) df alpha得分段
        if loopNum <= 150:
            alpha = 0.1
        elif 150 < loopNum <= 350:
            alpha = 0.05
        else:
            alpha = 0.01
        weightsMat = weightsMat - alpha * (2 / rNum) * diffFxMat
        # 检查均方差：f = (1/n)*∑(σ(zi)-yi)^2
        variance = (1 / rNum) * sum([value ** 2 for value in list(array(h - labelMat)[0])])
        print("Program in progress,the turn is {0}".format(loopNum + 1))
        print("Variance is {0}".format(variance))
    print(list(array(weightsMat.transpose())[0]))
    return weightsMat


# 我决定自己画一个图，书上那个我已经没法画了
def plot_fitline_3D(fileName=''):
    """
    # 全部在瞎画，没有任何意义，明天去改了
    :param fileName:
    :return:
    """
    dataMatrix, labelMatrix = read_dataset(fileName)
    dataMat = mat(dataMatrix)
    weightsMat = gradient_Ascent(dataMatrix, labelMatrix)
    weightsList = list(array(weightsMat.transpose())[0])
    Z_trans = (dataMat * weightsMat).transpose()
    h = sigmoid(Z_trans)
    hList = list(array(h)[0])
    # 绘制训练集的点集
    rNum = len(dataMatrix)
    xcord1 = []
    ycord1 = []
    zcord1 = []
    xcord2 = []
    ycord2 = []
    zcord2 = []
    for i in range(rNum):
        if int(labelMatrix[i]) == 1:
            xcord1.append(dataMatrix[i][0])
            ycord1.append(dataMatrix[i][1])
            zcord1.append(hList[i])
        else:
            xcord2.append(dataMatrix[i][0])
            ycord2.append(dataMatrix[i][1])
            zcord2.append(hList[i])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set(xlim=[-1.5, 1.5], ylim=[-10, 20], zlim=[0, 1])
    ax.scatter(xcord1, ycord1, zcord1, c='y')
    ax.scatter(xcord2, ycord2, zcord2, c='b')
    X = arange(-3, 3, 0.25)
    Y = arange(-15, 30, 0.25)
    X, Y = meshgrid(X, Y)
    Z = (1/(1+exp(-weightsList[0]*X-weightsList[1]*Y-weightsList[2])))
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='Oranges', antialiased=False)
    ax.set_zlabel('Sigema')
    ax.set_ylabel('x2')
    ax.set_xlabel('x1')
    plt.show()


file_name = 'testSet.txt'
# data_matrix, label_matrix = read_dataset(file_name)
# W = gradient_Ascent(data_matrix, label_matrix)

plot_fitline_3D(file_name)