"""
被导师指派去学labview，我觉得那玩意儿好坑，给的资料也都是些和控制系八竿子搭不上关系的，还好我早TM习惯自学了。
...
今天重新梳理 logistic回归(书里的理论解释的天马行空，我表示强烈抗议！)，参照网上转载的博文，原作者已经自己删了，恕我没法写出处。
函数的目的就是找到一个拟合线：z= w0*x0 + w1*x1 + w2*x2 +...+wn*xn (其中取 x0=1) 使 z尽可能的接近标签 y值，这里的我们最好使用阶跃函数来对样本分类
但是阶跃函数的跳变不易实现，取而代之使用sigmoid函数：

sigmoid函数：代替阶跃函数处理瞬间跳跃难以处理的问题
   h(z) = 1/(1+exp(-z)) 这个函数性能很利于计算:
   z<<0时σ(z)趋于0 z=0时σ(z)趋于0.5 z>>0时σ(z)趋于 1

其实我们使用这个函数得到的不是 0或 1，不能直接分类，事实上，这个函数有一层更深的意义(虽然我没搞懂)：
P(y=1|z) = h(z)  P(y=0|z) = 1 - h(z)
知道了样本的分布律后我们就可以使用常用的最优化方法：最大似然法

最大似然法：(X,W是包含一个样本中所有 xi和 wi的向量，我们假设有 m个样本)
  P(yj|zj) = P(yj|X;W) = (h(zj))^yj * (1-h(zj))^(1-yj)  (由于 y = 0、1上面的两个概率可以写成这一个)
  L(W) = P(y0|z0)*P(y1|z1)*...*P(yn|zn) = [(h(z0))^y0 * (1-h(z0))^(1-y0)]*...*[(h(zn))^yn * (1-h(zn))^(1-yn)]
  习惯性的取对数 l(W) = log(L(W)) = sum(yi*log(h(Xi)) + (1-yi)*log(1-h(Xi)))
  我们的目的是找到最大的 W 比较合适的方法是梯度上升，这里遵照博文用 Andrew Ng中的办法：
  取 J(W) = -(1/n)l(W) 换成求最小值，用梯度下降法

梯度下降法：wj：= Wj - α*(∂J(W)/∂wj)
  ∂J(W)/∂wj 的求解过程这里略过，具体参照博文
  最后的结果是∂J(W)/∂wj = (1/m) * sum(h(zi)-yi) *xij
  梯度算法的过程写为：wj: = wj - α*(1/m) * sum[(h(Xi)-yi)*xij]
  α本来就是带的待定系数，所以把(1/m)省略掉，最后的 wj更新式为：
  wj: = wj - α * sum(h(Xi)-yi) * Xj
  如果是用上升法那么是这样子的：wj: = wj + α * sum[(yi - h(Xi))*xij]  (α是个正数为前提)

上式是对单个 wj系数的求法，现在我们打算 n个系数一起求：
我们令 Xi = [xi0;xi1;...;xin] (其中 xi0为 1) 为第 i个样本的属性向量(列向量) Z = [z1;z2;...;zm] 是 Z阵
      W = [w0;w1;...;wn] 是系数向量 Y = [y1;y2;...;ym] 是标签向量 H = [h1;h2;...;hm]是sigmoid矩阵
      A = [X1';X2',...,Xm'] 把一个个 X向量横起来写，比较像原始的数据样式
我们用梯度下降法：
  首先算 Z：Z = A * W 再算 H：H = h(Z)
  h(Xi) - yi扩展到矩阵是：H - Y
  sum[(yi - h(Xi))*xij]扩展到矩阵是：A'*(H - Y)
  综合起来就是：W: = W - α * A'*(H - Y)

跑了几百次梯度循环后，绘制图形，分界线是按照 z=0 划分的，分类 1的 Z值会大于 0，分类 0 的 z值则会小于 0
所以画一条 z = 0的线会把两个类划分开来。
"""
from math import *
from numpy import *
import matplotlib.pyplot as plt


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
        # Xi = (1；x1;x2;...;xn) A = (X1';X2';...;Xn')
        dataMatrix.append([1.0] + [float(x) for x in lineList][:-1])
        # 标签行向量 W'，有需要时再转置
        labelMatrix.append(int(lineList[-1]))
    return dataMatrix, labelMatrix


# sigmoid的公式专门写个函数
def sigmoid(z):
    sigemaZ = 1.0 / (1 + exp(-z))
    return sigemaZ


# 梯度上升函数
def gradient_descent(dataMatrix, labelMatrix, alpha=0.001, maxCycles=500):
    """
    梯度下降函数，步长默认1,迭代次数默认300次
    :param maxCycles:
    :param dataMatrix:
    :param labelMatrix:
    :param alpha:
    :return:
    """
    # A阵
    A_mat = mat(dataMatrix)
    # Y向量
    Y_mat = mat(labelMatrix).transpose()
    # 获取属性矩阵的行(样本数)、列数(属性数)
    sampleNum, attriNum = shape(A_mat)
    # 载入步长和最大迭代次数
    alpha = alpha
    maxCycles = maxCycles
    # 初始化 W列向量，它和列长度和属性数相同
    W_mat = zeros((attriNum, 1))

    # 开始迭代计算 W: = W - α * A'*(H - Y)
    for loopNum in range(maxCycles):
        # Z = A * W
        Z_mat = A_mat * W_mat
        # H = A * W
        H_mat = sigmoid(Z_mat)
        # W: = W - α * A'*(H - Y)
        W_mat = W_mat - alpha * A_mat.transpose() * (H_mat - Y_mat)
        print("Program in progress,the turn is {0}".format(loopNum + 1))
    W_list = list(array(W_mat.transpose())[0])
    print(W_list)
    return W_list


def plot_fitline(fileName=''):
    dataMatrix, labelMatrix = read_dataset(fileName)
    dataList = list(array(dataMatrix))
    weights = gradient_descent(dataMatrix, labelMatrix)
    sampleNum = len(dataList)
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(sampleNum):
        if int(labelMatrix[i]) == 1:
            xcord1.append(dataList[i][1])
            ycord1.append(dataList[i][2])
        else:
            xcord2.append(dataList[i][1])
            ycord2.append(dataList[i][2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


file_name = 'testSet.txt'
# data_matrix, label_matrix = read_dataset(file_name)
# W = gradient_Ascent(data_matrix, label_matrix)
plot_fitline(file_name)
