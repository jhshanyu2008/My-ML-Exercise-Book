"""
重新梳理 logistic回归(书里的理论解释的天马行空，我表示强烈抗议！)，参照网上转载的博文，原作者已经自己删了，恕我没法写出处。
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
def read_dataset(fileName='testSet.txt'):
    """
    读取测试数据并转化为浮点数/整数
    :param fileName:
    :return:
    """
    dataMatrix = []
    labelVec = []
    fr = open(fileName)
    for frLine in fr.readlines():
        lineList = frLine.strip().split()
        # Xi = (1；x1;x2;...;xn) A = (X1';X2';...;Xn')
        dataMatrix.append([1.0] + [float(x) for x in lineList][:-1])
        # 标签行向量 W'，有需要时再转置
        labelVec.append(round(float(lineList[-1])))
    return dataMatrix, labelVec


# sigmoid的公式专门写个函数
def sigmoid(z):
    sigemaZ = 1.0 / (1 + exp(-z))
    return sigemaZ


# 梯度下降函数
def gradient_descent(dataMatrix, labelVec, alpha=0.0005, maxCycles=2000):
    """
    梯度下降函数，步长默认1,迭代次数默认300次
    :param maxCycles:
    :param dataMatrix:
    :param labelVec:
    :param alpha:
    :return W_list:
    """
    # A阵
    A_mat = mat(dataMatrix)
    # Y向量
    Y_mat = mat(labelVec).transpose()
    # 获取属性矩阵的行(样本数)、列数(属性数)
    sampleNum, attriNum = shape(A_mat)
    # 载入步长和最大迭代次数
    alpha = alpha
    maxCycles = maxCycles
    # 初始化 W列向量，它和列长度和属性数相同
    W_mat = ones((attriNum, 1))

    # 开始迭代计算 W: = W - α * A'*(H - Y)
    for loopNum in range(maxCycles):
        # Z = A * W
        Z_mat = A_mat * W_mat
        # H = h(Z)
        H_mat = sigmoid(Z_mat)
        # W: = W - α * A'*(H - Y)
        W_mat = W_mat - alpha * A_mat.transpose() * (H_mat - Y_mat)
        print("Program in progress,the turn is {0}".format(loopNum + 1))
    W_list = list(array(W_mat.transpose())[0])
    print(W_list)
    return W_list


# 随机梯度下降函数
def random_gradient_descent(dataMatrix, labelVec, alpha=0.01):
    """
    简化版梯度下降,这里的计算全部在数组域完成，不使用矩阵，缩短计算时间

    在完整的梯度下降法中，我们使用所有样本参与最大似然法的计算，这里我们就用了一个样本。
    为什么可以这么干呢？我分析了下觉得其实这种方法是用于给原来的训练集添加新的数据时用的
    比如原来有 n个数据，已经用梯度下降法做好了拟合，这 n个数据已经可以不用再管了，我们只要关注添加的新的数据对训练好的 W的影响。
    只用一个样本参与最大似然法显然就是假定：我们在原来拟合好的基础上新添加了一个样本，
    如此反复，这个过程重复了 n次，就相当于我们把 n个样本添加进了训练集

    事实上，这个函数的效果肯定是很不好的，因为就像假设的那样，需要在原拟合线的基础上修正，但在这个函数里：
    一.没有原始的拟合线。二.每次拟合过程只跑一遍。
    但是，即使是知道效果不好，也很难改变，因为你在使用这个方法的时候，已经默认之前的 n个训练集和这一个新的训练集拥有相同的权重了，
    如果出现了一个很不符常理数据，就会对结果造成重大影响，引起参数的剧烈波动，我觉得这个函数名很不名副其实。

    书中有介绍改进办法: random_gradient_descent_ver2
    事实上，我觉得改进法和本函数在思想上已经完全不同了，本函数是顺序添加样本，而改进版是随机取样并递减步长：
    这样子你会大概率在步长比较大的时候取到正常的数据，而取到高偏差样本时又因为步长变小波动不会大。
    :param dataMatrix:
    :param labelVec:
    :param alpha:
    :return W_list:
    """
    # 获取属性矩阵的行(样本数)、列数(属性数)
    sampleNum, attriNum = shape(dataMatrix)
    # 载入步长和最大迭代次数
    alpha = alpha
    # 初始化 W列向量，它和列长度和属性数相同
    W_Array = ones(attriNum)

    # 开始迭代计算 W: = W - α * (h - y) * X'
    for i in range(sampleNum):
        # 求 z 这里使用的是数组乘
        z_value = sum(dataMatrix[i] * W_Array)
        # h(Z) = sigmoid(z)
        h_value = sigmoid(z_value)
        # W: = W - α * (h - y) * X'
        W_Array = W_Array - alpha * (h_value - labelVec[i]) * array(dataMatrix[i])
        print("Program in progress,the turn is {0}".format(i + 1))
    W_list = list(W_Array)
    print(W_list)
    return W_list


# 改进型随机梯度下降法
def random_gradient_descent_ver2(dataMatrix, labelVec, alphaBase=0.001, maxCycles=150):
    """
    改进版本的随机梯度下降法：
    首先，它循环了 maxCycles次，而不是原始的 1次
    其次，迭代步长越来越小，高偏差值的样本波动也会随之越来越小
    最后，它使用随机取样，我们可以大概率在迭代中避开一些高偏差的样本，这样子即使偶尔取到了，由于步长已经变小，波动也不大
         如果真的非常不幸的在前期大量取到高偏差样本，大不了重来一次，要是再取到，可以去买彩票。

    书里的取样法我觉得很迷，不知道他怎么想的，我干脆就用真正的随机取样，所有样本一律平等，可能会有些样本根本没取到，
    但只要训练的次数够多，每个样本取到次数的期望值是一定的。
    :param maxCycles:
    :param dataMatrix:
    :param labelVec:
    :param alphaBase:
    :return W_list:
    """
    sampleNum, attriNum = shape(dataMatrix)
    # 初始化 W列向量，它和列长度和属性数相同
    W_Array = ones(attriNum)

    for j in range(maxCycles):
        for i in range(sampleNum):
            # 步长越来越小
            alpha = 4 / (1.0 + j + i) + alphaBase
            # 随机取样
            randomIndex = int(random.uniform(0, sampleNum))
            # 求 z 这里使用的是数组乘
            z_value = sum(dataMatrix[randomIndex] * W_Array)
            # h(Z) = sigmoid(z)
            h_value = sigmoid(z_value)
            # W: = W - α * (h - y) * X'
            W_Array = W_Array - alpha * (h_value - labelVec[randomIndex]) * array(dataMatrix[randomIndex])
        print("Program in progress,the turn is {0}".format((j + 1)*sampleNum))
    W_list = list(W_Array)
    print(W_list)
    return W_list


def plot_fitline(fileName='testSet.txt'):
    dataMatrix, labelVec = read_dataset(fileName)
    dataList = list(array(dataMatrix))
    # weights = gradient_descent(dataMatrix, labelVec)
    # weights = random_gradient_descent(dataMatrix, labelVec)
    weights = random_gradient_descent_ver2(dataMatrix, labelVec)
    sampleNum = len(dataList)
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(sampleNum):
        if int(labelVec[i]) == 1:
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


# file_name = 'testSet.txt'
# data_matrix, label_matrix = read_dataset(file_name)
# W = gradient_descent(data_matrix, label_matrix)
# plot_fitline(file_name)
