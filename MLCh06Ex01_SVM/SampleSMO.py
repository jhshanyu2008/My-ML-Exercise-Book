"""
今天学习支持向量机，某种意义上和 Logistic回归有点像，但使用的算法完全不同
还是使用线性方程来划分超平面，根据书上的写法，写成：W' * Xj + b = 0
这里我们不再使用 sigmoid函数获取概率计算最大似然参数，而是建立一种模型：计算样本空间中任意点 Xj到超平面的距离：
    r = |W'Xj + b|/||W|| 这其实就是点到直线距离计算法的一种延伸。

支持向量机也是二分类，分成 -1，+1两类，和 Logistic分类 0，1利于概率计算类似，支持向量机这么分也是为了计算简单：
    W'Xj + b >= 1 then yj = 1 ; W'Xj + b <= 1 then yj = -1
根据定义，样本中距离划分线最近点的距离是 1/||W||，换句话说，划分线穿过样本点的路径宽度是 2/||W||

现在我们要找到一个最优解使这个路径宽度最大从而使泛化效果尽可能的好：
    max( 2/||W|| ) 且满足 yj * (W' + b) >= 1

写成 min( 0.5 * ||W|| )这样子计算简单点，因为有约束条件，所以拉格朗日乘子法是个很自然的选择。
记为：L(W,b,A) = (||W||^2)/2 + sum(aj * (1 - yj * (W' + b)))  j是样本序号 A是 aj的向量
这里参照的是西瓜书，条件是：yj * (W' + b) >= 1 显然这是一个区域，但是上式是只考虑边界条件的，为什么可以这样子呢？
西瓜书里提及了 KKT条件，我承认没看懂，但是想一下建立这个模型我们需要哪些数据的话，显然————我们只需要边界的数据就足够了。
事实上，在计算中会选取特定的方法使其他非边界点得到 0系数，它们无法影响模型。

先撇开疑惑把上式计算一下：
    ∂L/∂W = W - sum(aj * yj * Xj) 令它为 0我们的得到：W* = sum(aj * yj * Xj)
    ∂L/∂b = 0 - sum(aj * yj) 令他为 0我们得到：0 = sum(aj * yj)
回带入原方程得到(i = j in range (m))
    L(W*,b,A) = 0.5 * W'W +  sum(aj * (1 - yj * (W' + b)))
             = sum(aj) - 0.5 * sum(sum(ai * aj * Xi' * Xj))
             = H(X,A) 且 sum(aj * yj) = 0 ;aj >= 0
    根据拉格朗日函数的性质有 0.5*||W|| > L(W,b,A) H函数是 L函数的下界它，所以有：
    0.5||W|| >= L(W*,b,A) >= H(X,A) H函数即是原函数的一个对偶函数，原函数的最小值即对偶函数的最大值：
    min(0.5 * ||W||) = max(H(X,A)) = max( sum(aj) - 0.5 * sum(sum(ai * aj * Xi' * Xj)) )
    如果解得出 aj那皆大欢喜，我们就可以得到模型：f(Xi) = W' * X + b = sum(aj * yj * Xj' * Xi) + b
    最关键是...这 TM该怎么求 aj 和 b？

最著名的方法就是 SMO，它的基本思路是：
    先固定 aj之外的所有参数，然后求 aj的极值，因为有条件 sum(aj * yj) = 0 所以 aj一定可以由其他变量导出。
    在初始化后执行以下两个步骤：
    1.选取一对需要更新的变量 ai和 aj；
    2.固定 ai和 aj以外的参数，求解 max(H(X,A))，获得更新后的 ai和 aj
    【注意】a的选取若是不满足 KTT条件，函数数值会增大，且违背程度越大，增大的越多。
           我们要先选取一个最违背 KTT条件的 a，第二个选使目标函数增长最快的变量(一个最大，一个最快...好抽象)
           但是上面那个选法计算量太大，SMO给出的优化法是：选两个变量所对应样本的间隔最大的，换句话说，这俩样本差别很大，
           更新它们对应的参数会最大的改变函数值。
KTT条件：aj >= 0
        yj * f(Xj) - 1 >= 0
        aj * (yj * f(Xj) - 1) = 0

理清这个算法的本质：
    算法的目的：找到所有满足 KTT条件的 a，有两种可能：
    yj * f(Xj) - 1 > 0 对于这种情况，分类本身就没有问题，要满足条件三，只要直接令 a为 0
    yj * f(Xj) - 1 < 0 的这些点是不满足分类需求的，我们需要修正 W和 b使得它们变成 yj * f(Xj) - 1 = 0
总结一下就是：
    aj = 0  yj * f(Xj) - 1 > 0 分类正确
    1 <= aj <= c  yj * f(Xj) - 1 = 0 边界上，支持向量
    aj = c  yj * f(Xj) - 1 < 0 边界内，需要调整
    【注意】这里多了个参数 c，c的意义我也深究不了，网上是这么解释的：C越大表示惩罚越大,造成过拟合，反之是欠拟合，姑且记着先。
所以以下三种情况是需要修正的：
    yj * f(Xj) - 1 <= 0 ; aj < c
    yj * f(Xj) - 1 >= 0 ; aj > 0
    yj * f(Xj) - 1 <= 0 ; aj = 0 or c

具体的计算方法呢...好复杂，具体参考博客：http://blog.csdn.net/on2way/article/details/47730367
有空再慢慢补充，先按照算法把函数写完先。

现在分析下 SMO的高效原理：
    选取 ai和aj后固定其他参数意味着 sum(aj * yj) = 0条件变成了：ai * yi + aj * aj = c (c是其他项的和)
    把这个条件和 H函数联立，消去 ai我们得到一个只有一个参数的方程，而且它只有一个约束条件 aj>=0
    偏移量 b怎么办呢？我们注意到所有的支持向量点都在边界上，有：yi(f(Xj) = W' * Xj + b) = 1
    即：yi * (sum(aj * yj * Xj' * X) + b) = 1 理论上随便选一个支持向量都能求出 b，但我们喜欢取平均值：
    b = average(bs) (bs是各个支持向量机求出的 b值)

西瓜书上关于线性模型的内容就是上面这些了，回到机器学习实战上(一如既往天马行空般的理论解释，我想不吐槽了)
    简化版 SMO：选 a的时候方法：遍历每一个 a，在剩下的 a中随机选择一个 a，在满足约束条件下同时改变两个 a
"""
from numpy import *
import matplotlib.pyplot as plt


# 辅助函数，从文本读取数据，输出属性矩阵和标签向量
def load_dataset(fileName='testSet.txt'):
    dataMatrix = []
    labelVec = []
    fr = open(fileName)
    for line in fr.readlines():
        lineList = line.strip().split('\t')
        # Xj = (x1;x2;...;xn) A = (X1';X2';...;Xn')
        dataMatrix.append([float(x) for x in lineList][:-1])
        # 标签行向量 Y'，有需要时再转置
        labelVec.append(float(lineList[-1]))
    return dataMatrix, labelVec


# 辅助函数，从文本读取数据，但只利用其中两个样本属性
def load_special_dataset(fileName='testSet.txt', attriA=0, attriB=1):
    dataMatrix = []
    labelVec = []
    fr = open(fileName)
    for line in fr.readlines():
        lineList = line.strip().split()
        # Xj = (x1;x2;...;xn) A = (X1';X2';...;Xn')
        dataMatrix.append([float(lineList[attriA]), float(lineList[attriB])])
        # 标签行向量 Y'，有需要时再转置
        labelVec.append(float(lineList[-1]))
    return dataMatrix, labelVec


# 辅助函数，用于某区间范围内随机选取一个和 a不同的 a
def random_2rdAlpha(i, sampleSum):
    j = i
    while j == i:
        j = int(random.uniform(0, sampleSum))
    return j


# 辅助函数，调整大于 highLimit和小于 lowLimit的 a
def clip_alpha(aj, highLimit, lowLimit):
    if aj > highLimit:
        aj = highLimit
    elif aj < lowLimit:
        aj = lowLimit
    return aj


def SVM_simpleSMO(dataMatrix, labelVec, c, faultToler=0.001, maxCycle=50):
    """
    简化版 SMO，主要的简化是指 alpha的选取，这里是遍历所有 a挑出需要改变的 ai，再在范围内随机挑一个 aj参与变换
    本函数通过————检测到连续 maxCycle次所有 alpha全部未改变————来判断结束条件，
    随着样本数量上升，本函数的计算时间会大大延长，所以这个方法并不是很好。
    :param dataMatrix:
    :param labelVec:
    :param c:
    :param faultToler:
    :param maxCycle:
    :return:
    """
    # A阵 A = (X1';X2';...;Xn')
    A_mat = mat(dataMatrix)
    # Y阵
    Y_vec = mat(labelVec).transpose()
    b = 0
    # 获取训练集的行数(样本)和列数(属性)
    sampleNum, attributeNum = shape(dataMatrix)
    # α阵初始化，它和样本数量一样多
    Alpha_mat = mat(zeros((sampleNum, 1)))
    # 记录 遍历 a后所有 a都未改变的次数，而且只要中途有改变，则归零
    unchangedSum = 0
    # 累计未变化 maxCyCle次后跳出循环
    while unchangedSum < maxCycle:
        # 记录 a是否变化的标志位
        alphaPairsChanged = 0
        for i in range(sampleNum):
            # f(Xi) = sum(j=1-m)(aj * yj * Xj') * Xi + b 其中 multiply(Alpha_mat, Y_vec)是数乘不是矩阵乘法
            funcXi = float(multiply(Alpha_mat, Y_vec).T * A_mat * A_mat[i, :].T) + b
            # 计算误差
            errorXi = funcXi - float(Y_vec[i])
            # 以下三种情况是需要修正的：
            # yi * f(Xi) - 1 = yi * (f(Xi) - yi) = yi * exi <= 0 ; ai < c
            # yi * f(Xi) - 1 = yi * (f(Xi) - yi) = yi * exi >= 0 ; ai > 0
            # yi * f(Xi) - 1 <= 0 ; ai = 0 or c
            # 这里使用 faultToler 替代 0呢...我也不太清楚原理，而且书上不写第三种情况，我决定加上去(和第一种合并)
            if ((Y_vec[i] * errorXi < -faultToler) and (0 <= Alpha_mat[i] <= c)) or \
                    ((Y_vec[i] * errorXi > faultToler) and (Alpha_mat[i] > 0)):
                # 随机选取第二个 alpha
                j = random_2rdAlpha(i, sampleNum)
                # f(Xj) = sum(i=1-m)(ai * yi * Xi' * Xj) + b
                funcXj = float(multiply(Alpha_mat, Y_vec).T * A_mat * A_mat[j, :].T) + b
                errorXj = funcXj - float(Y_vec[j])
                # 处理前先把两个 alpha保存备份下，用copy函数确保是深度复制
                Alpha_1Old = Alpha_mat[i].copy()
                Alpha_2Old = Alpha_mat[j].copy()
                # 根据 ai + aj 值固定及 ai属于0-c 得出的 aj的上下限
                if Y_vec[i] != Y_vec[j]:
                    lowLimit = max(Alpha_mat[j] - Alpha_mat[i], 0)
                    highLimit = min(c, c + Alpha_mat[j] - Alpha_mat[i])
                else:
                    lowLimit = max(Alpha_mat[j] + Alpha_mat[i] - c, 0)
                    highLimit = min(c, Alpha_mat[j] + Alpha_mat[i])
                # 如果 lowLimit = highLimit 那说明 aj和 ai是同一个，跳过。
                if lowLimit == highLimit:
                    print("L==H")
                    continue

                # η = 2K(x1,x2)−K(x1,x1)−K(x2,x2)  η就是 eta K(x,y)指下x,y向量的点乘
                eta = 2.0 * A_mat[i, :] * A_mat[j, :].T - A_mat[i, :] * A_mat[i, :].T - A_mat[j, :] * A_mat[j, :].T
                # 这个为社么 η必须小于 0还没懂...
                if eta >= 0:
                    print("eta>=0")
                    continue

                # a2new = α2old − y2*(Ex1−Ex2)η
                Alpha_mat[j] = Alpha_2Old - Y_vec[j] * (errorXi - errorXj) / eta
                # 修正一下 aj，不能越界
                Alpha_mat[j] = clip_alpha(Alpha_mat[j], highLimit, lowLimit)
                # 变化的太小则忽略这次变化
                if abs(Alpha_mat[j] - Alpha_2Old) < 0.00001:
                    print("j not moving enough")
                    continue
                # α1new = α1old + y1y2(α2old−α2new)
                Alpha_mat[i] = Alpha_1Old + Y_vec[j] * Y_vec[i] * (Alpha_2Old - Alpha_mat[j])

                # 现在求 b
                # b1new = bold−E1−y1(α1new−α1old)K(x1, x1)−y2(α2new−α2old)K(x1, x2)
                # b2new = bold−E2−y1(α1new−α1old)K(x1, x2)−y2(α2new−α2old)K(x2, x2)
                b1 = b - errorXi - Y_vec[i] * (Alpha_mat[i] - Alpha_1Old) * A_mat[i, :] * A_mat[i, :].T \
                     - Y_vec[j] * (Alpha_mat[j] - Alpha_2Old) * A_mat[i, :] * A_mat[j, :].T
                b2 = b - errorXj - Y_vec[i] * (Alpha_mat[i] - Alpha_1Old) * A_mat[i, :] * A_mat[j, :].T \
                     - Y_vec[j] * (Alpha_mat[j] - Alpha_2Old) * A_mat[j, :] * A_mat[j, :].T
                # 若 0 ≤ α1new ≤ c 取 b1
                # 若 0 ≤ α2new ≤ c 取 b2
                # 其他情况 取 (b1+b2)/2
                if 0 < Alpha_mat[i] < c:
                    b = b1
                elif 0 < Alpha_mat[j] < c:
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                # 能走到这步说明 alpha和 b确实改变了，标志位置一
                alphaPairsChanged += 1
                print("Alpha[{0}]:{1} pairs changed Alpha[{2}]:{3}".format(i, Alpha_mat[i], j, Alpha_mat[j]))
        # 所有alpha都没改变，则累加器加一
        if alphaPairsChanged == 0:
            unchangedSum += 1
            print("Alpha unchanged:{} times".format(unchangedSum))
        else:
            unchangedSum = 0
    return b, Alpha_mat


# 这是 Logistic回归里复制过来的，由于是平面图最多只有 2个属性，所以只挑选两个属性用于计算
def plot_fitline(fileName='testSet.txt', attriA=0, attriB=1):
    dataMatrix, labelVec = load_special_dataset(fileName, attriA, attriB)
    dataArray = array(dataMatrix)
    sampleNum = len(dataArray)
    b, Alpha_mat = SVM_simpleSMO(dataMatrix, labelVec, c=0.6, faultToler=0.001, maxCycle=50)
    # W = sum(aj * yj * Xj) = （Alpha_vec 数乘 Y_vec）' * A_vec
    W_List = list(array(multiply(Alpha_mat, mat(labelVec).transpose()).T * mat(dataMatrix))[0])
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(sampleNum):
        if int(labelVec[i]) == 1:
            xcord1.append(dataArray[i, 0])
            ycord1.append(dataArray[i, 1])
        else:
            xcord2.append(dataArray[i, 0])
            ycord2.append(dataArray[i, 1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-2.0, 10.0, 0.1)
    y0 = (-float(b) - W_List[0] * x) / W_List[1]
    ax.plot(x, y0)
    y1 = (1 - float(b) - W_List[0] * x) / W_List[1]
    ax.plot(x, y1, ':')
    y2 = (-1 - float(b) - W_List[0] * x) / W_List[1]
    ax.plot(x, y2, ':')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


# plot_fitline('testSet.txt')
