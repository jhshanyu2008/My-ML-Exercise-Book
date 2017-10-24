"""
完整的 SMO算法
就图画来看，性能怎么说呢，图不好看，但分类还算准
第一次随机到的 aj直接大范围影响性能，基本上跑个7、8次会出现一次性能好的情况。
除了调整 c和 faultToler的大小外
我感觉内循环次数越多，越准确，但是大多数实验都是3、4次就结束了，能自动循环到 7次的基本性能就比较好
我尝试在开始阶段增加几次随机 aj，再增加强制循环的次数,确实比原来的靠谱一点。
这应该就是速度和性能的取舍了
"""
from SampleSMO import *


# 写一个专门的类，保存 SMO任务的各种数据，包括 A阵，Y阵，Alpha阵等等
class SMO_struct:
    def __init__(self, fileName="testSet", c=200, faultToler=0.0001, maxCycle=100, kType=('linear', 0)):
        self.dataMatrix, labelVec = load_dataset(fileName)
        self.c = c
        self.faultToler = faultToler
        self.maxCycle = maxCycle
        self.A_mat = mat(self.dataMatrix)
        self.Y_vec = mat(labelVec).transpose()
        self.sampleSum = shape(self.A_mat)[0]
        self.Alpha_vec = mat(zeros((self.sampleSum, 1)))
        self.b = 0
        self.lestLoop = 7
        self.randAjSum = 5
        # 误差缓存，第一列是是否有效的标志位，第二列是实际的 E值
        self.ErrCache_mat = mat(zeros((self.sampleSum, 2)))
        self.K = mat(zeros((self.sampleSum, self.sampleSum)))
        for i in range(self.sampleSum):
            # 一次计算一整列的 K
            self.K[:, i] = kernelTrans(self.A_mat, self.A_mat[i, :], kType)


# 书上程序新增的一个辅助计算 K(xi,xj)的函数，我先不改直接拿来用
def kernelTrans(A_mat, A, kType):
    m, n = shape(A_mat)
    K = mat(zeros((m, 1)))
    # linear kernel 其实我们就用这个
    if kType[0] == 'linear':
        K = A_mat * A.T
    elif kType[0] == 'kernel':
        for j in range(m):
            deltaRow = A_mat[j, :] - A
            K[j] = deltaRow * deltaRow.T
        K = exp(K / (-1 * kType[1] ** 2))
    else:
        raise NameError('The Kernel is not recognized')
    return K


# 计算拟合误差
def get_fit_error(smoTask, k):
    # f(Xj) = sum(i=1-m)(ai * yi * Xi') * Xj + b
    funcXk = float(multiply(smoTask.Alpha_vec, smoTask.Y_vec).T * smoTask.K[:, k] + smoTask.b)
    errXk = funcXk - float(smoTask.Y_vec[k])
    return errXk


# PlattSMO挑选 aj 在每次优化中采用最大步长
def select_2rdAlpha(i, smoTask, errXi):
    suitJ = -1
    # 最大步长寄存器
    maxDeltaErr = 0
    errXj = 0
    # 跟新错误缓存
    smoTask.ErrCache_mat[i] = [1, errXi]
    print("++++update aj:a{0} = {1}".format(i, float(smoTask.Alpha_vec[i])))
    # mat.A和 mat.getA()和 array(mat) 相同
    # nozero 返回一个元祖，元祖有两个元素，第一个返回所有非零元素的行数，第二个返回列数
    # 检查第 0列中非 0元素的个数，即检查各个标志位
    validEcacheList = nonzero(smoTask.ErrCache_mat[:, 0].A)[0]
    # 若所含非 0个数大于 1，即除了 ErrCache_mat[i]外还有其他项被标志为 1
    # 这里有个很巧妙的设计，validEcacheList > n 在程序开始后会取至少有 (n+1)/2 个随机 aj
    print("----len of validEcacheList is:{0}".format(len(validEcacheList)))
    if (len(validEcacheList)) > (smoTask.randAjSum * 2 - 1):
        # 遍历这些含非零值的行序号
        for k in validEcacheList:
            if k == i:
                continue
            errXk = get_fit_error(smoTask, k)
            # 检查两个样本的步长 |Ei-Ej|，我也不是很懂步长的定义...
            deltaErr = abs(errXi - errXk)
            # 总之选择最大步长的 j
            if deltaErr > maxDeltaErr:
                suitJ = k
                maxDeltaErr = deltaErr
                errXj = errXk
        print("----Choose a suitable j:{0}".format(suitJ))
        return suitJ, errXj
    # 否则随便取一个，一般是第一次循环，因为默认所有 alpha为 0
    else:
        j = random_2rdAlpha(i, smoTask.sampleSum)
        errXj = get_fit_error(smoTask, j)
        print("----Choose a 【random】 j:{0}".format(j))
    return j, errXj


# 更新 SMO类里的 ErrCache_mat
def update_errCacheMat(smoTask, k):
    errXk = get_fit_error(smoTask, k)
    smoTask.ErrCache_mat[k] = [1, errXk]


# 算法的内循环方法(SampleSMO中 while内的 for循环)，单独写一个函数
def plattSMO_inner_loop(i, smoTask):
    errXi = get_fit_error(smoTask, i)
    # 以下三种情况是需要修正的：
    # yi * f(Xi) - 1 = yi * (f(Xi) - yi) = yi * exi <= 0 ; ai < c
    # yi * f(Xi) - 1 = yi * (f(Xi) - yi) = yi * exi >= 0 ; ai > 0
    # yi * f(Xi) - 1 <= 0 ; ai = 0 or c 边界条件算法里会另外考虑，所以不写
    # 这里使用 faultToler 替代 0呢...我也不太清楚原理
    print("----y * f(x) - 1 = {0} and a{1} = {2}".
          format(float(smoTask.Y_vec[i] * errXi), i, float(smoTask.Alpha_vec[i])))
    if ((smoTask.Y_vec[i] * errXi < -smoTask.faultToler) and (smoTask.Alpha_vec[i] < smoTask.c)) or \
            ((smoTask.Y_vec[i] * errXi > smoTask.faultToler) and (smoTask.Alpha_vec[i] > 0)):
        # 选取第二个 alpha
        j, errXj = select_2rdAlpha(i, smoTask, errXi)
        # 处理前先把两个 alpha保存备份下，用copy函数确保是深度复制
        Alpha_1Old = smoTask.Alpha_vec[i].copy()
        Alpha_2Old = smoTask.Alpha_vec[j].copy()
        # 根据 ai + aj 值固定及 ai属于0-c 得出的 aj的上下限
        if smoTask.Y_vec[i] != smoTask.Y_vec[j]:
            lowLimit = max(smoTask.Alpha_vec[j] - smoTask.Alpha_vec[i], 0)
            highLimit = min(smoTask.c, smoTask.c + smoTask.Alpha_vec[j] - smoTask.Alpha_vec[i])
        else:
            lowLimit = max(smoTask.Alpha_vec[j] + smoTask.Alpha_vec[i] - smoTask.c, 0)
            highLimit = min(smoTask.c, smoTask.Alpha_vec[j] + smoTask.Alpha_vec[i])
        if lowLimit == highLimit:
            print("----L==H")
            return 0

        # η = 2K(x1,x2)−K(x1,x1)−K(x2,x2)  η就是 eta K(x,y)指下x,y向量的点乘
        eta = 2.0 * smoTask.K[i, j] - smoTask.K[i, i] - smoTask.K[j, j]
        # 这个为社么 η必须小于 0还没懂...
        if eta >= 0:
            print("----eta>=0")
            return 0

        # a2new = α2old − y2*(Ex1−Ex2)η
        smoTask.Alpha_vec[j] -= smoTask.Y_vec[j] * (errXi - errXj) / eta
        # 修正一下 aj，不能越界
        smoTask.Alpha_vec[j] = clip_alpha(smoTask.Alpha_vec[j], highLimit, lowLimit)
        print("----aj = a{0} = {1}".format(j, float(smoTask.Alpha_vec[j])))
        # 更新误差缓存
        update_errCacheMat(smoTask, j)
        print("++++update aj:a{0} = {1}".format(j, float(smoTask.Alpha_vec[j])))
        if abs(smoTask.Alpha_vec[j] - Alpha_2Old) < 0.00001:
            print("----j not moving enough")
            return 0

        # 更新 ai
        smoTask.Alpha_vec[i] += smoTask.Y_vec[j] * smoTask.Y_vec[i] * (Alpha_2Old - smoTask.Alpha_vec[j])
        update_errCacheMat(smoTask, i)
        print("++++update aj:a{0} = {1}".format(i, float(smoTask.Alpha_vec[i])))

        # 现在求 b
        # b1new = bold−E1−y1(α1new−α1old)K(x1, x1)−y2(α2new−α2old)K(x1, x2)
        # b2new = bold−E2−y1(α1new−α1old)K(x1, x2)−y2(α2new−α2old)K(x2, x2)
        b1 = smoTask.b - errXi - \
             smoTask.Y_vec[i] * (smoTask.Alpha_vec[i] - Alpha_1Old) * smoTask.K[i, i] - \
             smoTask.Y_vec[j] * (smoTask.Alpha_vec[j] - Alpha_2Old) * smoTask.K[i, j]
        b2 = smoTask.b - errXj - \
             smoTask.Y_vec[i] * (smoTask.Alpha_vec[i] - Alpha_1Old) * smoTask.K[i, j] - \
             smoTask.Y_vec[j] * (smoTask.Alpha_vec[j] - Alpha_2Old) * smoTask.K[j, j]
        # 若 0 ≤ α1new ≤ c 取 b1
        # 若 0 ≤ α2new ≤ c 取 b2
        # 其他情况 取 (b1+b2)/2
        if 0 < smoTask.Alpha_vec[i] < smoTask.c:
            smoTask.b = b1
        elif 0 < smoTask.Alpha_vec[j] < smoTask.c:
            smoTask.b = b2
        else:
            smoTask.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0


# 主算法
def SVM_plattSMO(smoTask):
    """
    主算法
    :param smoTask:
    """
    # 记录 遍历 a后所有 a都未改变的次数，而且只要中途有改变，则归零
    innerLoopSum = 0
    entireSet = True
    alphaPairsChanged = 0
    lestLoop = smoTask.lestLoop
    # 累计未变化 maxCyCle次后跳出循环
    while ((innerLoopSum < smoTask.maxCycle) and ((alphaPairsChanged > 0) or entireSet)) or (lestLoop > 0):
        lestLoop -= 1
        alphaPairsChanged = 0
        # 完整循环
        if entireSet:
            for i in range(smoTask.sampleSum):
                alphaPairsChanged += plattSMO_inner_loop(i, smoTask)
                print("【FullSet】 innerLoop:{0},i:{1},pairs changed:{0}".format(innerLoopSum, i, alphaPairsChanged))
            innerLoopSum += 1
        # 非边界循环
        else:
            # nonzero函数的 * 相当于 and
            AlphaNonBoundList = list(nonzero((0 < smoTask.Alpha_vec.A) * (smoTask.Alpha_vec.A < smoTask.c))[0])
            for i in AlphaNonBoundList:
                alphaPairsChanged += plattSMO_inner_loop(i, smoTask)
                print("【Non bound】,innerLoop:{0},i:{1},pairs changed:{0}".format(innerLoopSum, i, alphaPairsChanged))
            innerLoopSum += 1
        # entireSet用于在完整循环和非边界循环间切换，一直到非边界循环没有变化后才切换回完整循环
        if entireSet:
            entireSet = False
        elif alphaPairsChanged == 0:
            entireSet = True
        print("\n***** Inner loop times {0} ******\n".format(innerLoopSum))
    print("\n******PlattSMO finished******")


def plattSMO_fitline(smoTask):
    dataArray = smoTask.A_mat.A
    sampleNum = len(dataArray)
    SVM_plattSMO(smoTask)
    # W = sum(aj * yj * Xj) = （Alpha_vec 数乘 Y_vec）' * A_vec
    W_List = list(array(multiply(smoTask.Alpha_vec, smoTask.Y_vec).T * smoTask.A_mat)[0])
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(sampleNum):
        if int(smoTask.Y_vec[i]) == 1:
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
    y0 = (-float(smoTask.b) - W_List[0] * x) / W_List[1]
    ax.plot(x, y0)
    y1 = (1 - float(smoTask.b) - W_List[0] * x) / W_List[1]
    ax.plot(x, y1, ':')
    y2 = (-1 - float(smoTask.b) - W_List[0] * x) / W_List[1]
    ax.plot(x, y2, ':')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


# test_smoTask = SMO_struct(fileName="testSet.txt")
# plattSMO_fitline(test_smoTask)
