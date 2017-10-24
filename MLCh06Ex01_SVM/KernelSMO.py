"""
今天来学习核转换函数
思路其实不麻烦：
把非线性的超平面点集映射到更高维度上以实现点集合的线性化
映射的方法不经相同，但理论上一定能找到，我们计算时不需要知道究竟怎么找的。
因为在求解 aj和 b时我们都是使用内积的方式 k(ai,aj)计算出来的，这事实上是个黑盒子，如果我们不关注内积的计算过程，
只要知道我们通过一个算法得到的新内积结果 C2是个由高维映射内积得到的，那直接拿来用就好了。
这个算法的选取就很迷了...数学原理我也看不懂
总之常用的黑匣子是“径向基函数”，数学上称为“核函数”，就是机器学习书上的那个公式：
k(X1,X2) = exp(-||X1-X2||^2 / (2*σ^2))
这个 σ是核函数的关键，σ越小，映射的维度越大，甚至于可以到无穷维。反之，维度就越小。
从实验结果看，不论是过高维度还是过低维度结果都不好，每个数据集都有一个适合它的维度，至于怎么找，这就是机器学习的又一个坑了(没人知道)

代码都基本是重复的，所以除了核函数的计算会单独写，其他全部引用。
"""
from PlattSMO import *


class kernelSMO_struct:
    def __init__(self, dataMatrix, labelList, c=200, faultToler=0.001, maxCycle=100, kType=('kernel', 1.3)):
        self.dataMatrix = dataMatrix
        self.labelList = labelList
        self.c = c
        self.faultToler = faultToler
        self.maxCycle = maxCycle
        self.A_mat = mat(self.dataMatrix)
        self.Y_vec = mat(labelList).transpose()
        self.sampleSum = shape(self.A_mat)[0]
        self.Alpha_vec = mat(zeros((self.sampleSum, 1)))
        self.b = 0
        self.lestLoop = 7
        self.randAjSum = 5
        self.kType = kType
        # 误差缓存，第一列是是否有效的标志位，第二列是实际的 E值
        self.ErrCache_mat = mat(zeros((self.sampleSum, 2)))
        self.K = mat(zeros((self.sampleSum, self.sampleSum)))
        for i in range(self.sampleSum):
            # 每个样本 Xi 和其他所有样本的内积写成 K的第 i列，k是一个方阵
            self.K[:, i] = kernelTrans(self.A_mat, self.A_mat[i, :], kType)


# 核函数辅助计算
def kernelTrans(A_mat, A, kType):
    """
    核函数计算辅助式，kType[0]指定类型，kType[1]指定 σ
    这里一次处理一整列的 K
    :param A_mat:
    :param A:
    :param kType:
    :return: K_vec
    """
    sampleSum, featSum = shape(A_mat)
    # 给内积预留的列向量
    K_vec = mat(zeros((sampleSum, 1)))
    # 不使用核函数时
    if kType[0] == 'linear':
        K_vec = A_mat * A.T
    # 使用核函数
    elif kType[0] == 'kernel':
        for j in range(sampleSum):
            # X1 - X2
            deltaRow = A_mat[j, :] - A
            # ||X1-X2||^2
            K_vec[j] = deltaRow * deltaRow.T
        # exp(-||X1-X2||^2 / (2*σ^2))
        # 【注意】区别于 matlab中的矩阵求逆，这里的除法是数除
        K_vec = exp(K_vec / (-1 * kType[1] ** 2))
    else:
        raise NameError('The Kernel is not recognized')
    return K_vec


# testSetRBF有两个数据集 我决定把它们写到一起
def load_total_dataset(file1st='', file2rd=''):
    data1st, lab1st = load_dataset(file1st)
    data2rd, lab2rd = load_dataset(file2rd)
    dataMatrix = data1st + data2rd
    labelVec = lab1st + lab2rd
    return dataMatrix, labelVec


# 核函数的主函数，鉴于样本太少，我不分训练测试了，一起当训练集用算了
def test_kernel_SMO(smoTask):
    """
    直接使用 PlattSMO中的类对象作输入
    生成拟合线，计算错误率
    :param smoTask:
    :return:
    """
    # 使用 plattSMO
    SVM_plattSMO(smoTask)

    # 找到不是 0的 a值
    svIndex = nonzero(smoTask.Alpha_vec.A > 0)[0]
    # 获取支持向量，a不是零的全是支持向量
    sVs = smoTask.A_mat[svIndex]
    labelSV = smoTask.Y_vec[svIndex]
    print("\nThere are {0} Support Vectors".format(shape(sVs)[0]))

    errorCount = 0
    for i in range(smoTask.sampleSum):
        # 只使用支持向量来计算，减小计算量
        # 原本是这样子的：
        # W = sum(ai * yi * Xi) = A' * (Alpha_vec 数乘 Y_vec)
        # predict = f(Xi) = A[i,:] * W + b = A[i,:] * A' * (Alpha_vec 数乘 Y_vec) + b
        # predict = kernel[:,i]' * (Alpha_vec 数乘 Y_vec) + b 当然这里的 kernel使用核函数计算不用内积
        kernelEval = kernelTrans(sVs, smoTask.A_mat[i, :], smoTask.kType)
        predict = kernelEval.T * multiply(labelSV, smoTask.Alpha_vec[svIndex]) + smoTask.b
        # sign函数使 0为 0、负数为 -1、正数为 +1
        if sign(predict) != sign(smoTask.labelList[i]):
            errorCount += 1
    print("\nThe training error rate is: {0}%".format(float(errorCount * 100) / smoTask.sampleSum))


data_matrix, label_list = load_total_dataset('testSetRBF.txt', 'testSetRBF2.txt')
# c=200极其重要，σ相对来说越小效率越高
# 这两个参数选的蛮不错的，就是速度慢了点
test_smoTask = kernelSMO_struct(data_matrix, label_list, c=200, kType=('kernel', 0.1))
test_kernel_SMO(test_smoTask)
