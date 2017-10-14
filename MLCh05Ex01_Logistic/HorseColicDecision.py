"""
利用 Logistic回归分析疝气病马的死亡率
"""

from Logistic import *

train_file = "horseColicTraining.txt"
test_file = "horseColicTest.txt"


def data_classcify(W_list, testVec):
    """
    计算 z值，大于 0是类别 1，小于 0是类别 0
    书中使用 sigmoid函数，我觉的这步有点多余
    :param W_list: 
    :param testVec: 
    :return classcifyResult:0 or 1: 
    """
    z = sum(array(W_list) * array(testVec))
    if z > 0:
        return 1
    else:
        return 0


def horse_colic_Test(trainFile, testFile):
    """
    20多维的数据，图没法画了
    实验证明还是原始的 Logistic梯度下降法最靠谱
    :param trainFile:
    :param testFile:
    :return errorRate:
    """
    # 先读取训练集的数据
    trainDataMatrix, trainLabelVec = read_dataset(trainFile)
    # 开始训练
    W_list = gradient_descent(trainDataMatrix, trainLabelVec, maxCycles=10000)
    # W_list = random_gradient_descent(trainDataMatrix, trainLabelVec)
    # W_list = random_gradient_descent_ver2(trainDataMatrix, trainLabelVec, maxCycles=1000)

    # 再读取测试集的数据
    testDataMatrix, testLabelVec = read_dataset(testFile)
    # 开始校验
    errorNum = 0
    for i in range(len(testLabelVec)):
        classcifyResult = data_classcify(W_list, testDataMatrix[i])
        if classcifyResult != testLabelVec[i]:
            errorNum += 1
    errorRate = errorNum / len(testLabelVec)
    return errorRate


def horse_colic_multitest(trainFile, testFile, testTimes=5):
    """
    多次测试，计算平均错误率
    :param testTimes:
    :param trainFile:
    :param testFile:
    """
    errorRateSum = 0.0
    for i in range(testTimes):
        errorRateSum += horse_colic_Test(trainFile, testFile)
    print("\nThe average error rate of this test is {0}%".format(100 * errorRateSum / testTimes))


horse_colic_multitest(train_file, test_file, testTimes=5)

# 书中留的习题就不写了，把 hose_colic_Test函数拆开，测试的部分剥离开来就好了。
