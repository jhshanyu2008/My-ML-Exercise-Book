"""
使用 KernelSMO处理第二章的 kNN手写识别
由于使二分类，我们只选 0和 9的
"""
from KernelSMO import *
import os


# 先把处理文本的几个函数复制过来
# 为了配合SVM的函数，我全部列表操作了

# 读取处理过的图像文本，转化为向量
def turn_into_list(filename):
    """
    读取处理过的图像文本，转化为向量
    （把一个32x32的矩阵信息存进1x1024的列表中）
    :param filename:
    :return returnList:
    """
    # 新建一个1x1024的零向量
    returnList = []
    file = open(filename)
    # 依次读取各行各位的数据，转换成浮点数存入列表中
    for fileLine in file.readlines():
        for j in range(len(fileLine[0:-1])):
            returnList.append(float(fileLine[j]))
    return returnList


# 生成特征二维列表和标签列表
def create_dataset(folderAddr=''):
    # 载入样本文件列表
    dataFileList = os.listdir(folderAddr)
    # 样本数量
    sampleSum = len(dataFileList)
    # 这次新建特征二维列表和标签列表
    dataMatrix = []
    labelList = []
    # 读取所有训练集数据
    for i in range(sampleSum):
        # 读取文件名
        fileName = dataFileList[i]
        # 从文件名中读出该训练样本的标签,先去掉后缀，再读取标签
        label = float((fileName.split('.')[0]).split('_')[0])
        # 只要 1-9的
        if label == 0:
            labelList.append(-1.0)
        elif label == 9.0:
            labelList.append(1.0)
        else:
            continue
        # 读取该训练样本的数据
        dataMatrix.append(turn_into_list('{0}/{1}'.format(folderAddr, fileName)))
        print("loading Data......{0}/{1}".format(folderAddr, fileName))
    return dataMatrix, labelList


# 我在 KernelSMO中没有写过测试集的函数，这里得写一个
def test_kernelSMO(trainedTask, testTask):
    # 找到不是 0的 a值
    Alpha_vec = trainedTask.Alpha_vec
    b = trainedTask.b
    kType = trainedTask.kType
    # 获取支持向量，a不是零的全是支持向量
    svIndex = nonzero(Alpha_vec.A > 0)[0]
    sVs = trainedTask.A_mat[svIndex]
    labelSV = trainedTask.Y_vec[svIndex]
    errorCount = 0
    for i in range(testTask.sampleSum):
        # predict =  A[i,:](测试) * W(训练) + b(训练) = A[i,:](测试) * A' * (Alpha_vec 数乘 Y_vec) + b 其他全训练
        kernelEval = kernelTrans(sVs, testTask.A_mat[i, :], kType)
        predict = kernelEval.T * multiply(labelSV, Alpha_vec[svIndex]) + b
        # sign函数使 0为 0、负数为 -1、正数为 +1
        if sign(predict) != sign(testTask.labelList[i]):
            errorCount += 1
    print("\nThe test error rate is: {0}%".format(float(errorCount * 100) / testTask.sampleSum))


# 开始操作，先生成各个训练和测试集
train_matrix, train_label = create_dataset(folderAddr="digits/trainingDigits")
test_matrix, test_label = create_dataset(folderAddr="digits/testDigits")
# 再创建两个训练对象
train_smoTask = kernelSMO_struct(train_matrix, train_label, c=200, kType=('kernel', 10))
test_smoTask = kernelSMO_struct(test_matrix, test_label)
# 开始测试，先是训练集
test_kernel_SMO(train_smoTask)
# 接着是测试集
test_kernelSMO(train_smoTask, test_smoTask)
