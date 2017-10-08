"""
测试邮件分类
"""
import DataProcess
import os
import random


def get_email_data(testRadio):
    """
    trainRadio 训练集和测试集的分段，0.1即 10%测试，其他训练
    ham分类为 0，spam分类为 1
    :param testRadio:
    :return testList, testCategory, trainList, trainCategory:
    """
    # 读取两个文件中的文件列表,文件名按照 windows的排序来有利于找文件
    hamFileList = sorted(os.listdir('email/ham'), key=lambda filename: int(filename.split('.')[0]))
    spamFileList = sorted(os.listdir('email/spam'), key=lambda filename: int(filename.split('.')[0]))
    # 根据分段百分比，从各自文件夹中随机抽取
    hamFileLen = len(hamFileList)
    spamFileLen = len(spamFileList)
    hamFileTestLen = round(testRadio * hamFileLen)
    spamFileTestLen = round(testRadio * spamFileLen)
    # 初始化ham的训练和测试集
    hamTrainList = []
    hamTestList = []
    hamTestFileList = []
    hamTestCategory = [0] * hamFileTestLen
    hamTrainCategory = [0] * (hamFileLen - hamFileTestLen)
    # 按照比例随机选取训练集和测试集，首先先生成随机数
    hamNumList = sorted(random.sample(range(0, hamFileLen), hamFileTestLen))
    for i in range(hamFileLen):
        if hamNumList.__contains__(i):
            hamTestFileList.append('email/ham/{0}'.format(hamFileList[i]))
            hamTestList.append(DataProcess.parse_to_wordList(open('email/ham/{0}'.format(hamFileList[i])).read()))
        else:
            hamTrainList.append(DataProcess.parse_to_wordList(open('email/ham/{0}'.format(hamFileList[i])).read()))
    # 同理，对spam也作相应处理
    spamTrainList = []
    spamTestList = []
    spamTestFileList = []
    spamTestCategory = [1] * spamFileTestLen
    spamTrainCategory = [1] * (spamFileLen - spamFileTestLen)
    spamNumList = sorted(random.sample(range(0, spamFileLen), spamFileTestLen))
    for i in range(spamFileLen):
        if spamNumList.__contains__(i):
            spamTestFileList.append('email/spam/{0}'.format(spamFileList[i]))
            spamTestList.append(DataProcess.parse_to_wordList(open('email/spam/{0}'.format(spamFileList[i])).read()))
        else:
            spamTrainList.append(DataProcess.parse_to_wordList(open('email/spam/{0}'.format(spamFileList[i])).read()))

    # 构建完整的训练集和测试集
    testFileList = hamTestFileList + spamTestFileList
    testList = hamTestList + spamTestList
    testCategory = hamTestCategory + spamTestCategory
    trainList = hamTrainList + spamTrainList
    trainCategory = hamTrainCategory + spamTrainCategory

    return testFileList, testList, testCategory, trainList, trainCategory


# 测试主函数
def test_email_filter(testRadio):
    """
    主函数，显示分类结果，打印错误率
    :param testRadio:
    """
    testFileList, testList, testCategory, trainList, trainCategory = get_email_data(testRadio)
    classifyResult = DataProcess.classify_main(testList, trainList, trainCategory, model='set')
    errorCount = 0
    # 用文本显示分类，而不是'0'、'1'
    classifyClass = {0: 'ham',
                     1: 'spam'}
    for i in range(len(testCategory)):
        print("The {0} is classified as {1}".format(testFileList[i], classifyClass[classifyResult[i]]))
        if classifyResult[i] != testCategory[i]:
            errorCount += 1
    errorRate = float(errorCount / len(testCategory))
    # 打印错误率
    print("The error rate of this test is {}%".format(errorRate * 100))
