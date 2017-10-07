"""
本例将制作一个快速的关键词过滤器，这里存放和数据操作有关的各种操作
"""
from NaiveBayes import *


# 这是一个测试用的训练原始样本
def test_dataSet():
    """
    生成测试样本:
    trainList记录了一个6个样本的原始数据
    trainCategory则指示这6个样本的标签('1'为"侮辱",'0'为"非侮辱")
    :return: 
    """
    trainList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    trainCategory = [0, 1, 0, 1, 0, 1]
    return trainList, trainCategory


# 创建一个包含所有文本的不重复词的列表
def create_vocabList(dataSet):
    """
    使用集合的特性（成员不重复）生成全词汇列表
    :param dataSet:
    :return: list()
    """
    # 先建立一个空集合
    vocabSet = set([])
    # 再把文档（按行分成多个列表）转化成集合取交集
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


# 检查输入的文档中是否出现词列表中的词汇
# 这里只分析单个词汇是否出现，不关心次数
def set_of_wordsVec(vocabList, inputSet):
    """
    检查输入文档中的所有词汇是否存在于词汇列表中
    存在就在词汇列表的标识向量中把相应位置置 1，最后返回该向量
    这个函数把原始数据整理成了可以使用的训练数据
    :param vocabList:
    :param inputSet:
    :return returnVec:
    """
    # 先创建一个长度和词汇列表相同的一维全 0列表，
    # 【注意】这种初始化方法只适用于一维列表，二维及以上绝不能使用
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        # if word in vocabList 这样写也可以，虽然我不习惯
        if vocabList.__contains__(word):
            # 如果出现，相应标识位置 1
            returnVec[vocabList.index(word)] = 1
        else:
            print("The word {0} is not in my Vocabulary".format(word))
    return returnVec


# 检查输入的文档中是否出现词列表中的词汇
# 这里在分析词汇是否出现的同时还关注出现的次数，相当于为每个词添加了权重
def bag_of_wordsVec(vocabList, inputSet):
    """
    内容和set_of_wordsVec(vocabList, inputSet)基本相同，除了置一处换成了累加
    :param vocabList:
    :param inputSet:
    :return returnVec:
    """
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        # if word in vocabList 这样写也可以，虽然我不习惯
        if vocabList.__contains__(word):
            # 如果出现，相应标识位置 1
            returnVec[vocabList.index(word)] = +1
        else:
            print("The word {0} is not in my Vocabulary".format(word))
    return returnVec


# 把原始训练集和测试集转换成数矩阵
def get_data_matrix(testList, trainList, model='set'):
    """
    把原始数据转换成可用的训练集和测试集，输出全是 0、1的向量集合
    :param model:
    :param testList:
    :param trainList:
    :return testMatrix, trainMatrix:
    """
    print("The vocablist is based on {0} model.".format(model))
    modelDic = {'set': set_of_wordsVec,
                'bag': bag_of_wordsVec}
    transferModel = modelDic[model]

    my_vocabList = create_vocabList(trainList)
    trainMatrix = []
    testMatrix = []
    for trainDoc in trainList:
        trainMatrix.append(transferModel(my_vocabList, trainDoc))
    for testDoc in testList:
        testMatrix.append(transferModel(my_vocabList, testDoc))
    return testMatrix, trainMatrix


# 调用 NaiveBayes对测试集分类
def test_naive_bayes(testMatrix, trainMatrix, trainCategory):
    """
    使用贝叶斯分类，返回分类结果向量
    :param testMatrix:
    :param trainMatrix:
    :param trainCategory:
    :return classify_return:
    """
    Pln_wiOfNormal, Pln_wiOfAbuse, P_abusive = get_train_prob(trainMatrix, trainCategory)
    classify_result = []
    for test_vec in testMatrix:
        classify_result.append(classify_naive_bayes(test_vec, Pln_wiOfNormal, Pln_wiOfAbuse, P_abusive))
    return classify_result


# 主分类函数
def classify_main(testList, trainList, trainCategory, model='set'):
    """
    # 主函数，打印分类结果
    :param model:
    :param testList:
    :param trainList: 
    :param trainCategory: 
    """
    testMatrix, trainMatrix = get_data_matrix(testList, trainList, model)
    classify_result = test_naive_bayes(testMatrix, trainMatrix, trainCategory)
    return classify_result


# 生成各种测试数据
test_list = [['love', 'my', 'dalmation'],
             ['stupid', 'garbage'],
             ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
             ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
             ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
             ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
             ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him']]
# train_list, train_category = test_dataSet()
# test_vocabList = create_vocabList(train_list)
# test_matrix, train_matrix = get_data_matrix(test_list, train_list, model='set')
# get_train_prob(train_matrix, train_category)
# classify_main(test_list, train_list, train_category, model='set')
