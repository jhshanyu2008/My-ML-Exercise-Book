"""
朴素贝叶斯：
假设样本的所有属性同等重要，本例中使用基于贝努力模型的实现方法：
   并不考虑词在文档中出现的次数，只考虑是否出现，相当于假设词是等权重的。

贝叶斯的基本公式：
P(ci|w) = (P(w|ci)*P(ci))/P(w) 这里的 w是一个向量
其中 P(w|ci):把 w展开 P(w|ci)写作 P(w0,w1,w2...wN|ci)
            这里假设所有的词都是独立的话有: P(w0,w1,w2...wN|ci) = P(wo|ci)*P(w1|ci)*...*P(wN|ci)

由于 P(w)对于任何分类都是一样的，所以比较 P(ci|w)相当于比较 P(ci|w)*P(w) = P(w|ci)*P(c)
即贝叶斯判定准则：h(w) = max(P(ci)P(w|ci))
【注意】由于可能存在 P(wi|ci) = 0的情况，所以计算时 P(wi|ci)的分子 +1，分母 +2,这称为"拉普拉斯修正"
【注意】由于 P(w|ci)数值会很小，所以采用对数计算 ln(P(w|ci)) = ln(P(w1|ci)) + ln(P(w2|ci))+...+ln(P(wN|ci))
       贝叶斯判别式变成：H(w) = ln(P(wi|ci)) + ln(P(ci))

"""
from numpy import *


# 获取 ln(P(wi|ci)) 和 P(abusive)
def get_train_prob(trainMatrix, trainCategory):
    """
    train_matrix记录了所有训练数据，已事先处理成 0，1的向量形式
       处理方式参见函数set_of_wordsVec(vocabList, inputSet) 每一行都来自一个样本
    train_category则标记了这些样本的对应标签('1'为"侮辱",'0'为"非侮辱")
    P1指示侮辱性文档，P0指示非侮辱性，出于计算考虑，全部用向量批量计算
    :param trainMatrix:
    :param trainCategory:
    :return Pln_wiOfNormal 即 ln(P(wi|0)), Pln_wiOfAbuse 即 ln(P(wi|1)0, P_abusive 即 P(1):
    """
    num_trainDocs = len(trainMatrix)
    num_words = len(trainMatrix[0])
    # 计算任意文档属于侮辱性文档的概率 P(abusive) = 侮辱性文档总数/总样本数,即概率 P(1)
    P_abusive = sum(trainCategory) / float(num_trainDocs)
    # 记录各个词汇在各自类别中出现总次数的两个向量
    # 【注意】为了不使某一个词汇概率为 0，用 1作初始化
    # 【注意】zeros()、ones()都是函数，绝对不可以用连等初始化，否则会引用同一个内存区的数据
    P0Vec_num = ones(num_words)
    P1Vec_num = ones(num_words)
    # 记录两个类别各自的词汇总数
    # 【注意】同样为了不使某一词汇概率为 0，用 2初始化
    # 【注意】变量赋值时是可以用连等的 P0_Denom = P1_Denom = 2 但我不推荐，一不小心函数也连等了那就开心了
    P0_Denom = 2
    P1_Denom = 2
    for i in range(num_trainDocs):
        if trainCategory[i] == 1:
            P1Vec_num += trainMatrix[i]
            P1_Denom += sum(trainMatrix[i])
        else:
            P0Vec_num += trainMatrix[i]
            P0_Denom += sum(trainMatrix[i])
    # 得到的两个向量即每个词占各自类别总词汇的比率，即概率 P(wi|ci) 分类ci："1(侮辱)"和"0(非侮辱)"
    # 【注意】后面计算 P(w|ci)时很多概率相乘数字可能会极小，因此换用对数 ln(P(wi|ci))来处理
    Pln_wiOfAbuse = log(P1Vec_num / P1_Denom)
    Pln_wiOfNormal = log(P0Vec_num / P0_Denom)
    return Pln_wiOfNormal, Pln_wiOfAbuse, P_abusive


# 贝叶斯分类
def classify_naive_bayes(test_vec, Pln_wiOfNormal, Pln_wiOfAbuse, P_abusive):
    """
    贝叶斯判别式：H(w) = ln(P(wi|ci)) + ln(P(ci))
    :return:
    """
    P_normal = 1 - P_abusive
    P_1 = sum(test_vec * Pln_wiOfAbuse) + log(P_abusive)
    P_0 = sum(test_vec * Pln_wiOfNormal) + log(P_normal)
    if P_1 > P_0:
        return 1
    else:
        return 0
