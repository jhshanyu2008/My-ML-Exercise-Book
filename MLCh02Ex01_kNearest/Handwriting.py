"""
机器学习第二章实战：识别手写算法
使用k临近算法

输入：新采集的手写数据
输出：由算法推断的数字

@原作者：pbharrin
@注释修改：shanyu
"""

from kNN import *
import os

classify_k = 3


def img2vector(filename):
    """
    读取处理过的图像文本，转化为向量
    （把一个32x32的矩阵信息存进1x1024的行向量中）
    :param filename:
    :return return_vect:
    """
    # 新建一个1x1024的零向量
    return_vect = zeros((1, 1024))
    fr = open(filename)
    # 依次读取各行各位的数据，存入向量中
    for i in range(32):
        line_str = fr.readline()
        for j in range(32):
            return_vect[0, 32*i+j] = int(line_str[j])
    return return_vect


def handwriting_test(k):
    """
    读取测试集的手写样本数据
    并用测试集预测它的标签
    检测使用k临近算法的正确率
    """
    global classify_k
    classify_k = k
    # 存储测试集的标签
    handw_labels = []
    # 存储训练集所在文件夹的文件信息
    training_file_list = os.listdir('digits/trainingDigits')
    # 存储训练集文件数量
    m_training = len(training_file_list)
    # 这次新建一个存储所有样本属性的矩阵
    training_mat = zeros((m_training, 1024))
    # 读取所有训练集数据
    for i in range(m_training):
        # 读取文件名
        file_name_str = training_file_list[i]
        # 去掉文件名的后缀
        file_str = file_name_str.split('.')[0]
        # 从文件名中读出该训练样本的标签
        class_num_str = int(file_str.split('_')[0])
        handw_labels.append(class_num_str)
        # 读取该训练样本的数据
        training_mat[i, :] = img2vector('digits/trainingDigits/{0}'.format(file_name_str))

    error_mat = {}
    error_count = 0
    # 读取所有测试集数据，进行验证
    test_file_list = os.listdir('digits/testDigits')
    m_test = len(test_file_list)
    for i in range(m_test):
        file_name_str = test_file_list[i]
        file_str = file_name_str.split('.')[0]
        class_num_str = int(file_str.split('_')[0])
        vector_test = img2vector('digits/testDigits/{0}'.format(file_name_str))
        """
        调用classify0函数，使用k临近算法判断测试样本属性
        """
        classifier_result = classify0(vector_test,
                                      training_mat,
                                      handw_labels,
                                      classify_k)
        # 在窗口输出测试信息
        print("The classifier came back with: {0},the real answer is {1}".\
              format(classifier_result, class_num_str))
        if classifier_result != class_num_str:
            error_mat['{0}'.format(file_name_str)] = '{0}'.format(classifier_result)
            error_count += 1

    print("\nHere are the list of wrong-classified items:")
    for file_name, misclassified_label in error_mat.items():
        print("{0} was misclassified as {1}.".format(file_name, misclassified_label))
    print("\nThe total number of error is : {0}".format(error_count))
    print("\nThe total error rate is : {0}%".format(100*error_count/float(m_test)))



