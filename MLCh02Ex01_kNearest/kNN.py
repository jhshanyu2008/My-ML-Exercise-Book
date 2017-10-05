"""
机器学习与实战第二章实例。
kNN：k临近算法

输入：in_x：用于分类的输入向量（1xN的矩阵）
     data_set：训练样本集（MxN的矩阵）
     labels：数据集的标签（1xM向量）
     k：用于比较的临近数据个数（应该是个奇数）

输出：最可能的类别标签

思路：采集数据，建立属性和标签矩阵,属性矩阵需要首先归一化
     输入测试样本，计算它和所有训练集的欧式距离
     距离从小到大排序后选取k个开始投票
     投票最多者确认为测试样本的标签

@原作者：pbharrin
@注释修改：shanyu
"""

# import matplotlib.pyplot as plt
# 科学计算包
from numpy import *
import operator

#两个classify0的参数
classify0_k = 3


def create_data_set_test():
    """
    函数作用：构建一组训练数据（训练样本），共4个样本 
    同时给出了这4个样本的标签，及labels 
    test_group = array([  
                [1.0, 1.1],  
                [1.0, 1.0],  
                [0. , 0. ],  
                [0. , 0.1]  
    ])  
    """
    test_group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    test_labels = ['A', 'A', 'B', 'B']
    return test_group, test_labels


def classify0(in_x, data_set, label, k):
    """
    k-紧邻算法
    """
    # 数据集的长度（第1维是子集的数量即行数M）
    data_set_size = data_set.shape[0]

    """
    inX只是个1xN的矩阵，要把它变成MxN型才能和数据集操作
    比如inX = [0,1],data_set就用函数返回的结果，那么
    tile(in_x, (3,1))= [[ 0, 1],
                        [ 0, 1],
                        [ 0, 1]]
    和数据集作差之后得到diffMat
    diff_mat 即完成了分类的输入和所有测试样本各属性的作差"""
    diff_mat = tile(in_x, (data_set_size, 1)) - data_set
    # 这些差再作平方
    sq_diff_mat = diff_mat ** 2
    # 再把每行全加起来，axis=1表示按照横轴
    sq_distances = sq_diff_mat.sum(axis=1)
    # 最后开个根号
    distance = sq_distances ** 0.5
    """
    总结上面的过程就是求测试样本和训练集各个成员的范数，也称欧式距离
    """

    # distance从小到排序，记录排序后各元素排序前的序号成sortedDistIndices
    sorted_dist_indices = distance.argsort()
    # 元组列表，存放最终的分类结果及相应的结果投票数
    class_count = {}
    # 选择距离最小的k个点，统计各个标签的票数
    for i in range(k):
        vote_label = label[sorted_dist_indices[i]]
        # 返回classCount中相应标签的票数，如果不存在就返回0并新建该元素，最后加上一票
        class_count[vote_label] = class_count.get(vote_label, 0) + 1
    # 对得到的classCount排序(所有元素，第1维的数据即票数为依据，反向排序即从高到低)
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def file2matrix(filename):
    """
    从文本读取数据
    """
    # 打开文件，类似于c#数据流
    file_source = open(filename)
    # 读取每一行的数据，存成矩阵
    array_file_lines = file_source.readlines()
    # 读取arrayFileLines的元素个数，即文本的行数
    number_of_lines = len(array_file_lines)
    # 读取arrayFileLines第一个元素的子元素个数，即文本的列数
    array_file_column = len(array_file_lines[0].split())
    # 新建一个和文本元素数量相同的矩阵
    return__mat = zeros((number_of_lines, array_file_column - 1))
    class_label_vector = []
    index = 0
    for line in array_file_lines:
        # 去掉前后的指定字符
        line = line.strip()
        # 存取每行的各个元素
        list_form_line = line.split()
        # 把属性数据存入returnMat
        return__mat[index][:] = list_form_line[0:-1]
        # 把标签值存入classLabelVector
        class_label_vector.append(int(list_form_line[-1]))
        index = index + 1
    return return__mat, class_label_vector


def auto_norm(data_set):
    """
    归一化特征值：(old_value - min_value) / (max_value - min_value)
    把输入的特征值转化成小区间的值
    """
    # min,max(0)是挑选出每一列中的最小值组成一行
    # min,max(1)是挑选出每一行中的最小值组成一行
    min_vals = data_set.min(0)
    max_vals = data_set.max(0)
    # max_value - min_value
    ranges = max_vals - min_vals
    # 获取原data_set的行数
    m = data_set.shape[0]
    # old_value - min_value
    norm_data_set = data_set - tile(min_vals, (m, 1))
    # 再除以range,这里是数字除,矩阵除使用linalg.solve(matA,matB)
    norm_data_set = norm_data_set / tile(ranges, (m, 1))
    return norm_data_set, ranges, min_vals


def dating_class_test(ho_radio, k):
    """
    验证经过k临近算法处理后的标签和原始标签的误差率
    ho_radio训练集分段的系数（0.1即10%测试，90%训练）
    """
    global classify0_k
    # 更新全局参数
    classify0_k = k
    # 读取训练集的属性和标签
    dating_data_mat, dating_labels = file2matrix('datingTestSet.txt')
    # 把训练集的属性作归一化处理
    norm_mat, ranges, min_vals = auto_norm(dating_data_mat)
    # 读取训练集的行数
    m = norm_mat.shape[0]
    # 把数据集分段
    num_test_vecs = int(m * ho_radio)
    # 用于记录错误
    error_count = 0.0
    for i in range(num_test_vecs):
        # 测试集是从第1到第m*num_test_vec行，即10%的数据用于测试
        # 训练集是从第m*num_test_vecs行到第m行，即90%的数据用于训练
        classifier_result = classify0(norm_mat[i, :],
                                      norm_mat[num_test_vecs:m, :],
                                      dating_labels[num_test_vecs:m], k)
        print("The classifier came back with :{0}, the real answer is {1}" \
              .format(classifier_result, dating_labels[i]))
        if classifier_result != dating_labels[i]:
            error_count += 1
    print("The total error rate is : {0}%".format(100 * error_count / float(num_test_vecs)))


def classify_person():
    """
    应用分类
    """
    global classify0_k
    result_list = ['not at all', 'in small doses', 'in large doses']
    ff_miles = float(input("frequent flier miles earned per year?"))
    percentage_video = float(input("percentage of time spent playing computer game?"))
    ice_cream = float(input("liters of ice cream consumed per year?"))
    dating_data_mat, dating_labels = file2matrix('datingTestSet.txt')
    norm_mat, ranges, min_vals = auto_norm(dating_data_mat)
    input_data = array([ff_miles, percentage_video, ice_cream])
    norm_input = (input_data - min_vals)/ranges
    classify_result = classify0(norm_input, norm_mat, dating_labels, classify0_k)
    print("You will probably like this person:{0}".format(result_list[classify_result-1]))


# fig = plt.figure()
# ax = fig.add_subplotperson()
