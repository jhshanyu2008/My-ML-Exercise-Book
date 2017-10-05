"""
机器学习实战第二章实例：决策树

输入：
输出：

决策树的一般流程：
(1)收集数据：可以使用任何方法。
(2)准备数据：树构造算法只适用于标称型数据，因此数值型数据必须离散化。
(3)分析数据：可以使用任何方法，构造树完成之后，我们应该检查图形是否符合预期。
(4)训练算法：构造树的数据结构。
(5)测试算法：使用经验树计算错误率。
(6)使用算法：可以适用于任何监督学习算法，用以更好地理解数据的内在含义。

使用ID3算法划分数据集，原则是：将无序的数据变得更加有序。
如果待分类的事务可能划分在多个分类之中，定义符号l(xi)为：
l(Xi) = -log2(P(xi)) P(xi)是选择该分类的概率。
为了计算香农熵，我们需要计算所有类别所有可能值包含的信息期望值，通过下面的公式得到：
H = -sum[i=1 to n](P(xi)*log2(P(xi)))
得到香农熵后就可按最大信息增益法划分数据集。

最大信息增益法：
选取第 n 个属性，再选取它的一个属性值，提取值是这个的样本并去掉这个属性列组成新数据集
计算这个数据集的香农值记为 H（j），样本出现这个值的概率是 P(j),则该属性的香农熵为：
H(n) = sum(P(j)*H(j))
信息增益为：H - H(n) 遍历 n个属性，选取最大那个。


@原作者：pbharrin
@注释修改：shanyu
"""
from math import *
import operator
import pickle


# 生成测试数据集
def create_test_dataset():
    """
    用于创建自检参数
    :return: dataset, labels
    """
    dataset = [[1, 1, 'yes'],
               [1, 2, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    feature_labels = ['no surfacing', 'flippers']
    return dataset, feature_labels


# 读取外部用数据集
def read_dataset(filename):
    """
    读取外部数据集
    :return:
    """
    # 打开文件，类似于c#数据流
    file_source = open(filename)
    # 读取每一行的数据，存成矩阵
    feature_labels = file_source.readline().split()
    raw_dataset = file_source.readlines()[1:]
    dataset = []
    for line in raw_dataset:
        # 存取每行的各个元素
        dataset.append(line.strip().split())
    return dataset, feature_labels


# 获取数据集的香农熵
def calc_shannon_ent(data_set):
    """
    计算数据集的香农熵
    :param data_set:
    :return shannon_ent:
    """
    # 得到数据集的样本总数
    num_sample = len(data_set)
    # 新建一个存储标签的字典，存储数据集最后列的标签及出现次数
    label_counts = {}
    for feat_vec in data_set:
        # 提取每个样本的最后一个元素，即该样本的标签
        current_label = feat_vec[-1]
        # 判断字典label_counts是否存储该元素，不存在就新建
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        # 当然存在就键值加一
        label_counts[current_label] += 1
    shannon_ent = 0.0
    for key in label_counts:
        # 计算各个P(xi)
        prob = float(label_counts[key]) / num_sample
        # 计算l(xi) 即l(Xi) = -log2(P(xi))
        # 最后计算香农熵，即H = -sum[i=1 to n](P(xi)*log2(P(xi)))
        shannon_ent -= prob * log(prob, 2)
    return shannon_ent


# 按具体的特征划分并获取精简数据集
def split_dataset(data_set, axis, value):
    """
    划分data_set数据集,选取特征的坐标 axis，该特征的指定值 value
    最后得到data_set[axis]=value的所有向量组成的精简（去掉data_set[axis]列）矩阵
    :param data_set:
    :param axis:
    :param value:
    :return: ret_dataset
    """
    ret_dataset = []
    for feat_vec in data_set:
        # 判断是否和指定值相同
        if feat_vec[axis] == value:
            # 去除了feat_vec[axis]处的数据组成新向量
            reduced_feat_vec = feat_vec[:axis]
            reduced_feat_vec.extend(feat_vec[axis + 1:])
            # 把简化后的向量添加进返回列表
            ret_dataset.append(reduced_feat_vec)
    return ret_dataset


# 根据香农熵选择最好的划分方式
def choose_best_feature_split(data_set):
    """
    根据香农熵选择最好的划分属性
    :param data_set:
    :return best_feature:
    """
    # 获取特征数量（数据集最后一个元素是标签）
    num_features = len(data_set[0]) - 1
    # 获取数据集的总体香农熵
    base_entropy = calc_shannon_ent(data_set)
    best_InfoGain = 0.0
    best_feature = -1
    for i in range(num_features):
        # 提取data_set第i列的所有数据组成feature_list
        feature_list = [example[i] for example in data_set]
        # 把向量变成集合,所有同名变量全部合并为一个元素
        feature_value_set = set(feature_list)
        new_entropy = 0.0
        for value in feature_value_set:
            # 获取i位置值为value的精简矩阵
            sub_dataset = split_dataset(data_set, i, value)
            prob = len(sub_dataset) / float(len(data_set))
            """
            value[j]的香农熵：
            Hj = -sum[i=1 to n](P(xi)*log2(P(xi)))
            prob即P(j) = 属性为value的样本占总样本的概率
            data_set[i]的香农熵即把所有value[j]的香农熵加权求和：
            H = sum(P(j)*H(j))
            """
            new_entropy += prob * calc_shannon_ent(sub_dataset)
        # 获取信息增益
        infoGain = base_entropy - new_entropy
        # 逐一比较获取最大的信息增益
        if infoGain > best_InfoGain:
            best_InfoGain = infoGain
            best_feature = i
    return best_feature


# 投票找出列表中出现次数最多的值
def majority_count(class_list):
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1
    sorted_class_count = sorted(class_count.items(),
                                # 按照键值从大到小排序
                                key=operator.itemgetter(1),
                                reverse=True)
    return sorted_class_count[0][0]


# 正式创建树
def create_tree(data_set, feature_labels):
    """
    创建关于属性的完整树，树枝顺序遵从最大信息增益法
    :param feature_labels:
    :param data_set:
    :return my_tree: 
    """
    # 获取数据集的标签列表
    class_list = [example[-1] for example in data_set]
    # 如果列表第一个元素的数量和列表长度相等，即列表只含一种标签
    if class_list.count(class_list[0]) == len(class_list):
        # 那就直接返回标签，不用划分了
        return class_list[0]
    # 如果数据集的样本只有一个元素，即没有属性只有标签
    if len(data_set[0]) == 1:
        # 那也不用属性划分了，直接找到最多的标签值返回
        return majority_count(class_list)

    # 获取最适合于划分的属性序号,进而获取该标签
    best_feature = choose_best_feature_split(data_set)
    # 从特征标签中获取该标签的具体表达
    best_feature_label = feature_labels[best_feature]
    # 定义树(多维字典)
    my_tree = {best_feature_label: {}}
    # 删除已经使用的属性
    del (feature_labels[best_feature])
    # 获取当前选定属性的所有值，然后转成集合
    feature_value_set = set([example[best_feature] for example in data_set])
    """
    用一个递归创建完整树：
    从一个属性、属性值组合开始作为树枝构建精简矩阵，用这个矩阵再次构建树（再探出树枝）
    ...如此继续，一直到函数最上方的两个if生效时回溯最后的标签值（字典键值）。
    回溯到上一级的树枝再伸出另一根树枝（如果有的话），最后一步步回溯到树干，形成完整的树。
    """
    for value in feature_value_set:
        # 读取剩下的特征标签
        sub_labels = feature_labels[:]
        my_tree[best_feature_label][value] = create_tree(
            split_dataset(data_set, best_feature, value), sub_labels)
    return my_tree


# 获取叶的数量
def get_num_of_leafs(my_tree):
    """
    获取叶（终端节点）的数目
    :param my_tree:
    :return num_of_leafs:
    """
    num_of_leafs = 0
    # 返回树第一层的第一根树枝(索引key)
    first_branch = list(my_tree.keys())[0]
    # 返回此树枝分叉出的所有枝叶
    second_dic = my_tree[first_branch]

    for key in list(second_dic.keys()):
        # 判断每个分支下面是叶(value)，还是继续分支(dict)
        if type(second_dic[key]).__name__ == 'dict':
            # 用一个递归遍历所有的分支，统计叶的数量
            num_of_leafs += get_num_of_leafs(second_dic[key])
        else:
            num_of_leafs += 1
    return num_of_leafs


# 获取树的单层中最大的分支数
def get_num_of_maxBranch(my_tree):
    """
    获取树的单层最大分支数
    :param my_tree:
    :return max_branch:
    """
    max_branch = 1
    next_branch = 0
    first_branch = list(my_tree.keys())[0]
    second_dict = my_tree[first_branch]
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            next_branch += 1
            inner_maxbranch = get_num_of_maxBranch(second_dict[key])
            if inner_maxbranch > next_branch:
                max_branch = inner_maxbranch
            else:
                max_branch = next_branch
    return max_branch


# 获取树的层数和单层中最大的分指数
def get_tree_depth(my_tree):
    """
    获取树的目录层数，方法和叶相似
    :param my_tree:
    :return tree_depth:
    """
    tree_depth = 0
    first_branch = list(my_tree.keys())[0]
    second_dict = my_tree[first_branch]
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            thisDepth = 1 + get_tree_depth(second_dict[key])
        else:
            thisDepth = 1
        if thisDepth > tree_depth:
            tree_depth = thisDepth
    return tree_depth


# 获取待预测的样本的样本标签
def tree_classify(my_tree, featLabel_list, featValue_vec):
    """
    featLabel_list 存储了所有的属性标签，原始顺序
    featValue_vec 存储了一个样本各个属性组成的向量,原始顺序
    函数的目的是在树中追踪这个属性向量走过的路径找到最后的样本标签
    这也是用决策树做预测的基本执行程序
    :param my_tree:
    :param featLabel_list:
    :param featValue_vec:
    :return:
    """
    classLabel = ''
    # 获取当前层的属性标签
    first_str = list(my_tree.keys())[0]
    # 获取此树的下一级数据
    second_dict = my_tree[first_str]
    # 获取当前属性在属性列表中的索引
    feature_index = featLabel_list.index(first_str)
    """
    检查此标签下的各个元素(可能有多个叶和枝)，找到和样本属性值匹配的树的位置：
    检查树中该位置的类型，是树枝那递归继续匹配下一个属性直到找到叶输出预测的样本标签。
    """
    for key in list(second_dict.keys()):
        if featValue_vec[feature_index] == key:
            if type(second_dict[key]).__name__ == 'dict':
                classLabel = tree_classify(second_dict[key], featLabel_list, featValue_vec)
            else:
                classLabel = second_dict[key]
    return classLabel


# 存储树
def store_tree(my_tree, file_name):
    """
    存储树(二进制存储)
    :param my_tree: 
    :param file_name: 
    """
    file_write = open(file_name, "wb")
    pickle.dump(my_tree, file_write)
    file_write.close()


# 读取树
def read_tree(file_name):
    """
    读取树(读取二进制文本)
    :param file_name:
    :return: my_tree
    """
    file_read = open(file_name, "rb")
    my_tree = pickle.load(file_read)
    return my_tree


# 创建测试数据
dataset, feature_list = read_dataset('lenses.txt')
get_tree = create_tree(dataset, feature_list)

test_dataset, test_feature_labels = create_test_dataset()
test_tree = create_tree(test_dataset, test_feature_labels)