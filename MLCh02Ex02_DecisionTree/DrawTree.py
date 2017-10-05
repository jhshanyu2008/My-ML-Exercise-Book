"""
存储一些绘制树的函数
"""

# coding:utf-8
from DecisionTree import *


class DrawTree:
    import DecisionTree as treelib
    import matplotlib.pyplot as plt

    # 用来正常显示中文标签和负号
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    ax = plt.subplot(111, frameon=False)
    test_tree = treelib.test_tree

    __input_tree = test_tree
    __node_content = 'test node'
    __current_point = (1, 1)
    __original_point = (0, 0)

    __total_wide = 1
    __total_depths = 1
    __max_branch = 1
    __x_off = 1
    __y_off = 1
    __used_leafs = 0
    __used_branch = []
    __orig_pointlist = []

    # fc控制的注解框内的颜色深度
    style_decisionNode = dict(boxstyle="sawtooth,pad=0.6", fc="0.8")
    style_leafNode = dict(boxstyle="round,pad=0.6", fc="0.8")
    style_arrowArgs = dict(arrowstyle="<|-", connectionstyle="arc3,rad=-0.2", fc="w")

    def __init__(self, input_tree=test_tree):
        self.__input_tree = input_tree

    def plot_node(self, content=__node_content,
                  current_point=__current_point,
                  original_point=__original_point,
                  node_type=style_decisionNode,
                  arrow_props=style_arrowArgs
                  ):
        """
        绘制箭头注释
        annotate()文本注释属性，此属性会被matplotlib自动识别
        xytext注释坐标，textcoords指定注释相对坐标的位置，axes fraction指箭头左侧
        xycoords和 textcoords属性一样，xy指定箭头出发位置
        va指注释的垂直位置，ha指水平位置，bbox指注释的文本框外观
        如果有arrowprops属性即意味着画一条从xy到xytext的箭头
        """
        self.ax.annotate(content, xytext=current_point, textcoords='axes fraction',
                         xy=original_point, xycoords='axes fraction', size=15,
                         va='center', ha='center', bbox=node_type, arrowprops=arrow_props)

    def create_test_plot(self):
        fig = self.plt.figure(1, facecolor='white')
        fig.clf()
        """
        ax是subplot意为“创建子图”，111即1x1分割界面取第一个图，frameon = False 即无边缘
        """
        self.ax = self.plt.subplot(111, frameon=False)
        self.plot_node(u'决策点', current_point=(0.5, 0.1), original_point=(0.1, 0.5),
                       node_type=self.style_decisionNode)
        self.plot_node(u'叶节点', current_point=(0.8, 0.1), original_point=(0.3, 0.8),
                       node_type=self.style_leafNode)
        self.plt.show()

    def plot_midtext(self, original_point=(0, 0), current_point=(1, 1), content='test'):
        """
        绘制箭头中间的注释
        """
        x_mid = (original_point[0] - current_point[0]) / 2.0 + current_point[0]
        y_mid = (original_point[1] - current_point[1]) / 2.0 + current_point[1]
        self.ax.text(x_mid, y_mid, content, va="center", ha="center", rotation=30, size=12)

    def plot_tree(self, my_tree, node_text):
        """
        用递归绘制各个枝叶
        :param my_tree:
        :param node_text:
        """
        # 获取当前枝的叶数和层数
        num_leafs = self.treelib.get_num_of_leafs(my_tree)
        num_depths = self.treelib.get_tree_depth(my_tree)
        # 确定此分支的当前坐标
        x = 0 + (self.__used_leafs + num_leafs / 2) * self.__x_off
        y = 0 + num_depths * self.__y_off
        self.__current_point = (x, y)

        # 获取此分支的横纵序列
        orig_hori = self.__total_depths - num_depths + 1
        self.__used_branch[orig_hori] += 1
        orig_vert = self.__used_branch[orig_hori] - 1
        # 上一级分支的横纵序列
        front_orig_hori = orig_hori - 1
        front_orig_vert = self.__used_branch[front_orig_hori] - 1

        # 把当前坐标存入出发坐标列表
        self.__orig_pointlist[orig_hori].append(self.__current_point)
        # 调用上一级的出发坐标,绘制上一级分支到此分支的注释
        self.__original_point = self.__orig_pointlist[front_orig_hori][front_orig_vert]
        # 首先绘制箭头中间的文字
        self.plot_midtext(original_point=self.__original_point,
                          current_point=self.__current_point, content=node_text)
        # 再画“分支——>分支”的图示(先提取此分支的属性标签)
        first_str = list(my_tree.keys())[0]
        self.plot_node(first_str, original_point=self.__original_point,
                       current_point=self.__current_point, node_type=self.style_decisionNode)

        second_dict = my_tree[first_str]
        for key in list(second_dict.keys()):
            if type(second_dict[key]).__name__ == 'dict':
                # 发现是分支时，递归函数
                self.plot_tree(second_dict[key], str(key))
            else:
                # 发现是叶时，使用叶的坐标原则
                x = 0 + self.__used_leafs * self.__x_off
                y = 0 + (num_depths - 1) * self.__y_off
                self.__used_leafs += 1
                self.__current_point = (x, y)
                # 以当前的分支坐标作为出发点，绘制注释
                self.__original_point = \
                    self.__orig_pointlist[orig_hori][orig_vert]
                self.plot_node(second_dict[key], current_point=self.__current_point,
                               original_point=self.__original_point, node_type=self.style_leafNode)
                self.plot_midtext(current_point=self.__current_point,
                                  original_point=self.__original_point, content=str(key))

    def create_whole_tree(self, my_tree=__input_tree):
        """
        创建完整的树
        :param my_tree:
        绘制的思路是：
        根据树的叶数和层树把界面分割成__x_off长，__y_off宽的一个个网格,网格的左下点为叶记录坐标.
        当一个点为叶时：该点x坐标为：0+__used_point*__x_off 即在已经绘制的叶的最右侧
                       该点y坐标为：上一级分支的下一层
        当一个点为枝时：该点x坐标为：0+(__used_point+num_leaf/2)*__x_off
                         即在已绘制叶的右侧、自己所含叶将占据空间的中间，y坐标同叶
        和叶坐标从左到右一个个绘制不同，分支坐标要跨位置调用，所以需要一个二维列表__orig_pointlist储存各个分支坐标
        """
        # 初始化图
        fig = self.plt.figure(1, facecolor='white')
        fig.clf()
        axprops = dict(xticks=[], yticks=[])
        self.ax = self.plt.subplot(111, frameon=False, **axprops)
        # 获取树的叶数和层数和单层最大分支数
        self.__total_wide = self.treelib.get_num_of_leafs(my_tree)
        self.__total_depths = self.treelib.get_tree_depth(my_tree)
        self.__max_branch = self.treelib.get_tree_depth(my_tree)
        # 计算分割网格的长宽
        self.__x_off = 1.0 / float(self.__total_wide)
        self.__y_off = 1.0 / float(self.__total_depths)
        # 已用叶数及已用分叉数清零
        self.__used_leafs = 0
        self.__used_branch = []
        # 预输入初始坐标
        self.__original_point = self.__current_point = (0.5, 1.0)
        # 把出发坐标存入列表，方便以后调用
        self.__orig_pointlist = []
        for i in range(self.__total_wide + 1):
            self.__orig_pointlist.append([])
            self.__used_branch.append(0)
        self.__orig_pointlist[0].append(self.__original_point)
        self.__used_branch[0] = 1
        # 绘制树
        self.plot_tree(my_tree, '')
        self.plt.show()

# 测试数据
test = DrawTree()
test.create_whole_tree(get_tree)
