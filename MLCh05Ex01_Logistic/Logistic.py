
"""
sigmoid函数：代替阶跃函数处理瞬间跳跃难以处理的问题
   σ(z) = 1/(1+exp(-z)) 这个函数性能很利于计算:
   z<<0时σ(z)趋于0 z=0时σ(z)趋于0.5 z>>0时σ(z)趋于1

在线性模型中输入：
   z = w0*x0 + w1*x1 + w2*x2 +...+ wn*xn  其中wi是我们要找的系数，xi是各个属性
   用矩阵保存wi、xi 记作 W和X(都是列向量) z = W'*X  W'意为W的转置

梯度上升法：
   梯度预示着函数的前进方向,只要该点可微一定有梯度值例如：(∂f(x,y)/∂x, ∂f(x,y)/∂y)
   给定一个步长按梯度显示的方向前进，只要步长足够小最终一定会到达函数的极值点。

步长记作α，梯度算法函数取点的迭代公式即为：ω:= ω + α*df(ω)
"""


# 读取测试数据
def read_dataSet(fileName=''):
    """
    读取测试数据
    :param fileName:
    :return:
    """
    dataMatrix = []
    labelMatrix = []
    fr = open(fileName)
    for line in fr.readlines():
        lineList = line.strip().split()
        dataMatrix.append([1.0] + [float(x) for x in lineList][:-1])
        labelMatrix.append(int(lineList[-1]))
    return dataMatrix, labelMatrix


a, b = read_dataSet('testSet.txt')
print(a)
print(b)
