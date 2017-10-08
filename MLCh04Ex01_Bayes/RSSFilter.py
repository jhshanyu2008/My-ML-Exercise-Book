"""
Rss测试
"""
import feedparser
import DataProcess
import random

source_A = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
source_B = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')


# 遍历文本，选取出现次数最高的一组单词
def get_mostFreq_words(vocabList, fullText):
    """
    遍历文本，选取出现次数最高的一组单词
    :param vocabList:
    :param fullText:
    :return:
    """
    # 新建一个包含所有单词与其出现次数组合的字典
    freqDict = {}
    for words in vocabList:
        freqDict[words] = fullText.count(words)
    # 从大到小排序生成元祖列表
    sortedFreqDict = sorted(freqDict.items(), key=lambda x: int(x[1]), reverse=True)
    return sortedFreqDict[:100]


def RSS_filter(sourceA, sourceB, testRadio):
    """
    sourceA, sourceB是两个原始文本来源
    设定sourceA为'1'类，sourceB为'0'类
    testRadio 和邮件测试一样，用于分段训练和测试集
    这次我决定和书中一样从整合的数据集中抽取，而不是像邮件测试那样从每个类别分别抽取
    :param testRadio:
    :param sourceA:
    :param sourceB:
    """
    docList = []
    classList = []
    # 这里将在一维列表中存储所有文本的单词
    fullText = []
    minLen = min(len(sourceA['entries']), len(sourceB['entries']))
    for i in range(minLen):
        # 读取来自source的文本，只提取'summary'下的文本，转换成词汇列表
        wordList = DataProcess.parse_to_wordList(sourceA['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        # sourceA分类为'1'
        classList.append(1)
        wordList = DataProcess.parse_to_wordList(sourceB['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        # sourceB分类为'0'
        classList.append(0)
    # 挑选出测试集和训练集
    testList = []
    trainList = []
    testCategory = []
    trainCategory = []
    docLen = len(docList)
    testNumList = sorted(random.sample(range(0, docLen), round(testRadio * docLen)))
    for i in range(docLen):
        if testNumList.__contains__(i):
            testList.append(docList[i])
            testCategory.append(classList[i])
        else:
            trainList.append(docList[i])
            trainCategory.append(classList[i])

    # 生成训练集词汇列表
    adjVocabList = DataProcess.create_vocabList(trainList)
    # 我们打算把出现频率最高的词汇删除了
    topWords = get_mostFreq_words(adjVocabList, fullText)
    for word in topWords:
        if adjVocabList.__contains__(word[0]):
            adjVocabList.remove(word[0])

    # 调用分类的函数,这里按是否去掉高频词汇算两次，对比结果
    # 去掉高频词汇的
    testMatrix, trainMatrix = \
        DataProcess.get_data_matrix(testList, trainList, model='bag', vocabModel='extra', extraVocabList=adjVocabList)
    classifyResult = DataProcess.test_naive_bayes(testMatrix, trainMatrix, trainCategory)
    # 不去高频词汇的
    testMatrix2, trainMatrix2 = \
        DataProcess.get_data_matrix(testList, trainList, model='bag')
    classifyResult2 = DataProcess.test_naive_bayes(testMatrix2, trainMatrix2, trainCategory)
    errorCount = 0
    errorCount2 = 0
    # 用文本显示分类，而不是'0'、'1'
    classifyClass = {0: 'sourceB-sfbay',
                     1: 'sourceA-newyork'}
    for i in range(len(testCategory)):
        print("The {0}\n is classified as {1}，and the real source is {2}"
              .format(testList[i], classifyClass[classifyResult[i]], classifyClass[testCategory[i]]))
        print("The {0}\n is classified as {1}，and the real source is {2}"
              .format(testList[i], classifyClass[classifyResult2[i]], classifyClass[testCategory[i]]))
        if classifyResult[i] != testCategory[i]:
            errorCount += 1
        if classifyResult2[i] != testCategory[i]:
            errorCount2 += 1
    errorRate = float(errorCount / len(testCategory))
    errorRate2 = float(errorCount2 / len(testCategory))
    # 打印错误率
    print("The error rate of this test is {}%".format(errorRate * 100))
    print("The error rate of this test is {}%".format(errorRate2 * 100))


RSS_filter(source_A, source_B, 0.1)
# 函数结构和书中不太一样了，所以最后的表征词汇就略过了，没有十分新奇的内容
