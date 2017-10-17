# @Time    : 2017/10/6 16:31
# @Author  : Jalin Hu
# @File    : bayes.py
# @Software: PyCharm


import re
import random
from bayes_set import *


def text_parse(bigstring):
    words = re.split(r'\W*', bigstring)
    return [word.lower() for word in words if len(word) > 1]


# def load_data_set():
#     posting_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
#                     ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
#                     ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
#                     ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
#                     ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
#                     ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
#     class_vect = [0, 1, 0, 1, 0, 1]
#     return posting_list, class_vect


if __name__ == '__main__':
    # postingList, classVec = load_data_set()
    postingList = []
    classVec = []
    for i in range(1, 26):
        with open('./spam/%d.txt' % i, 'r') as f:
            text = f.read()
            postingList.append(text_parse(text))
            classVec.append(1)
        with open('./ham/%d.txt' % i, 'r') as f:
            text = f.read()
            postingList.append(text_parse(text))
            classVec.append(0)
    myVocabList = create_vlcab_list(postingList)
    print('词库是：', myVocabList, '\n', '词库的长度是：', len(myVocabList))
    trainsetindex = list(range(50))
    testsetindex = []
    for i in range(10):
        randomindex = int(random.uniform(0, len(trainsetindex)))
        testsetindex.append(randomindex)
        del (trainsetindex[i])

    trainMat = []
    errorcount = 0
    for postinDoc in postingList:
        trainMat.append(bagof_word2vec(myVocabList, postinDoc))  # 训练集样本化向量化
    pvec, pbusive, category_set = trainNB0(trainMat, classVec)

    for i in testsetindex:
        thisDoc = numpy.array(bagof_word2vec(myVocabList, postingList[i]))  # 测试样本向量化
        if classifyNB(thisDoc, pvec, pbusive, category_set) != classVec[i]:
            errorcount += 1
            print('错误的测试集：', postingList[i])
    accuricy = float(errorcount / len(testsetindex) * 100)
    print('错误率是：%.2f%%' % accuricy)
