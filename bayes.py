# @Time    : 2017/10/6 16:31
# @Author  : Jalin Hu
# @File    : bayes.py
# @Software: PyCharm


import numpy


def load_data_set():
    posting_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                    ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                    ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                    ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_vect = [0, 1, 0, 1, 0, 1]
    return posting_list, class_vect


def create_vlcab_list(data_set):
    vocab_set = set()
    for document in data_set:
        vocab_set = vocab_set | set(document)
    return list(vocab_set)


def setof_word2vec(vocablist, inputset):
    returnvec = [0] * len(vocablist)
    for word in inputset:
        if word in vocablist:
            returnvec[vocablist.index(word)] = 1
        else:
            print('word:', word, 'is not in the list')
    return returnvec


def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numwords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)  # 计算p(ci),文档属于侮辱类的概率
    p0Num = numpy.zeros(numwords)
    p1Num = numpy.zeros(numwords)
    p0denom = 0.0
    p1denom = 0.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0denom += sum(trainMatrix[i])
    p1vect = p1Num / p1denom
    p0vect = p0Num / p0denom
    return p1vect, p0vect, pAbusive


def classifyNB(vec2classify, p0vec, p1vec, pClass1):
    p1 = sum(vec2classify*p1vec)*pClass1
    p0 = sum(vec2classify*p0vec)*pClass1
    if p1 > p0:
        return 1
    else:
        return 0


if __name__ == '__main__':
    postingList, classVec = load_data_set()
    myVocabList = create_vlcab_list(postingList)
    print('myVocabList:\n', myVocabList)
    trainMat = []
    for postinDoc in postingList:
        trainMat.append(setof_word2vec(myVocabList, postinDoc))
    p1V, p0V, pAb = trainNB0(trainMat, classVec)

    testEntry = ['love', 'my', 'dalmation']  # 测试样本1
    thisDoc = numpy.array(setof_word2vec(myVocabList, testEntry))  # 测试样本向量化
    if classifyNB(thisDoc, p0V, p1V, pAb):
        print(testEntry, '属于侮辱类')  # 执行分类并打印分类结果
    else:
        print(testEntry, '属于非侮辱类')  # 执行分类并打印分类结果
    testEntry = ['stupid', 'garbage']  # 测试样本2
    thisDoc = numpy.array(setof_word2vec(myVocabList, testEntry))  # 测试样本向量化
    if classifyNB(thisDoc, p0V, p1V, pAb):
        print(testEntry, '属于侮辱类')  # 执行分类并打印分类结果
    else:
        print(testEntry, '属于非侮辱类')  # 执行分类并打印分类结果
