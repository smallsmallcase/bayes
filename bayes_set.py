# @Time    : 2017/10/17 20:11
# @Author  : Jalin Hu
# @File    : bayes_set.py
# @Software: PyCharm
import numpy
import collections
def create_vlcab_list(data_set):
    vocab_set = set()
    for document in data_set:
        vocab_set = vocab_set | set(document)
    return list(vocab_set)


def bagof_word2vec(vocablist, inputset):
    returnvec = [0] * len(vocablist)
    for word in inputset:
        if word in vocablist:
            returnvec[vocablist.index(word)] += 1
        else:
            print('word:', word, 'is not in the list')
    return returnvec


def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numwords = len(trainMatrix[0])
    trainCategory_set = set(trainCategory)
    typetime = collections.defaultdict(int)
    pbusive = collections.defaultdict(float)
    pNum_dict = collections.defaultdict(int)
    pdenom_dict = collections.defaultdict(float)
    pvect_dict = collections.defaultdict(float)
    for i in trainCategory:
        typetime[i] += 1
    for type in trainCategory_set:
        pbusive[type] = typetime[type]/float(numTrainDocs)
        pNum_dict[type] = numpy.ones(numwords)
        pdenom_dict[type] = 2.0
    # pAbusive = sum(trainCategory) / float(numTrainDocs)  # 计算p(ci),文档属于侮辱类的概率
    # p0Num = numpy.zeros(numwords)
    # p1Num = numpy.zeros(numwords)
    # p0denom = 0.0
    # p1denom = 0.0

    # p0Num = numpy.ones(numwords)
    # p1Num = numpy.ones(numwords)
    # p0denom = 2.0
    # p1denom = 2.0
    for i in range(numTrainDocs):
        for type in trainCategory_set:
            if trainCategory[i] == type:
                pNum_dict[type] += trainMatrix[i]
                pdenom_dict[type] += sum(trainMatrix[i])
                pvect_dict[type] = numpy.log(pNum_dict[type]/pdenom_dict[type])
                # p1Num += trainMatrix[i]
                # p1denom += sum(trainMatrix[i])
            # else:
            #     p0Num += trainMatrix[i]
            #     p0denom += sum(trainMatrix[i])
    # p1vect = numpy.log(p1Num / p1denom)
    # p0vect = numpy.log(p0Num / p0denom)
    return pvect_dict, pbusive, trainCategory_set


def classifyNB(vec2classify, pvec, p_before, category_set):
    p_after_dict = collections.defaultdict(float)
    for type in category_set:
        p_after_dict[type] = sum(vec2classify*pvec[type]) + numpy.log(p_before[type])
    p_after_list = sorted(p_after_dict.items(), key=lambda e: e[1], reverse=True)
    return p_after_list[0][0]

    # p1 = sum(vec2classify * p1vec) + numpy.log(p_before)
    # p0 = sum(vec2classify * p0vec) + numpy.log(1 - pClass1)
    # if p1 > p0:
    #     return 1
    # else:
    #     return 0