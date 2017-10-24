# @Time    : 2017/10/23 17:43
# @Author  : Jalin Hu
# @File    : sinanewswithsklarn.py
# @Software: PyCharm

import os
import jieba
import random
from bayes_set import *
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt


def textprocess(foldpath, testsize=0.1):
    foldlist = os.listdir(foldpath)  # 获取Sample下面的子文件夹,共9个
    datalist = []
    classlist = []
    for fold in foldlist:
        newfoldpath = os.path.join(foldpath, fold)  # 生成新的文件夹路径
        filelist = os.listdir(newfoldpath)
        for file in filelist:
            with open(os.path.join(newfoldpath, file), 'r', encoding='utf-8') as f:
                sequence = f.read()
            datalist.append(jieba.lcut(sequence, cut_all=False))
            classlist.append(fold)
    # return datalist, classlist
    data_class_list = list(zip(datalist, classlist))
    # print(data_class_list)
    random.shuffle(data_class_list)
    index = int(len(data_class_list) * testsize) + 1  # 训练集和测试集区分的索引
    traindatalist, trainclasslist = zip(*(data_class_list[index:]))  # 训练集解压缩
    testdatalist, testclasslist = zip(*(data_class_list[:index]))  # 测试集解压缩

    # 统计训练集词频
    allworddict = collections.defaultdict(int)  # 创建默认字典
    for word_list in traindatalist:
        for word in word_list:
            # allworddict[word] = allworddict.get(allworddict[word], 0) + 1
            allworddict[word] += 1
            # if word in allworddict.keys():
            #     allworddict[word] += 1
            # else:
            #     allworddict[word] = 1
    # 根据键的值倒序排列
    all_word_sorted = sorted(allworddict.items(), key=lambda e: e[1], reverse=True)
    all_word_list, all_word_nums = zip(*all_word_sorted)
    all_word_list = list(all_word_list)
    return all_word_list, traindatalist, trainclasslist, testdatalist, testclasslist


'''函数说明:读取文件里的内容，并去重

Parameters:
    words_file - 文件路径
Returns:
    words_set - 读取的内容的set集合'''


def make_word_set(word_file):
    word_set = set()
    with open(word_file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            word = line.strip()
            if len(word) > 0:
                word_set.add(word)
    return word_set


'''函数说明:文本特征选取

Parameters:
    all_words_list - 训练集所有文本列表
    deleteN - 删除词频最高的deleteN个词
    stopwords_set - 指定的结束语
Returns:
    feature_words - 特征集'''


def word_dict(all_words_list, deleteN, stopwords_set):
    n = 0
    feature_words = []
    for i in range(deleteN, len(all_words_list), 1):
        if n > 1000:  # 特征维度要小于1000
            break
        else:
            if not all_words_list[i].isdigit() and all_words_list[i] not in stopwords_set and 1 < len(
                    all_words_list[i]) < 5:
                feature_words.append(all_words_list[i])
        n += 1
    return feature_words


def TextFeatures(train_data_list, test_data_list, feature_words):
    def text_features(text, feature_words):
        feature = [0] * len(feature_words)
        for word in text:
            if word in feature_words:
                feature[feature_words.index(word)] += 1
        return feature
        # text_words = set(text)
        # feature = [1 if word in feature_words else 0 for word in text_words]
        # return feature

    train_feature_list = [text_features(text, feature_words) for text in train_data_list]
    test_feature_list = [text_features(text, feature_words) for text in test_data_list]
    return train_feature_list, test_feature_list


def TextClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list):
    clf = MultinomialNB()
    clf.fit(train_feature_list, train_class_list)
    test_accuracy = clf.score(test_feature_list, test_class_list)
    return test_accuracy


if __name__ == '__main__':
    all_word_list, traindatalist, trainclasslist, testdatalist, testclasslist = textprocess('./SogouC/Sample')
    stop_word_file = './stopwords_cn.txt'
    stop_word_set = make_word_set(stop_word_file)
    deleteNs = range(0, 1000, 20)
    # accuricy_list = []
    # for deleteN in deleteNs:
    #     feature_words = word_dict(all_word_list, deleteN, stop_word_set)
    #     train_feature_list, test_feature_list = TextFeatures(traindatalist, testdatalist, feature_words)
    #     accuricy = TextClassifier(train_feature_list, test_feature_list, list(trainclasslist), list(testclasslist))
    #     accuricy_list.append(accuricy)
    #     print(accuricy)
    # plt.figure()
    # plt.plot(deleteNs, accuricy_list)
    # plt.xlabel('deleteNs')
    # plt.ylabel('accuricy')
    # plt.show()
    feature_words = word_dict(all_word_list, 100, stop_word_set)
    train_feature_list, test_feature_list = TextFeatures(traindatalist, testdatalist, feature_words)
    accuricy = TextClassifier(train_feature_list, test_feature_list, list(trainclasslist), list(testclasslist))
    print(accuricy)