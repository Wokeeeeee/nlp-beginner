import math

import jieba
from collections import Counter

import pandas
import pandas as pd
import numpy as np


def load_vocab():
    """
    载入自定义词表
    :return: word_dict 词表
    """
    word_dict = []
    with open("data/vocab.txt", 'r', encoding='utf8') as vocab:
        seg_stop_words = {" ", "，", "。", "“", "”", '“', "？", "！", "：", "《", "》", "、", "；", "·", "‘ ", "’",
                          "──", ",", ".", "?", "!", "`", "~", "@", "#", "$", "%", "^", "&", "*", "(", ")", "-",
                          "_", "+", "=", "[", "]", "{", "}", '"', "'", "<", ">", "\\", "|" "\r", "\n", "\t"}
        for word in vocab:
            word = word.strip('\n')
            if word not in seg_stop_words:
                word_dict.append(word.strip())
    print("finish loading vocab")
    return word_dict


def load_train():
    """
    载入训练数据
    :return: train_data[label,text_a,words_count]--->[标签,原文,分词后计数]
    """
    label = []
    words = []
    train_data = pd.read_csv("data/train.tsv", sep='\t')
    words_count = []
    sum = 0
    jieba.load_userdict("data/vocab.txt")  # 使用jieba载入自定义词表并分词
    for sentence in train_data["text_a"]:
        # 二元语法（2-grams）集合：{"STARTThe", "The cat", "cat", "cat sat", "sat", "sat on", "on", "on the", "the", "the mat", "matEND"}
        raw = jieba.lcut(sentence)
        sum += len(raw)
        raw.insert(0, "START")
        raw.append("END")
        gram_2 = []
        for i in range(len(raw)):
            if i == len(raw) - 1:
                gram_2.append(raw[i])  # END  the last one
                break
            gram_2.append(raw[i])  # START
            gram_2.append(raw[i] + raw[i + 1])  # the cat
        words_count.append(Counter(gram_2))
    # print(words_count)
    train_data["word_count"] = words_count
    print("finish loading train data")
    return train_data, sum


# 统计次数也打表格
# count_table = {}


def getWkOfCj(wk, wk_1, cj):
    '''
    p(wi|wi-1)=count(wi-1 wi,c)+1/count(wi-1,c)+v
    :param wk:
    :param wk_1:
    :param cj:
    :return:
    '''
    cnt_ww_1 = 0
    cnt_w_1 = 0
    cnt_ww_0 = 0
    cnt_w_0 = 0

    if wk + wk_1 not in count_table.keys() or wk_1 not in count_table.keys():
        for i in range(len(train_data["label"])):  # 遍历每一行
            if train_data["label"][i] == 1:
                cnt_ww_1 += train_data["word_count"][i][wk + wk_1]  # wkwk-1在第i个句子中出现次数
                cnt_w_1 += train_data["word_count"][i][wk_1]  # wk-1在第i个句子中出现次数
            else:
                cnt_ww_0 += train_data["word_count"][i][wk + wk_1]  # wkwk-1在第i个句子中出现次数
                cnt_w_0 += train_data["word_count"][i][wk_1]  # wk-1在第i个句子中出现次数
        if wk + wk_1 not in count_table.keys():
            count_table[wk + wk_1] = [cnt_ww_1, cnt_ww_0]
        if wk_1 not in count_table.keys():
            count_table[wk_1] = [cnt_w_1, cnt_w_0]
    else:
        cnt_ww_1, cnt_ww_0 = count_table[wk + wk_1]
        cnt_w_1, cnt_w_0 = count_table[wk_1]
    if cj == 1:
        p = np.log((cnt_ww_1 + 1) / (cnt_w_1 + len(words_dic)))  # 平滑
    else:
        p = np.log((cnt_ww_0 + 1) / (cnt_w_0 + len(words_dic)))
    return p


# table = {}
def load_table():
    table = {}
    with open("pro_table.txt", 'r', encoding='utf8') as t:
        for word in t:
            word = word.strip('\n').split(",")
            if word[0] != '' and len(word) == 3:
                # print(word)
                table[word[0]] = [float(word[1]), float(word[2])]
    # print(table)
    return table


def getProbOfCi(words):
    '''
    一个句子为类别1和0的可能性
    :param words: 某句子
    :param i: 类别i
    :return: p
    '''
    # 类别概率
    label_count = Counter(train_data["label"])
    p0 = np.log(label_count[0] / sum(label_count.values()))
    p1 = np.log(label_count[1] / sum(label_count.values()))

    # 添加首尾标志位
    words.insert(0, "START")
    words.append("END")
    for i in range(len(words) - 1):
        pw0 = getWkOfCj(words[i], words[i + 1], 0)
        pw1 = getWkOfCj(words[i], words[i + 1], 1)
        p0 += pw0
        p1 += pw1
    return p1, p0


def deleteStopWord(words):
    # delete stop word
    # 这里只去掉标点符号
    seg_stop_words = {" ", "，", "。", "“", "”", '“', "？", "！", "：", "《", "》", "、", "；", "·", "‘ ", "’",
                      "──", ",", ".", "?", "!", "`", "~", "@", "#", "$", "%", "^", "&", "*", "(", ")", "-",
                      "_", "+", "=", "[", "]", "{", "}", '"', "'", "<", ">", "\\", "|" "\r", "\n", "\t"}
    result = []
    for w in words:
        if w not in seg_stop_words:
            result.append(w)
        if w in seg_stop_words:
            result.append("START")  # 处理符号位
            result.append("END")
    return result


def load_test():
    test_data = pd.read_csv("data/test.tsv", sep='\t')
    words_cut = []
    jieba.load_userdict("data/vocab.txt")  # 使用jieba载入自定义词表并分词
    for sentence in test_data["text_a"]:
        words_cut.append(jieba.lcut(sentence))
    test_data["word_cut"] = words_cut
    print("finish loading test data")
    return test_data


def load_count_table():
    count_table = {}
    lines = np.array(pd.read_csv("count_table.csv"))
    for w in lines:
        if len(w) == 3:
            count_table[w[0]] = [float(w[1]), float(w[2])]
    return count_table


def predict(test_data):
    result = []
    index = 0
    for words in test_data["word_cut"]:
        index += 1
        if index % 100 == 0: print("predicting", index, "/", len(test_data["word_cut"]))
        p1, p0 = getProbOfCi(words)
        if p1 > p0:
            result.append(1)
        else:
            result.append(0)
    pd.DataFrame.from_dict(count_table, orient='index').to_csv("count_table.csv")
    return result


count_table = load_count_table()
words_dic = load_vocab()
train_data, words_sum = load_train()
test_data = load_test()


# table = load_table()

def calculateARF(result):
    tp, fp, fn, tn = 0, 0, 0, 0
    for i in range(len(result)):
        if test_data["label"][i] == 1:
            if result[i] == 1:
                tp += 1
            else:
                fn += 1
        else:
            if result[i] == 1:
                fp += 1
            else:
                tn += 1
    accuracy = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * accuracy * recall / (accuracy + recall)
    print(accuracy, recall, f1)


result = predict(test_data)
print(result)
calculateARF(result)


def complexity1Gram(test_data):
    result = []
    for words in test_data["word_cut"]:
        result.append(calculateComplex1Gram(words[1:len(words) - 1]))
    return sum(result) / len(result)


def calculateComplex1Gram(words):
    '''
    传入一句话,计算1元语法困惑度
    :param words:
    :return:
    '''
    p = 0
    for i in range(len(words)):
        if words[i] in count_table.keys():
            p += np.log((count_table[words[i]][0] + count_table[words[i]][1] + 1) / words_sum)
        else:
            cnt = 0
            for j in range(len(train_data["label"])):  # 遍历每一行
                cnt += train_data["word_count"][j][words[i]]  # wkwk-1在第i个句子中出现次数
            p += np.log(cnt / words_sum)
    complex = np.exp(-p / len(words))
    return complex


def complexity2Gram(test_data):
    result = []
    for words in test_data["word_cut"]:
        result.append(calculateComplex1Gram(words))
    return sum(result) / len(result)


def calculateComplex2Gram(words):
    '''
    传入一句话，用计算二元语法困惑度
    :param words:
    :return:
    '''
    p = 0
    for i in range(len(words) - 1):
        p += np.log(
            (count_table[words[i] + words[i + 1]][0] + count_table[words[i] + words[i + 1]][1] + 1) / (
                    count_table[words[i]][0] + count_table[words[i]][1] + len(words_dic)))

    complex = np.exp(-p / len(words))
    return complex

print(complexity1Gram(test_data))
print(complexity2Gram(test_data))
# print(calculateComplex(test_data["word_cut"].values[0]))
