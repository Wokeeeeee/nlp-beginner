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


def load_stopws():
    stopwords = [line.strip("\n").strip() for line in open("data/stop.txt", 'r', encoding='utf-8').readlines()]
    return stopwords


def load_train():
    """
    载入训练数据
    :return: train_data[label,text_a,words_count]--->[标签,原文,分词后计数]
    """
    label = []
    words = []
    train_data = pd.read_csv("data/train.tsv", sep='\t')
    words_count = []
    jieba.load_userdict("data/vocab.txt")  # 使用jieba载入自定义词表并分词
    for sentence in train_data["text_a"]:
        words_count.append(Counter(jieba.cut(sentence)))
    train_data["word_count"] = words_count
    print("finish loading train data")
    return train_data


def getWkOfCj(wk, cj):
    '''
    计算某单词wk为类别cj的可能性 p（wk|cj）
    :param wk:
    :param cj:
    :return:
    '''
    # count(w,c)
    cnt_wc = 0
    cnt_c = 0
    cnt_all = 0
    for i in range(len(train_data["label"])):
        cnt_all += sum(train_data["word_count"][i].values())
        if train_data["label"][i] == cj:
            cnt_wc += train_data["word_count"][i][wk]
            cnt_c += sum(train_data["word_count"][i].values())
    p = np.log((cnt_wc + 1) / (cnt_c + len(words_dic)))
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


def getProbOfCi(words, i):
    '''
    计算某句子word为类别i的可能性
    :param words:
    :param i:
    :return:
    '''
    label_count = Counter(train_data["label"])
    p = np.log(label_count[i] / sum(label_count.values()))
    for w in words:
        if w not in table.keys():
            # print(table["word"].values)
            pw1 = getWkOfCj(w, 1)
            pw0 = getWkOfCj(w, 0)
            table[w] = [pw1, pw0]
            p += pw1 if i == 1 else pw0
        else:
            line = table[w]
            p += line[0] if i == 1 else line[1]
    return p


def deleteStopWord(words):
    # delete stop word and repeated word
    result = []
    for w in words:
        if w not in stop_dic and w not in result:
            result.append(w)
    return result


def load_test():
    test_data = pd.read_csv("data/test.tsv", sep='\t')
    words_cut = []
    jieba.load_userdict("data/vocab.txt")  # 使用jieba载入自定义词表并分词
    for sentence in test_data["text_a"]:
        words_cut.append(deleteStopWord(jieba.lcut(sentence)))
    test_data["word_cut"] = words_cut
    print("finish loading test data")
    return test_data


def predict(test_data):
    result = []
    index = 0
    for words in test_data["word_cut"]:
        if index % 100 == 0: print(index)
        index += 1
        p0 = getProbOfCi(words, 0)
        p1 = getProbOfCi(words, 1)
        if p1 > p0:
            result.append(1)
        else:
            result.append(0)
    pd.DataFrame.from_dict(table, orient='index').to_csv("pro_table.txt")
    return result


words_dic = load_vocab()
stop_dic = load_stopws()
train_data = load_train()
test_data = load_test()
table = load_table()

def calculateARF(result):
    tp,fp,fn,tn=0,0,0,0
    for i in range(len(result)):
        if test_data["label"][i]==1:
            if result[i]==1:tp+=1
            else:fn+=1
        else:
            if result[i]==1:fp+=1
            else:tn+=1
    accuracy=tp/(tp+fp)
    recall=tp/(tp+fn)
    f1=2*accuracy*recall/(accuracy+recall)
    print(accuracy,recall,f1)

result=predict(test_data)
calculateARF(result)
# words = test_data["word_cut"][0]
# # print(words)
# p0 = getProbOfCi(words, 0)
# p1 = getProbOfCi(words, 1)
# if p1 > p0:
#     print(1)
# else:
#     print(0)
