def add2transDict(pos1, pos2, transDict):
    if pos2.isnumeric() or pos1.isnumeric(): return
    if pos1 in transDict:
        if pos2 in transDict[pos1]:
            transDict[pos1][pos2] += 1
        else:
            transDict[pos1][pos2] = 1
    else:
        transDict[pos1] = {pos2: 1}


def add2emitDict(pos, word, emitDict):
    if pos in emitDict:
        if word in emitDict[pos]:
            emitDict[pos][word] += 1
        else:
            emitDict[pos][word] = 1
    else:
        emitDict[pos] = {word: 1}


def clearINFS(infs):
    new_infs = []
    for s in infs:
        if s.find("[") != -1:
            s = s[s.find("[") + 1:]
        if s.find("]") != -1:
            s = s[:s.find("]")]
        s = s.strip()
        if s == "": continue
        new_infs.append(s)
    return new_infs


def training(infile, transDict, emitDict):
    fdi = open(infile, 'r', encoding='utf-8')
    for line in fdi:
        infs = line.strip().split()
        infs = clearINFS(infs)
        wpList = [["__NONE__", "__start__"]] + [s.split("/") for s in infs] + [["__NONE_", "__end__"]]
        for i in range(1, len(wpList)):
            pre_pos = wpList[i - 1][1]  # 前面一个词性（隐藏状态 y_t-1）
            cur_pos = wpList[i][1]  # 当前词性状态 y_t
            word = wpList[i][0]  # 当前观测值(发射值) x_t
            if word == "" or cur_pos == "" or pre_pos == "":
                continue
            add2transDict(pre_pos, cur_pos, transDict)  # 统计转移频次
            add2emitDict(cur_pos, word, emitDict)  # 统计发射频次
        add2transDict("__end__", "__end__", transDict)
    fdi.close()


def getPosNumList(transDict):
    pnList = []
    for pos in transDict:
        num = sum(transDict[pos].values())
        pnList.append([pos, num])  # 前一个词性出现了多少次
    pnList.sort(key=lambda infs: (infs[1]), reverse=True)
    return pnList


def getTotalWordNum(emitDict):
    total_word_num = 0
    for pos in emitDict:
        total_word_num += sum(list(emitDict[pos].values()))
    return total_word_num


def exportmodel(transDict, emitDict, model_file):
    pnList = getPosNumList(transDict)

    # 状态集合
    fdo = open(model_file, 'w', encoding='utf-8')
    total = sum([num for pos, num in pnList])  # 所有词性的出现次数
    for pos, num in pnList:
        fdo.write("pos_set\t%s\t%d\t%f\n" % (pos, num, num / total))

    # 转移概率
    total_word_num = getTotalWordNum(emitDict)  # {cur_pos, {word, count}}
    for pos1, num1 in pnList:  # 前一个词性，频次
        if pos1 == "__end__":
            continue
        smoothing_factor = 1.0
        tmpList = []
        for pos2, _ in pnList:
            if pos2 == "__start__":
                continue
            if pos2 in transDict[pos1]:
                tmpList.append([pos2, transDict[pos1][pos2] + smoothing_factor])
            else:
                tmpList.append([pos2, smoothing_factor])
        denominator = sum([infs[1] for infs in tmpList])
        for pos2, numerator in tmpList:
            fdo.write("trans_prob\t%s\t%s\t%f\n" % (pos1, pos2, math.log(numerator / denominator)))

    # 发射概率
    for pos, _ in pnList:
        if pos == "__start__" or pos == "__end__":
            continue
        wnList = list(emitDict[pos].items())
        wnList.sort(key=lambda infs: infs[1], reverse=True)
        num = sum([num for _, num in wnList])
        smoothing_factor = num / total_word_num
        tmpList = []
        for word, num in wnList:
            tmpList.append([word, num + smoothing_factor])
        tmpList.append(["__NEW__", smoothing_factor])
        # pos词性下，发射其他未统计到的词时的概率给个平滑
        denominator = sum([infs[1] for infs in tmpList])
        for word, numerator in tmpList:
            fdo.write("emit_prob\t%s\t%s\t%f\n" % (pos, word, math.log(numerator / denominator)))
    fdo.close()


import sys
import os
import math


#
# transDict = {}  # 转移
# emitDict = {}  # 发射
#
# model_file = "./model_file.txt"
# rootdir = "D:\PycharmProjects\\nlp-beginner\people-2014\\train\\0123"
# txt_list = os.listdir(rootdir)
# for i in range(0, len(txt_list)):
#     path = os.path.join(rootdir, txt_list[i])
#     if os.path.isfile(path):
#         training(path, transDict, emitDict)
#         # print(path)
# exportmodel(transDict, emitDict, model_file)  # 输出到文件


def add2transDictWithProb(pos1, pos2, prob, transDict):
    if pos1 in transDict:
        transDict[pos1][pos2] = prob
    else:
        transDict[pos1] = {pos2: prob}


def add2emitDictWithProb(pos, word, prob, emitDict):
    if pos in emitDict:
        emitDict[pos][word] = prob
    else:
        emitDict[pos] = {word: prob}


def loadModel(infile, gPosList, transDict, emitDict):
    fdi = open(infile, 'r', encoding='utf-8')
    for line in fdi:
        infs = line.strip().split()
        if infs[0] == "pos_set":
            pos = infs[1]
            if pos != "__start__" and pos != "__end__":
                gPosList.append(pos)
        if infs[0] == "trans_prob":
            pos1 = infs[1]
            pos2 = infs[2]
            prob = float(infs[3])
            add2transDictWithProb(pos1, pos2, prob, transDict)
        if infs[0] == "emit_prob":
            pos = infs[1]
            word = infs[2]
            prob = float(infs[3])
            add2emitDictWithProb(pos, word, prob, emitDict)
    fdi.close()


def getSequence(infs):
    sequence = []
    for s in infs:
        if s.find("[") != -1:
            s = s[s.find("[") + 1:]
        if s.find("]") != -1:
            s = s[:s.find("]")]
        s = s.strip()
        if s == "": continue
        sequence.append(s)
    return sequence


def getWords(infs):
    sequence = []
    for s in infs:
        if s.find("[") != -1:
            s = s[s.find("[") + 1:]
        if s.find("]") != -1:
            s = s[:s.find("]")]
        s = s.strip()
        if s == "": continue
        sequence.append(s.split("/")[0])
    return sequence


def getEmitProb(emitDict, pos, word):
    if word in emitDict[pos]:
        return emitDict[pos][word]
    else:
        return emitDict[pos]["__NEW__"]


def predict4one(words, gPosList, transDict, emitDict, results):
    if words == []:
        return
    prePosDictList = []
    for i in range(len(words)):  # 遍历单词，相当于时间i
        prePosDict = {}
        for pos in gPosList:  # 遍历词性，即状态
            if i == 0:  # 初始时刻
                trans_prob = transDict["__start__"][pos]
                emit_prob = getEmitProb(emitDict, pos, words[i])
                total_prob = trans_prob + emit_prob  # 概率之前取了log，logA+logB = logAB
                prePosDict[pos] = [total_prob, "__start__"]
            else:
                emit_prob = getEmitProb(emitDict, pos, words[i])
                max_total_prob = -10000000.0
                max_pre_pos = ""
                for pre_pos in prePosDictList[i - 1]:  # 动态规划：全局最大->局部最大->在前一次里面找最大的
                    pre_prob = prePosDictList[i - 1][pre_pos][0]
                    trans_prob = transDict[pre_pos][pos]
                    total_prob = pre_prob + trans_prob + emit_prob
                    if max_pre_pos == "" or total_prob > max_total_prob:
                        max_total_prob = total_prob
                        max_pre_pos = pre_pos
                prePosDict[pos] = [max_total_prob, max_pre_pos]
        prePosDictList.append(prePosDict)
    max_total_prob = -10000000.0
    max_pre_pos = ""
    for pre_pos in prePosDictList[len(prePosDictList) - 1]:  # 最后一列
        pre_prob = prePosDictList[len(prePosDictList) - 1][pre_pos][0]
        trans_prob = transDict[pre_pos]["__end__"]
        total_prob = pre_prob + trans_prob
        if max_pre_pos == "" or total_prob > max_total_prob:
            max_total_prob = total_prob
            max_pre_pos = pre_pos
    posList = [max_pre_pos]  # 最优路径
    indx = len(prePosDictList) - 1
    max_pre_pos = prePosDictList[indx][max_pre_pos][1]
    indx -= 1
    while indx >= 0:
        posList.append(max_pre_pos)
        max_pre_pos = prePosDictList[indx][max_pre_pos][1]
        indx -= 1
    if len(posList) == len(words):
        posList.reverse()
        for i in range(len(posList)):
            results.append(words[i] + "/" + posList[i])


def accuracy(answer, standard):
    cnt = 0
    for i in range(len(answer)):
        if answer[i].split("/")[1] == standard[i].split("/")[1]:
            cnt += 1
    return cnt / len(answer)


def predict(infile, gPosList, transDict, emitDict, outfile):
    fdi = open(infile, 'r', encoding='utf-8')
    fdo = open(outfile, "w", encoding='utf-8')
    sequence = []
    answer = []
    for line in fdi:
        infs = line.strip().split()
        words = getWords(infs)
        results = []
        predict4one(words, gPosList, transDict, emitDict, results)
        sequence += getSequence(infs)
        answer += results
        fdo.write(" ".join(results) + "\n")
    print(accuracy(answer,sequence))
    fdo.close()
    fdi.close()


import sys
import math

infile = "./people-2014/test/0123/c1001-24200318.txt"
model_file = "model_file.txt"
outfile = "./result.txt"
gPosList = []
transDict = {}
emitDict = {}
loadModel(model_file, gPosList, transDict, emitDict)
predict(infile, gPosList, transDict, emitDict, outfile)
