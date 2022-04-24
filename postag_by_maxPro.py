# staForPosDistribution.py
import sys

def add2posDict(pos, pDict):
    if pos in pDict:
        pDict[pos] += 1
    else:
        pDict[pos] = 1


def sta(infile, pDict):
    fdi = open(infile, 'r', encoding='utf-8')
    for line in fdi:
        infs = line.strip().split()
        posList = [s.split("/")[1] for s in infs]  # 词性list
        for pos in posList:
            add2posDict(pos, pDict)  # 统计各个词性的次数
            add2posDict("all", pDict)  # 总的次数
    fdi.close()


def out(pDict):
    oList = list(pDict.items())
    oList.sort(key=lambda infs: (infs[1]), reverse=True)  # 按匿名函数排序
    total = oList[0][1]
    for pos, num in oList:
        print("%s\t%.4f" % (pos, num / total))  # 打印 词性，对应频率


try:
    infile = "./people-2014/train/0101/c1002-23995935.txt"
except:
    sys.stderr.write("\tpython " + sys.argv[0] + " infile\n")
    sys.exit(-1)
pDict = {}
sta(infile, pDict)  # 统计训练集中的语料出现频率
out(pDict)  # 打印输出


# trainByMaxProb.py
def staForWordToPosDict(infile, word2posDict):
    fdi = open(infile, 'r', encoding='utf-8')
    for line in fdi:
        infs = line.strip().split()
        for s in infs:#数据集里会有[]存在，这里不处理这些符合词性
            if s[0] == "[":
                s = s[1:]
            if s[-4:] == "]/nz":
                s = s[:-4]

            w_p = s.split("/")
            if len(w_p) == 2:
                addToDic(w_p[0], w_p[1], word2posDict)
    fdi.close()


def addToDic(word, pos, word2posDict):
    if word in word2posDict:
        if pos in word2posDict[word]:
            word2posDict[word][pos] += 1
        else:
            word2posDict[word][pos] = 1
    else:
        word2posDict[word] = {pos: 1}


def getMaxProbPos(posDict):
    total = sum(posDict.values())
    max_num = -1
    max_pos = ""
    for pos in posDict:
        if posDict[pos] > max_num:
            max_num = posDict[pos]
            max_pos = pos
    return max_pos, max_num / total


def out4model(word2posDict, model_file):
    wordNumList = [[word, sum(word2posDict[word].values())] for word in word2posDict]
    # [[word, counts]] 两重列表，单词 & 其所有词性下的频次总和
    wordNumList.sort(key=lambda infs: (infs[1]), reverse=True)  # 按counts降序
    fdo = open(model_file, "w", encoding='utf-8')
    for word, num in wordNumList:
        pos, prob = getMaxProbPos(word2posDict[word])
        # 单词可能有多个词性，出现最多的词性，及其概率(最大)
        if word != "" and pos != "":
            fdo.write("%s\t%d\t%s\t%f\n" % (word, num, pos, prob))
        # 写入文件			单词、 出现次数、出现最多的词性、该词性的概率
    fdo.close()


import os

# model_file = "./model_file.txt"
# rootdir = "D:\PycharmProjects\\nlp-beginner\people-2014\\train\\0123"
# txt_list = os.listdir(rootdir)
# word2posDict = {}
# for i in range(0, len(txt_list)):
#     path = os.path.join(rootdir, txt_list[i])
#     if os.path.isfile(path):
#         staForWordToPosDict(path, word2posDict)
#         print(path)
# out4model(word2posDict, model_file)  # 输出到文件


# predictByMaxProb.py
def loadModel(model_file, word2posDict):  # 加载训练模型
    fdi = open(model_file, 'r', encoding='utf-8')
    for line in fdi:
        infs = line.strip().split()
        if len(infs) == 4:
            word = infs[0]
            pos = infs[2]
            word2posDict[word] = pos  # 从模型读取单词，和其最大概率的词性
        else:
            sys.stderr.write("format error in " + model_file + "\n")
            sys.stderr.write(line)
            sys.exit(-1)
    fdi.close()


def getWords(infs):
    sequence=[]
    for s in infs:
        if s[0] == "[":
            s = s[1:]
        if s[-4:] == "]/nz":
            s = s[:-4]
        sequence.append(s.split("/")[0])
    return sequence


def predict(infile, word2posDict, outfile):
    fdi = open(infile, 'r', encoding='utf-8')
    fdo = open(outfile, 'w', encoding='utf-8')
    for line in fdi:
        infs = line.strip().split()
        # 盖住答案，闭卷考试
        words = getWords(infs)  # 只获取输入文件的单词
        results = []
        for word in words:
            if word in word2posDict:  # 从模型中获取它的最大概率词性
                results.append(word + "/" + word2posDict[word])
            else:
                results.append(word + "/unknown")
        fdo.write(" ".join(results) + "\n")  # 写入输出文件
    fdo.close()
    fdi.close()


import sys

infile = "./people-2014/test/0123/c1001-24200318.txt"
model_file = "./model_file.txt"
outfile = "./result.txt"

word2posDict = {}
loadModel(model_file, word2posDict)  # 加载训练模型
predict(infile, word2posDict, outfile)  # 输出


