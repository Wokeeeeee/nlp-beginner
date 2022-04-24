# HMM实现中文分词和词性标注

分词：word_seg.py

词性标注：postag_by_hmm.py

## 目的

分词和词性标注

1.使用分词工具(jieba等等)或者自己实现课程讲的任意分词和词性标注方法进行数据集的分词和词性标注实验；

2.对测试集分别进行分词和词性标注性能评估，评估指标至少包括准确率，召回率，F-测度；

## 原理

### HMM

时序的概率模型，描述由一个隐藏的马尔可夫链接随机生成不可观察的状态序列，再有状态序列生成一个观测序列的过程。

#### *μ*(*A*,*B*,*π*)

- 初始状态概率向量：句子的第一个字属于BEMS四种状态的概率
- 状态转移概率矩阵：如果前一个字的位置是B，那么后一个字位置为BEMS的概率各是多少？
- 观测概率矩阵： 在状态b条件下，观察值为耀的概率



#### 三类基本问题

1. 给定模型*μ*=(*A*,*B*,*π*)，计算某个观察序列*O=o1o2 …… oT*的概率*P*(*O* | *μ*)=?--->向前算法
2. 给定模型*μ*=(*A*,*B*,π)和观察序列*O=o1o2 …… oT* ，如何有效地确定一个状态序列*Q=q1q2 …… qT* ，以便最好地解释观察序列？arg max⁡ P(Q | O)=？---> Viterbi算法
3. 给定一个观察序列*O=o1o2 …… oT* ，如何找到一个能够最好地解释这个观察序列的模型，即如何调节模型参数*μ*=(*A*,*B*,π)，使得*P*(*O*| *μ*)最大化？--->Baum-Welch算法



#### Viterbi

分词和词性标注都属于问题2，采用viterbi算法，用动态规划去求解最优路径。

![img](https://le1pgsvahu.feishu.cn/space/api/box/stream/download/asynccode/?code=Yjg0MDJlYzcyMjFlYjUwZjg5N2VkZGExNGEzOTEyNWZfeEFEdm9aUXhoMVh5aTFJUEtBVEt4cmNERUpwUmFNeDNfVG9rZW46Ym94Y25vNlNPcW9WdHNtQUxRNzdWamNhRHEyXzE2NTA3OTI0Mjk6MTY1MDc5NjAyOV9WNA)



如果从起点A经过P、H到达终点G是一条最短路径，那么，由A出发经过P到达H所走的这条子路径，对于从A出发到H的所有可能的路径来说，必定也是**最短路径**

全局最优→局部最优

局部最优是全局最优的必要条件



## 实验内容

### 中文分词

#### BMES

汉语句子作为输入，BEMS序列串作为输出

- B->汉语的起始词
- M->中间字
- E->结束词
- S->单字成词



#### 基于HMM的中文分词

1. ##### 导入已知分好的词序，训练HMM的三个矩阵。

初始状态概率矩阵：统计所有训练样本中分别以状态S、B、M、E为初始状态的样本的数量，之后分别除以训练词语总数，就可以得到初始概率分布。

状态转移概率矩阵：统计所有样本中，从状态S转移到B的出现次数，再除以S出现的总次数，便得到由S转移到B的转移概率，其他同理。

观测概率矩阵：统计训练数据中，状态为j并观测为k的频数，除以训练数据中状态j出现的次数，其他同理。



先统计频数，然后得到概率。

```Python
def training(self, observes, states):
    for i in range(len(states)):
        if i == 0:
            self.init_vec[states[0]] += 1
            self.state_count[states[0]] += 1
        else:
            self.trans_mat[states[i - 1]][states[i]] += 1
            self.state_count[states[i]] += 1
        if observes[i] not in self.emit_mat[states[i]]:
            self.emit_mat[states[i]][observes[i]] = 1
        else:
            self.emit_mat[states[i]][observes[i]] += 1
            
            

def get_prob(self):
    init_vec_prob = {}
    trans_mat_prob = {}
    emit_mat_prob = {}
    # 初始概率分布
    for vec in self.init_vec:
        init_vec_prob[vec] = self.init_vec[vec] / sum(self.init_vec.values())

    # 状态转移概率分布
    for key1 in self.trans_mat:
        trans_mat_prob[key1] = {}
        for key2 in self.trans_mat[key1]:
            trans_mat_prob[key1][key2] = self.trans_mat[key1][key2] / (
                self.state_count[key1] if self.state_count[key1] != 0 else max(self.state_count.values()))
    # 观察概率
    for key1 in self.emit_mat:
        emit_mat_prob[key1] = {}
        for key2 in self.emit_mat[key1]:
            emit_mat_prob[key1][key2] = self.emit_mat[key1][key2] / (
                self.state_count[key1] if self.state_count[key1] != 0 else max(self.state_count.values()))
    return init_vec_prob, trans_mat_prob, emit_mat_prob
```





1. ##### 将HMM模型的三个矩阵带入viterbi算法

第二类问题：

给定模型*μ=*(*A,B,π*)和观察序列*O=o1o2 …… oT* ，如何确定最优的状态序列*Q=q1q2 …… qT* 



###### 最优路径求解

**最优问题：**

如果从起点A经过P、H到达终点G是一条最短路径，那么，由A出发经过P到达H所走的这条子路径，对于从A出发到H的所有可能的路径来说，必定也是最短路径

**动态规划：**

- 在每个节点中仅存储从起点到当前节点的最优路径
- 每增加一个节点，都把它跟各个前驱节点的最优路径连接起来，找出连接后的最优路径，仅把它存储起来



###### viterbi算法步骤

![img](https://le1pgsvahu.feishu.cn/space/api/box/stream/download/asynccode/?code=MzU0ZDI0M2RmODE4Y2JjZDc1OTNkMDNlZDU5YjgyNjhfNTBoSFJxODNMaElaMnRkdHRBUGY3SVl6WngzNmNGMTFfVG9rZW46Ym94Y25BeWFlclAxNWRIMzlGSnVhamdLYXRoXzE2NTA3OTI0Mjk6MTY1MDc5NjAyOV9WNA)

```Apache
def viterbi(self, sequence, DEFAULT):
    tab = [{}]
    path = {}

    init_vec, trans_mat, emit_mat = self.get_prob()
    # init
    for state in self.states:
        tab[0][state] = init_vec[state] * emit_mat[state].get(sequence[0])
        path[state] = [state]

    # dp
    for t in range(1, len(sequence)):
        tab.append({})
        new_path = {}
        for state1 in STATES:
            items = []
            for state2 in STATES:  # state2为后一个
                if tab[t - 1][state2] == 0:
                    continue
                else:
                    prob = tab[t - 1][state2] * trans_mat[state2].get(state1, DEFAULT) * emit_mat[state1].get(sequence[t], DEFAULT)
                    items.append((prob, state2))
            best = max(items)
            tab[t][state1] = best[0]
            new_path[state1] = path[best[1]] + [state1]
        path = new_path
    prob, state = max([(tab[len(sequence) - 1][state], state) for state in STATES])
    return prob, state, path
```





#### NLP中的评价指标


$$
Precision=\frac{|A\cap B|}{|B|}\;
$$
  
$$
Recall=\frac{|A\cap B|}{|A|}
$$

$$
F-score=\frac{2 \times Precision \times Recall}{Precision+Recall}
$$



```Python
def prf(gold: str, pred: str):
    A_size, B_size, A_cap_B_size, OOV, IV, OOV_R, IV_R = 0, 0, 0, 0, 0, 0, 0
    A, B = set(to_region(gold)), set(to_region(pred))
    A_size += len(A)
    B_size += len(B)
    A_cap_B_size += len(A & B)
    p, r = A_cap_B_size / B_size * 100, A_cap_B_size / A_size * 100
    return p, r, 2 * p * r / (p + r)
```



#### 结果

##### 参数估计

训练得到的初始矩阵和状态转移矩阵概率.



| trans_mat_prob |         |         |         |         |
| -------------- | ------- | ------- | ------- | ------- |
|                | S       | M       | B       | E       |
| S              | 0.38441 | 0.0     | 0.60212 | 0.0     |
| M              | 0.0     | 0.31506 | 0.0     | 0.68493 |
| B              | 0.0     | 0.14315 | 0.0     | 0.85684 |
| E              | 0.37346 | 0.0     | 0.60165 | 0.0     |



| init_vec_prob |          |
| ------------- | -------- |
| S             | 0.346124 |
| M             | 0        |
| B             | 0.653875 |
| E             | 0        |



##### 分词序列

> 民族 复兴 迫切 需要 培养造 就 德才 兼备 的 人才 。 希望 你们 继续 发扬 严谨 治学 、 甘为 人梯 的 精神 ， 坚持 特色 、 争创 一流 ， 培养 更 多 具有 为国 奉献 钢筋 铁骨 的 高 素质 人才 ， 促进 钢铁 产业 创新 发展 、 绿色 低碳 发展 ， 为 铸 就 科技 强国 、 制造 强国 的 钢铁 脊梁 作出 新 的 更 大 的 贡献 ！



##### 指标

jieba分词结果作为标准

| P    | 85.93 |
| ---- | ----- |
| R    | 76.39 |
| F1   | 80.88 |





### 基于HMM的中文词性标注

数据集为课程提供的people-2014文件夹。在实际操作过程中，没有考虑符合词性的参数估计和词性标注。例如[人民/n 生活/vn 水平/n]/nz 只考虑了人民/n 生活/vn 水平/n，没有考虑人民生活水平\nz。

也尝试了基于最大概率的标注方法，但效果不佳，且不能应对新词出现的问题，标注会产生\unknown。



#### 训练参数矩阵

采用people-2014/train/0123文件夹下的所有txt训练，这里用了一个简单的遍历问价夹下的所有txt文件。



统计状态转移的频数，这里采用了"__start__"和"__end__"关键词来统计首尾位置上出现某词语的概率。由此，初始矩阵可以和概率转移矩阵合并。

```Python
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
```

在导出模型参数阶段将统计的频数转换为概率的对数，之后运算过程中求概率相乘即求对数相加。



#### HMM标注

测试集采用的people-2014/0123/c1001-24200318.txt

和HMM分词思路相同，通过动态规划的方式寻找最优路径，从前一列的最优路径上继续延申下一词的最优路径。

```Prolog
def predict4one(words, gPosList, transDict, emitDict, results):
    if words == []:
        return
    prePosDictList = []
    for i in range(len(words)):  # 遍历单词
        prePosDict = {}
        for pos in gPosList:  # 遍历词性
            if i == 0:  # 初始矩阵
                trans_prob = transDict["__start__"][pos]
                emit_prob = getEmitProb(emitDict, pos, words[i])
                total_prob = trans_prob + emit_prob  
                prePosDict[pos] = [total_prob, "__start__"]
            else:
                emit_prob = getEmitProb(emitDict, pos, words[i])
                max_total_prob = -10000000.0
                max_pre_pos = ""
                for pre_pos in prePosDictList[i - 1]:  
# 动态规划：全局最大->局部最大->在前一次里面找最大的
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
```



#### 结果和分析

截取的一段分词结果：

> 本报/rz 北京/ns 1月22日/t 电/n （/w 记者/nnt 朱剑红/nr ）/w 我国/n “/w 十二五/m ”/w 规划/n 提出/v 的/ude1 24/m 个/q 主要/b 指标/n ，/w 绝大部分/m 的/ude1 实施/vn 进度/n 好/a 于/p 预期/vn ，/w 氮氧化物/n 排放/vn 总量/n 减少/v 、/w 化石/n 能源/n 占/v 一次/mq 能源/n 消费/vn 比重/n 、/w 单位/n GDP/x 能源/n 消耗/v 降低/vn 、/w 单位/n GDP/x 二氧化碳/n 排放/vn 降低/vn 等/udeng 四/m 个/q 指标/n 完成/v 的/ude1 进度/n 滞后/v 于/p 预期/vn 。/w 这/rzv 是/vshi 国家/n 发改委/nis 有关/vn 负责人/nnt 今天/t 在/p “/w 宏观/n 经济/n 形势/n 和/cc 政策/n ”/w 新闻/n 发布会/n 上/f 透露/v 的/ude1 。/w 国家/n 发改委/nis 已/d 组织/n 有关/vn 部门/n 和/cc 有关方面/nz 对/p “/w 十二五/m ”/w 规划/n 的/ude1 实施/vn 情况/n 进行/vn 了/ule 全面/ad 的/ude1 分析/vn 和/cc 评估/vn ，/w 评估/vn 报告/n 在/p 修改/v 完成/vn 后/f 将/d 对外/vn 公开/ad 。/w

> 在/p 发展/vn 目标/n 方面/n ，/w “/w 十二五/m ”/w 规划/n 提出/v 来/vf 的/ude1 GDP/x 增长/v 预期/vn 性/ng 指标/n 是/vshi 年均/v 增长/v 7%/m ，/w 在/p 过去/vf 三年/t 里/f ，/w 分别/d 实现/v 了/ule 9.2%/m 、/w 7.7%/m 和/cc 7.7%/m ，/w 完成/v 目标/n 没有/v 任何/rz 问题/n 。/w 其他/rzv 指标/n 有/vyou 两个/mq 已经/d 提前/vd 完成/v ，/w 一/d 是/vshi 每/rz 万/d 人/n 发明/v 专利/n 拥有量/n ，/w 二/m 是/vshi 森林/n 蓄积量/nz 。/w

准确率为0.97440585