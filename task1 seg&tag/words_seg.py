# 分词工具
STATES = {'B', 'M', 'E', 'S'}

def get_tags(src):
    '''
    :param src: 传入短语  eg.我们 一起 吃饭
    :return:
    '''
    tags = []
    if len(src) == 1:
        tags = ['S']
    elif len(src) == 2:
        tags = ['B', 'E']
    else:
        tags.append('B')
        tags.extend(['M'] * (len(src) - 2))
        tags.append('E')
    return tags


def stateToToken(state, path, sequence):
    fenci = ''
    for i in range(len(path[state])):
        j = path[state][i]
        if j == 'B':
            fenci = fenci + sequence[i]
        else:
            if j == 'M':
                fenci = fenci + sequence[i]
            else:
                fenci = fenci + sequence[i] + ' '
    return fenci


class HMM:
    def __init__(self):
        self.trans_mat = {}
        self.emit_mat = {}
        self.init_vec = {}
        self.state_count = {}
        self.states = STATES
        self.setup()

    def setup(self):
        for state in self.states:
            self.trans_mat[state] = {}
            for next_state in self.states:
                self.trans_mat[state][next_state] = 0.0
            self.emit_mat[state] = {}
            self.init_vec[state] = 0
            self.state_count[state] = 0

    def load_data(self, filename):
        lines = open(filename, 'r', encoding='utf-8').readlines()
        seg_stop_words = {" ", "，", "。", "“", "”", '“', "？", "！", "：", "《", "》", "、", "；", "·", "‘ ", "’",
                          "──", ",", ".", "?", "!", "`", "~", "@", "#", "$", "%", "^", "&", "*", "(", ")", "-",
                          "_", "+", "=", "[", "]", "{", "}", '"', "'", "<", ">", "\\", "|" "\r", "\n", "\t"}
        for line in lines:
            line = line.strip()
            if not line: continue

            # 中文合集
            observes = []
            for i in range(len(line)):
                if line[i] not in seg_stop_words:
                    observes.append(line[i])

            # 中文对应的BMES合集
            states = []
            for word in line.split("  "):
                if word not in seg_stop_words:
                    states.extend(get_tags(word))
            if (len(observes) >= len(states)):
                self.training(observes,states)

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
        # 初始概率分布：统计所有训练样本中分别以状态S、B、M、E为初始状态的样本的数量，之后分别除以训练词语总数
        for vec in self.init_vec:
            init_vec_prob[vec] = self.init_vec[vec] / sum(self.init_vec.values())

        # 状态转移概率分布： 从状态S转移到B的出现次数，再除以S出现的总次数
        for key1 in self.trans_mat:
            trans_mat_prob[key1] = {}
            for key2 in self.trans_mat[key1]:
                trans_mat_prob[key1][key2] = self.trans_mat[key1][key2] / (
                    self.state_count[key1] if self.state_count[key1] != 0 else max(self.state_count.values()))
        # 观察概率：状态为j并观测为k的频数，除以训练数据中状态j出现的次数，其他同理。
        for key1 in self.emit_mat:
            emit_mat_prob[key1] = {}
            for key2 in self.emit_mat[key1]:
                emit_mat_prob[key1][key2] = self.emit_mat[key1][key2] / (
                    self.state_count[key1] if self.state_count[key1] != 0 else max(self.state_count.values()))
        return init_vec_prob, trans_mat_prob, emit_mat_prob

    def viterbi(self, sequence, DEFAULT):
        tab = [{}]
        path = {}

        init_vec, trans_mat, emit_mat = self.get_prob()
        # init
        for state in self.states:
            tab[0][state] = init_vec[state] * emit_mat[state].get(sequence[0],DEFAULT)
            path[state] = [state]

        # dp
        for t in range(1, len(sequence)):
            tab.append({})
            new_path = {}
            for state1 in STATES:
                items = []
                for state2 in STATES:  # state2为前一个
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

    def do_predict(self,text):
        seg_stop_words = {" ", "，", "。", "“", "”", '“', "？", "！", "：", "《", "》", "、", "；", "·", "‘ ", "’",
                          "──", ",", ".", "?", "!", "`", "~", "@", "#", "$", "%", "^", "&", "*", "(", ")", "-",
                          "_", "+", "=", "[", "]", "{", "}", '"', "'", "<", ">", "\\", "|" "\r", "\n", "\t"}

        text_token=""
        start=0
        for i in range(len(text)):
            if text[i] in seg_stop_words:
                _,state,path=self.viterbi(text[start:i],DEFAULT=0.000001)
                text_token+=stateToToken(state, path, text[start:i])+text[i]+" "
                start=i+1
        return text_token

    def printHMM(self):
        print(self.get_prob()[0])
        print(self.get_prob()[1])


if __name__ == '__main__':
    sequence = '民族复兴迫切需要培养造就德才兼备的人才。希望你们继续发扬严谨治学、甘为人梯的精神，坚持特色、争创一流，培养更多具有为国奉献钢筋铁骨的高素质人才，促进钢铁产业创新发展、绿色低碳发展，为铸就科技强国、制造强国的钢铁脊梁作出新的更大的贡献！'
    training_file = 'pku_training.utf8'
    hmm = HMM()
    hmm.load_data(training_file)
    result= hmm.do_predict(sequence)
    print(result)


import jieba

seg_list = jieba.cut(sequence, cut_all=False)
standard_answer=" ".join(seg_list)  # 全模式


def to_region(segment_str):
    # input : 就读 于 中国人民大学    # return：[(0,1),(2,2),(3,8)] 闭区间
    region = []
    start = 0
    for word in segment_str.split(" "):
        end = start + len(word)
        region.append((start, end - 1))
        start = end
    return region

def prf(gold: str, pred: str):
    """
    计算P、R、F1
    :param gold: 标准答案文件，比如“商品 和 服务”
    :param pred: 分词结果文件，比如“商品 和服 务”
    :param dic: 词典
    :return: (P, R, F1, OOV_R, IV_R)
    """
    A_size, B_size, A_cap_B_size, OOV, IV, OOV_R, IV_R = 0, 0, 0, 0, 0, 0, 0
    A, B = set(to_region(gold)), set(to_region(pred))
    A_size += len(A)
    B_size += len(B)
    A_cap_B_size += len(A & B)
    p, r = A_cap_B_size / B_size * 100, A_cap_B_size / A_size * 100
    return p, r, 2 * p * r / (p + r)


print(standard_answer)

print(to_region(result))
print(to_region(standard_answer))

print(prf(result,standard_answer))