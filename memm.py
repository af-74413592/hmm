import pickle
from tqdm import tqdm
import numpy as np
import os

def make_label(text_str):
    text_len = len(text_str)
    if text_len == 1:
        return "S"
    else:
        return "B" + "M" * (text_len -2) + "E"

def text_to_state(file="all_train_text.txt"):

    if os.path.exists("all_train_state.txt"):
        return
    input_file= open(file,'r',encoding= "utf-8")
    all_data = input_file.read().split("\n")
    with open("all_train_state.txt",'w',encoding="utf-8") as f2:
        for d_index,data in tqdm(enumerate(all_data)):
            if data:
                state_ = ""
                for w in data.split(" "):
                    if w:
                        state_ = state_ + make_label(w) + " "
                if d_index != len(all_data) - 1:
                    state_ = state_.strip() + "\n"
                f2.write(state_)
    input_file.close()

class MEMM():
    def __init__(self,file_text = "all_train_text.txt",file_state = "all_train_state.txt"):
        self.all_states = open(file_state, "r", encoding="utf-8").read().split("\n")
        self.all_texts = open(file_text, "r", encoding="utf-8").read().split("\n")
        self.states_to_index = {"B": 0, "M": 1, "S": 2, "E": 3}
        self.index_to_states = ["B", "M", "S", "E"]
        self.len_states = len(self.states_to_index)

        self.init_matrix = np.zeros((self.len_states))
        #不需要转换矩阵
        #self.transfer_matrix = np.zeros((self.len_states, self.len_states))

        # 发射矩阵, 使用的3级 字典嵌套
        self.emit_matrix = {"B": {"B":{},"M":{},"S":{},"E":{},"start":{}}, "M": {"B":{},"M":{},"S":{},"E":{},"start":{}}, "S": {"B":{},"M":{},"S":{},"E":{},"start":{}}, "E": {"B":{},"M":{},"S":{},"E":{},"start":{}}}

    def cal_init_matrix(self, state):
        self.init_matrix[self.states_to_index[state[0]]] += 1 # BMSE 四种状态, 对应状态出现 1次 就 +1

    # 计算转移矩阵(不需要)
    # def cal_transfer_matrix(self, states):
    #     sta_join = "".join(states)        # 状态转移 从当前状态转移到后一状态, 即 从 sta1 每一元素转移到 sta2 中
    #     sta1 = sta_join[:-1]
    #     sta2 = sta_join[1:]
    #     for s1, s2 in zip(sta1, sta2):   # 同时遍历 s1 , s2
    #         self.transfer_matrix[self.states_to_index[s1],self.states_to_index[s2]] += 1

    # 计算发射矩阵
    # 计算给定标签下，观察值概率矩阵观察值是<St,Ot+1>而不是HMM的<Ot+1>
    def cal_emit_matrix(self,word_0, words, state_0, states):
        last_state = state_0[0]
        for index,(word, state) in enumerate(zip("".join(words), "".join(states))):
            if index == 0:
                self.emit_matrix[state_0[0]]["start"][word_0[0]] = self.emit_matrix[state_0[0]]["start"].get(word_0[0],0) + 1
                self.emit_matrix[state_0[0]]["start"]["total"] = self.emit_matrix[state_0[0]]["start"].get("total",0) + 1
            else:
                self.emit_matrix[state][last_state][word] = self.emit_matrix[state][last_state].get(word,0) + 1
                self.emit_matrix[state][last_state]["total"] = self.emit_matrix[state][last_state].get("total",0) + 1
                last_state = state

    # 将矩阵归一化
    def normalize(self):
        self.init_matrix = self.init_matrix/np.sum(self.init_matrix)
        #self.transfer_matrix = self.transfer_matrix/np.sum(self.transfer_matrix,axis = 1,keepdims = True)
        for state0,dict0 in self.emit_matrix.items():
            for state1,dict1 in dict0.items():
                for word,t in dict1.items():
                    if word!= "total":
                        self.emit_matrix[state0][state1][word] = t/dict1["total"]*1000

    def train(self):
        for words, states in tqdm(zip(self.all_texts, self.all_states)):  # 按行读取文件, 调用3个矩阵的求解函数
            words = words.split(" ")  # 在文件中 都是按照空格切分的
            states = states.split(" ")
            self.cal_init_matrix(states[0])  # 计算三大矩阵
            #self.cal_transfer_matrix(states)
            self.cal_emit_matrix(words[0], words, states[0], states)
        self.normalize()  # 矩阵求完之后进行归一化

def viterbi(text, memm):
    states = memm.index_to_states
    emit_p = memm.emit_matrix
    #trans_p = memm.transfer_matrix
    start_p = memm.init_matrix
    V = []  #[{}]里面存储每一层tag和对应的最大概率值
    start_path = {}
    for y in states:
        neverSeen = text[0] not in emit_p[y]["start"]
        if not neverSeen:
            start_path[y] = start_p[memm.states_to_index[y]] * emit_p[y]["start"][text[0]]
        else:
            start_path[y] = 0
    V.append(start_path)
    path = []  # 存储每一层的tag最大概率值对应的前一个tag
    pre_key = ""
    for i in range(1, len(text)):
        next_dict = {}
        new_path = {}
        for state in states:
            temp = []
            max = 0
            for key in V[i - 1].keys():
                neverSeen = text[i] not in emit_p[state][key].keys()
                if not neverSeen:
                    value = V[i - 1][key] * emit_p[state][key][text[i]]
                else:
                    value = 0
                if value > max:
                    max = value
                    pre_key = key
                temp.append(value)
            next_dict[state] = max
            new_path[state] = pre_key
        path.append(new_path)
        V.append(next_dict)
    # 寻找路径
    max = 0
    end = ""
    for key in V[-1].keys():
        if V[-1][key] > max:
            end = key
    result = []
    result.append(end)
    for i in range(len(V) - 2, -1, -1):
        for key in path[i].keys():
            if key == result[len(V) - i - 2]:
                result.append(path[i][key])
    result.reverse()
    print(result)

if __name__ == '__main__':
    memm = MEMM()
    memm.train()
    test_str = u"今天天气不错"
    viterbi(test_str,memm)