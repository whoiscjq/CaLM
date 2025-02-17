import json
from ananke.graphs import DAG
from ananke.identification import OneLineID
from ananke.models import binary_nested
from ananke.models.binary_nested import initialize_q_vector, permutations
from ananke.identification.opt_adjust import OptAdjustment
import random
import math
from collections import OrderedDict
import pandas as pd
import networkx as nx
from dowhy.causal_identifier.auto_identifier import (AutoIdentifier, 
    BackdoorAdjustment, EstimandType, identify_ate_effect, identify_frontdoor, 
    identify_nde_effect, identify_mediation)
from dowhy.causal_graph import CausalGraph
from enum import Enum
import sys
import string
import os
from copy import deepcopy
import argparse


random.seed(1023)

DEFAULT_NODE_VALUE = ["low", "high"]
DEFAULT_NODE_VALUE_CN = ["低", "高"]
DEFAULT_NODE_NAME_LENGTH = 4
RANDOM_NODE_NAME = [
    "sales performance",
    "income level",
    "education level",
    "appearance",
    "talent",
    "physical health",
    "blood pressure",
    "rainfall",
    "temperature",
    "employee performance",
    "market demand",
    "air pressure",
    "traffic congestion",
    "wheather condition",
    "amount of exercise",
    "government policies",
    "job satisfaction",
    "humidity level",
    "quality of teaching",
    "mental health",
    "work-life balance",
    "severity of respiratory illnesses",
    "stress level"
]
RANDOM_NODE_NAME_CN = [
    "销售表现",
    "收入水平",
    "教育水平",
    "外貌水平",
    "天赋水平",
    "身体健康水平",
    "血压",
    "降雨量",
    "温度",
    "雇员表现水平",
    "市场需求水平",
    "气压",
    "交通拥堵水平",
    "天气状况",
    "运动量",
    "政府政策",
    "工作满意度",
    "湿度水平",
    "教学质量水平",
    "精神健康水平",
    "工作生活平衡水平",
    "呼吸系统疾病严重程度",
    "压力水平"
]

class Mode(Enum):
    REAL = 1
    RANDOM = 2
    FAKE = 3

class TaskType(Enum):
    ATE = 1
    CDE = 2
    NDE = 3
    NIE = 4
    PN = 5
    PS = 6
    ETT = 7
    BAS = 8  # backdoor adjustment set
    CB = 9  # collider bias

def count_words(x):
    return len(x.split(' '))


class Causal_Question:
    def __init__(self, 
            nodes, 
            edges, 
            mode=Mode.RANDOM,
            node_name=None,
            node_value=None,
            edge_sign=None,
            thresh=0.001,
            sample_num=50000,
            max_adjust_set=1,
            node_name_CN=None,
            node_value_CN=None,
            with_CN=False,
            max_edge_num=10
        ):
        self.max_edge_num = max_edge_num
        self.with_CN = with_CN
        self.nodes = nodes
        self.edges = edges
        self.mode = mode
        self.sample_num = sample_num
        self.thresh = thresh
        self.max_adjust_set = max_adjust_set
        self._gen_graph_params(node_name, node_value, edge_sign, node_name_CN, node_value_CN)  # 赋予节点含义，生成SCM参数
        self._construct_graph()  # 构造图
        self._init_identifier()  # 初始化identifier
        self._gen_scm_description()  # 生成两种SCM描述
        # print(self.function_description)
        # print(self.text_description)
        self._sample_data()  # 采样数据
        # print(self.data)
       
    
    @staticmethod
    def sigmoid(x):
        if x >= 0:
            return 1 / (1 + math.exp(-x))
        else:
            tmp = math.exp(x)
            return tmp / (tmp + 1)

    @staticmethod
    def gen_random_string(N):
        return "".join(random.choice(string.ascii_lowercase) for _ in range(N))

    @staticmethod
    def make_str_from_list(strs, CN_flag=False):
        x = ", ".join(strs)
        index = x.rfind(",")
        if index > 0:
            if CN_flag:
                return x[:index] + "且" + x[index+2:]
            else:
                return x[:index] + " and" + x[index+1:]
        else:
            return x

    def _gen_cond_prob_description(self, cond_dict, out_dict, number):
        out_str = self.make_str_from_list([f"{self.node_name[t]} being {self.node_value[t][out_dict[t]]} ({t}={out_dict[t]})" for t in out_dict])
        if cond_dict:
            cond_str = self.make_str_from_list([f"{self.node_name[t]} being {self.node_value[t][cond_dict[t]]} ({t}={cond_dict[t]})" for t in cond_dict])
            return f"For those with {cond_str}, the probability of {out_str} is {number:.4f}. "
        else:
            return f"The probability of {out_str} is {number:.4f}. "

    def _gen_cond_prob_math(self, cond_dict, out_dict, number):
        out_str = ",".join([f"{key}={out_dict[key]}" for key in out_dict])
        if cond_dict:
            cond_str = ",".join([f"{key}={cond_dict[key]}" for key in cond_dict])
            return f"P({out_str}|{cond_str})={number:.4f}; "
        else:
            return f"P({out_str})={number:.4f}; "

    def _gen_cond_prob_description_CN(self, cond_dict, out_dict, number):
        out_str = self.make_str_from_list(
            [f"{self.node_name_CN[t]}为{self.node_value_CN[t][out_dict[t]]}({t}={out_dict[t]})" for t in out_dict],
            CN_flag=True
        )
        if cond_dict:
            cond_str = self.make_str_from_list(
                [f"{self.node_name_CN[t]}为{self.node_value_CN[t][cond_dict[t]]}({t}={cond_dict[t]})" for t in cond_dict],
                CN_flag=True
            )
            return f"在{cond_str}的条件下, {out_str}的概率为{number:.4f}。"
        else:
            return f"{out_str}的概率为{number:.4f}。"

    def _gen_graph_params(self, node_name, node_value, edge_sign, node_name_CN, node_value_CN):
        if self.mode == Mode.REAL:
            assert node_name is not None and node_value is not None and edge_sign is not None
            self.node_name = node_name
            self.node_value = node_value
            self.edge_sign = edge_sign
            if self.with_CN:
                assert node_name_CN is not None and node_value_CN is not None
                self.node_name_CN = node_name_CN
                self.node_value_CN = node_value_CN
                for key in self.node_value_CN:
                    while self.node_value_CN[key][0][-1] == "的":
                        self.node_value_CN[key][0] = self.node_value_CN[key][0][:-1]
                    while self.node_value_CN[key][1][-1] == "的":
                        self.node_value_CN[key][1] = self.node_value_CN[key][1][:-1]
                # print(self.node_name_CN, self.node_value_CN)
        elif self.mode == Mode.FAKE:
            self.node_name = {}
            self.node_value = {}
            for node in self.nodes:
                self.node_name[node] = self.gen_random_string(DEFAULT_NODE_NAME_LENGTH)
                self.node_value[node] = DEFAULT_NODE_VALUE
            self.edge_sign = {}
            for edge in self.edges:
                edge_sign[edge] = random.choice([1, -1])
            if self.with_CN:
                self.node_name_CN = deepcopy(self.node_name)
                self.node_value_CN = {}
                for node in self.nodes:
                    self.node_value_CN[node] = DEFAULT_NODE_VALUE_CN
        elif self.mode == Mode.RANDOM:
            self.node_name = {}
            self.node_value = {}
            select_indexes = random.sample(list(range(len(RANDOM_NODE_NAME))), len(self.nodes))
            select_names = [RANDOM_NODE_NAME[index] for index in select_indexes]
            for i, node in enumerate(self.nodes):
                self.node_name[node] = select_names[i]
                self.node_value[node] = DEFAULT_NODE_VALUE
            self.edge_sign = {}
            for edge in self.edges:
                edge_sign[edge] = random.choice([1, -1])
            if self.with_CN:
                self.node_name_CN = {}
                self.node_value_CN = {}
                select_names_CN = [RANDOM_NODE_NAME_CN[index] for index in select_indexes]
                for i, node in enumerate(self.nodes):
                    self.node_name_CN[node] = select_names_CN[i]
                    self.node_value_CN[node] = DEFAULT_NODE_VALUE_CN
        else:
            assert False, f"unknown mode: {mode}"
        
        self.node_bias = {}
        for node in self.nodes:
            self.node_bias[node] = random.gauss(0, 1)
        self.edge_weight = {}
        for edge in self.edges:
            self.edge_weight[edge] = abs(random.gauss(0, 1) + 1) * edge_sign[edge]

    def _construct_graph(self):
        edges = []
        for e in self.edges:
            x, y = e.split("->")
            edges.append((x, y))
        assert len(edges) <= self.max_edge_num

        # ananke graph
        self.graph = DAG(vertices=self.nodes, di_edges=edges)
        # networkx graph
        self.nx_graph = nx.DiGraph()
        self.nx_graph.add_nodes_from(self.nodes)
        self.nx_graph.add_edges_from(edges)
        # dowhy graph
        self.dowhy_graph = CausalGraph(
            treatment_name=None,
            outcome_name=None,
            graph="\n".join(nx.generate_gml(self.nx_graph)),
            observed_node_names=self.nodes
        )
    
    def _init_identifier(self):
        self.autoId_exhaustive = AutoIdentifier(
            estimand_type=EstimandType.NONPARAMETRIC_ATE,
            backdoor_adjustment=BackdoorAdjustment.BACKDOOR_EXHAUSTIVE
        )
        self.autoId_min = AutoIdentifier(
            estimand_type=EstimandType.NONPARAMETRIC_ATE,
            backdoor_adjustment=BackdoorAdjustment.BACKDOOR_MIN
        )
        self.autoId_max = AutoIdentifier(
            estimand_type=EstimandType.NONPARAMETRIC_ATE,
            backdoor_adjustment=BackdoorAdjustment.BACKDOOR_MAX
        )

    def _gen_scm_description(self):
        function_description = ""
        text_description = ""
        for node in self.nodes:
            parents = list(self.graph.parents(node))
            if not parents:
                p = self.sigmoid(self.node_bias[node])
                text_description += self._gen_cond_prob_description(None, {node: 1}, p) + "\n"
                function_description += f"P({node}=1)={p:.4f}\n"
            else:
                p_str = ",".join(parents)
                w_str = " + ".join([f"{self.edge_weight[p+'->'+node]:.4f} * {p}" for p in parents])
                function_description += f"P({node}=1|{p_str})=sigmoid({self.node_bias[node]} + {w_str})\n"

                all_parents_values = [dict(zip(parents, x)) for x in permutations(len(parents), 2)]
                for parents_value in all_parents_values:
                    s = self.node_bias[node]
                    for p in parents_value:
                        if parents_value[p] == 1:
                            s += self.edge_weight[p+"->"+node]
                    p = self.sigmoid(s)
                    text_description += self._gen_cond_prob_description(parents_value, {node: 1}, p) + "\n"
        self.function_description = function_description
        self.text_description = text_description

    def _sample_data(self):
        Stats = {}
        for _ in range(self.sample_num):
            parent_count = {}
            queue = []
            result = {}
            for node in self.nodes:
                parent_count[node] = len(self.graph.parents(node))
                if parent_count[node] == 0:
                    queue.append(node)
            while queue:
                node = queue[0]
                del queue[0]
                for child in self.graph.children(node):
                    parent_count[child] -= 1
                    if parent_count[child] == 0:
                        queue.append(child)
                s = self.node_bias[node]
                for parent in self.graph.parents(node):
                    if result[parent] == 1:
                        s += self.edge_weight[parent + "->" + node]
                p = self.sigmoid(s)
                if random.uniform(0, 1) < p:
                    result[node] = 1
                else:
                    result[node] = 0

            tmp = "".join([str(result[node]) for node in self.nodes])
            if tmp not in Stats:
                Stats[tmp] = 1
            else:
                Stats[tmp] += 1
        self.data = []
        for key in Stats:
            tmp = {}
            for i, char in enumerate(key):
                tmp[self.nodes[i]] = int(char)
            tmp["count"] = Stats[key]
            self.data.append(tmp)

    def _Count(self, values):
        count = 0
        total = 0
        for item in self.data:
            total += item["count"]
            flag = True
            for key in values:
                if item[key] != values[key]:
                    flag = False
                    break
            if flag:
                count += item["count"]
        return count / total

    def do_calculas(self, t, tv, o, ov, CN_flag=False):
        o_str, t_str = f"{o}={ov}", f"{t}={tv}"
        one_id = OneLineID(graph=self.graph, treatments=[t], outcomes=[o])
        assert one_id.id()

        if o not in self.graph.descendants(t):
            # P(Y=y|do(X=x)) = P(Y=y)
            result = self._Count({o: ov})
            if CN_flag:
                background = self._gen_cond_prob_description_CN(None, {o: ov}, result)
            else:
                background = self._gen_cond_prob_description(None, {o: ov}, result)
            Reason = f"P({o_str}|do({t_str}))=P({o_str})={result:.4f}\n"
            background_math = self._gen_cond_prob_math(None, {o: ov}, result)
            return result, Reason, [background], [backgrounds_math]

        tmp = self.autoId_min.identify_backdoor(
            graph=self.dowhy_graph,
            treatment_name=t,
            outcome_name=o
        )
        adjust_set = tmp[0]['backdoor_set']
        if len(adjust_set) > self.max_adjust_set: return None, None, None, None
        if not adjust_set:
            # P(Y=y|do(X=x)) = P(Y=y|X=x)
            result = self._Count({o: ov, t: tv}) / max(1/self.sample_num, self._Count({t: tv}))
            if CN_flag:
                background = self._gen_cond_prob_description_CN({t: tv}, {o: ov}, result)
            else:
                background = self._gen_cond_prob_description({t: tv}, {o: ov}, result)
            Reason = f"P({o_str}|do({t_str}))=P({o_str}|{t_str})={result:.4f}"
            background_math = self._gen_cond_prob_math({t: tv}, {o: ov}, result)
            return result, Reason, [background], [background_math]
        else:
            # P(Y=y|do(X=x)) = sum_{A} P(Y=y|X=x, A=a)*P(A=a)
            a_str = ','.join(list(adjust_set))
            result = 0
            background = []
            background_math = []
            Reason = f"P({o_str}|do({t_str}))=sum_{{{a_str}}} P({o_str}|{t_str},{a_str})*P({a_str})"
            all_adjust_values = [dict(zip(list(adjust_set), x)) for x in permutations(len(adjust_set), 2)]
            str_items, num_items, value_items = [], [], []
            for adjust_value in all_adjust_values:
                av_str = ','.join([f"{key}={adjust_value[key]}" for key in adjust_value])
                A = self._Count({o: ov, t : tv, **adjust_value})/ max(1/self.sample_num, self._Count({t: tv, **adjust_value}))
                B = self._Count(adjust_value)
                result += A * B
                if CN_flag:
                    background.append(self._gen_cond_prob_description_CN({t:tv, **adjust_value}, {o: ov}, A))
                    background.append(self._gen_cond_prob_description_CN(None, adjust_value, B))
                else:
                    background.append(self._gen_cond_prob_description({t:tv, **adjust_value}, {o: ov}, A))
                    background.append(self._gen_cond_prob_description(None, adjust_value, B))
                
                background_math.append(self._gen_cond_prob_math({t:tv, **adjust_value}, {o: ov}, A))
                background_math.append(self._gen_cond_prob_math(None, adjust_value, B))
                str_items.append(f"P({o_str}|{t_str},{av_str})*P({av_str})")
                num_items.append(f"{A:.4f}*{B:.4f}")
                value_items.append(f"P({o_str}|{t_str},{av_str})={A:.4f}\tP({av_str})={B:.4f}")

            Reason += f"={'+'.join(str_items)}\n"
            Reason += '\t'.join(value_items) + '\n'
            Reason += f"P({o_str}|do({t_str}))={'+'.join(num_items)}={result:.4f}"
            return result, Reason, background, background_math

    def gen_background(self):
        nodes_num = len(self.nodes)
        nodes_str = ", ".join(self.nodes)

        edges_num = len(self.edges)
        edges_str = ", ".join(self.edges)
        names_str = self.make_str_from_list([f"{node} represents {self.node_name[node]}" for node in self.nodes])
        return {
            "graph": f"Given a causal graph with {nodes_num} nodes {nodes_str}, and {edges_num} edges {edges_str}.",
            "real_world_meaning": f"The real world meaning of each node is defined as follows: {names_str}. ",
            "data_info": "",
            "data_info_math": "",
        }
    
    def gen_background_CN(self):
        nodes_num = len(self.nodes)
        nodes_str = ", ".join(self.nodes)

        edges_num = len(self.edges)
        edges_str = ", ".join(self.edges)
        names_str = ", ".join([f"{node}代表{self.node_name_CN[node]}" for node in self.nodes])
        return {
            "graph": f"给定一个因果图，它具有{nodes_num}个节点{nodes_str}和{edges_num}条边{edges_str}。",
            "real_world_meaning": f"每个节点的真实世界含义如下：{names_str}。",
            "data_info": "",
            "data_info_math": "",
        }

    # treatment effect of treated
    def gen_ett_question(self, t, tv, o, ov):
        background = self.gen_background()
        o_str, t_str, at_str = f"{o}={ov}", f"{t}={tv}", f"{t}={1-tv}"
        t_name, o_name, tv_name, ov_name = self.node_name[t], self.node_name[o], self.node_value[t][tv], self.node_value[o][ov]
        atv_name = self.node_value[t][1-tv]
        Instruction = f"Consider the effect of treatment on the treated (ETT) of {t_name} on {o_name}."
        Question = f"For those with {t_name} being {tv_name}, if their {t_name} had been {atv_name}, would the {o_name} have been more likely to be {ov_name}?"
        Reason = {}
        if self.with_CN:
            background_CN = self.gen_background_CN()
            t_name_CN, o_name_CN, tv_name_CN, ov_name_CN = self.node_name_CN[t], self.node_name_CN[o], self.node_value_CN[t][tv], self.node_value_CN[o][ov]
            atv_name_CN = self.node_value_CN[t][1-tv]
            Instruction_CN = f"考虑{t_name_CN}作用于{o_name_CN}的“对被干预者的干预效果”(effect of treatment on the treated, ETT)。"
            Question_CN = f"对于那些{t_name_CN}为{tv_name_CN}，假如{t_name_CN}为{atv_name_CN}，那么{o_name_CN}更有可能为{ov_name_CN}吗？"
            Reason_CN = {}

        # Step 1
        if o not in self.graph.descendants(t):
            Answer = "No"
            Reason["Step 1"] = f"Check whether treatment ({t}) is a cause of outcome ({o}). Node {t} is not a cause of node {o} because there is no directed path from {t} to {o}. So the answer is No."
            # return [(Instruction, Question, Answer, Reason, 0, background)]
            result_EN = {
                "Background": deepcopy(background),
                "Instruction": Instruction,
                "Question": Question,
                "Answer": Answer,
                "Reason": deepcopy(Reason),
                "Type": TaskType.ETT.name,
                "Mode": self.mode.name,
                "prob": 0,
            }
            if self.with_CN:
                Answer_CN = "否"
                Reason_CN["Step 1"] = f"检查干预对象({t})是否为观察结果({o})的一个原因。因为从{t}到{o}没有有向路径，所以节点{t}不是节点{o}的原因。"
                result_CN = {
                    "Background": deepcopy(background_CN),
                    "Instruction": Instruction_CN,
                    "Question": Question_CN,
                    "Answer": Answer_CN,
                    "Reason": deepcopy(Reason_CN),
                    "Type": TaskType.ETT.name,
                    "Mode": self.mode.name,
                    "prob": 0,
                }
                return [(result_EN, result_CN)]
            else:
                return [result_EN]

        one_path = list(nx.all_simple_paths(self.nx_graph, t, o))[0]
        one_path = "->".join(one_path)
        Reason["Step 1"] = f"Check whether treatment ({t}) is a cause of outcome ({o}). Node {t} is a cause of node {o} because there is one or more directed paths from {t} to {o} (e.g. {one_path})."
        if self.with_CN:
            Reason_CN["Step 1"] = f"检查干预对象({t})是否为观察结果({o})的一个原因。因为从{t}到{o}有一条或多条有向路径(例如{one_path})，所以节点{t}是节点{o}的原因。"

        # Step 2. check identification
        one_id = OneLineID(graph=self.graph, treatments=[t], outcomes=[o])
        if not one_id.id():
            Answer = "Not sure"
            Reason["Step 2"] = f"Identification of the Causal Effect. P({o}|do({t})) can not be identified, so the answer is not sure."
            result_EN = {
                "Background": deepcopy(background),
                "Instruction": Instruction,
                "Question": Question,
                "Answer": Answer,
                "Reason": deepcopy(Reason),
                "Type": TaskType.ETT.name,
                "Mode": self.mode.name,
                "prob": None,
            }
            if self.with_CN:
                Answer_CN = "不确定"
                Reason_CN["Step 2"] = f"因果效应的可识别性。P({o}|do({t}))不可被识别，所以答案为不确定。"
                result_CN = {
                    "Background": deepcopy(background_CN),
                    "Instruction": Instruction_CN,
                    "Question": Question_CN,
                    "Answer": Answer_CN,
                    "Reason": deepcopy(Reason_CN),
                    "Type": TaskType.ETT.name,
                    "Mode": self.mode.name,
                    "prob": None,
                }
                return [(result_EN, result_CN)]
            else:
                return [result_EN]

        Reason["Step 2"] = f"Identification of the Causal Effect. P({o}|do({t})) can be identified."
        if self.with_CN:
            Reason_CN["Step 2"] = f"因果效应的可识别性。P({o}|do({t}))可被识别。"

        # Step 3. find backdoor set
        # tmp = self.autoId_exhaustive.identify_backdoor(
        tmp = self.autoId_min.identify_backdoor(
            graph=self.dowhy_graph,
            treatment_name=t,
            outcome_name=o
        )
        adjust_sets = [set(item['backdoor_set']) for item in tmp]

        results = []
        for adjust_set in adjust_sets:
            if len(adjust_set) > self.max_adjust_set: continue
            background["data_info"] = ''
            background["data_info_math"] = ''
            if self.with_CN:
                background_CN["data_info"] = ''
                background_CN["data_info_math"] = ''
            if not adjust_set:
                Reason["Step 3"] = f"Find a valid backdoor adjustment set: empty set."
                A = self._Count({t: tv, o: ov}) / max(1/self.sample_num, self._Count({t: tv}))
                B = self._Count({t: 1-tv, o: ov}) / max(1/self.sample_num, self._Count({t: 1-tv}))
                ETT = A - B
                Reason["Step 4"] = f"ETT=E[{o}_{{{t_str}}}-{o}_{{{at_str}}}|{t_str}]=P({o_str}|{t_str})-P({o_str}|{at_str})"
                Reason["Step 5"] = f"P({o_str}|{t_str})={A:.4f}\tP({o_str}|{at_str})={B:.4f}"
                Reason["Step 6"] = f"ETT={A:.4f}-{B:.4f}={ETT:.4f}"

                background["data_info"] += self._gen_cond_prob_description({t: tv}, {o: ov}, A)
                background["data_info"] += self._gen_cond_prob_description({t: 1 - tv}, {o: ov}, B)
                background["data_info_math"] += self._gen_cond_prob_math({t: tv}, {o: ov}, A)
                background["data_info_math"] += self._gen_cond_prob_math({t: 1 - tv}, {o: ov}, B)

                if self.with_CN:
                    Reason_CN["Step 3"] = f"找到一个合法的后门调整集合：空集。"
                    Reason_CN["Step 4"] = Reason["Step 4"]
                    Reason_CN["Step 5"] = Reason["Step 5"]
                    Reason_CN["Step 6"] = Reason["Step 6"]

                    background_CN["data_info"] += self._gen_cond_prob_description_CN({t: tv}, {o: ov}, A)
                    background_CN["data_info"] += self._gen_cond_prob_description_CN({t: 1 - tv}, {o: ov}, B)
                    background_CN["data_info_math"] += self._gen_cond_prob_math({t: tv}, {o: ov}, A)
                    background_CN["data_info_math"] += self._gen_cond_prob_math({t: 1 - tv}, {o: ov}, B)                
            else:
                a_str = ','.join(list(adjust_set))
                Reason["Step 3"] = f"Find a valid backdoor adjustment set: {{{a_str}}}."
                Reason["Step 4"] = f"ETT=E[{o}_{{{t_str}}}-{o}_{{{at_str}}}|{t_str}]=P({o_str}|{t_str})-sum_{{{a_str}}} P({o_str}|{at_str},{a_str})*P({a_str}|{t_str})"
                
                P0 = self._Count({t: tv, o: ov}) / max(1/self.sample_num, self._Count({t: tv}))
                background["data_info"] += self._gen_cond_prob_description({t: tv}, {o: ov}, P0)
                background["data_info_math"] += self._gen_cond_prob_math({t: tv}, {o: ov}, P0)
                if self.with_CN:
                    Reason_CN["Step 3"] = f"找到一个合法的后门调整集合：{{{a_str}}}。"
                    Reason_CN["Step 4"] = Reason["Step 4"]
                    background_CN["data_info"] += self._gen_cond_prob_description_CN({t: tv}, {o: ov}, P0)
                    background_CN["data_info_math"] += self._gen_cond_prob_math({t: tv}, {o: ov}, P0)

                all_adjust_values = [dict(zip(list(adjust_set), x)) for x in permutations(len(adjust_set), 2)]
                ETT, str_items, num_items, value_items = P0, [], [], []
                for adjust_value in all_adjust_values:
                    av_str = ','.join([f"{key}={adjust_value[key]}" for key in adjust_value])

                    A = self._Count({**adjust_value, t: 1-tv, o: ov}) / max(1/self.sample_num, self._Count({**adjust_value, t: 1-tv}))
                    B = self._Count({**adjust_value, t: tv}) / max(1/self.sample_num, self._Count({t: tv}))
                    str_items.append(f"P({o_str}|{at_str},{av_str})*P({av_str}|{t_str})")
                    num_items.append(f"{A:.4f}*{B:.4f}")
                    value_items.append(f"P({o_str}|{at_str},{av_str})={A:.4f}\tP({av_str}|{t_str})={B:.4f}")
                    ETT -= A * B

                    background["data_info"] += self._gen_cond_prob_description({**adjust_value, t: 1 - tv}, {o: ov}, A)
                    background["data_info"] += self._gen_cond_prob_description({t: tv}, adjust_value, B)
                    background["data_info_math"] += self._gen_cond_prob_math({**adjust_value, t: 1 - tv}, {o: ov}, A)
                    background["data_info_math"] += self._gen_cond_prob_math({t: tv}, adjust_value, B)
                    if self.with_CN:
                        background_CN["data_info"] += self._gen_cond_prob_description_CN({**adjust_value, t: 1 - tv}, {o: ov}, A)
                        background_CN["data_info"] += self._gen_cond_prob_description_CN({t: tv}, adjust_value, B)
                        background_CN["data_info_math"] += self._gen_cond_prob_math({**adjust_value, t: 1 - tv}, {o: ov}, A)
                        background_CN["data_info_math"] += self._gen_cond_prob_math({t: tv}, adjust_value, B)
                Reason["Step 4"] += f"=P({o_str}|{t_str})-[{'+'.join(str_items)}]"
                Reason["Step 5"] = f"P({o_str}|{t_str})={P0:.4f}\t" + '\t'.join(value_items)
                Reason["Step 6"] = f"ETT={P0:.4f}-[{'+'.join(num_items)}]={ETT:.4f}"
                if self.with_CN:
                    Reason_CN["Step 4"] = Reason["Step 4"]
                    Reason_CN["Step 5"] = Reason["Step 5"]
                    Reason_CN["Step 6"] = Reason["Step 6"]
            # Step 7. decide the final answer
            if abs(ETT) > self.thresh:
                rel = '>' if ETT > 0 else '<'
            else:
                rel = '~='
            Answer = 'Yes' if rel == '<' else 'No'
            Reason["Step 7"] = f"ETT={ETT:.4f}{rel}0 so the answer is {Answer}."
            result_EN = {
                "Background": deepcopy(background),
                "Instruction": Instruction,
                "Question": Question,
                "Answer": Answer,
                "Reason": deepcopy(Reason),
                "Type": TaskType.ETT.name,
                "Mode": self.mode.name,
                "prob": ETT,
            }
            if self.with_CN:
                Answer_CN = '是' if rel == '<' else '否'
                Reason_CN["Step 7"] = f"ETT={ETT:.4f}{rel}0，所以答案为{Answer_CN}。"
                result_CN = {
                    "Background": deepcopy(background_CN),
                    "Instruction": Instruction_CN,
                    "Question": Question_CN,
                    "Answer": Answer_CN,
                    "Reason": deepcopy(Reason_CN),
                    "Type": TaskType.ETT.name,
                    "Mode": self.mode.name,
                    "prob": ETT
                }
                results.append((result_EN, result_CN))
            else:
                results.append(result_EN)
        return results

    # average treatment effect
    def gen_ate_question(self, t, tv, o, ov):
        # Background
        background = self.gen_background()
        # Generate Question
        o_str, t_str, at_str = f"{o}={ov}", f"{t}={tv}", f"{t}={1-tv}"
        t_name, o_name, tv_name, ov_name = self.node_name[t], self.node_name[o], self.node_value[t][tv], self.node_value[o][ov]
        Question = f"If {t_name} is changed to be {tv_name}, will {o_name} be more likely to be {ov_name}?"
        Instruction = f"Consider the average treatment effect (ATE) of {t_name} on {o_name}."
        # Init Reason
        Reason = {}
        if self.with_CN:
            background_CN = self.gen_background_CN()
            t_name_CN, o_name_CN, tv_name_CN, ov_name_CN = self.node_name_CN[t], self.node_name_CN[o], self.node_value_CN[t][tv], self.node_value_CN[o][ov]
            Question_CN = f"如果{t_name_CN}被改变为{tv_name_CN}，那么{o_name_CN}更有可能为{ov_name_CN}吗？"
            Instruction_CN = f"考虑{t_name_CN}作用于{o_name_CN}的“平均干预效果”(average treatment effect, ATE)。"
            Reason_CN = {}
            

        # Step 1
        if o not in self.graph.descendants(t):
            Answer = "No"
            Reason["Step 1"] = f"Check whether treatment ({t}) is a cause of outcome ({o}). Node {t} is not a cause of node {o} because there is no directed path from {t} to {o}. So the answer is No."
            result_EN = {
                "Background": deepcopy(background),
                "Instruction": Instruction,
                "Question": Question,
                "Answer": Answer,
                "Reason": deepcopy(Reason),
                "Type": TaskType.ATE.name,
                "Mode": self.mode.name,
                "prob": 0,
            }
            if self.with_CN:
                Answer_CN = "否"
                Reason_CN["Step 1"] = f"检查干预对象({t})是否为观察结果({o})的一个原因。因为从{t}到{o}没有有向路径，所以节点{t}不是节点{o}的原因。"
                result_CN = {
                    "Background": deepcopy(background_CN),
                    "Instruction": Instruction_CN,
                    "Question": Question_CN,
                    "Answer": Answer_CN,
                    "Reason": deepcopy(Reason_CN),
                    "Type": TaskType.ATE.name,
                    "Mode": self.mode.name,
                    "prob": 0
                }
                return [(result_EN, result_CN)]
            else:
                return [result_EN]

        one_path = list(nx.all_simple_paths(self.nx_graph, t, o))[0]
        one_path = "->".join(one_path)
        Reason["Step 1"] = f"Check whether treatment ({t}) is a cause of outcome ({o}). Node {t} is a cause of node {o} because there is one or more directed paths from {t} to {o} (e.g. {one_path})."
        if self.with_CN:
            Reason_CN["Step 1"] = f"检查干预对象({t})是否为观察结果({o})的一个原因。因为从{t}到{o}有一条或多条有向路径(例如{one_path})，所以节点{t}是节点{o}的原因。"

        # Step 2. check identification
        one_id = OneLineID(graph=self.graph, treatments=[t], outcomes=[o])
        if not one_id.id():
            Answer = "Not sure"
            Reason["Step 2"] = f"Identification of the Causal Effect. P({o}|do({t})) can not be identified, so the answer is not sure."
            result_EN = {
                "Background": deepcopy(background),
                "Instruction": Instruction,
                "Question": Question,
                "Answer": Answer,
                "Reason": deepcopy(Reason),
                "Type": TaskType.ATE.name,
                "Mode": self.mode.name,
                "prob": None,
            }
            if self.with_CN:
                Answer_CN = "不确定"
                Reason_CN["Step 2"] = f"因果效应的可识别性。P({o}|do({t}))不可被识别，所以答案为不确定。"
                result_CN = {
                    "Background": deepcopy(background_CN),
                    "Instruction": Instruction_CN,
                    "Question": Question_CN,
                    "Answer": Answer_CN,
                    "Reason": deepcopy(Reason_CN),
                    "Type": TaskType.ATE.name,
                    "Mode": self.mode.name,
                    "prob": None,
                }
                return [(result_EN, result_CN)]
            else:
                return [result_EN]
        Reason["Step 2"] = f"Identification of the Causal Effect. P({o}|do({t})) can be identified."
        if self.with_CN:
            Reason_CN["Step 2"] = f"因果效应的可识别性。P({o}|do({t}))可被识别。"

        # Step 3. find backdoor set
        # tmp = self.autoId_exhaustive.identify_backdoor(
        tmp = self.autoId_min.identify_backdoor(
            graph=self.dowhy_graph,
            treatment_name=t,
            outcome_name=o
        )
        adjust_sets = [set(item["backdoor_set"]) for item in tmp]
        results = []
        for adjust_set in adjust_sets:
            if len(adjust_set) > self.max_adjust_set: continue
            background["data_info"] = ""
            background["data_info_math"] = ""
            if self.with_CN:
                background_CN["data_info"] = ""
                background_CN["data_info_math"] = ""
            a_str = ",".join(list(adjust_set))
            if not adjust_set:
                A = self._Count({t: tv, o: ov}) / max(1/self.sample_num, self._Count({t: tv}))
                B = self._Count({t: 1-tv, o: ov}) / max(1/self.sample_num, self._Count({t: 1-tv}))
                ATE = A - B
                Reason["Step 3"] = f"Find a valid backdoor adjustment set: empty set."
                Reason["Step 4"] = f"ATE=P({o_str}|do({t_str}))-P({o_str}|do({at_str}))=P({o_str}|{t_str})-P({o_str}|{at_str})"
                Reason["Step 5"] = f"P({o_str}|{t_str})={A:.4f}\tP({o_str}|{at_str})={B:.4f}"
                Reason["Step 6"] = f"ATE={A:.4f}-{B:.4f}={ATE:.4f}"
                background["data_info"] += self._gen_cond_prob_description({t:tv}, {o:ov}, A)
                background["data_info"] += self._gen_cond_prob_description({t:1-tv}, {o:ov}, B)
                background["data_info_math"] += self._gen_cond_prob_math({t:tv}, {o:ov}, A)
                background["data_info_math"] += self._gen_cond_prob_math({t:1-tv}, {o:ov}, B)
                if self.with_CN:
                    Reason_CN["Step 3"] = f"找到一个合法的后门调整集合：空集。"
                    Reason_CN["Step 4"] = Reason["Step 4"]
                    Reason_CN["Step 5"] = Reason["Step 5"]
                    Reason_CN["Step 6"] = Reason["Step 6"]
                    background_CN["data_info"] += self._gen_cond_prob_description_CN({t:tv}, {o:ov}, A)
                    background_CN["data_info"] += self._gen_cond_prob_description_CN({t:1-tv}, {o:ov}, B)
                    background_CN["data_info_math"] += self._gen_cond_prob_math({t:tv}, {o:ov}, A)
                    background_CN["data_info_math"] += self._gen_cond_prob_math({t:1-tv}, {o:ov}, B)
            else:
                Reason["Step 3"] = f"Find a valid backdoor adjustment set: {{{a_str}}}."
                all_adjust_values = [dict(zip(list(adjust_set), x)) for x in permutations(len(adjust_set), 2)]
                Reason["Step 4"] = f"ATE=P({o_str}|do({t_str}))-P({o_str}|do({at_str}))=sum_{{{a_str}}} [P({o_str}|{t_str},{a_str})*P({a_str})-P({o_str}|{at_str},{a_str})*P({a_str})]"

                if self.with_CN:
                    Reason_CN["Step 3"] = f"找到一个合法的后门调整集合：{{{a_str}}}。"

                str_items = []
                num_items = []
                value_items = []
                ATE = 0
                for adjust_value in all_adjust_values:
                    av_str = ",".join([f"{key}={adjust_value[key]}" for key in adjust_value])
                    A = self._Count({**adjust_value, t: tv, o: ov}) / max(1/self.sample_num, self._Count({**adjust_value, t: tv}))
                    B = self._Count(adjust_value)
                    C = self._Count({**adjust_value, t: 1-tv, o: ov}) / max(1/self.sample_num, self._Count({**adjust_value, t: 1-tv}))
                    str_items.append(f"[P({o_str}|{t_str},{av_str})*P({av_str})-P({o_str}|{at_str},{av_str})*P({av_str})]")
                    num_items.append(f"[{A:.4f}*{B:.4f}-{C:.4f}*{B:.4f}]")
                    value_items.append(f"P({o_str}|{t_str},{av_str})={A:.4f}\tP({o_str}|{at_str},{av_str})={C:.4f}\tP({av_str})={B:.4f}")
                    ATE += A * B - C * B

                    background["data_info"] += self._gen_cond_prob_description({t:tv, **adjust_value}, {o:ov}, A)
                    background["data_info"] += self._gen_cond_prob_description({t:1-tv, **adjust_value}, {o:ov}, C)
                    background["data_info"] += self._gen_cond_prob_description(None, adjust_value, B)
                    background["data_info_math"] += self._gen_cond_prob_math({t:tv, **adjust_value}, {o:ov}, A)
                    background["data_info_math"] += self._gen_cond_prob_math({t:1-tv, **adjust_value}, {o:ov}, C)
                    background["data_info_math"] += self._gen_cond_prob_math(None, adjust_value, B)
                    if self.with_CN:
                        background_CN["data_info"] += self._gen_cond_prob_description_CN({t:tv, **adjust_value}, {o:ov}, A)
                        background_CN["data_info"] += self._gen_cond_prob_description_CN({t:1-tv, **adjust_value}, {o:ov}, C)
                        background_CN["data_info"] += self._gen_cond_prob_description_CN(None, adjust_value, B)
                        background_CN["data_info_math"] += self._gen_cond_prob_math({t:tv, **adjust_value}, {o:ov}, A)
                        background_CN["data_info_math"] += self._gen_cond_prob_math({t:1-tv, **adjust_value}, {o:ov}, C)
                        background_CN["data_info_math"] += self._gen_cond_prob_math(None, adjust_value, B)
                Reason["Step 4"] += f"={'+'.join(str_items)}"
                Reason["Step 5"] = "\t".join(value_items)
                Reason["Step 6"] = f"ATE={'+'.join(num_items)}={ATE:.4f}"
                if self.with_CN:
                    Reason_CN["Step 4"] = Reason["Step 4"]
                    Reason_CN["Step 5"] = Reason["Step 5"]
                    Reason_CN["Step 6"] = Reason["Step 6"]
            # Step 5. decide the final answer
            if abs(ATE) > self.thresh:
                rel = ">" if ATE > 0 else "<"
            else:
                rel = "~="
            Answer = "Yes" if rel == ">" else "No"
            Reason["Step 7"] = f"ATE={ATE:.4f}{rel}0 so the answer is {Answer}."
            result_EN = {
                "Background": deepcopy(background),
                "Instruction": Instruction,
                "Question": Question,
                "Answer": Answer,
                "Reason": deepcopy(Reason),
                "Type": TaskType.ATE.name,
                "Mode": self.mode.name,
                "prob": ATE,
            }
            if self.with_CN:
                Answer_CN = "是" if rel == ">" else "否"
                Reason_CN["Step 7"] = f"ATE={ATE:.4f}{rel}0 所以答案为{Answer_CN}。"
                result_CN = {
                    "Background": deepcopy(background_CN),
                    "Instruction": Instruction_CN,
                    "Question": Question_CN,
                    "Answer": Answer_CN,
                    "Reason": deepcopy(Reason_CN),
                    "Type": TaskType.ATE.name,
                    "Mode": self.mode.name,
                    "prob": ATE
                }
                results.append((result_EN, result_CN))
            else:
                results.append(result_EN)
        return results

    # natural indirect effect
    def gen_nie_question(self, t, tv, o, ov):
        # Background
        background = self.gen_background()
        # Generate Question
        o_str, t_str, at_str = f"{o}={ov}", f"{t}={tv}", f"{t}={1-tv}"
        t_name, o_name, tv_name, ov_name = self.node_name[t], self.node_name[o], self.node_value[t][tv], self.node_value[o][ov]
        Instruction = f"Consider the natural indirect effect (NIE) of {t_name} on {o_name}."
        Question =  f"Suppose {t_name} is held constant and the mediator changes to whatever value it would have attained under {t_name} changing to be {tv_name}, would {o_name} have been more likely to be {ov_name}?"
        # Init Reason
        Reason = {}
        if self.with_CN:
            background_CN = self.gen_background_CN()
            t_name_CN, o_name_CN, tv_name_CN, ov_name_CN = self.node_name_CN[t], self.node_name_CN[o], self.node_value_CN[t][tv], self.node_value_CN[o][ov]
            Instruction_CN = f"考虑{t_name_CN}作用于{o_name_CN}的“自然间接效果”(natural indirect effect, NIE)。"
            Question_CN =  f"假如{t_name_CN}保持不变，而所有中间变量被改变为当它们在{t_name_CN}变化为{tv_name_CN}下的取值，那么{o_name_CN}更有可能为{ov_name_CN}吗？"
            Reason_CN ={}

        # Step 1. Check whether there is an indirect connect between treatment and outcome.
        M = set()
        for c in self.nodes:
            if c == t or c == o: continue
            if c in self.graph.descendants(t) and o in self.graph.descendants(c):
                M.add(c)
        M_str = ",".join(list(M))

        if len(M) == 0:
            Answer = "No"
            Reason["Step 1"] = f"Check whether there is an indirect connect between treatment ({t}) and outcome ({o}). There is no directed path with three or more vertices from {t} to {o}. So the answer is No."
            result_EN = {
                "Background": deepcopy(background),
                "Instruction": Instruction,
                "Question": Question,
                "Answer": Answer,
                "Reason": deepcopy(Reason),
                "Type": TaskType.NIE.name,
                "Mode": self.mode.name,
                "prob": 0,
            }
            if self.with_CN:
                Answer_CN = "否"
                Reason_CN["Step 1"] = f"检查从干预对象({t})到观察结果({o})是否有间接因果关系。从{t}到{o}不存在节点数大于等于3的有向路径，因此答案为否。"
                result_CN = {
                    "Background": deepcopy(background_CN),
                    "Instruction": Instruction_CN,
                    "Question": Question_CN,
                    "Answer": Answer_CN,
                    "Reason": deepcopy(Reason_CN),
                    "Type": TaskType.NIE.name,
                    "Mode": self.mode.name,
                    "prob": 0,
                }
                return [(result_EN, result_CN)]
            else:
                return [result_EN]

        if len(M) > 1: # 当前不支持mediator数量大于0
            return []

        Reason["Step 1"] = f"Check whether there is an indirect connect between treatment ({t}) and outcome ({o}). There exists directed path(s) with three or more vertices from {t} to {o} (e.g. {t}->{list(M)[0]}->{o})"
        Reason["Step 2"] = f"Find a valid mediator set: {{{M_str}}}."
        if self.with_CN:
            Reason_CN["Step 1"] = f"检查从干预对象({t})到观察结果({o})是否有间接因果关系。从{t}到{o}存在节点数大于等于3的有向路径(例如 {t}->{list(M)[0]}->{o})。"
            Reason_CN["Step 2"] = f"找到一个合法的中间变量集合: {{{M_str}}}。"

        results1 = self.autoId_min.identify_backdoor(
                graph=self.dowhy_graph,
                treatment_name=[t],
                outcome_name=list(M)[0]
            )
        results2 = self.autoId_min.identify_backdoor(
            graph=self.dowhy_graph,
            treatment_name=list(M) + [t],
            outcome_name=o
        )
        adjust_set1, adjust_set2 = results1[-1]["backdoor_set"], results2[-1]["backdoor_set"]
        adjust_set = set(adjust_set1).union(set(adjust_set2))
        if len(adjust_set) > self.max_adjust_set: return []

        backgrounds, backgrounds_math, backgrounds_CN = [], [], []
        if len(adjust_set) == 0:
            Reason["Step 3"] = f"Find a valid backdoor adjustment set: empty set."
            Reason["Step 4"] = f"NIE=sum_{{{M_str}}} P({o_str}|{at_str},{M_str})*[P({M_str}|{t_str})-P({M_str}|{at_str})]"

            all_mediator_values = [dict(zip(list(M), x)) for x in permutations(len(M), 2)]
            NIE, str_items, num_items, value_items = 0, [], [], []
            mediator_flag = 0
            for mediator_value in all_mediator_values:
                
                MV_str = ",".join([f"{key}={mediator_value[key]}" for key in mediator_value])
                A = self._Count({**mediator_value, t: 1 - tv, o: ov}) / max(1/self.sample_num, self._Count({**mediator_value, t: 1 - tv}))
                B = self._Count({**mediator_value, t: tv}) / max(1/self.sample_num, self._Count({t: tv}))
                C = self._Count({**mediator_value, t: 1 - tv}) / max(1/self.sample_num, self._Count({t: 1 - tv}))
                NIE += A * (B - C)
                str_items.append(f"P({o_str}|{at_str},{MV_str})*[P({MV_str}|{t_str})-P({MV_str}|{at_str})]")
                num_items.append(f"{A:.4f}*({B:.4f}-{C:.4f})")
                value_items.append(f"P({o_str}|{at_str},{MV_str})={A:.4f}\tP({MV_str}|{t_str})={B:.4f}\tP({MV_str}|{at_str})={C:.4f}")
                
                backgrounds.append(self._gen_cond_prob_description({**mediator_value, t: 1 - tv}, {o: ov}, A))
                backgrounds_math.append(self._gen_cond_prob_math({**mediator_value, t: 1 - tv}, {o: ov}, A))
                if self.with_CN:
                    backgrounds_CN.append(self._gen_cond_prob_description_CN({**mediator_value, t: 1 - tv}, {o: ov}, A))

                if mediator_flag == 0:
                    backgrounds.append(self._gen_cond_prob_description({t: tv}, mediator_value, B))
                    backgrounds.append(self._gen_cond_prob_description({t: 1 - tv}, mediator_value, C))
                    backgrounds_math.append(self._gen_cond_prob_math({t: tv}, mediator_value, B))
                    backgrounds_math.append(self._gen_cond_prob_math({t: 1 - tv}, mediator_value, C))
                    if self.with_CN:
                        backgrounds_CN.append(self._gen_cond_prob_description_CN({t: tv}, mediator_value, B))
                        backgrounds_CN.append(self._gen_cond_prob_description_CN({t: 1 - tv}, mediator_value, C))

                mediator_flag += 1             

            Reason["Step 4"] += f"={'+'.join(str_items)}"
            Reason["Step 5"] = "\t".join(value_items)
            Reason["Step 6"] = f"NIE={'+'.join(num_items)}={NIE:.4f}"
            if self.with_CN:
                Reason_CN["Step 3"] = f"找到一个合法的后门调整集合：空集。"
                Reason_CN["Step 4"] = Reason["Step 4"]
                Reason_CN["Step 5"] = Reason["Step 5"]
                Reason_CN["Step 6"] = Reason["Step 6"]

        else:
            a_str = ",".join(list(adjust_set))
            Reason["Step 3"] = f"Find a valid backdoor adjustment set: {{{a_str}}}."

            Reason["Step 4"] = f"NIE=sum_{{{M_str}}}sum_{{{a_str}}} P({o_str}|{at_str},{M_str},{a_str})*[P({M_str}|{t_str},{a_str})-P({M_str}|{at_str},{a_str})]*P({a_str})"

            all_mediator_values = [dict(zip(list(M), x)) for x in permutations(len(M), 2)]
            all_adjust_values = [dict(zip(list(adjust_set), x)) for x in permutations(len(adjust_set), 2)]

            NIE, str_items, num_items, value_items = 0, [], [], []
            
            adjust_flag = 0
            for adjust_value in all_adjust_values:
                mediator_flag = 0
                D = self._Count(adjust_value)
                if adjust_flag == 0:
                    backgrounds.append(self._gen_cond_prob_description(None, adjust_value, D))
                    backgrounds_math.append(self._gen_cond_prob_math(None, adjust_value, D))
                    if self.with_CN:
                        backgrounds_CN.append(self._gen_cond_prob_description_CN(None, adjust_value, D))

                for mediator_value in all_mediator_values:

                    MV_str = ",".join([f"{key}={mediator_value[key]}" for key in mediator_value])
                    av_str = ",".join([f"{key}={adjust_value[key]}" for key in adjust_value])

                    A = self._Count({o: ov, t: 1 - tv, **adjust_value, **mediator_value}) / max(1/self.sample_num, self._Count({t: 1 - tv, **adjust_value, **mediator_value}))
                    B = self._Count({**mediator_value, t: tv, **adjust_value}) / max(1/self.sample_num, self._Count({t: tv, **adjust_value}))
                    C = self._Count({**mediator_value, t: 1 - tv, **adjust_value}) / max(1/self.sample_num, self._Count({t: 1 - tv, **adjust_value}))
                    

                    str_items.append(f"P({o_str}|{at_str},{MV_str},{av_str})*[P({MV_str}|{t_str},{av_str})-P({MV_str}|{at_str},{av_str})]*P({av_str})")
                    num_items.append(f"{A:.4f}*[{B:.4f}-{C:.4f}]*{D:.4f}")
                    value_items.append(f"P({o_str}|{at_str},{MV_str},{av_str})={A:.4f}\tP({MV_str}|{t_str},{av_str})={B:.4f}\tP({MV_str}|{at_str},{av_str})={C:.4f}\tP({av_str})={D:.4f}")
                    NIE += A * (B - C) * D

                    backgrounds.append(self._gen_cond_prob_description({**mediator_value, t: 1 - tv, **adjust_value}, {o: ov}, A))
                    backgrounds_math.append(self._gen_cond_prob_math({**mediator_value, t: 1 - tv, **adjust_value}, {o: ov}, A))
                    if self.with_CN:
                        backgrounds_CN.append(self._gen_cond_prob_description_CN({**mediator_value, t: 1 - tv, **adjust_value}, {o: ov}, A))

                    if mediator_flag == 0:
                        backgrounds.append(self._gen_cond_prob_description({t: tv, **adjust_value}, mediator_value, B))
                        backgrounds.append(self._gen_cond_prob_description({t: 1 - tv, **adjust_value}, mediator_value, C))
                        backgrounds_math.append(self._gen_cond_prob_math({t: tv, **adjust_value}, mediator_value, B))
                        backgrounds_math.append(self._gen_cond_prob_math({t: 1 - tv, **adjust_value}, mediator_value, C))
                        if self.with_CN:
                            backgrounds_CN.append(self._gen_cond_prob_description_CN({t: tv, **adjust_value}, mediator_value, B))
                            backgrounds_CN.append(self._gen_cond_prob_description_CN({t: 1 - tv, **adjust_value}, mediator_value, C))
                    
                    mediator_flag += 1
                adjust_flag += 1
        
            Reason["Step 4"] += f"={'+'.join(str_items)}"
            Reason["Step 5"] = "\t".join(value_items)
            Reason["Step 6"] = f"NIE={'+'.join(num_items)}={NIE:.4f}"
            if self.with_CN:
                Reason_CN["Step 3"] = f"找到一个合法的后门调整集合：{{{a_str}}}。"
                Reason_CN["Step 4"] = Reason["Step 4"]
                Reason_CN["Step 5"] = Reason["Step 5"]
                Reason_CN["Step 6"] = Reason["Step 6"]
        if abs(NIE) > self.thresh:
            rel = ">" if NIE > 0 else "<"
        else:
            rel = "~="
        Answer = "Yes" if rel == ">" else "No"
        Reason["Step 7"] = f"NIE={NIE:.4f}{rel}0 so the answer is {Answer}."
        background["data_info"] += ''.join(backgrounds)
        background["data_info_math"] += ''.join(backgrounds_math)
        result_EN = {
            "Background": deepcopy(background),
            "Instruction": Instruction,
            "Question": Question,
            "Answer": Answer,
            "Reason": deepcopy(Reason),
            "Type": TaskType.NIE.name,
            "Mode": self.mode.name,
            "prob": NIE,
        }
        if self.with_CN:
            background_CN["data_info"] += ''.join(backgrounds_CN)
            background_CN["data_info_math"] += ''.join(backgrounds_math)
            Answer_CN = "是" if rel == ">" else "否"
            Reason_CN["Step 7"] = f"NIE={NIE:.4f}{rel}0，所以答案为{Answer_CN}。"
            result_CN = {
                "Background": deepcopy(background_CN),
                "Instruction": Instruction_CN,
                "Question": Question_CN,
                "Answer": Answer_CN,
                "Reason": deepcopy(Reason_CN),
                "Type": TaskType.NIE.name,
                "Mode": self.mode.name,
                "prob": NIE,
            }
            return [(result_EN, result_CN)]
        else:
            return [result_EN]

    # natural direct effect
    def gen_nde_question(self, t, tv, o, ov):
        # Background
        background = self.gen_background()
        # Generate Question
        o_str, t_str, at_str = f"{o}={ov}", f"{t}={tv}", f"{t}={1-tv}"
        t_name, o_name, tv_name, ov_name = self.node_name[t], self.node_name[o], self.node_value[t][tv], self.node_value[o][ov]
        Instruction = f"Consider the natural direct effect (NDE) of {t_name} on {o_name}."
        Question =  f"Suppose the mediator keeps constant when {t_name} is changed to be {tv_name}, would the {o_name} have been more likely to be {ov_name}?"
        # Init Reason
        Reason = {}
        if self.with_CN:
            background_CN = self.gen_background_CN()
            t_name_CN, o_name_CN, tv_name_CN, ov_name_CN = self.node_name_CN[t], self.node_name_CN[o], self.node_value_CN[t][tv], self.node_value_CN[o][ov]
            Instruction_CN = f"考虑{t_name_CN}作用于{o_name_CN}的“自然直接效果”(natural direct effect, NDE)。"
            Question_CN =  f"假如所有中间变量保持不变，而{t_name_CN}变化为{tv_name_CN}，那么{o_name_CN}更有可能为{ov_name_CN}吗？"
            Reason_CN ={}
            


        # Step 1. Check whether there is a direct connect between treatment and outcome.
        if o not in self.graph.children(t):
            Answer = "No"
            Reason["Step 1"] = f"Check whether there is an edge from treatment ({t}) to outcome ({o}). The edge {t}->{o} does not exist. So the answer is No."
            result_EN = {
                "Background": deepcopy(background),
                "Instruction": Instruction,
                "Question": Question,
                "Answer": Answer,
                "Reason": deepcopy(Reason),
                "Type": TaskType.NDE.name,
                "Mode": self.mode.name,
                "prob": 0,
            }
            if self.with_CN:
                Answer_CN = "否"
                Reason_CN["Step 1"] = f"检查从干预对象({t})到观察结果({o})是否存在边。边{t}->{o}不存在，因此答案为否。"
                result_CN = {
                    "Background": deepcopy(background_CN),
                    "Instruction": Instruction_CN,
                    "Question": Question_CN,
                    "Answer": Answer_CN,
                    "Reason": deepcopy(Reason_CN),
                    "Type": TaskType.NDE.name,
                    "Mode": self.mode.name,
                    "prob": 0,
                }
                return [(result_EN, result_CN)]
            else:
                return [result_EN]

        Reason["Step 1"] = f"Check whether there is an edge from treatment ({t}) to outcome ({o}). The edge {t}->{o} exists."
        if self.with_CN:
            Reason_CN["Step 1"] = f"检查从干预对象({t})到观察结果({o})是否存在边。边{t}->{o}存在。"

        # Step 2. Find the mediator
        M = set()
        for c in self.nodes:
            if c == t or c == o: continue
            if c in self.graph.descendants(t) and o in self.graph.descendants(c):
                M.add(c)
        M_str = ",".join(list(M))

        if len(M) == 0:  # 没有中间节点，NDE = ATE
            Reason["Step 2"] = f"Find a valid mediator set: empty set."
            if self.with_CN:
                Reason_CN["Step 2"] = f"找到一个合法的中间变量集合: 空集。"
            results = self.autoId_min.identify_backdoor(
                graph=self.dowhy_graph,
                treatment_name=t,
                outcome_name=o
            )
            adjust_set = results[0]["backdoor_set"]
            if len(adjust_set) > self.max_adjust_set: return []

            if len(adjust_set) == 0:
                Reason["Step 3"] = "Find a valid backdoor adjustment set: empty set."
                if self.with_CN:
                    Reason_CN["Step 3"] = f"找到一个合法的后门调整集合：空集。"
                A = self._Count({t: tv, o: ov}) / max(1/self.sample_num, self._Count({t: tv}))
                B = self._Count({t: 1-tv, o: ov}) / max(1/self.sample_num, self._Count({t: 1-tv}))
                NDE = A - B
                Reason["Step 4"] = f"NDE=P({o_str}|{t_str})-P({o_str}|{at_str})"
                Reason["Step 5"] = f"P({o_str}|{t_str})={A:.4f}\tP({o_str}|{at_str})={B:.4f}"
                Reason["Step 6"] = f"NDE={A:.4f}-{B:.4f}={NDE:.4f}"

                background["data_info"] += self._gen_cond_prob_description({t: tv}, {o: ov}, A)
                background["data_info"] += self._gen_cond_prob_description({t: 1 - tv}, {o: ov}, B)
                background["data_info_math"] += self._gen_cond_prob_math({t: tv}, {o: ov}, A)
                background["data_info_math"] += self._gen_cond_prob_math({t: 1 - tv}, {o: ov}, B)
                if self.with_CN:
                    background_CN["data_info"] += self._gen_cond_prob_description_CN({t: tv}, {o: ov}, A)
                    background_CN["data_info"] += self._gen_cond_prob_description_CN({t: 1 - tv}, {o: ov}, B)
                    background_CN["data_info_math"] += self._gen_cond_prob_math({t: tv}, {o: ov}, A)
                    background_CN["data_info_math"] += self._gen_cond_prob_math({t: 1 - tv}, {o: ov}, B)
            else:
                a_str = ",".join(list(adjust_set))
                Reason["Step 3"] = f"Find a valid backdoor adjustment set: {{{a_str}}}."
                if self.with_CN:
                    Reason_CN["Step 3"] = f"找到一个合法的后门调整集合：{{{a_str}}}。"
                all_adjust_values = [dict(zip(list(adjust_set), x)) for x in permutations(len(adjust_set), 2)]
                Reason["Step 4"] = f"NDE=sum_{{{a_str}}} [P({o_str}|{t_str},{a_str})-P({o_str}|{at_str},{a_str})]*P({a_str})"

                NDE, str_items, num_items, value_items = 0, [], [], []
                adjust_flag = 0
                for adjust_value in all_adjust_values:
                    av_str = ",".join([f"{key}={adjust_value[key]}" for key in adjust_value])
                    A = self._Count({**adjust_value, t: tv, o: ov}) / max(1/self.sample_num, self._Count({**adjust_value, t: tv}))
                    B = self._Count(adjust_value)
                    C = self._Count({**adjust_value, t: 1-tv, o: ov}) / max(1/self.sample_num, self._Count({**adjust_value, t: 1-tv}))
                    str_items.append(f"[P({o_str}|{t_str},{av_str})-P({o_str}|{at_str},{av_str})]*P({av_str})")
                    num_items.append(f"[{A:.4f}-{C:.4f}]*{B:.4f}")
                    value_items.append(f"P({o_str}|{t_str},{av_str})={A:.4f}\tP({o_str}|{at_str},{av_str})={C:.4f}\tP({av_str})={B:.4f}")
                    NDE += (A - C) * B

                    background["data_info"] += self._gen_cond_prob_description({t: tv, **adjust_value}, {o: ov}, A)
                    background["data_info"] += self._gen_cond_prob_description({t: 1 - tv, **adjust_value}, {o: ov}, C)                    
                    background["data_info_math"] += self._gen_cond_prob_math({t: tv, **adjust_value}, {o: ov}, A)
                    background["data_info_math"] += self._gen_cond_prob_math({t: 1 - tv, **adjust_value}, {o: ov}, C)
                    if self.with_CN:
                        background_CN["data_info"] += self._gen_cond_prob_description_CN({t: tv, **adjust_value}, {o: ov}, A)
                        background_CN["data_info"] += self._gen_cond_prob_description_CN({t: 1 - tv, **adjust_value}, {o: ov}, C)
                        background_CN["data_info_math"] += self._gen_cond_prob_math({t: tv, **adjust_value}, {o: ov}, A)
                        background_CN["data_info_math"] += self._gen_cond_prob_math({t: 1 - tv, **adjust_value}, {o: ov}, C)
                        
                    if adjust_flag == 0:
                        background["data_info"] += self._gen_cond_prob_description(None, adjust_value, B)
                        background["data_info_math"] += self._gen_cond_prob_math(None, adjust_value, B)
                        if self.with_CN:
                            background_CN["data_info"] += self._gen_cond_prob_description_CN(None, adjust_value, B)
                            background_CN["data_info_math"] += self._gen_cond_prob_math(None, adjust_value, B)

                    adjust_flag += 1             
                Reason["Step 4"] += f"={'+'.join(str_items)}"
                Reason["Step 5"] = '\t'.join(value_items)
                Reason["Step 6"] = f"NDE={'+'.join(num_items)}={NDE:.4f}"
        elif len(M) > 1:  # 超过一个中间节点，暂时无法处理
            return []
        else:
            Reason["Step 2"] = f"Find a valid mediator set: {{{M_str}}}."
            if self.with_CN:
                Reason_CN["Step 2"] = f"找到一个合法的中间变量集合: {{{M_str}}}。"
            results1 = self.autoId_min.identify_backdoor(
                graph=self.dowhy_graph,
                treatment_name=[t],
                outcome_name=list(M)[0]
            )
            results2 = self.autoId_min.identify_backdoor(
                graph=self.dowhy_graph,
                treatment_name=list(M) + [t],
                outcome_name=o
            )
            adjust_set1, adjust_set2 = results1[-1]["backdoor_set"], results2[-1]["backdoor_set"]
            adjust_set = set(adjust_set1).union(set(adjust_set2))
            if len(adjust_set) > self.max_adjust_set: return []
            if len(adjust_set) == 0:
                Reason["Step 3"] = f"Find a valid backdoor adjustment set: empty set."
                if self.with_CN:
                    Reason_CN["Step 3"] = f"找到一个合法的后门调整集合：空集。"
                Reason["Step 4"] = f"NDE=sum_{{{M_str}}} [P({o_str}|{t_str},{M_str})-P({o_str}|{at_str},{M_str})]*P({M_str}|{at_str})"
                
                all_mediator_values = [dict(zip(list(M), x)) for x in permutations(len(M), 2)]
                NDE, str_items, num_items, value_items = 0, [], [], []
                mediator_flag = 0
                for mediator_value in all_mediator_values:
                    MV_str = ",".join([f"{key}={mediator_value[key]}" for key in mediator_value])
                    A = self._Count({**mediator_value, t: tv, o: ov}) / max(1/self.sample_num, self._Count({**mediator_value, t: tv}))
                    B = self._Count({**mediator_value, t: 1 - tv, o: ov}) / max(1/self.sample_num, self._Count({**mediator_value, t: 1 - tv}))
                    C = self._Count({**mediator_value, t: 1 - tv}) / max(1/self.sample_num, self._Count({t: 1 - tv}))
                    NDE += (A - B) * C
                    str_items.append(f"[P({o_str}|{t_str},{MV_str})-P({o_str}|{at_str},{MV_str})]*P({MV_str}|{at_str})")
                    num_items.append(f"({A:.4f}-{B:.4f})*{C:.4f}")
                    value_items.append(f"P({o_str}|{t_str},{MV_str})={A:.4f}\tP({o_str}|{at_str},{MV_str})={B:.4f}\tP({MV_str}|{at_str})={C:.4f}")

                    background["data_info"] += self._gen_cond_prob_description({**mediator_value, t: tv}, {o: ov}, A)
                    background["data_info"] += self._gen_cond_prob_description({**mediator_value, t: 1 - tv}, {o: ov}, B)
                    background["data_info_math"] += self._gen_cond_prob_math({**mediator_value, t: tv}, {o: ov}, A)
                    background["data_info_math"] += self._gen_cond_prob_math({**mediator_value, t: 1 - tv}, {o: ov}, B)
                    if self.with_CN:
                        background_CN["data_info"] += self._gen_cond_prob_description_CN({**mediator_value, t: tv}, {o: ov}, A)
                        background_CN["data_info"] += self._gen_cond_prob_description_CN({**mediator_value, t: 1 - tv}, {o: ov}, B)
                        background_CN["data_info_math"] += self._gen_cond_prob_math({**mediator_value, t: tv}, {o: ov}, A)
                        background_CN["data_info_math"] += self._gen_cond_prob_math({**mediator_value, t: 1 - tv}, {o: ov}, B)

                    if mediator_flag == 0:
                        background["data_info"] += self._gen_cond_prob_description({t: 1 - tv}, mediator_value, C)
                        background["data_info_math"] += self._gen_cond_prob_math({t: 1 - tv}, mediator_value, C)
                        if self.with_CN:
                            background_CN["data_info"] += self._gen_cond_prob_description_CN({t: 1 - tv}, mediator_value, C)
                            background_CN["data_info_math"] += self._gen_cond_prob_math({t: 1 - tv}, mediator_value, C)
                        
                    mediator_flag += 1
                Reason["Step 4"] += f"={'+'.join(str_items)}"
                Reason["Step 5"] = '\t'.join(value_items)
                Reason["Step 6"] = f"={'+'.join(num_items)}={NDE:.4f}"
            else:
                a_str = ",".join(list(adjust_set))
                Reason["Step 3"] = f"Find a valid backdoor adjustment set: {{{a_str}}}."
                if self.with_CN:
                    Reason_CN["Step 3"] = f"找到一个合法的后门调整集合：{{{a_str}}}。"
                Reason["Step 4"] = f"NDE=sum_{{{M_str}}}sum_{{{a_str}}} [P({o_str}|{t_str},{M_str},{a_str})-P({o_str}|{at_str},{M_str},{a_str})]*P({M_str}|{at_str},{a_str})*P({a_str})"

                all_mediator_values = [dict(zip(list(M), x)) for x in permutations(len(M), 2)]
                all_adjust_values = [dict(zip(list(adjust_set), x)) for x in permutations(len(adjust_set), 2)]

                NDE, str_items, num_items, value_items = 0, [], [], []
                adjust_flag = 0
                for adjust_value in all_adjust_values:
                    
                    D = self._Count(adjust_value)
                    if adjust_flag == 0:
                        background["data_info"] += self._gen_cond_prob_description(None, adjust_value, D)
                        background["data_info_math"] += self._gen_cond_prob_math(None, adjust_value, D)
                        if self.with_CN:
                            background_CN["data_info"] += self._gen_cond_prob_description_CN(None, adjust_value, D)
                            background_CN["data_info_math"] += self._gen_cond_prob_math(None, adjust_value, D)
                    
                    mediator_flag = 0
                    for mediator_value in all_mediator_values:
                    
                        MV_str = ",".join([f"{key}={mediator_value[key]}" for key in mediator_value])
                        av_str = ",".join([f"{key}={adjust_value[key]}" for key in adjust_value])

                        A = self._Count({**mediator_value, t: tv, **adjust_value, o: ov}) / max(1/self.sample_num, self._Count({**mediator_value, t: tv, **adjust_value}))
                        B = self._Count({**mediator_value, t: 1 - tv, **adjust_value, o: ov}) / max(1/self.sample_num, self._Count({**mediator_value, t: 1 - tv, **adjust_value}))
                        C = self._Count({**mediator_value, t: 1 - tv, **adjust_value}) / max(1/self.sample_num, self._Count({t: 1 - tv, **adjust_value}))
                        
                        str_items.append(f"[P({o_str}|{t_str},{MV_str},{av_str}) - P({o_str}|{at_str},{MV_str},{av_str})]*P({MV_str}|{at_str},{av_str})*P({av_str})")
                        num_items.append(f"({A:.4f}-{B:.4f})*{C:.4f}*{D:.4f}")
                        value_items.append(f"P({o_str}|{t_str},{MV_str},{av_str})={A:.4f}\tP({o_str}|{at_str},{MV_str},{av_str})={B:.4f}\tP({MV_str}|{at_str},{av_str})={C:.4f}\tP({av_str})={D:.4f}")

                        NDE += (A - B) * C * D
                        background["data_info"] += self._gen_cond_prob_description({**mediator_value, t: tv, **adjust_value}, {o: ov}, A)
                        background["data_info"] += self._gen_cond_prob_description({**mediator_value, t: 1 - tv, **adjust_value}, {o: ov}, B)
                        background["data_info_math"] += self._gen_cond_prob_math({**mediator_value, t: tv, **adjust_value}, {o: ov}, A)
                        background["data_info_math"] += self._gen_cond_prob_math({**mediator_value, t: 1 - tv, **adjust_value}, {o: ov}, B)
                        
                        if self.with_CN:
                            background_CN["data_info"] += self._gen_cond_prob_description_CN({**mediator_value, t: tv, **adjust_value}, {o: ov}, A)
                            background_CN["data_info"] += self._gen_cond_prob_description_CN({**mediator_value, t: 1 - tv, **adjust_value}, {o: ov}, B)
                            background_CN["data_info_math"] += self._gen_cond_prob_math({**mediator_value, t: tv, **adjust_value}, {o: ov}, A)
                            background_CN["data_info_math"] += self._gen_cond_prob_math({**mediator_value, t: 1 - tv, **adjust_value}, {o: ov}, B)
                            
                        if mediator_flag == 0:
                            background["data_info"] += self._gen_cond_prob_description({t: 1 - tv, **adjust_value}, mediator_value, C)
                            background["data_info_math"] += self._gen_cond_prob_math({t: 1 - tv, **adjust_value}, mediator_value, C)
                            if self.with_CN:
                                background_CN["data_info"] += self._gen_cond_prob_description_CN({t: 1 - tv, **adjust_value}, mediator_value, C)
                                background_CN["data_info_math"] += self._gen_cond_prob_math({t: 1 - tv, **adjust_value}, mediator_value, C)

                        mediator_flag += 1
                    adjust_flag += 1
                                                       

                Reason["Step 4"] += f"={'+'.join(str_items)}"
                Reason["Step 5"] = '\t'.join(value_items)
                Reason["Step 6"] = f"NDE={'+'.join(num_items)}={NDE:.4f}"
        if self.with_CN:
            Reason_CN["Step 4"] = Reason["Step 4"]
            Reason_CN["Step 5"] = Reason["Step 5"]
            Reason_CN["Step 6"] = Reason["Step 6"]
        # Step 7. decide the final answer
        if abs(NDE) > self.thresh:
            rel = ">" if NDE > 0 else "<"
        else:
            rel = "~="
        Answer = "Yes" if rel == ">" else "No"
        Reason["Step 7"] = f"NDE={NDE:.4f}{rel}0 so the answer is {Answer}."
        result_EN = {
            "Background": deepcopy(background),
            "Instruction": Instruction,
            "Question": Question,
            "Answer": Answer,
            "Reason": deepcopy(Reason),
            "Type": TaskType.NDE.name,
            "Mode": self.mode.name,
            "prob": NDE,
        }
        if self.with_CN:
            Answer_CN = "是" if rel == ">" else "否"
            Reason_CN["Step 7"] = f"NDE={NDE:.4f}{rel}0，所以答案为{Answer_CN}。"
            result_CN = {
                "Background": deepcopy(background_CN),
                "Instruction": Instruction_CN,
                "Question": Question_CN,
                "Answer": Answer_CN,
                "Reason": deepcopy(Reason_CN),
                "Type": TaskType.NDE.name,
                "Mode": self.mode.name,
                "prob": NDE
            }
            return [(result_EN, result_CN)]
        else:
            return [result_EN]     

    # controlled direct effect
    def gen_cde_question(self, t, tv, o, ov):
        background = self.gen_background()
        o_str, t_str, at_str = f"{o}={ov}", f"{t}={tv}", f"{t}={1-tv}"
        t_name, o_name, tv_name, ov_name = self.node_name[t], self.node_name[o], self.node_value[t][tv], self.node_value[o][ov]
        if o not in self.graph.descendants(t):
            return []
        Reason = {}
        if self.with_CN:
            background_CN = self.gen_background_CN()
            t_name_CN, o_name_CN, tv_name_CN, ov_name_CN = self.node_name_CN[t], self.node_name_CN[o], self.node_value_CN[t][tv], self.node_value_CN[o][ov]
            Reason_CN = {}

        M = set()
        for c in self.nodes:
            if c == t or c == o: continue
            if c in self.graph.descendants(t) and o in self.graph.descendants(c):
                M.add(c)
        
        if len(M) == 0:
            return []

        M_str = ','.join(list(M))
        MV = random.choice([dict(zip(list(M), x)) for x in permutations(len(M), 2)])
        MV_str = ','.join([f"{key}={MV[key]}" for key in MV])
        MV_name_list = [f"{self.node_name[key]} being {self.node_value[key][MV[key]]}" for key in MV]
        MV_name = self.make_str_from_list(MV_name_list)

        Instruction = f"Consider the controlled direct effect (CDE) of {t_name} on {o_name}."
        Question = f"Conditioned on {MV_name}, if the {t_name} had been {tv_name}, would the {o_name} have been more likely to be {ov_name}?"

        if self.with_CN:
            MV_name_list_CN = [f"{self.node_name_CN[key]}为{self.node_value_CN[key][MV[key]]}" for key in MV]
            MV_name_CN = self.make_str_from_list(MV_name_list_CN, CN_flag=True)
            Instruction_CN = f"考虑{t_name_CN}作用于{o_name_CN}的“受控直接效果”(controlled direct effect, CDE)。"
            Question_CN = f"在{MV_name_CN}的条件下，假如{t_name_CN}为{tv_name_CN}，那么{o_name_CN}更有可能为{ov_name_CN}吗？"

        # Step 1. Check whether there is a direct connect between treatment and outcome.
        if o not in self.graph.children(t):
            Answer = "No"
            Reason["Step 1"] = f"Check whether there is an edge from treatment ({t}) to outcome ({o}). The edge {t}->{o} does not exist."
            result_EN = {
                "Background": deepcopy(background),
                "Instruction": Instruction,
                "Question": Question,
                "Answer": Answer,
                "Reason": deepcopy(Reason),
                "Type": TaskType.CDE.name,
                "Mode": self.mode.name,
                "prob": 0,
            }
            if self.with_CN:
                Answer_CN = "否"
                Reason_CN["Step 1"] = f"检查从干预对象({t})到观察结果({o})是否存在边。边{t}->{o}不存在，因此答案为否。"
                result_CN = {
                    "Background": deepcopy(background_CN),
                    "Instruction": Instruction_CN,
                    "Question": Question_CN,
                    "Answer": Answer_CN,
                    "Reason": deepcopy(Reason_CN),
                    "Type": TaskType.CDE.name,
                    "Mode": self.mode.name,
                    "prob": 0,            
                }
                return [(result_EN, result_CN)]
            else:
                return [result_EN]
        Reason["Step 1"] = f"Check whether there is an edge from treatment ({t}) to outcome ({o}). The edge {t}->{o} exists."
        if self.with_CN:
            Reason_CN["Step 1"] = f"检查从干预对象({t})到观察结果({o})是否存在边。边{t}->{o}存在。"

        # Step 2. check identification
        one_id = OneLineID(graph=self.graph, treatments=list(M) + [t], outcomes=[o])
        if not one_id.id():
            Answer = "Not sure"
            Reason["Step 2"] = f"Identification of the Causal Effect. P({o}|do({t},{M_str})) can not be identified, so the answer is Not sure."
            result_EN = {
                "Background": deepcopy(background),
                "Instruction": Instruction,
                "Question": Question,
                "Answer": Answer,
                "Reason": deepcopy(Reason),
                "Type": TaskType.CDE.name,
                "Mode": self.mode.name,
                "prob": None
            }
            if self.with_CN:
                Answer_CN = "不确定"
                Reason_CN["Step 2"] = f"因果效应的可识别性。P({o}|do({t},{M_str}))不可被识别，所以答案为不确定。"
                result_CN = {
                    "Background": deepcopy(background_CN),
                    "Instruction": Instruction_CN,
                    "Question": Question_CN,
                    "Answer": Answer_CN,
                    "Reason": deepcopy(Reason_CN),
                    "Type": TaskType.CDE.name,
                    "Mode": self.mode.name,
                    "prob": None              
                }
                return [(result_EN, result_CN)]
            else:
                return [result_EN]

        Reason["Step 2"] = f"Identification of the Causal Effect. P({o}|do({t},{M_str})) can be identified."
        if self.with_CN:
            Reason_CN["Step 2"] = f"因果效应的可识别性。P({o}|do({t},{M_str}))可被识别。"

        # Step 3. find backdoor set
        # tmp = self.autoId_exhaustive.identify_backdoor(
        tmp = self.autoId_min.identify_backdoor(
            graph=self.dowhy_graph,
            treatment_name=list(M) + [t],
            outcome_name=o
        )
        adjust_sets = [set(item['backdoor_set']) for item in tmp]

        results = []
        for adjust_set in adjust_sets:
            if len(adjust_set) > self.max_adjust_set: continue 
            a_str = ','.join(list(adjust_set))
            background["data_info"] = ""
            background["data_info_math"] = ""
            if self.with_CN:
                background_CN["data_info"] = ""
                background_CN["data_info_math"] = ""
            if not adjust_set:
                Reason["Step 3"] = f"Find a valid backdoor adjustment set: empty set."
                if self._Count({t: tv, **MV}) == 0: return []
                A = self._Count({t: tv, o: ov, **MV}) / max(1/self.sample_num, self._Count({t: tv, **MV}))
                if self._Count({t: 1-tv, **MV}) == 0: return []
                B = self._Count({t: 1-tv, o: ov, **MV}) / max(1/self.sample_num, self._Count({t: 1-tv, **MV}))
                CDE = A - B
                Reason["Step 4"] = f"CDE=P({o_str}|do({t_str},{MV_str}))-P({o_str}|do({at_str},{MV_str}))=P({o_str}|{t_str},{MV_str})-P({o_str}|{at_str},{MV_str})"
                Reason["Step 5"] = f"P({o_str}|{t_str},{MV_str})={A:.4f}\tP({o_str}|{at_str},{MV_str})={B:.4f}"
                Reason["Step 6"] = f"CDE={A:.4f}-{B:.4f}={CDE:.4f}"

                background["data_info"] += self._gen_cond_prob_description({t: tv, **MV}, {o: ov}, A)
                background["data_info"] += self._gen_cond_prob_description({t: 1 - tv, **MV}, {o: ov}, B)

                background["data_info_math"] += self._gen_cond_prob_math({t: tv, **MV}, {o: ov}, A)
                background["data_info_math"] += self._gen_cond_prob_math({t: 1 - tv, **MV}, {o: ov}, B)
                if self.with_CN:
                    Reason_CN["Step 3"] = f"找到一个合法的后门调整集合：空集。"
                    background_CN["data_info"] += self._gen_cond_prob_description_CN({t: tv, **MV}, {o: ov}, A)
                    background_CN["data_info"] += self._gen_cond_prob_description_CN({t: 1 - tv, **MV}, {o: ov}, B)

                    background_CN["data_info_math"] += self._gen_cond_prob_math({t: tv, **MV}, {o: ov}, A)
                    background_CN["data_info_math"] += self._gen_cond_prob_math({t: 1 - tv, **MV}, {o: ov}, B)
            else:
                Reason["Step 3"] = f"Find a valid backdoor adjustment set: {{{a_str}}}."
                if self.with_CN:
                     Reason_CN["Step 3"] = f"找到一个合法的后门调整集合：{{{a_str}}}。"
                all_adjust_values = [dict(zip(list(adjust_set), x)) for x in permutations(len(adjust_set), 2)]
                Reason["Step 4"] = f"CDE=P({o_str}|do({t_str},{MV_str}))-P({o_str}|do({at_str},{MV_str}))=sum_{{{a_str}}} [P({o_str}|{t_str},{MV_str},{a_str})-P({o_str}|{at_str},{MV_str},{a_str})]*P({a_str})"
                CDE, str_items, num_items, value_items = 0, [], [], []
                for adjust_value in all_adjust_values:
                    av_str = ','.join([f"{key}={adjust_value[key]}" for key in adjust_value])
                    if self._Count({**adjust_value, t: tv, **MV}) == 0: return []
                    A = self._Count({**adjust_value, t: tv, o: ov, **MV}) / max(1/self.sample_num, self._Count({**adjust_value, t: tv, **MV}))
                    B = self._Count(adjust_value)
                    if self._Count({**adjust_value, t: 1-tv, **MV}) == 0: return []
                    C = self._Count({**adjust_value, t: 1-tv, o: ov, **MV}) / max(1/self.sample_num, self._Count({**adjust_value, t: 1-tv, **MV}))
                    str_items.append(f"[P({o_str}|{t_str},{MV_str},{av_str})-P({o_str}|{at_str},{MV_str},{av_str})]*P({av_str})")
                    num_items.append(f"[{A:.4f}-{C:.4f}]*{B:.4f}")
                    value_items.append(f"P({o_str}|{t_str},{MV_str},{av_str})={A:.4f}\tP({av_str})={B:.4f}\tP({o_str}|{at_str},{MV_str},{av_str})={C:.4f}")
                    CDE += (A - C) * B

                    background["data_info"] += self._gen_cond_prob_description({t: tv, **MV, **adjust_value}, {o: ov}, A)
                    background["data_info"] += self._gen_cond_prob_description({t: 1 - tv, **MV, **adjust_value}, {o: ov}, C)
                    background["data_info"] += self._gen_cond_prob_description(None, adjust_value, B)

                    background["data_info_math"] += self._gen_cond_prob_math({t: tv, **MV, **adjust_value}, {o: ov}, A)
                    background["data_info_math"] += self._gen_cond_prob_math({t: 1 - tv, **MV, **adjust_value}, {o: ov}, C)
                    background["data_info_math"] += self._gen_cond_prob_math(None, adjust_value, B)
                    if self.with_CN:
                        background_CN["data_info"] += self._gen_cond_prob_description_CN({t: tv, **MV, **adjust_value}, {o: ov}, A)
                        background_CN["data_info"] += self._gen_cond_prob_description_CN({t: 1 - tv, **MV, **adjust_value}, {o: ov}, C)
                        background_CN["data_info"] += self._gen_cond_prob_description_CN(None, adjust_value, B) 

                        background_CN["data_info_math"] += self._gen_cond_prob_math({t: tv, **MV, **adjust_value}, {o: ov}, A)
                        background_CN["data_info_math"] += self._gen_cond_prob_math({t: 1 - tv, **MV, **adjust_value}, {o: ov}, C)
                        background_CN["data_info_math"] += self._gen_cond_prob_math(None, adjust_value, B)                   
                Reason["Step 4"] += f"={'+'.join(str_items)}"
                Reason["Step 5"] = '\t'.join(value_items)
                Reason["Step 6"] = f"CDE={'+'.join(num_items)}={CDE:.4f}"
            if self.with_CN:
                Reason_CN["Step 4"] = Reason["Step 4"]
                Reason_CN["Step 5"] = Reason["Step 5"]
                Reason_CN["Step 6"] = Reason["Step 6"]
            # Step 7. decide the final answer
            if abs(CDE) > self.thresh:
                rel = '>' if CDE > 0 else '<'
            else:
                rel = '~='
            Answer = 'Yes' if rel == '>' else 'No'
            Reason["Step 7"] = f"CDE={CDE:.4f}{rel}0 so the answer is {Answer}."
            result_EN = {
                "Background": deepcopy(background),
                "Instruction": Instruction,
                "Question": Question,
                "Answer": Answer,
                "Reason": deepcopy(Reason),
                "Type": TaskType.CDE.name,
                "Mode": self.mode.name,
                "prob": CDE,
            }
            if self.with_CN:
                Answer_CN = '是' if rel == '>' else '否'
                Reason_CN["Step 7"] = f"CDE={CDE:.4f}{rel}0，所以答案为{Answer_CN}。"
                result_CN = {
                    "Background": deepcopy(background_CN),
                    "Instruction": Instruction_CN,
                    "Question": Question_CN,
                    "Answer": Answer_CN,
                    "Reason": deepcopy(Reason_CN),
                    "Type": TaskType.CDE.name,
                    "Mode": self.mode.name,
                    "prob": CDE,
                }
                results.append((result_EN, result_CN))
            else:
                results.append(result_EN)
        return results

    # probability of necessity
    def _gen_pn_ub_question(self, t, o):
        # PN = P(Y_0=0|X=1, Y=1)
        # lower bound: [P(Y=1)-P(Y=1|do(X=0))]/P(X=1,Y=1)
        # upper bound: [P(Y=0|do(X=0))-P(X=0,Y=0)]/P(X=1,Y=1)
        background = self.gen_background()
        backgrounds = []
        backgrounds_math = []
        Reason = {}
        t_name, o_name, tp_name, op_name, tn_name, on_name = self.node_name[t], self.node_name[o], self.node_value[t][1], self.node_value[o][1], self.node_value[t][0], self.node_value[o][0]

        Instruction = f"Consider the probability of necessity (PN) of {t_name} on {o_name}."
        Question = f"Given that {t_name} was {tp_name} and {o_name} was {op_name}, what is the upper bound of the probability of the {o_name} would have been {on_name} if the {t_name} had been {tn_name}?"
    
        do_X0_Y0, reason, backgrounds_new, backgrounds_math_new = self.do_calculas(t, 0, o, 0)
        if do_X0_Y0 is None: return []
        Reason["Step 1"] = f"Calculate P({o}=0|do({t}=0))\n{reason}"

        for b in backgrounds_new:
            if b not in backgrounds:
                backgrounds.append(b)
        for b in backgrounds_math_new:
            if b not in backgrounds_math:
                backgrounds_math.append(b)

        P_X0_Y0 = self._Count({t: 0, o: 0})
        P_X1_Y1 = self._Count({t: 1, o: 1})

        for tmp in [self._gen_cond_prob_description(None, {t: 0, o: 0}, P_X0_Y0),
            self._gen_cond_prob_description(None, {t: 1, o: 1}, P_X1_Y1)]:
            if tmp not in backgrounds:
                backgrounds.append(tmp)
        background["data_info"] += ''.join(backgrounds)

        for tmp in [self._gen_cond_prob_math(None, {t: 0, o: 0}, P_X0_Y0),
            self._gen_cond_prob_math(None, {t: 1, o: 1}, P_X1_Y1)]:
            if tmp not in backgrounds_math:
                backgrounds_math.append(tmp)
        background["data_info_math"] += ''.join(backgrounds_math)

        U = (do_X0_Y0 - P_X0_Y0) / P_X1_Y1
        UB = min(1, U)

        Reason["Step 2"] = f"Upper bound of PN: min{{1, [P({o}=0)|do({t}=0)-P({t}=0,{o}=0)]/P({t}=1,{o}=1)}}\n=min{{1, ({do_X0_Y0:.4f}-{P_X0_Y0:.4f})/{P_X1_Y1:.4f}}}\n=min{{1, {U:.4f}}}\n={UB:.4f}"

        Answer = f"{UB:.4f}"
        result_EN = {
            "Background": deepcopy(background),
            "Instruction": Instruction,
            "Question": Question,
            "Answer": Answer,
            "Reason": deepcopy(Reason),
            "Type": TaskType.PN.name,
            "Mode": self.mode.name,
            "prob": UB,
        }

        if self.with_CN:
            background_CN = self.gen_background_CN()
            background_CN["data_info_math"] = deepcopy(background["data_info_math"])
            backgrounds_CN = []
            Reason_CN = {}
            t_name_CN, o_name_CN, tp_name_CN, op_name_CN, tn_name_CN, on_name_CN = self.node_name_CN[t], self.node_name_CN[o], self.node_value_CN[t][1], self.node_value_CN[o][1], self.node_value_CN[t][0], self.node_value_CN[o][0]
            Instruction_CN = f"考虑{t_name_CN}作用于{o_name_CN}的必要性概率(probability of necessity, PN)。"
            Question_CN = f"给定{t_name_CN}为{tp_name_CN}且{o_name_CN}为{op_name_CN}, 那么假如{t_name_CN}为{tn_name_CN}，此时{o_name_CN}为{on_name_CN}的概率的上界是多少？"

            _, reason, backgrounds_new_CN, _ = self.do_calculas(t, 0, o, 0, CN_flag=True)
            Reason_CN["Step 1"] = f"计算P({o}=0|do({t}=0))\n{reason}"

            for b in backgrounds_new_CN:
                if b not in backgrounds_CN:
                    backgrounds_CN.append(b)

            for tmp in [
                self._gen_cond_prob_description_CN(None, {t: 0, o: 0}, P_X0_Y0), 
                self._gen_cond_prob_description_CN(None, {t: 1, o: 1}, P_X1_Y1)]:
                if tmp not in backgrounds_CN:
                    backgrounds_CN.append(tmp)
            background_CN["data_info"] += ''.join(backgrounds_CN)

            Reason_CN["Step 2"] = f"PN的上界: min{{1, [P({o}=0)|do({t}=0)-P({t}=0,{o}=0)]/P({t}=1,{o}=1)}}\n=min{{1, ({do_X0_Y0:.4f}-{P_X0_Y0:.4f})/{P_X1_Y1:.4f}}}\n=min{{1, {U:.4f}}}\n={UB:.4f}"

            Answer_CN = f"{UB:.4f}"
            result_CN = {
                "Background": deepcopy(background_CN),
                "Instruction": Instruction_CN,
                "Question": Question_CN,
                "Answer": Answer_CN,
                "Reason": deepcopy(Reason_CN),
                "Type": TaskType.PN.name,
                "Mode": self.mode.name,
                "prob": UB
            }

            return [(result_EN, result_CN)]
        else:
            return [result_EN]

    def _gen_pn_lb_question(self, t, o):
        # PN = P(Y_0=0|X=1, Y=1)
        # lower bound: [P(Y=1)-P(Y=1|do(X=0))]/P(X=1,Y=1)
        # upper bound: [P(Y=0|do(X=0))-P(X=0,Y=0)]/P(X=1,Y=1)
        background = self.gen_background()
        backgrounds = []
        backgrounds_math = []
        Reason = {}
        t_name, o_name, tp_name, op_name, tn_name, on_name = self.node_name[t], self.node_name[o], self.node_value[t][1], self.node_value[o][1], self.node_value[t][0], self.node_value[o][0]

        Instruction = f"Consider the probability of necessity (PN) of {t_name} on {o_name}."
        Question = f"Given that {t_name} was {tp_name} and {o_name} was {op_name}, what is the lower bound of the probability of the {o_name} would have been {on_name} if the {t_name} had been {tn_name}?"

        do_X0_Y1, reason, backgrounds_new, backgrounds_math_new = self.do_calculas(t, 0, o, 1)
        if do_X0_Y1 is None: return []
        Reason["Step 1"] = f"Calculate P({o}=1|do({t}=0))\n{reason}"
        
        for b in backgrounds_new:
            if b not in backgrounds:
                backgrounds.append(b)
        for b in backgrounds_math_new:
            if b not in backgrounds_math:
                backgrounds_math.append(b)

        P_Y1 = self._Count({o: 1})
        P_X1_Y1 = self._Count({t: 1, o: 1})

        for tmp in [
            self._gen_cond_prob_description(None, {o: 1}, P_Y1),
            self._gen_cond_prob_description(None, {t: 1, o: 1}, P_X1_Y1)]:
            if tmp not in backgrounds:
                backgrounds.append(tmp)
        background["data_info"] += ''.join(backgrounds)

        for tmp in [
            self._gen_cond_prob_math(None, {o: 1}, P_Y1),
            self._gen_cond_prob_math(None, {t: 1, o: 1}, P_X1_Y1)]:            
            if tmp not in backgrounds_math:
                backgrounds_math.append(tmp)
        background["data_info_math"] += ''.join(backgrounds_math)

        L = (P_Y1 - do_X0_Y1) / P_X1_Y1
        LB = max(0, L)
        
        Reason["Step 2"] = f"Lower bound of PN: max{{0, [P({o}=1)-P({o}=1|do({t}=0))]/P({t}=1,{o}=1)}}\n=max{{0, ({P_Y1:.4f}-{do_X0_Y1:.4f})/{P_X1_Y1:.4f}}}\n=max{{0, {L:.4f}}}\n={LB:.4f}"


        Answer = f"{LB:.4f}"
        result_EN = {
            "Background": deepcopy(background),
            "Instruction": Instruction,
            "Question": Question,
            "Answer": Answer,
            "Reason": deepcopy(Reason),
            "Type": TaskType.PN.name,
            "Mode": self.mode.name,
            "prob": LB,
        }

        if self.with_CN:
            background_CN = self.gen_background_CN()
            background_CN["data_info_math"] = deepcopy(background["data_info_math"])
            backgrounds_CN = []
            Reason_CN = {}
            t_name_CN, o_name_CN, tp_name_CN, op_name_CN, tn_name_CN, on_name_CN = self.node_name_CN[t], self.node_name_CN[o], self.node_value_CN[t][1], self.node_value_CN[o][1], self.node_value_CN[t][0], self.node_value_CN[o][0]
            Instruction_CN = f"考虑{t_name_CN}作用于{o_name_CN}的必要性概率(probability of necessity, PN)。"
            Question_CN = f"给定{t_name_CN}为{tp_name_CN}且{o_name_CN}为{op_name_CN}, 那么假如{t_name_CN}为{tn_name_CN}，此时{o_name_CN}为{on_name_CN}的概率的下界是多少？"

            _, reason, backgrounds_new_CN, _ = self.do_calculas(t, 0, o, 1, CN_flag=True)
            Reason_CN["Step 1"] = f"计算P({o}=1|do({t}=0))\n{reason}"

            for b in backgrounds_new_CN:
                if b not in backgrounds_CN:
                    backgrounds_CN.append(b)

            for tmp in [
                self._gen_cond_prob_description_CN(None, {o: 1}, P_Y1),
                self._gen_cond_prob_description_CN(None, {t: 1, o: 1}, P_X1_Y1)]:
                if tmp not in backgrounds_CN:
                    backgrounds_CN.append(tmp)
            background_CN["data_info"] += ''.join(backgrounds_CN)

            Reason_CN["Step 2"] = f"PN的下界: max{{0, [P({o}=1)-P({o}=1|do({t}=0))]/P({t}=1,{o}=1)}}\n=max{{0, ({P_Y1:.4f}-{do_X0_Y1:.4f})/{P_X1_Y1:.4f}}}\n=max{{0, {L:.4f}}}\n={LB:.4f}"

            Answer_CN = f"{LB:.4f}"
            result_CN = {
                "Background": deepcopy(background_CN),
                "Instruction": Instruction_CN,
                "Question": Question_CN,
                "Answer": Answer_CN,
                "Reason": deepcopy(Reason_CN),
                "Type": TaskType.PN.name,
                "Mode": self.mode.name,
                "prob": LB
            }

            return [(result_EN, result_CN)]
        else:
            return [result_EN]

    # probability of sufficiency
    def _gen_ps_ub_question(self, t, o):
        # PS = P(Y_1=1|X=0, Y=0)
        # lower bound: [P(Y=0)-P(Y=0|do(X=1))]/P(X=0,Y=0)
        # upper bound: [P(Y=1|do(X=1))-P(X=1,Y=1)]/P(X=0,Y=0)
        background = self.gen_background()
        backgrounds = []
        backgrounds_math = []
        Reason = {}

        t_name, o_name, tp_name, op_name, tn_name, on_name = self.node_name[t], self.node_name[o], self.node_value[t][1], self.node_value[o][1], self.node_value[t][0], self.node_value[o][0]

        Instruction = f"Consider the probability of sufficiency (PS) of {t_name} on {o_name}."
        Question = f"Given that {t_name} was {tn_name} and {o_name} was {on_name}, what is the upper bound of the probability that {o_name} would have been {op_name} if the {t_name} had been {tp_name}?"

        do_X1_Y1, reason, backgrounds_new, backgrounds_math_new = self.do_calculas(t, 1, o, 1)
        if do_X1_Y1 is None: return []
        Reason["Step 1"] = f"Calculate P({o}=1|do({t}=1))\n{reason}"
        for b in backgrounds_new:
            if b not in backgrounds:
                backgrounds.append(b)
        for b in backgrounds_math_new:
            if b not in backgrounds_math:
                backgrounds_math.append(b)

        P_X0_Y0 = self._Count({t: 0, o: 0})
        P_X1_Y1 = self._Count({t: 1, o: 1})

        for tmp in [
            self._gen_cond_prob_description(None, {t: 0, o: 0}, P_X0_Y0),
            self._gen_cond_prob_description(None, {t: 1, o: 1}, P_X1_Y1)]:
            if tmp not in backgrounds:
                backgrounds.append(tmp)
        background["data_info"] += ''.join(backgrounds)
        for tmp in [
            self._gen_cond_prob_math(None, {t: 0, o: 0}, P_X0_Y0),
            self._gen_cond_prob_math(None, {t: 1, o: 1}, P_X1_Y1)]:
            if tmp not in backgrounds_math:
                backgrounds_math.append(tmp)
        background["data_info_math"] += ''.join(backgrounds_math)


        U = (do_X1_Y1 - P_X1_Y1) / P_X0_Y0
        UB = min(1, U)

        Reason["Step 2"] = f"Upper bound of PS: min{{1, [P({o}=1)|do({t}=1)-P({t}=1,{o}=1)]/P({t}=0,{o}=0)}}\n=min{{1, ({do_X1_Y1:.4f}-{P_X1_Y1:.4f})/{P_X0_Y0:.4f}}}\n=min{{1, {U:.4f}}}\n={UB:.4f}"

        Answer = f"{UB:.4f}"
        result_EN = {
            "Background": deepcopy(background),
            "Instruction": Instruction,
            "Question": Question,
            "Answer": Answer,
            "Reason": deepcopy(Reason),
            "Type": TaskType.PS.name,
            "Mode": self.mode.name,
            "prob": UB,
        }
        if self.with_CN:
            background_CN = self.gen_background_CN()
            background_CN["data_info_math"] = deepcopy(background["data_info_math"])
            backgrounds_CN = []
            Reason_CN = {}

            t_name_CN, o_name_CN, tp_name_CN, op_name_CN, tn_name_CN, on_name_CN = self.node_name_CN[t], self.node_name_CN[o], self.node_value_CN[t][1], self.node_value_CN[o][1], self.node_value_CN[t][0], self.node_value_CN[o][0]

            Instruction_CN = f"考虑{t_name_CN}作用于{o_name_CN}的充分性概率(probability of sufficiency, PS)。"
            Question_CN = f"给定{t_name_CN}为{tn_name_CN}且{o_name_CN}为{on_name_CN}, 假如{t_name_CN}为{tp_name_CN}，此时{o_name_CN}为{op_name_CN}的概率的上界是多少？"

            _, reason, backgrounds_new, _ = self.do_calculas(t, 1, o, 1, CN_flag=True)
            Reason_CN["Step 1"] = f"计算P({o}=1|do({t}=1))\n{reason}"
            for b in backgrounds_new:
                if b not in backgrounds_CN:
                    backgrounds_CN.append(b)
            for tmp in [
                self._gen_cond_prob_description_CN(None, {t: 0, o: 0}, P_X0_Y0),
                self._gen_cond_prob_description_CN(None, {t: 1, o: 1}, P_X1_Y1)]:
                if tmp not in backgrounds_CN:
                    backgrounds_CN.append(tmp)

            background_CN["data_info"] += ''.join(backgrounds_CN)

            Reason_CN["Step 2"] = f"PS的上界: min{{1, [P({o}=1)|do({t}=1)-P({t}=1,{o}=1)]/P({t}=0,{o}=0)}}\n=min{{1, ({do_X1_Y1:.4f}-{P_X1_Y1:.4f})/{P_X0_Y0:.4f}}}\n=min{{1, {U:.4f}}}\n={UB:.4f}"

            Answer_CN = f"{UB:.4f}"
            result_CN = {
                "Background": deepcopy(background_CN),
                "Instruction": Instruction_CN,
                "Question": Question_CN,
                "Answer": Answer_CN,
                "Reason": deepcopy(Reason_CN),
                "Type": TaskType.PS.name,
                "Mode": self.mode.name,
                "prob": UB,
            }
            return [(result_EN, result_CN)]
        else:
            return [result_EN]
    
    def _gen_ps_lb_question(self, t, o):
        # PS = P(Y_1=1|X=0, Y=0)
        # lower bound: [P(Y=0)-P(Y=0|do(X=1))]/P(X=0,Y=0)
        # upper bound: [P(Y=1|do(X=1))-P(X=1,Y=1)]/P(X=0,Y=0)
        background = self.gen_background()
        backgrounds = []
        backgrounds_math = []
        Reason = {}

        t_name, o_name, tp_name, op_name, tn_name, on_name = self.node_name[t], self.node_name[o], self.node_value[t][1], self.node_value[o][1], self.node_value[t][0], self.node_value[o][0]

        Instruction = f"Consider the probability of sufficiency (PS) of {t_name} on {o_name}."
        Question = f"Given that {t_name} was {tn_name} and {o_name} was {on_name}, what is the lower bound of the probability that {o_name} would have been {op_name} if the {t_name} had been {tp_name}?"

        do_X1_Y0, reason, backgrounds_new, backgrounds_math_new = self.do_calculas(t, 1, o, 0)
        if do_X1_Y0 is None: return []
        Reason["Step 1"] = f"Calculate P({o}=0|do({t}=1))\n{reason}"
        for b in backgrounds_new:
            if b not in backgrounds:
                backgrounds.append(b)
        for b in backgrounds_math_new:
            if b not in backgrounds_math:
                backgrounds_math.append(b)

        P_Y0 = self._Count({o: 0})
        P_X0_Y0 = self._Count({t: 0, o: 0})
        
        for tmp in [
            self._gen_cond_prob_description(None, {o: 0}, P_Y0),
            self._gen_cond_prob_description(None, {t: 0, o: 0}, P_X0_Y0)]:
            if tmp not in backgrounds:
                backgrounds.append(tmp)
        background["data_info"] += ''.join(backgrounds)

        for tmp in [
            self._gen_cond_prob_math(None, {o: 0}, P_Y0),
            self._gen_cond_prob_math(None, {t: 0, o: 0}, P_X0_Y0)]:
            if tmp not in backgrounds_math:
                backgrounds_math.append(tmp)
        background["data_info_math"] += ''.join(backgrounds_math)


        L = (P_Y0 - do_X1_Y0) / P_X0_Y0
        LB = max(0, L)

        Reason["Step 2"] = f"Lower bound of PS: max{{0, [P({o}=0)-P({o}=0|do({t}=1))]/P({t}=0,{o}=0)}}\n=max{{0, ({P_Y0:.4f}-{do_X1_Y0:.4f})/{P_X0_Y0:.4f}}}\n=max{{0, {L:.4f}}}\n={LB:.4f}"

        Answer = f"{LB:.4f}"
        result_EN = {
            "Background": deepcopy(background),
            "Instruction": Instruction,
            "Question": Question,
            "Answer": Answer,
            "Reason": deepcopy(Reason),
            "Type": TaskType.PS.name,
            "Mode": self.mode.name,
            "prob": LB,
        }
        if self.with_CN:
            background_CN = self.gen_background_CN()
            background_CN["data_info_math"] = deepcopy(background["data_info_math"])
            backgrounds_CN = []
            Reason_CN = {}

            t_name_CN, o_name_CN, tp_name_CN, op_name_CN, tn_name_CN, on_name_CN = self.node_name_CN[t], self.node_name_CN[o], self.node_value_CN[t][1], self.node_value_CN[o][1], self.node_value_CN[t][0], self.node_value_CN[o][0]

            Instruction_CN = f"考虑{t_name_CN}作用于{o_name_CN}的充分性概率(probability of sufficiency, PS)。"
            Question_CN = f"给定{t_name_CN}为{tn_name_CN}且{o_name_CN}为{on_name_CN}, 假如{t_name_CN}为{tp_name_CN}，此时{o_name_CN}为{op_name_CN}的概率的下界是多少？"

            _, reason, backgrounds_new, _ = self.do_calculas(t, 1, o, 0, CN_flag=True)
            Reason_CN["Step 1"] = f"计算P({o}=0|do({t}=1))\n{reason}"
            for b in backgrounds_new:
                if b not in backgrounds_CN:
                    backgrounds_CN.append(b)
            
            for tmp in [
                self._gen_cond_prob_description_CN(None, {o: 0}, P_Y0),
                self._gen_cond_prob_description_CN(None, {t: 0, o: 0}, P_X0_Y0)]:
                if tmp not in backgrounds_CN:
                    backgrounds_CN.append(tmp)

            background_CN["data_info"] += ''.join(backgrounds_CN)

            Reason_CN["Step 2"] = f"PS的下界: max{{0, [P({o}=0)-P({o}=0|do({t}=1))]/P({t}=0,{o}=0)}}\n=max{{0, ({P_Y0:.4f}-{do_X1_Y0:.4f})/{P_X0_Y0:.4f}}}\n=max{{0, {L:.4f}}}\n={LB:.4f}"

            Answer_CN = f"{LB:.4f}"
            result_CN = {
                "Background": deepcopy(background_CN),
                "Instruction": Instruction_CN,
                "Question": Question_CN,
                "Answer": Answer_CN,
                "Reason": deepcopy(Reason_CN),
                "Type": TaskType.PS.name,
                "Mode": self.mode.name,
                "prob": LB,
            }
            return [(result_EN, result_CN)]
        else:
            return [result_EN]

    # backdoor adjustment set
    def _gen_bas_question_reverse(self, t, o):
        background = self.gen_background()
        t_name, o_name = self.node_name[t], self.node_name[o]
        Reason = {}
        if self.with_CN:
            background_CN = self.gen_background_CN()
            t_name_CN, o_name_CN = self.node_name_CN[t], self.node_name_CN[o]
            Reason_CN = {}

        results = self.autoId_min.identify_backdoor(
            graph=self.dowhy_graph,
            treatment_name=t,
            outcome_name=o
        )
        adjust_set = results[0]["backdoor_set"]
        if len(adjust_set) > self.max_adjust_set: return []
        
        if len(adjust_set) == 0:
            # randomly sample a node
            c = random.choice(list(set(self.nodes) - {t, o}))
            adj_name = self.node_name[c]
            Reason["Step 1"] = f"Find a valid backdoor adjustment set: empty set. So method 2 is preferred and the answer is yes."
            Answer = 'Yes'
            if self.with_CN:
                adj_name_CN = self.node_name_CN[c]
                Reason_CN["Step 1"] = f"找到一个合法的后门调整集合：空集。因此方法2更好，答案为是。"
                Answer_CN = "是"
        else:
            a_str = ','.join(list(adjust_set))
            adj_name = self.make_str_from_list([self.node_name[_] for _ in list(adjust_set)])
            Reason["Step 1"] = f"Find a valid backdoor adjustment set: {{{a_str}}}. So method 1 is preferred and the answer is no."
            Answer = 'No'
            if self.with_CN:
                adj_name_CN = self.make_str_from_list([self.node_name_CN[_] for _ in list(adjust_set)], CN_flag=True)
                Reason_CN["Step 1"] = f"找到一个合法的后门调整集合：{{{a_str}}}。因此方法1更好，答案为否。"
                Answer_CN = "否"
            
        Instruction = f"Consider the backdoor adjustment set for treatment {t_name} ({t}) and outcome {o_name} ({o})."
        Question = f"Method 1: We look at how {t_name} correlates with {o_name} case by case according to {adj_name}.\nMethod 2: We look directly at how {t_name} correlates with {o_name} in general.\nTo understand how {t_name} affects {o_name}, is it more correct to use the Method 2 than Method 1?"
        if self.with_CN:
            Instruction_CN = f"考虑关于干预对象{t_name_CN}({t})和观察节点{o_name_CN}({o})的后门调整集合。"
            Question_CN = f"方法1: 根据{adj_name_CN}的情况，我们逐个研究{t_name_CN}和{o_name_CN}的关联。\n方法2：我们直接研究一般情况下{t_name_CN}和{o_name_CN}的关联。\n为了理解{t_name_CN}如何影响{o_name_CN}，使用方法2比方法1更准确吗？"

        result_EN = {
            "Background": deepcopy(background),
            "Instruction": Instruction,
            "Question": Question,
            "Answer": Answer,
            "Reason": deepcopy(Reason),
            "Type": TaskType.BAS.name,
            "Mode": self.mode.name,
            "prob": None
        }
        if self.with_CN:
            result_CN = {
                "Background": deepcopy(background_CN),
                "Instruction": Instruction_CN,
                "Question": Question_CN,
                "Answer": Answer_CN,
                "Reason": deepcopy(Reason_CN),
                "Type": TaskType.BAS.name,
                "Mode": self.mode.name,
                "prob": None
            }
            return [(result_EN, result_CN)]
        else:
            return [result_EN]


    # backdoor adjustment set
    def gen_bas_question(self, t, o):
        background = self.gen_background()
        t_name, o_name = self.node_name[t], self.node_name[o]
        Reason = {}
        if self.with_CN:
            background_CN = self.gen_background_CN()
            t_name_CN, o_name_CN = self.node_name_CN[t], self.node_name_CN[o]
            Reason_CN = {}

        results = self.autoId_min.identify_backdoor(
            graph=self.dowhy_graph,
            treatment_name=t,
            outcome_name=o
        )
        adjust_set = results[0]["backdoor_set"]
        if len(adjust_set) > self.max_adjust_set: return []
        
        if len(adjust_set) == 0:
            # randomly sample a node
            c = random.choice(list(set(self.nodes) - {t, o}))
            adj_name = self.node_name[c]
            Reason["Step 1"] = f"Find a valid backdoor adjustment set: empty set. So method 2 is preferred and the answer is no."
            Answer = 'No'
            if self.with_CN:
                adj_name_CN = self.node_name_CN[c]
                Reason_CN["Step 1"] = f"找到一个合法的后门调整集合：空集。因此方法2更好，答案为否。"
                Answer_CN = "否"
        else:
            a_str = ','.join(list(adjust_set))
            adj_name = self.make_str_from_list([self.node_name[_] for _ in list(adjust_set)])
            Reason["Step 1"] = f"Find a valid backdoor adjustment set: {{{a_str}}}. So method 1 is preferred and the answer is yes."
            Answer = 'Yes'
            if self.with_CN:
                adj_name_CN = self.make_str_from_list([self.node_name_CN[_] for _ in list(adjust_set)], CN_flag=True)
                Reason_CN["Step 1"] = f"找到一个合法的后门调整集合：{{{a_str}}}。因此方法1更好，答案为是。"
                Answer_CN = "是"
            
        Instruction = f"Consider the backdoor adjustment set for treatment {t_name} ({t}) and outcome {o_name} ({o})."
        Question = f"Method 1: We look at how {t_name} correlates with {o_name} case by case according to {adj_name}.\nMethod 2: We look directly at how {t_name} correlates with {o_name} in general.\nTo understand how {t_name} affects {o_name}, is it more correct to use the Method 1 than Method 2?"
        if self.with_CN:
            Instruction_CN = f"考虑关于干预对象{t_name_CN}({t})和观察节点{o_name_CN}({o})的后门调整集合。"
            Question_CN = f"方法1: 根据{adj_name_CN}的情况，我们逐个研究{t_name_CN}和{o_name_CN}的关联。\n方法2：我们直接研究一般情况下{t_name_CN}和{o_name_CN}的关联。\n为了理解{t_name_CN}如何影响{o_name_CN}，使用方法1比方法2更准确吗？"

        result_EN = {
            "Background": deepcopy(background),
            "Instruction": Instruction,
            "Question": Question,
            "Answer": Answer,
            "Reason": deepcopy(Reason),
            "Type": TaskType.BAS.name,
            "Mode": self.mode.name,
            "prob": None
        }
        if self.with_CN:
            result_CN = {
                "Background": deepcopy(background_CN),
                "Instruction": Instruction_CN,
                "Question": Question_CN,
                "Answer": Answer_CN,
                "Reason": deepcopy(Reason_CN),
                "Type": TaskType.BAS.name,
                "Mode": self.mode.name,
                "prob": None
            }
            tmp = [(result_EN, result_CN)]
        else:
            tmp = [result_EN]
        return tmp + self._gen_bas_question_reverse(t, o)

    # collider bias
    def gen_cb_question(self, v, t, o, value):
        assert v in self.graph.descendants(t)
        assert v in self.graph.descendants(o)
        assert t not in self.graph.descendants(o)
        assert o not in self.graph.descendants(t)
    
        background = self.gen_background()
        v_name, t_name, o_name, vv_name = self.node_name[v], self.node_name[t], self.node_name[o], self.node_value[v][value]
        v_str, o_str = f"{v}={value}", f"{o}={1}"

        A = self._Count({v: value, t: 1, o: 1}) / max(1/self.sample_num, self._Count({v: value, t: 1}))
        B = self._Count({v: value, t: 0, o: 1}) / max(1/self.sample_num, self._Count({v: value, t: 0}))
        background["data_info"] += self._gen_cond_prob_description({v: value, t: 1}, {o: 1}, A)
        background["data_info"] += self._gen_cond_prob_description({v: value, t: 0}, {o: 1}, B)

        background["data_info_math"] += self._gen_cond_prob_math({v: value, t: 1}, {o: 1}, A)
        background["data_info_math"] += self._gen_cond_prob_math({v: value, t: 0}, {o: 1}, B)
        Reason = {}

        Instruction = ""
        if self.with_CN:
            Instruction_CN = ""
            background_CN = self.gen_background_CN()
            v_name_CN, t_name_CN, o_name_CN, vv_name_CN = self.node_name_CN[v], self.node_name_CN[t], self.node_name_CN[o], self.node_value_CN[v][value]
            background_CN["data_info"] += self._gen_cond_prob_description_CN({v: value, t: 1}, {o: 1}, A)
            background_CN["data_info"] += self._gen_cond_prob_description_CN({v: value, t: 0}, {o: 1}, B)

            background_CN["data_info_math"] += self._gen_cond_prob_math({v: value, t: 1}, {o: 1}, A)
            background_CN["data_info_math"] += self._gen_cond_prob_math({v: value, t: 0}, {o: 1}, B)            
            Reason_CN = {}

        
        results = []
        for i in range(4):
            if i == 0:
                Question = f"Given that {v_name} is {vv_name}, does it mean that {t_name} affects {o_name}?"
                Reason["Step 1"] = f"P({o_str}|do({t}=1),{v_str})-P({o_str}|do({t}=0),{v_str})=P({o_str}|{v_str})-P({o_str}|{v_str})=0"
                Reason["Step 2"] = f"Conditioned on {v}, {t} does not affect {o}, so the answer is no."
                Answer = "No"
                if self.with_CN:
                    Question_CN = f"给定{v_name_CN}为{vv_name_CN}的条件下，{t_name_CN}会影响{o_name_CN}吗？"
                    Reason_CN["Step 1"] = Reason["Step 1"]
                    Reason_CN["Step 2"] = f"在{v}已知的条件下，{t}不会影响{o}，因此答案为否。"
                    Answer_CN = "否"
            elif i == 1:
                Question = f"Given that {v_name} is {vv_name}, does it mean that {t_name} does not affect {o_name}?"
                Reason["Step 1"] = f"P({o_str}|do({t}=1),{v_str})-P({o_str}|do({t}=0),{v_str})=P({o_str}|{v_str})-P({o_str}|{v_str})=0"
                Reason["Step 2"] = f"Conditioned on {v}, {t} does not affect {o}, so the answer is yes."
                Answer = "Yes"
                if self.with_CN:
                    Question_CN = f"给定{v_name_CN}为{vv_name_CN}的条件下，{t_name_CN}不会影响{o_name_CN}吗？"
                    Reason_CN["Step 1"] = Reason["Step 1"]
                    Reason_CN["Step 2"] = f"在{v}已知的条件下，{t}不会影响{o}，因此答案为是。"
                    Answer_CN = "是"
            elif i == 2:
                Question = f"Given that {v_name} is {vv_name}, does it mean that {t_name} is correlated with {o_name}?"
                Reason["Step 1"] = f"P({o_str}|{t}=1,{v_str})-P({o_str}|{t}=0,{v_str})={A:.4f}-{B:.4f}={A-B:.4f}!=0"
                Reason["Step 2"] = f"Conditioned on {v}, {t} is correlated with {o}, so the answer is yes."
                Answer = "Yes"
                if self.with_CN:
                    Question_CN = f"给定{v_name_CN}为{vv_name_CN}的条件下，{t_name_CN}是和{o_name_CN}相关的吗？"
                    Reason_CN["Step 1"] = Reason["Step 1"]
                    Reason_CN["Step 2"] = f"在{v}已知的条件下，{t}和{o}是相关的，因此答案为是。"
                    Answer_CN = "是"
            elif i == 3:
                Question = f"Given that {v_name} is {vv_name}, does it mean that {t_name} is independet of {o_name}?"
                Reason["Step 1"] = f"P({o_str}|{t}=1,{v_str})-P({o_str}|{t}=0,{v_str})={A:.4f}-{B:.4f}={A-B:.4f}!=0"
                Reason["Step 2"] = f"Conditioned on {v}, {t} is not independent of {o}, so the answer is no."
                Answer = "No"
                if self.with_CN:
                    Question_CN = f"给定{v_name_CN}为{vv_name_CN}的条件下，{t_name_CN}是和{o_name_CN}独立的吗？"
                    Reason_CN["Step 1"] = Reason["Step 1"]
                    Reason_CN["Step 2"] = f"在{v}已知的条件下，{t}和{o}是相关的，因此答案为否。"
                    Answer_CN = "否"
            result_EN = {
                "Background": deepcopy(background),
                "Instruction": Instruction,
                "Question": Question,
                "Answer": Answer,
                "Reason": deepcopy(Reason),
                "Type": TaskType.CB.name,
                "Mode": self.mode.name,
                "prob": None
            }
            if self.with_CN:
                result_CN = {
                    "Background": deepcopy(background_CN),
                    "Instruction": Instruction_CN,
                    "Question": Question_CN,
                    "Answer": Answer_CN,
                    "Reason": deepcopy(Reason_CN),
                    "Type": TaskType.CB.name,
                    "Mode": self.mode.name,
                    "prob": None
                }
                results.append((result_EN, result_CN))
            else:
                results.append((result_EN))

        return results
        

    def generate_quesiton(self, task_types):
        all_results = []
        for task_type in task_types:
            if task_type == TaskType.CB:
                for v in self.nodes:
                    parents = self.graph.parents(v)
                    if len(parents) >= 2:
                        for t in parents:
                            for o in parents:
                                if o in self.graph.descendants(t) or t in self.graph.descendants(o): continue
                                for v0 in self.graph.descendants(v):
                                    for value in [0, 1]:
                                        results = getattr(self, f"gen_{task_type.name.lower()}_question")(v0, t, o, value)
                                        all_results.append(results)
            else:
                for t in self.nodes:
                    for o in self.nodes:
                        if (task_type == TaskType.ATE or task_type == TaskType.ETT or task_type == TaskType.CDE) and (t == o): continue
                        if (task_type == TaskType.NIE or task_type == TaskType.NDE or task_type == TaskType.PN or task_type == TaskType.PS or task_type == TaskType.BAS) and (o not in self.graph.descendants(t) or o == t): continue

                        if task_type == TaskType.PN or task_type == TaskType.PS:
                            for tmp in ['lb', 'ub']:
                                results = getattr(self, f"_gen_{task_type.name.lower()}_{tmp}_question")(t, o)
                                all_results.append(results)
                        elif task_type == TaskType.BAS:
                            results = getattr(self, f"gen_{task_type.name.lower()}_question")(t, o)
                            all_results.append(results)
                        else:
                            for tv, ov in permutations(2, 2):
                                results = getattr(self, f"gen_{task_type.name.lower()}_question")(t, tv, o, ov)
                                if len(results) == 0: continue
                                all_results.append(results)

        return all_results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str
    )
    parser.add_argument(
        "--output_prefix",
        type=str
    )
    parser.add_argument(
        "--node_num",
        type=int
    )
    return parser.parse_args()


def main(input_file, output_prefix, node_num):
    random.seed(1023)
    #task_types = [TaskType.ATE, TaskType.NDE, TaskType.NIE, TaskType.ETT, TaskType.CDE, TaskType.BAS, TaskType.CB, TaskType.PN, TaskType.PS]
    task_types = [TaskType.NDE]
    stats = {}
    N = node_num

    os.system(f'mkdir -p {output_prefix}')
    for task_type in task_types:
        stats[task_type.name] = {
            'Yes': 0,
            'No': 0,
        }

    max_length_EN = 0
    max_length_CN = 0

    jsObj = json.loads(open(input_file).read())
    for graph_id, graph in enumerate(jsObj):
        for mode in [Mode.REAL, Mode.FAKE, Mode.RANDOM]:
            output_file = os.path.join(output_prefix, 'English', f'N_{N}_graph_{graph_id}_{mode.name}.json')
            output_CN_file = os.path.join(output_prefix, 'Chinese', f'N_{N}_graph_{graph_id}_{mode.name}.json')
            os.system('mkdir -p {}'.format(os.path.dirname(output_file)))
            os.system('mkdir -p {}'.format(os.path.dirname(output_CN_file)))
            id = 0
            question_id = 0
            edge_sign = {}
            for edge in graph['edges']:
                edge_sign[edge] = graph['edge_sign'][edge]['value']

            causal_question = Causal_Question(
                graph["nodes"],
                graph["edges"],
                mode=mode,
                node_name=graph["node_name"],
                node_value=graph["node_value"],
                edge_sign=edge_sign,
                with_CN=True,
                node_name_CN=graph["node_name_CN"],
                node_value_CN=graph['node_value_CN']
            )

            if mode == Mode.REAL:
                fig = causal_question.graph.draw()
                fig.render(directory='tmp', filename=f'N_{N}_graph_{graph_id}', format='svg')
            final = {
                "Graph": {
                    "node": causal_question.nodes,
                    "edge": causal_question.edges,
                    "node_name": causal_question.node_name,
                    "node_value": causal_question.node_value,
                },
                "Data": {
                    "edge_sign": causal_question.edge_sign,
                    "node_bias": causal_question.node_bias,
                    "edge_weight": causal_question.edge_weight,
                    "function_description": causal_question.function_description,
                    "text_description": causal_question.text_description,
                    "Statistic_data": causal_question.data,
                },
                "Questions": [
                ]
            }
            final_CN = deepcopy(final)
            final_CN['Graph']['node_name'] = causal_question.node_name_CN
            final_CN['Graph']['node_value'] = causal_question.node_value_CN

            all_results = causal_question.generate_quesiton(task_types=task_types)
            for results in all_results:
                for result, result_CN in results:
                    assert 'prob' in result, result
                    assert 'prob' in result_CN, result_CN
                    final["Questions"].append({
                        "id": id,
                        "question_id": question_id,
                        **result
                    })
                    final_CN["Questions"].append({
                        "id": id,
                        "question_id": question_id,
                        **result_CN
                    })
                    if result['Type'] != 'PN' and result['Type'] != 'PS':
                        stats[result['Type']][result['Answer']] += 1
                    else:
                        stats[result['Type']]['Yes'] += 1
                    
                    max_length_EN = max(max_length_EN, count_words(result["Background"]["data_info"]))

                    max_length_CN = max(max_length_CN, len(result_CN["Background"]["data_info"]))

                    id += 1
                question_id += 1


            with open(output_file, "w") as f:
                f.write(json.dumps(final, indent="  ") + "\n")

            with open(output_CN_file, "w") as f:
                f.write(json.dumps(final_CN, indent="  ", ensure_ascii=False) + "\n")

    print('max_length_EN:', max_length_EN)
    print('max_length_CN:', max_length_CN)
    
    s_key = ''
    s_value = ''
    for key in stats:
        s_key += key + '\t'
        T, F = stats[key]["Yes"], stats[key]["No"]
        s_value += f"{T}:{F}\t"
    print(s_key)
    print(s_value)


if __name__ == "__main__":
    args = parse_args()
    main(args.input_file, args.output_prefix, args.node_num)
