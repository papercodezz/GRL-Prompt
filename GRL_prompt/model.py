import math
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import dgl
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.datasets import DBLP
from torch_geometric.nn import HGTConv, Linear
import os.path as osp

class HGT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers, data):
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in data.node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, data.metadata(),
                           num_heads)
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        x_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in x_dict.items()
        }

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        return {
            node_type: self.lin(x)
            for node_type, x in x_dict.items()
        }


class sim_cal_model(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        # print("model_config:", model_config)
        self.model = model = BertModel.from_pretrained(model_path)

    def get_bert_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True).to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # 取 [CLS] token 的输出作为句子的表示
        return outputs.last_hidden_state[:, 0, :].cpu().numpy()

    def cal_sim(self, text1, text2):
        vec1 = self.get_bert_embedding(text1)
        vec2 = self.get_bert_embedding(text2)
        sim = cosine_similarity(vec1, vec2)
        return sim[0][0]
    

class Embedding(nn.Module):
    def __init__(self,
                 model_config="bert-uncased-small",
                 add_linear=False,
                 embedding_size=128,
                 freeze_encoder=True) -> None:
        super(Embedding, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("../pretrain/{}".format(model_config))
        print("model_config:", model_config)
        self.model = AutoModelForTokenClassification.from_pretrained("../pretrain/{}".format(model_config))

        # Freeze transformer encoder and only train the linear layer
        if freeze_encoder:
            for param in self.model.parameters():
                param.requires_grad = False

        if add_linear:
            # Add an additional small, adjustable linear layer on top of BERT tuned through RL
            self.embedding_size = embedding_size
            self.linear = nn.Linear(self.model.config.hidden_size,
                                    embedding_size)  # 768 for bert-base-uncased, distilbert-base-uncased
        else:
            self.linear = None

    def forward(self, input_list):
        input = self.tokenizer(input_list, truncation=True, padding=True, return_tensors="pt").to(self.model.device)
        # print(f"input: {input}")
        output = self.model(**input, output_hidden_states=True)
        # Get last layer hidden states
        last_hidden_states = output.hidden_states[-1]
        # Get [CLS] hidden states
        sentence_embedding = last_hidden_states[:, 0, :]  # len(input_list) x hidden_size
        # print(f"sentence_embedding: {sentence_embedding}")

        if self.linear:
            sentence_embedding = self.linear(sentence_embedding)  # len(input_list) x embedding_size

        return sentence_embedding


class Data():
    def __init__(self, cand):
        self.num_node = len(cand) + 1
        self.node_types = ["q", "can"]

    def metadata(self):
        return (self.node_types, [("q", "to", "can"), ("can", "to", "q"), ("can", "to", "can")])


def all_connect(num_can):
    rlt = [[], []]
    for i in range(num_can):
        for j in range(0, i):
            if i != j:
                rlt[0].append(i)
                rlt[1].append(j)
                rlt[0].append(j)
                rlt[1].append(i)
    return rlt


def construct_edge_index_dict(num_can):
    edge_index_dict = dict()
    edge_index_dict[("q", "to", "can")] = torch.LongTensor([[0 for _ in range(num_can)], [i for i in range(num_can)]])
    edge_index_dict[("can", "to", "q")] = torch.LongTensor([[i for i in range(num_can)], [0 for _ in range(num_can)]])
    edge_index_dict[("can", "to", "can")] = torch.LongTensor(all_connect(num_can))
    return edge_index_dict


def construct_graph_input(can, query):
    x_dict = dict(q=query, can=can)
    return x_dict, construct_edge_index_dict(len(can))


class SimPredictor(nn.Module):
    def __init__(self, in_features, device = 'cpu'):
        super().__init__()
        self.W = nn.Linear(in_features, in_features).to(device)

    def forward(self,q,can):
        return F.sigmoid(torch.matmul(self.W(can),q.T)/math.sqrt(256))


class SinePredictor(nn.Module):
    def __init__(self, in_features, device = 'cpu'):
        super().__init__()
        self.W = nn.Linear(in_features, 1).to(device)

    def forward(self, graph, h):
        s = h[graph.edges()[0]]
        o = h[graph.edges()[1]]
        score = self.W(torch.sin(s-o)) # 90 * 1
        score = F.softmax(score.reshape(-1, 2), dim=1).reshape(-1, 1)
        return score, score> 0.5

class PolicyNet(nn.Module):
    def __init__(self, data,
                 model_config="bert-uncased-small",
                 add_linear=False,
                 embedding_size=128,
                 freeze_encoder=True, hidden_channels=64, out_channels=128, num_heads=2, num_layers=2, device = 'cpu' ) -> None:
        super(PolicyNet, self).__init__()
        self.emb = Embedding(model_config, add_linear, embedding_size, freeze_encoder)
        self.gnn = HGT(hidden_channels=hidden_channels, out_channels=out_channels, num_heads=num_heads,
                       num_layers=num_layers, data=data)
        self.linkpred = SinePredictor(out_channels*2, device)
        self.selectn = SimPredictor(out_channels*2, device)
        self.device = device

    def forward(self, q_batch, can_list):
        q_emb = self.emb(q_batch)  # batch_size * hidden_size(768)
        can_emb = self.emb(can_list) # num_can * hidden_size(768)
        x_dict, edge_index_dict = construct_graph_input(can_emb, q_emb) # x_dict: 2, edge_index_dict: 3
        node_emb = self.gnn(x_dict, edge_index_dict) # node_emb: dict, node_emb-q: num_can * out_channels, node_emb-can: batch_size * out_channels
        
        g =  dgl.graph((edge_index_dict[("can", "to", "can")][0],edge_index_dict[("can", "to", "can")][1]), idtype=torch.int32, device=self.device)
       
        can = torch.cat([can_emb, node_emb['can']], -1).to(self.device)  # num_can * (hidden_size+out_channels)
        q = torch.cat([q_emb, node_emb['q']], -1).to(self.device)   # batch_size * (hidden_size+out_channels)
        link_pred,edge_index = self.linkpred(g, can)
        select_score = self.selectn(q,can) # num_can * batch_size

        return (link_pred,edge_index,select_score,edge_index_dict)


def test_policy_network():
    test_pids = [1]
    cand_pids = [0, 2, 4]
    problems = [
        "This is problem 0", "This is the first question", "Second problem is here", "Another problem",
        "This is the last problem"
    ]
    ctxt_list = [problems[pid] for pid in test_pids]
    cands_list = [problems[pid] for pid in cand_pids]
    data = Data(cands_list)
    model = PolicyNet(data, model_config="bert-uncased-small", add_linear=False, device = device)
    link_pred,edge_index,select_score = model(ctxt_list, cands_list)
    print(f"scores: {scores}")
    for i, test_pid in enumerate(test_pids):
        print(f"test_problem: {problems[test_pid]}")
        scores = scores[i, :].tolist()
        cand_rank = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        cand_pids = [cand_pids[cid] for cid in cand_rank]
        print(f"====== candidates rank: {[problems[pid] for pid in cand_pids]}")

device = torch.device("cuda:" + "0" if torch.cuda.is_available() else "cpu")  # one GPU


if __name__ == "__main__":
    test_policy_network()