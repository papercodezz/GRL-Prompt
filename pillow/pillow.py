import math
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np
import torch.nn.init as init


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


class MatchingNet(nn.Module):
    def __init__(self, device):
        super(MatchingNet, self).__init__()
        self.linear11 = nn.Linear(128,1024).to(device)
        self.linear12 = nn.Linear(1024,128).to(device)
        self.linear21 = nn.Linear(128, 1024).to(device)
        self.linear22 = nn.Linear(1024, 128).to(device)
        # Initialize the parameters
        self._initialize_weights()

    def _initialize_weights(self):
        # Apply initialization to each layer
        for layer in [self.linear11, self.linear12, self.linear21, self.linear22]:
            init.xavier_uniform_(layer.weight)  # Xavier/Glorot initialization for weights
            if layer.bias is not None:
                init.zeros_(layer.bias)  


    def forward(self,q_emb,p_l_emb,cand_emb):
        if p_l_emb != []:
            # q_emb += torch.mean(p_l_emb,1).reshape(-1,1)
            q_emb = q_emb + torch.mean(p_l_emb,1).reshape(-1,1)
        q_proj = self.linear12(self.linear11(q_emb))
        cand_proj = self.linear22(self.linear21(cand_emb))

        score = F.softmax(torch.matmul(cand_proj,q_proj.T),dim=0)
        # max_scores, indices = torch.max(score, dim=0)
        # return indices, torch.log(max_scores)

        # 返回每个 query 的 top-2 候选项和对应的得分
        top_2_scores, top_2_indices = torch.topk(score, k=2, dim=0)
        return top_2_indices, top_2_scores


class Pillow(nn.Module):
    def __init__(self, 
                 model_config="bert-uncased-small",
                 add_linear=False,
                 embedding_size=128,
                 freeze_encoder=True, hidden_channels=64, out_channels=128, num_heads=2, num_layers=2, device = 'cpu' ) -> None:
        super(Pillow, self).__init__()
        self.emb = Embedding(model_config, add_linear, embedding_size, freeze_encoder).to(device)
        self.matching_net = MatchingNet(device)
        self.device = device
        self.m_shots = 2

        # Adding a dropout layer
        # self.dropout = nn.Dropout(p=0.3)

    def forward(self, q_batch, can_list):
        q_emb = self.emb(q_batch)
        can_emb = self.emb(can_list)

        # q_emb = self.dropout(q_emb)
        # can_emb = self.dropout(can_emb)

        # p_l = []
        # log_prob = 0
        p_l = torch.zeros((len(q_batch), self.m_shots), dtype=torch.int64, device=self.device)
        # p_l = np.zeros((len(q_batch),self.m_shots),dtype=int)
        log_prob = torch.zeros(len(q_batch), device=self.device)
        # log_prob = np.zeros(len(q_batch))
        for i in range(self.m_shots):
            if i == 0:
                p_l_emb = []
            else:
                #  p_l[:,i-1]
                the_p_l = [can_list[j] for j in p_l[:,i-1]]
                p_l_emb = self.emb(the_p_l).to(device)
                # p_l_emb = self.dropout(p_l_emb)
            
            # idx, prob_norm = self.matching_net(q_emb,p_l_emb,can_emb)
            top_2_indices, top_2_scores = self.matching_net(q_emb, p_l_emb, can_emb)
            print(f'top_2_indices shape:{top_2_indices.shape}\ntop_2_scores:{top_2_scores.shape}')

            if i==0:
                p_l[:,i] = top_2_indices[0]
                log_prob += sum(top_2_scores[0])
            else:
                for j in range(top_2_indices.shape[1]):
                    if top_2_indices[0][j] not in p_l[j][:i]:
                        p_l[j][i]= top_2_indices[0][j]
                        log_prob +=top_2_scores[0][j]
                    else:
                        p_l[j][i]=top_2_indices[1][j]
                        log_prob +=top_2_scores[1][j]


        return p_l, log_prob



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
    model = Pillow(data, model_config="bert-uncased-small", add_linear=False, device = device)
    p_l, reward_total = model(ctxt_list, cands_list)
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
