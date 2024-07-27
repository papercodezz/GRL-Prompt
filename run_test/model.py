from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch.nn as nn
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch


class policy_network(nn.Module):

    def __init__(self,
                 model_config="bert-base-uncased",
                 add_linear=False,
                 embedding_size=128,
                 freeze_encoder=True) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("/data/model/bert-base-uncased")
        print("model_config:", model_config)
        self.model = AutoModelForTokenClassification.from_pretrained("/data/model/bert-base-uncased")

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



def test_policy_network():
    test_pids = [1]
    cand_pids = [0, 2, 4]
    problems = [
        "This is problem 0", "This is the first question", "Second problem is here", "Another problem",
        "This is the last problem"
    ]
    ctxt_list = [problems[pid] for pid in test_pids]
    cands_list = [problems[pid] for pid in cand_pids]

    model = policy_network(model_config="bert-base-uncased", add_linear=True, embedding_size=256)
    scores = model(ctxt_list, cands_list).cpu().detach().numpy()
    print(f"scores: {scores}")
    for i, test_pid in enumerate(test_pids):
        print(f"test_problem: {problems[test_pid]}")
        scores = scores[i, :].tolist()
        cand_rank = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        cand_pids = [cand_pids[cid] for cid in cand_rank]
        print(f"====== candidates rank: {[problems[pid] for pid in cand_pids]}")


if __name__ == "__main__":
    test_policy_network()
