import os
import sys
import math
import json
import argparse
import random
import time
import torch
import openai
import requests
import numpy as np
import torch.nn.functional as F
sys.path.append("../")
from functools import lru_cache
from tools import utils
from base_prompt import *
from model import *

# from new_model import PolicyNet,Data
import networkx as nx
from topological_sort import return_order_prob
from collections import defaultdict
from transformers import BertTokenizer, BertModel
from fuzzywuzzy import fuzz
from retrying import retry

import sacrebleu
from rouge_score import rouge_scorer

openai.api_key = os.getenv("OPENAI_API_KEY")

def load_data(args):
    # problems = json.load(open("/data/PromptPG-main/data/dolly_train_data.json"))
    problems = json.load(open("/data/PromptPG-main/data/alpaca_train_data.json"))
    pids = list(problems.keys())

    samples = random.sample(pids, args.train_number + args.cand_number)  # random sample
    train_pids = samples[:args.train_number]
    cand_pids = samples[args.train_number:]

    return problems, cand_pids, train_pids


def get_output(prompt, args):
    if "gpt-4" in args.model:
        return call_gpt4(prompt, args.temperature, args.max_tokens, args.top_p, args.frequency_penalty,
                        args.presence_penalty)
    elif "gpt" in args.model:
        return call_gpt3(prompt, args.temperature, args.max_tokens, args.top_p, args.frequency_penalty,
                        args.presence_penalty)
    elif "llama" in args.model:
        return call_llama(prompt, args.temperature, args.max_tokens, args.top_p, args.frequency_penalty,
                        args.presence_penalty)
    else:
        print("please check generated model type!")
        sys.exit(1)


def call_gpt4(prompt, temperature, max_tokens, top_p, frequency_penalty, presence_penalty):
    patience = 100
    while True:
        try:
            response = openai.ChatCompletion.create(model='gpt-4o-mini',
                                                messages=[
                                                    {"role":"user","content":prompt}
                                                ],
                                                temperature=temperature,
                                                max_tokens=max_tokens,
                                                top_p=top_p,
                                                frequency_penalty=frequency_penalty,
                                                presence_penalty=presence_penalty,
                                                stop=["\n"],
                                                logprobs=True,
                                                )
            output = response["choices"][0]["message"]['content'].strip()
            print(f"response: {output}")
            token_logprobs =[i['logprob'] for i in  response["choices"][0]["logprobs"]["content"]]
            
            avg_log_prob = sum(token_logprobs) / len(token_logprobs)
            ppl = math.exp(-avg_log_prob)
            break
        except Exception as e:
            patience -= 1
            if not patience:
                print("!!! Running out of patience waiting for OpenAI")
            else:
                print(e)
                time.sleep(0.1)
    return output.strip(), ppl


@retry(stop_max_attempt_number=10, wait_fixed=200)
def call_llama(prompt, temperature, max_tokens, top_p, frequency_penalty, presence_penalty):
    # print(prompt)
    api_data = {
        "model": "llama",
        "prompt":prompt,
        "temperature": temperature,
        "n":1,
        "max_tokens": max_tokens,
        "frequency_penalty":frequency_penalty,
        "presence_penalty":presence_penalty,
        "stop":["\n"],
        "logprobs": True
    }
    try:
        openai_api_base = "http://0.0.0.0:9201/v1/completions"
        response = requests.post(url=openai_api_base, json=api_data, timeout=80)
        output=response.json()['choices'][0]['text']
        print(f"response: {output}")
        token_logprobs = response.json()['choices'][0]['logprobs']['token_logprobs']
        avg_log_prob = sum(token_logprobs) / len(token_logprobs)
        ppl = math.exp(-avg_log_prob)
    except Exception as e:
            print(e)

    return output.strip(),ppl


def call_gpt3(prompt, temperature, max_tokens, top_p, frequency_penalty, presence_penalty):
    patience = 100
    while True:
        try:
            response = openai.ChatCompletion.create(model='gpt-3.5-turbo-0125',
                                                messages=[
                                                    {"role":"user","content":prompt}
                                                ],
                                                temperature=temperature,
                                                max_tokens=max_tokens,
                                                top_p=top_p,
                                                frequency_penalty=frequency_penalty,
                                                presence_penalty=presence_penalty,
                                                stop=["\n"],
                                                logprobs=True,
                                                )
            output = response["choices"][0]["message"]['content'].strip()
            print(f"response: {output}")
            token_logprobs =[i['logprob'] for i in  response["choices"][0]["logprobs"]["content"]]
            
            avg_log_prob = sum(token_logprobs) / len(token_logprobs)
            ppl = math.exp(-avg_log_prob)
            break
        except Exception as e:
            patience -= 1
            if not patience:
                print("!!! Running out of patience waiting for OpenAI")
            else:
                print(e)
                time.sleep(0.1)
    return output.strip(), ppl



def cal_rouge(all_references, all_predictions, scorer):
    # 计算 ROUGE 分数
    rouge1_f1 = 0.0
    rouge2_f1 = 0.0
    rougeL_f1 = 0.0

    for ref, pred in zip(all_references, all_predictions):
        scores = scorer.score(ref, pred)
        rouge1_f1 += scores['rouge1'].fmeasure
        rouge2_f1 += scores['rouge2'].fmeasure
        rougeL_f1 += scores['rougeL'].fmeasure

    # 取平均值
    rouge1_f1 /= len(all_references)
    rouge2_f1 /= len(all_references)
    rougeL_f1 /= len(all_references)

    return rouge1_f1, rouge2_f1, rougeL_f1

def sample_policy(num_can,select_score,sample_round = 1000):
    rlt_sample = defaultdict(int)
    for _ in range(sample_round):
        selected = []
        sample_r = [np.random.rand(1) for _ in range(num_can)]
        for i in range(num_can):
            if sample_r[i][0] < select_score[i].detach().cpu().numpy():
                selected.append(i)
        rlt_sample[tuple(selected)] += 1
    max = 0
    rlt = []
    for k,v in rlt_sample.items():
        if v > max:
            if k != ():
                rlt = list(k)
    if rlt == []:
        rlt = [int(torch.argmax(select_score))]
    return rlt

def get_log_prob(rlt, s_index):
    log_prob = 0
    link_pred, edge_index, select_score, edge_index_dict = rlt
    edge_index = edge_index.squeeze().detach().cpu().numpy()
    num_can = select_score.shape[0]
    c_t_c = edge_index_dict[("can","to","can")]
    num_edge = c_t_c[0].shape[0]
    # sample_r = [np.random.rand(1) for _ in range(num_can)]
    # selected = []
    # for i in range(num_can):
    #     if sample_r[i].data < select_score[i].detach().numpy():
    #         selected.append(i)
    the_select_score = select_score[:,s_index].reshape(-1,1)
    selected = sample_policy(num_can,the_select_score)
    order,edge_rlt_index = return_order_prob(rlt,selected)
    # selected_edge_index = [True for _ in range(num_edge)]
    # selected_edge_index_af = [edge_index[i] and selected_edge_index[i]  for i in range(num_edge)]
    # for i in range(2):
    #     for j in range(num_edge):
    #         if c_t_c[i][j].detach().numpy() not in selected:
    #             selected_edge_index[j] = False
    # selected_edge = c_t_c[:,selected_edge_index_af].detach().numpy()
    # edge_list = []
    # for i in range(selected_edge.shape[1]):
    #     edge_list.append((selected_edge[0][i],selected_edge[1][i]))
    # DG = nx.DiGraph(edge_list)
    # sorted = list(nx.topological_sort(DG))
    # np.random.shuffle(selected)
    for ele in selected:
        log_prob += torch.log(the_select_score[ele])
    for ele in edge_rlt_index:
        if ele >= len(the_select_score):
            continue
        log_prob += torch.log(the_select_score[ele])
    return order, log_prob


def get_batch_reward_loss_new(the_output, cand_pids, pid_batch, option_batch, unit_batch, label_batch, args, sim_model):

    batch_loss = 0
    batch_reward = 0

    output_list = []
    ppl_list = []

    ## loop over the training examples
    for i in range(len(pid_batch)):

        prompt_sorted,log_prob = get_log_prob(the_output, i)

        shot_pids = [cand_pids[cid] for cid in prompt_sorted[:args.shot_number]]
        
        prompt = build_prompt(problems, shot_pids, pid_batch[i], args)
        print(f"prompt: {prompt}")
        # sys.exit(1)
        # get the output from GPT-3
        output, ppl = get_output(prompt, args)
        # output = ""
        output_list.append(output)
        ppl_list.append(ppl)
        prediction = output

        prediction_norm = prediction


        scores_sim = sim_model.cal_sim(prediction_norm, label_batch[i])
        scores_texual = fuzz.ratio(prediction_norm, label_batch[i])*1.0/100

        _reword_lambda=0.4
        _reward = _reword_lambda*scores_texual + (1-_reword_lambda)*scores_sim
        print(_reward)
        batch_reward += _reward
        temp_reward = (_reward - 0.5) * 2 * np.pi
        batch_loss -= np.tanh(temp_reward) * log_prob

        cids = []

    return cids, batch_reward, batch_loss, output_list, ppl_list


def policy_gradient_train_new(device,problems, train_pids, cand_pids, cand_examples, args):
    # REINFORCE
    # if os.path.exists(args.ckpt_path):
    #     print("!!! Model dir already exists. Consider load it instead of training again.")
    data = Data(cand_examples)
    policy_model = PolicyNet(data, model_config=args.model_config, add_linear=True, device = device)

    sim_model = sim_cal_model("../pretrain/bert-base-uncased").to(torch.device("cuda:0"))

    optimizer = torch.optim.Adam(policy_model.parameters(), lr=args.lr)

    train_samples, train_labels, units, options = [], [], [], []
    for pid in train_pids:
        train_samples.append(create_example_from_pid(
            pid, problems, args, test=True))  # Set test=True to avoid answer being added to the training input.
        answer_norm = problems[pid]['answer']
        train_labels.append(answer_norm)
        units.append("")
        options.append("")

    num_batch = math.ceil(len(train_samples) / args.batch_size)

    reward_history = []
    loss_history = []

    total_reward_history = []  # epoch based
    total_loss_history = []  # epoch based

    total_rouge1_history = []
    total_rouge2_history = []
    total_rougeL_history = []
    total_bleu_history = []
    total_ppl_history = []

    STOP_FLAG = False

    # 初始化 ROUGE 计算器
    the_rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    for epoch in range(args.epochs):
        logger.write(f"Epoch: {epoch}")

        total_train_reward = 0
        total_train_loss = 0

        # 用于存储所有预测和参考文本的列表
        all_references = []
        all_predictions = []
        all_ppls = []

        # We can simply set the batch_size to len(train_data) in few-shot setting.
        for batch_i in range(num_batch):
            logger.write(f"Batch: {batch_i}")
            train_batch = train_samples[batch_i * args.batch_size:(batch_i + 1) * args.batch_size]
            label_batch = train_labels[batch_i * args.batch_size:(batch_i + 1) * args.batch_size]
            pid_batch = train_pids[batch_i * args.batch_size:(batch_i + 1) * args.batch_size]
            unit_batch = units[batch_i * args.batch_size:(batch_i + 1) * args.batch_size]
            option_batch = options[batch_i * args.batch_size:(batch_i + 1) * args.batch_size]


            rlt = policy_model(train_batch, cand_examples)

            cids, reward, loss, predictions, ppls = get_batch_reward_loss_new(rlt, cand_pids, pid_batch, option_batch, unit_batch,
                                                       label_batch, args, sim_model)
            
            all_references.extend(label_batch)
            all_predictions.extend(predictions)
            all_ppls.extend(ppls)

            logger.write(f"cids for sample[-1] in batch: {cids}")
            # logger.write(f"Cand prob for sample[-1] in batch: {[round(x,5) for x in scores[-1, :].tolist()]}")
            logger.write(f"### reward for the batch: {reward}")
            logger.write(f"### loss for the batch: {loss}\n")

            the_rouge1_f1, the_rouge2_f1, the_rougeL_f1 = cal_rouge(label_batch, predictions, the_rouge_scorer)
            logger.write(f"### ROUGE for the batch: ROUGE-1: {the_rouge1_f1:.4f}, ROUGE-2: {the_rouge2_f1:.4f}, ROUGE-L: {the_rougeL_f1:.4f}\n")

            the_bleu = sacrebleu.corpus_bleu(label_batch, [predictions])
            logger.write(f"### BLEU for the batch: {the_bleu.score:.4f}\n")

            the_ppl = sum(ppls) / len(ppls)
            logger.write(f"### PPL for the batch: {the_ppl:.4f}\n")

            # linear layer has Weight and bias
            # prev_param = list(policy_model.linear.parameters())[0].clone()
            # print(f"prev_param: {prev_param.data}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # for each iteration/batch
            total_train_reward += reward
            total_train_loss += loss.item()

            reward_history.append(reward)
            loss_history.append(loss.item())

            if np.isnan(loss.item()):
                STOP_FLAG = True
                break
    #
    #     # for each epoch
        all_rouge1_f1, all_rouge2_f1, all_rougeL_f1 = cal_rouge(all_references, all_predictions, the_rouge_scorer)
        total_rouge1_history.append(all_rouge1_f1)
        total_rouge2_history.append(all_rouge2_f1)
        total_rougeL_history.append(all_rougeL_f1)

        all_bleu = sacrebleu.corpus_bleu(all_references, [all_predictions])
        total_bleu_history.append(all_bleu.score)

        total_ppl_history.append(sum(all_ppls) / len(all_ppls))

        total_reward_history.append(total_train_reward)
        total_loss_history.append(total_train_loss)
    
        best_reward = max(total_reward_history)
        best_loss = min(total_loss_history)
    
        best_reward_epoch = total_reward_history.index(best_reward)
        best_loss_epoch = total_loss_history.index(best_loss)

        best_rouge1 = max(total_rouge1_history)
        best_rouge2 = max(total_rouge2_history)
        best_rougeL = max(total_rougeL_history)
        best_bleu = max(total_bleu_history)
        best_ppl = min(total_ppl_history)
    
        logger.write("============================================")
        logger.write(f"### Epoch: {epoch} / {args.epochs}")
        logger.write(f"### Total reward: {total_train_reward}, " + f"Total loss: {round(total_train_loss,5)}, " +
                     f"Best reward: {best_reward} at epoch {best_reward_epoch}, " +
                     f"Best loss: {round(best_loss, 5)} at epoch {best_loss_epoch}\n")

        logger.write(f"### Total ROUGE-1: {all_rouge1_f1:.4f}, " + f"Total ROUGE-2: {all_rouge2_f1:.4f}, " +
                     f"Total ROUGE-L: {all_rougeL_f1:.4f}, " + f"Total BLEU: {all_bleu.score:.4f}\n")
        logger.write(f"### Best ROUGE-1: {best_rouge1:.4f}, " + f"Best ROUGE-2: {best_rouge2:.4f}, " +
                     f"Best ROUGE-L: {best_rougeL:.4f}, " + f"Best BLEU: {best_bleu:.4f}\n")
        logger.write(f"### Total BLEU: {all_bleu.score:.4f}, " + f"Best BLEU: {best_bleu:.4f}\n")
        logger.write(f"### Total PPL: {sum(all_ppls) / len(all_ppls):.4f}, " + f"Best PPL: {best_ppl:.4f}\n")

        # save every epoch for each component
        emb_file = os.path.join(args.ckpt_path, f"emb_{epoch}.pt")
        torch.save(policy_model.emb.linear.state_dict(), emb_file)
        logger.write(f"saved the emb to {emb_file}")

        gnn_file = os.path.join(args.ckpt_path, f"gnn_{epoch}.pt")
        torch.save(policy_model.gnn.state_dict(), gnn_file)

        linkpred_file = os.path.join(args.ckpt_path, f"linkpred_{epoch}.pt")
        torch.save(policy_model.linkpred.state_dict(), linkpred_file)

        selectn_file = os.path.join(args.ckpt_path, f"selectn_{epoch}.pt")
        torch.save(policy_model.selectn.state_dict(), selectn_file)

    
        # save best epoch
        if epoch == best_reward_epoch:
            emb_file = os.path.join(args.ckpt_path, "emb_best_reward.pt")
            torch.save(policy_model.emb.linear.state_dict(), emb_file)
            logger.write(f"saved the best reward ckpt to {emb_file}")

            gnn_file = os.path.join(args.ckpt_path, f"gnn_best_reward.pt")
            torch.save(policy_model.gnn.state_dict(), gnn_file)

            linkpred_file = os.path.join(args.ckpt_path, f"linkpred_best_reward.pt")
            torch.save(policy_model.linkpred.state_dict(), linkpred_file)

            selectn_file = os.path.join(args.ckpt_path, f"selectn_best_reward.pt")
            torch.save(policy_model.selectn.state_dict(), selectn_file)

    
        if epoch == best_loss_epoch:
            emb_file = os.path.join(args.ckpt_path, "emb_best_loss.pt")
            torch.save(policy_model.emb.linear.state_dict(), emb_file)
            logger.write(f"saved the best reward ckpt to {emb_file}")

            gnn_file = os.path.join(args.ckpt_path, f"gnn_best_loss.pt")
            torch.save(policy_model.gnn.state_dict(), gnn_file)

            linkpred_file = os.path.join(args.ckpt_path, f"linkpred_best_loss.pt")
            torch.save(policy_model.linkpred.state_dict(), linkpred_file)

            selectn_file = os.path.join(args.ckpt_path, f"selectn_best_loss.pt")
            torch.save(policy_model.selectn.state_dict(), selectn_file)
    
        # save reward and loss history
        history = {
            "reward_history": reward_history,
            "loss_history": loss_history,
            "total_reward_history": total_reward_history,
            "total_loss_history": total_loss_history,
            "total_rouge1_history": total_rouge1_history,
            "total_rouge2_history": total_rouge2_history,
            "total_rougeL_history": total_rougeL_history,
            "total_bleu_history": total_bleu_history,
            "total_ppl_history": total_ppl_history,
        }
        history_file = os.path.join(args.ckpt_path, "history.json")
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2, separators=(',', ': '))
    
        # print cache info
    
        if STOP_FLAG:
            break
    
    # save in the end
    # ckpt_file = os.path.join(args.ckpt_path, "ckpt_final.pt")
    # torch.save(policy_model.linear.state_dict(), ckpt_file)

    emb_file = os.path.join(args.ckpt_path, "emb_final.pt")
    torch.save(policy_model.emb.linear.state_dict(), emb_file)
    logger.write(f"saved the best reward ckpt to {emb_file}")

    gnn_file = os.path.join(args.ckpt_path, f"gnn_final.pt")
    torch.save(policy_model.gnn.state_dict(), gnn_file)

    linkpred_file = os.path.join(args.ckpt_path, f"linkpred_final.pt")
    torch.save(policy_model.linkpred.state_dict(), linkpred_file)

    selectn_file = os.path.join(args.ckpt_path, f"selectn_final.pt")
    torch.save(policy_model.selectn.state_dict(), selectn_file)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='../data/tabmwp')
    parser.add_argument('--model', type=str, default='gpt-4')
    parser.add_argument('--option_inds', type=list, default=["A", "B", "C", "D", "E", "F"])

    # User options
    parser.add_argument('--label', type=str, default='exp0')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument(
        '--prompt_format',
        type=str,
        default='Q-A',
        choices=['T-A', 'Q-A', 'Q-AS', 'Q-SA', 'TQ-A', 'TQ-AS', 'TQ-SA', 'QT-A', 'QT-AS', 'QT-SA', 'QTS-A', 'TQS-A'],
        help='prompt format template')
    parser.add_argument('--shot_number', type=int, default=2, help='Number of n-shot training examples.')
    parser.add_argument('--seed', type=int, default=1, help='random seed')

    # GPT-3 settings
    parser.add_argument('--engine', type=str, default='davinci-002', choices=['davinci-002', 'ada'])
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--max_tokens',
                        type=int,
                        default=512,
                        help='The maximum number of tokens allowed for the generated answer.')
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--frequency_penalty', type=float, default=0.0)
    parser.add_argument('--presence_penalty', type=float, default=0.0)

    # Policy gradient settings
    parser.add_argument('--gpu', type=str, default='-1')
    parser.add_argument('--model_config',
                        type=str,
                        default='bert-uncased-small',
                        choices=['distilbert-base-uncased', 'bert-base-uncased','bert-uncased-small'])
    parser.add_argument('--train_number', type=int, default=1, help='Number of training samples.')
    parser.add_argument('--cand_number', type=int, default=10, help='Number of candidate prompts.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate of policy network.')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs.')
    parser.add_argument('--embedding_size', type=int, default=128, help='Policy network final layer hidden state size.')
    parser.add_argument('--batch_size',
                        type=int,
                        default=1,
                        help='Policy network training batch size. Set to train_number by default.')
    parser.add_argument('--ckpt_root', type=str, default='../checkpoints')

    args = parser.parse_args()

    # print and save the args
    args.ckpt_path = os.path.join(args.ckpt_root, args.label)
    utils.create_dir(args.ckpt_path)
    _logger = utils.Logger(args.ckpt_path + '/args.txt')

    print('====Input Arguments====')
    _logger.write(json.dumps(vars(args), indent=2, sort_keys=False))

    return args


if __name__ == '__main__':

    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)  # CPU random seed
    torch.cuda.manual_seed(args.seed)  # GPU random seed
    torch.backends.cudnn.benchmark = True

    problems, cand_pids, train_pids = load_data(args)


    device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")  # one GPU

    ## construct candidate examples
    cand_examples = []
    for pid in cand_pids:
        example = create_example_from_pid(pid, problems, args, test=True)
        cand_examples.append(example)

    ## TRAINING
    logger = utils.Logger(os.path.join(args.ckpt_path, 'log.txt'))

    policy_gradient_train_new(device, problems, train_pids, cand_pids, cand_examples, args)
