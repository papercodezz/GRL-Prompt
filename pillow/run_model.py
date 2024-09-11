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
from utilities import extract_prediction, normalize_answer
from pillow import Pillow

# from new_model import PolicyNet,Data
import networkx as nx
from collections import defaultdict
from transformers import BertTokenizer, BertModel
from fuzzywuzzy import fuzz
from retrying import retry

import sacrebleu
from rouge_score import rouge_scorer
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from learn_policy import sim_cal_model


openai.api_key = os.getenv("OPENAI_API_KEY")



def load_data(args):
    # problems_test = json.load(open(os.path.join(args.data_root, f'problems_{args.test_split}.json')))
    # problems_train = json.load(open(os.path.join(args.data_root, f'problems_train.json')))
    problems_train = json.load(open("/data/PromptPG-main/data/dolly_train_data.json"))
    problems_test = json.load(open("/data/PromptPG-main/data/dolly_test_data.json"))
    problems = {**problems_test, **problems_train}

    # test problem ids
    test_pids = list(problems_test.keys())
    test_pids = test_pids[:args.test_number] if args.test_number > 0 else test_pids
    print(f"number of test problems: {len(test_pids)}\n")

    # pick up shot/in-context example candidates from the training set
    train_pids = list(problems_train.keys())

    cand_pids = random.sample(train_pids, args.cand_number)  # random sample

    return problems, test_pids, cand_pids, train_pids


@retry(stop_max_attempt_number=10, wait_fixed=200)
def get_llama_output(prompt,args):
    api_data = {
        "model": "llama",
        "prompt":prompt,
        "temperature": args.temperature,
        "n":1,
        "max_tokens": args.max_tokens,
        "frequency_penalty": args.frequency_penalty,
        "presence_penalty": args.presence_penalty,
        "stop":["\n"],
        "logprobs": True
    }
    openai_api_base = "http://0.0.0.0:9201/v1/completions"
    response = requests.post(url=openai_api_base, json=api_data, timeout=80)
    output=response.json()['choices'][0]['text']
    print(f"response: {output}")

    token_logprobs = response.json()['choices'][0]['logprobs']['token_logprobs']
    avg_log_prob = sum(token_logprobs) / len(token_logprobs)
    ppl = math.exp(-avg_log_prob)

    return output.strip(), ppl



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


def get_gpt4_output(prompt, args):
    patience = 100
    while True:
        try:
            response = openai.ChatCompletion.create(model="gpt-4o-mini-2024-07-18",
                                                messages=[
                                                    {"role":"user","content":prompt}
                                                ],
                                                temperature=args.temperature,
                                                max_tokens=args.max_tokens,
                                                top_p=args.top_p,
                                                frequency_penalty=args.frequency_penalty,
                                                presence_penalty=args.presence_penalty,
                                                stop=["\n"],
                                                logprobs=True,
                                                )
            output = response["choices"][0]["message"]['content'].strip()

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

def get_result_file(args):
    result_path = f"{args.output_root}/{args.model}"
    os.makedirs(result_path, exist_ok=True)

    result_file = "{}/{}_{}_{}_{}_seed_{}.json".format(result_path, args.label, args.test_split, args.prompt_format,
                                                       args.shot_number, args.seed)

    return result_file


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


def save_results(result_file, acc, correct, count, cand_pids, args, results):
    data = {}
    data['acc'] = acc
    data['correct'] = correct
    data['count'] = count
    data['cand_pids'] = cand_pids
    data['args'] = vars(args)
    data['results'] = results

    with open(result_file, 'w') as f:
        json.dump(data, f, indent=2, separators=(',', ': '))





def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='../data/tabmwp')
    parser.add_argument('--output_root', type=str, default='../results')
    parser.add_argument('--model', type=str, default='gpt3_rl')
    parser.add_argument('--option_inds', type=list, default=["A", "B", "C", "D", "E", "F"])

    # user options
    parser.add_argument('--label', type=str, default='exp0')
    parser.add_argument('--test_split', type=str, default='test', choices=['dev', 'dev1k', 'test', 'test1k'])
    parser.add_argument('--test_number', type=int, default=100, help='GPT-3 is expensive. -1 for the whole test set')
    parser.add_argument('--save_every', type=int, default=10, help='Save the result with every n examples.')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument(
        '--prompt_format',
        type=str,
        default='TQ-A',
        choices=['T-A', 'Q-A', 'Q-AS', 'Q-SA', 'TQ-A', 'TQ-AS', 'TQ-SA', 'QT-A', 'QT-AS', 'QT-SA', 'QTS-A', 'TQS-A'],
        help='prompt format template')
    parser.add_argument('--shot_number', type=int, default=2, help='Number of n-shot training examples.')
    parser.add_argument('--seed', type=int, default=1, help='random seed')

    # GPT-3 settings
    parser.add_argument('--engine', type=str, default='davinci-002', choices=['davinci-002', 'ada'])
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--max_tokens',
                        type=int,
                        default=1024,
                        help='The maximum number of tokens allowed for the generated answer.')
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--frequency_penalty', type=float, default=0.0)
    parser.add_argument('--presence_penalty', type=float, default=0.0)

    # Policy Model settings
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--model_config',
                        type=str,
                        default='bert-base-uncased',
                        choices=['distilbert-base-uncased', 'bert-base-uncased'])
    parser.add_argument('--cand_number', type=int, default=20, help='Number of candidate prompts.')
    parser.add_argument('--embedding_size', type=int, default=128, help='Policy network final layer hidden state size.')
    parser.add_argument('--ckpt_root', type=str, default='../checkpoints')
    parser.add_argument('--ckpt', type=str, default=None)

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=False))

    # https://pytorch.org/docs/stable/notes/randomness.html
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)  # CPU random seed
    torch.cuda.manual_seed(args.seed)  # GPU random seed
    torch.backends.cudnn.benchmark = True

    the_rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    # problems, test question ids, candidate prompt pids, RL training pids
    problems, pids, cand_pids, train_pids = load_data(args)

    result_file = get_result_file(args)

    # load the check point
    if os.path.exists(result_file):
        print("# The result file exists! We will load the learned check point!!!")
        check_point = json.load(open(result_file))
        results = check_point['results']
    else:
        results = {}

    total = len(pids)
    check_count = len(results)  # number of existing results
    correct = 0  # number of correct results

    # sim_cal_model
    sim_model = sim_cal_model("../pretrain/bert-base-uncased").to(torch.device("cuda:0"))

    cand_examples = []
    for pid in cand_pids:
        example = create_example_from_pid(pid, problems, args, test=True)
        cand_examples.append(example)
    
    # data = Data(cand_examples)
    # policy_model = PolicyNet(data, model_config=args.model_config, add_linear=True, device = device)
    policy_model = Pillow(model_config="bert-base-uncased", add_linear=True, device = torch.device("cuda:0"))
    if args.ckpt:
        # emb_file = os.path.join(args.ckpt_path, "emb_best_reward.pt")
        # torch.save(policy_model.emb.linear.state_dict(), emb_file)
        # logger.write(f"saved the best reward ckpt to {emb_file}")

        # gnn_file = os.path.join(args.ckpt_path, f"gnn_best_reward.pt")
        # torch.save(policy_model.gnn.state_dict(), gnn_file)

        # linkpred_file = os.path.join(args.ckpt_path, f"linkpred_best_reward.pt")
        # torch.save(policy_model.linkpred.state_dict(), linkpred_file)

        # selectn_file = os.path.join(args.ckpt_path, f"selectn_best_reward.pt")
        # torch.save(policy_model.selectn.state_dict(), selectn_file)
        emb_path = os.path.join(args.ckpt_root, args.ckpt, "pillow_best_reward.pt")
        # gnn_path = os.path.join(args.ckpt_root, args.ckpt, "gnn_best_reward.pt")
        # linkpred_path = os.path.join(args.ckpt_root, args.ckpt, "linkpred_best_reward.pt")
        # selectn_path = os.path.join(args.ckpt_root, args.ckpt, "selectn_best_reward.pt")
        # for component in [emb_path, gnn_path, linkpred_path, selectn_path]:
        #     if not os.path.exists(component):
        #         print(f"{component} does not exist!")
        #         exit()
        policy_model.load_state_dict(torch.load(emb_path))
        # policy_model.gnn.load_state_dict(torch.load(gnn_path))
        # policy_model.linkpred.load_state_dict(torch.load(linkpred_path))
        # policy_model.selectn.load_state_dict(torch.load(selectn_path))
    else:
        print(f"!!! Load the pre-traind model instead!")  # CHECK
        exit()

    # policy_model.eval()
    with torch.no_grad():
        for i, pid in enumerate(pids):
            count = i + 1  # number of current results
            print(count)
            problem = problems[pid]
            answer = problems[pid]['answer']
            # options = problems[pid]['choices']
            # unit = problems[pid]['unit']
            options = ""
            unit = ""

            example = create_example_from_pid(pid, problems, args, test=True)
            in_context_list, log_prob = policy_model([example], cand_examples)

            the_in_context = in_context_list[0,:].clone().detach().cpu().numpy()
            shot_pids = [cand_pids[j] for j in the_in_context]
            
            prompt = build_prompt(problems, shot_pids, pid, args)
            
            

            if pid in results:
                output = results[pid]
                print("output exists")
            else:
                if "gpt-4" in args.model:
                    output,ppl = get_gpt4_output(prompt, args)
                elif "gpt" in args.model:
                    output, ppl = call_gpt3(prompt, 0.0,1024,1,0,0)  # generate the output by GPT-3
                elif "llama" in args.model:
                    output, ppl = get_llama_output(prompt, args)
                else:
                    print("please check generated model type!")
                    sys.exit()
            
            prediction = output
            answer_norm = answer
            prediction_norm = prediction

            # save the results
            results[pid] = {}
            results[pid]["question"] = problem['question']
            results[pid]["shot_pids"] = shot_pids
            results[pid]["prompt"] = prompt
            results[pid]["answer"] = answer
            results[pid]["answer_norm"] = answer_norm
            results[pid]["output"] = output

            results[pid]["ppl"] = ppl
            results[pid]['rouge1'], results[pid]['rouge2'], results[pid]['rougeL'] = cal_rouge([answer], [prediction], the_rouge_scorer)
            results[pid]['bleu'] = sacrebleu.corpus_bleu([prediction], [[answer]]).score

            
            # print(f"prediction: {prediction}\nanswer: {answer}")

            scores_sim = sim_model.cal_sim(prediction_norm, answer_norm)
            scores_texual = fuzz.ratio(prediction_norm, answer_norm)*1.0/100

            _reword_lambda=0.6
            reward = _reword_lambda*scores_texual + (1-_reword_lambda)*scores_sim
            results[pid]["reward"] = reward
            
            correct+=reward if reward>0.5 else 0
            acc = correct / (i + 1) * 100

            if args.debug or i < 10:
                print("\n##################################")
                print(prompt, "\n")
                # print("[A] labeled answer (normalized):\t", answer_norm)
                # print("[P] predicted answer (normalized):\t", prediction_norm)
                print("[reward]:\t", results[pid]["reward"])
                print("")
                print("[A] labeled answer:\t", answer)
                print("[P] predicted answer:\t", prediction)
                print("[P] generated output:\t", output)
                print(f"[PPL]:\t{ppl:.4f}")
                print(f"[rouge1]:\t{results[pid]['rouge1']:.4f}")
                print(f"[rouge2]:\t{results[pid]['rouge2']:.4f}")
                print(f"[rougeL]:\t{results[pid]['rougeL']:.4f}")
                print(f"[bleu]:\t{results[pid]['bleu']:.4f}")
                

            if count % args.save_every == 0 or count == total:
                if count >= check_count:
                    # have new outputs
                    print(f"{count}/{total}, correct: {correct}, acc: {round(acc, 2)}%, saved to {result_file}")
                    save_results(result_file, acc, correct, count, cand_pids, args, results)
                else:
                    # no new outputs, just print the accuracy
                    print(f"{count}/{total}, correct: {correct}, acc: {round(acc, 2)}%")

            cand_pids = random.sample(train_pids, args.cand_number) 
            cand_examples = []
            for pid in cand_pids:
                example = create_example_from_pid(pid, problems, args, test=True)
                cand_examples.append(example)

