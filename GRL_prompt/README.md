### 模型和数据地址
使用到的模型地址：
bert-base-uncased: /data/model/bert-base-uncased
llama-2-7b-hf: /data/model/Llama-2-7b-hf

数据地址：
dolly: /data/PromptPG-main/data/dolly-all.json （全量数据）
alpaca：
    - /data/PromptPG-main/data/alpaca-all.json（全量数据）


### 模型训练
运行命令：
```
python learn_policy.py \
--model gpt-4 \
--label exp1 \
--ckpt_root ../checkpoints \
--shot_number 2 \
--prompt_format Q-A \
--seed 2 \
--model_config bert-base-uncased \
--train_number 200 \
--cand_number 20 \
--lr 0.001 \
--epochs 10 \
--embedding_size 128 \
--batch_size 20 \
--gpu 0
```

若使用gpt，把--model 后改为gpt或gpt-4

#### 注意

这里使用vllm服务提供llama推理接口，因此使用llama前要先启动vllm服务，启动步骤：
```
1. 开启tmux窗口，tmux a -t clash
2. 执行命令：CUDA_VISIBLE_DEVICES=0  python -m vllm.entrypoints.openai.api_server --model /data/model/llama --gpu-memory-utilization 0.7  --port 9201 --served-model-name llama
```
完成训练或推理任务后，在tmux窗口中按control+c 停止服务，避免占用显卡资源


## 模型推理
运行命令：
```
python run_model.py \
--model gpt-4 \
--label exp1 \
--ckpt_root ../checkpoints \
--test_split test \
--test_number -1 \
--shot_number 2 \
--prompt_format Q-A \
--seed 2 \
--cand_number 20 \
--embedding_size 128 \
--model_config bert-base-uncased \
--ckpt exp1/ \
--gpu 0
```

切换模型同上