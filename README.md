# COLING 2025
### Model and data address
The address of the model used：

bert-base-uncased: /data/model/bert-base-uncased
llama-2-7b-hf: /data/model/Llama-2-7b-hf

data download url


alpaca: https://huggingface.co/datasets/tatsu-lab/alpaca

dolly: https://huggingface.co/datasets/databricks/databricks-dolly-15k

data process
python ./data/test.py

data address:  
dolly:   
        /data/dolly-all.json （Full Data）  
        /data/dolly-all.json （Training Data）  
        /data/dolly-all.json （Test Data）  
alpaca：  
        /data/alpaca-all.json（Full Data）  
        /data/alpaca-train.json（Training Data）  
        /data/alpaca-test.json（Test Data）  


### Model Training

cd GRL_prompt

Run Command：
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

To use GPT-3, change --model to gpt
To use LLaMA, change --model to llama

#### Note

Here, the vllm service is used to provide the LLaMA inference interface. Therefore, before using LLaMA, the vllm service must be started first. The startup steps are as follows:
```
1. Open a tmux Window：tmux a -t clash
2. Execute Command：CUDA_VISIBLE_DEVICES=0  python -m vllm.entrypoints.openai.api_server --model /data/model/llama --gpu-memory-utilization 0.7  --port 9201 --served-model-name llama
```
After completing the training or inference task, press Control+C in the tmux window to stop the service to avoid occupying GPU resources.


## Model Inference

Run Command：
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
test_number being -1 means that all examples from the test set will be used.
Switch Models as Above
