import json
import random

# 读取JSON文件
with open('/data/PromptPG-main/data/dolly-all.json', 'r') as file:
    data = json.load(file)

# 获取数据的所有键
keys = list(data.keys())

# 打乱键的顺序
random.shuffle(keys)

# 按照7:3的比例划分训练集和测试集
split_point = int(len(keys) * 0.7)
train_keys = keys[:split_point]
test_keys = keys[split_point:]

# 创建训练集和测试集
train_data = {key: data[key] for key in train_keys}
test_data = {key: data[key] for key in test_keys}

# 保存训练集和测试集到新的JSON文件
with open('dolly_train_data.json', 'w') as train_file:
    json.dump(train_data, train_file, indent=4)

with open('dolly_test_data.json', 'w') as test_file:
    json.dump(test_data, test_file, indent=4)

print("训练集和测试集已成功划分并保存。")
