#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author：samge
# date：2024-06-18 11:21
# describe：

# 导入库
import os
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import torch
import wandb

# 检查PyTorch版本
print("PyTorch version:", torch.__version__)

wandb_api_key = os.getenv("WANDB_API_KEY", "")
wandb_username = os.getenv("WANDB_USERNAME", "")

if wandb_api_key and wandb_username:
    # 登录wandb
    wandb.login()
    # 初始化wandb项目
    wandb.init(project="imdb-sentiment-analysis", entity=wandb_username)

# 加载IMDB数据集
dataset = load_dataset('imdb')

# 划分训练集和测试集
train_dataset = dataset['train']
test_dataset = dataset['test']

# 预处理数据
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def preprocess_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

encoded_train_dataset = train_dataset.map(preprocess_function, batched=True)
encoded_test_dataset = test_dataset.map(preprocess_function, batched=True)

# 定义模型和训练参数
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    report_to="wandb",  # 启用wandb
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_train_dataset,
    eval_dataset=encoded_test_dataset
)

# 训练模型
trainer.train()

# 评估模型
results = trainer.evaluate()
print(results)

# 使用模型进行预测
def predict(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    return predictions

texts = ["I love this movie!", "This film was terrible."]
predictions = predict(texts)
print(predictions)

# 完成后结束wandb运行
wandb.finish()
