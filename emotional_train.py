#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author：samge
# date：2024-06-18 11:21
# describe：Script for training IMDb sentiment analysis using BERT

import os
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import torch
import wandb


CUDA_IS_AVAILABLE = torch.cuda.is_available()
device = "cuda" if CUDA_IS_AVAILABLE else "cpu"     # 设备类型


# 检查PyTorch版本
print("PyTorch version:", torch.__version__, "device：", device)


# 是否已经配置了wandb
def is_wandb_config_ok() -> bool:
    wandb_api_key = os.getenv("WANDB_API_KEY", "")
    wandb_username = os.getenv("WANDB_USERNAME", "")
    return wandb_api_key and wandb_username


# 登录wandb
def try_login_wandb():
    if not is_wandb_config_ok():
        return
    wandb.login()
    wandb.init(project="imdb-sentiment-analysis", entity=os.getenv("WANDB_USERNAME", ""))


# 训练
def train():
    # Load IMDb dataset
    dataset = load_dataset('imdb')

    # Split into training and testing datasets
    train_dataset = dataset['train']
    test_dataset = dataset['test']

    # Preprocess data
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def preprocess_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True)

    encoded_train_dataset = train_dataset.map(preprocess_function, batched=True)
    encoded_test_dataset = test_dataset.map(preprocess_function, batched=True)

    # Define model and training parameters
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    model.to(device)
    
    output_dir = './results'
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        report_to="wandb",  # Enable wandb integration
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_train_dataset,
        eval_dataset=encoded_test_dataset
    )

    # Train model
    trainer.train()

    # Finish wandb run
    wandb.finish() if is_wandb_config_ok() else None
    
    
if __name__ == "__main__":
    try_login_wandb()
    train()
    print("all done")
