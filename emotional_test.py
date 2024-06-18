#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author：samge
# date：2024-06-18 11:21
# describe：Script for testing IMDb sentiment analysis using trained BERT model

from transformers import BertForSequenceClassification, BertTokenizer
import torch


CUDA_IS_AVAILABLE = torch.cuda.is_available()
device = "cuda" if CUDA_IS_AVAILABLE else "cpu"     # 设备类型


# Function to predict using trained model
def predict(model, tokenizer, texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)  # Move inputs to the same device as the model
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    return predictions

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('./results/checkpoint-4500')  # Load from training output directory
model.to(device)

# Test texts
texts = ["I love this movie!", "This film was terrible."]

# Perform predictions
predictions = predict(model, tokenizer, texts)
print("0=负面，1=正面。\n结果 => ", predictions)
for i in range(len(texts)):
    print(f"【{predictions[i]}】{texts[i]}")
