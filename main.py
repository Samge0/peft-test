#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author：samge
# date：2024-06-05 13:51
# describe：
from transformers import AutoModelForSeq2SeqLM
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
    

if __name__ == "__main__":

    model_name_or_path = "bigscience/mt0-large"  # 替换为上述模型之一
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM, 
        inference_mode=False, 
        r=8, 
        lora_alpha=32, 
        lora_dropout=0.1
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
