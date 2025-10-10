# -*- coding: utf-8 -*-
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_path = "../model/Qwen3-0.6B"
lora_dir = "../finetune_res/output"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
model = PeftModel.from_pretrained(model, lora_dir).to(device)
print(model)
# 合并model, 同时保存 token
model = model.merge_and_unload()
model.save_pretrained("../model/lora_output")
tokenizer.save_pretrained("../model/lora_output")
