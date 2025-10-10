# -*- coding: utf-8 -*-
from torch.utils.data import Dataset
import torch
import json
import numpy as np


class QADataset(Dataset):
    def __init__(self, data_path, tokenizer, max_source_length, max_target_length) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length  # prompt 的最大长度
        self.max_target_length = max_target_length  # answer 的最大长度
        self.max_seq_length = self.max_source_length + self.max_target_length  # 总序列最大长度

        # 加载数据（保持原逻辑）
        self.data = []
        if data_path:
            with open(data_path, "r", encoding='utf-8') as f:
                for line in f:
                    line = line.strip()  # 去除首尾空白（避免空行干扰）
                    if not line:
                        continue
                    try:
                        json_line = json.loads(line)
                        # 确保数据包含 "question" 和 "answer" 字段
                        if "question" in json_line and "answer" in json_line:
                            self.data.append({
                                "question": json_line["question"],
                                "answer": json_line["answer"]
                            })
                    except json.JSONDecodeError:
                        print(f"跳过无效JSON行：{line}")  # 捕获JSON解析错误，避免程序崩溃
        print(f"数据加载完成，共 {len(self.data)} 条样本")

    def preprocess(self, question, answer):
        # 1. 构建对话模板（保持原逻辑）
        messages = [
            {"role": "system", "content": "你是一个医疗方面的专家，可以根据患者的问题进行解答。"},
            {"role": "user", "content": question}
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True  # 生成带 "assistant" 前缀的prompt
        )

        # 2. 编码 prompt（关键修改：添加 truncation=True + return_attention_mask=True）
        instruction = self.tokenizer(
            prompt,
            add_special_tokens=False,  # 不重复添加特殊token（chat_template已处理）
            max_length=self.max_source_length,  # 限制prompt最大长度
            truncation=True,  # 显式截断超长prompt，消除警告
            padding=False,  # 不自动填充（后续手动处理总长度）
            return_attention_mask=True  # 必须生成attention_mask，否则后续拼接会报错
        )

        # 3. 编码 answer（同样添加 truncation=True + return_attention_mask=True）
        response = self.tokenizer(
            answer,
            add_special_tokens=False,  # 不添加特殊token（避免与prompt冲突）
            max_length=self.max_target_length,  # 限制answer最大长度
            truncation=True,  # 显式截断超长answer
            padding=False,
            return_attention_mask=True
        )

        # 4. 拼接 prompt + answer + pad_token（优化截断逻辑）
        # 拼接input_ids（末尾加1个pad_token，确保序列有终止标识）
        input_ids = instruction["input_ids"] + response["input_ids"] + [self.tokenizer.pad_token_id]
        # 拼接attention_mask（pad_token对应mask为0，其他为1）
        attention_mask = instruction["attention_mask"] + response["attention_mask"] + [0]  # 注意：pad的mask是0！
        # 构建labels（prompt部分设为-100，不参与损失计算；answer部分正常，pad部分设为-100）
        labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [-100]

        # 5. 确保总长度不超过 max_seq_length（修复截断逻辑）
        if len(input_ids) > self.max_seq_length:
            # 截断到最大长度
            input_ids = input_ids[:self.max_seq_length]
            attention_mask = attention_mask[:self.max_seq_length]
            labels = labels[:self.max_seq_length]
            # 确保最后一个token是pad（可选，避免序列末尾是不完整的answer）
            if input_ids[-1] != self.tokenizer.pad_token_id:
                input_ids[-1] = self.tokenizer.pad_token_id
                attention_mask[-1] = 0
                labels[-1] = -100

        # 6. 处理长度不足的情况（可选：填充到max_seq_length，确保批次内张量长度一致）
        # （如果batch_size>1，必须填充，否则不同样本长度仍会不匹配）
        padding_length = self.max_seq_length - len(input_ids)
        if padding_length > 0:
            input_ids += [self.tokenizer.pad_token_id] * padding_length
            attention_mask += [0] * padding_length
            labels += [-100] * padding_length

        return input_ids, attention_mask, labels

    def __getitem__(self, index):
        item_data = self.data[index]
        input_ids, attention_mask, labels = self.preprocess(**item_data)

        # 转换为torch.LongTensor（保持原逻辑）
        return {
            "input_ids": torch.LongTensor(input_ids),  # 无需np.array中转，直接列表转张量
            "attention_mask": torch.LongTensor(attention_mask),
            "labels": torch.LongTensor(labels)
        }

    def __len__(self):
        return len(self.data)