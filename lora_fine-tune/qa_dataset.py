from torch.utils.data import Dataset
import torch
import json
import numpy as np


class QADataset(Dataset):
    def __init__(self, data_path, tokenizer, max_source_length, max_target_length) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length 
        self.max_target_length = max_target_length  
        self.max_seq_length = self.max_source_length + self.max_target_length  

        self.data = []
        if data_path:
            with open(data_path, "r", encoding='utf-8') as f:
                for line in f:
                    line = line.strip() 
                    if not line:
                        continue
                    try:
                        json_line = json.loads(line)
                        if "question" in json_line and "answer" in json_line:
                            self.data.append({
                                "question": json_line["question"],
                                "answer": json_line["answer"]
                            })
                    except json.JSONDecodeError:
                        print(f"跳过无效JSON行：{line}")  
        print(f"数据加载完成，共 {len(self.data)} 条样本")

    def preprocess(self, question, answer):
        messages = [
            {"role": "system", "content": "你是一个医疗方面的专家，可以根据患者的问题进行解答。"},
            {"role": "user", "content": question}
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True  
        )

        instruction = self.tokenizer(
            prompt,
            add_special_tokens=False,  
            max_length=self.max_source_length,  
            truncation=True,  
            padding=False, 
            return_attention_mask=True  
        )

        response = self.tokenizer(
            answer,
            add_special_tokens=False,  
            max_length=self.max_target_length, 
            truncation=True,  
            padding=False,
            return_attention_mask=True
        )

        input_ids = instruction["input_ids"] + response["input_ids"] + [self.tokenizer.pad_token_id]
        attention_mask = instruction["attention_mask"] + response["attention_mask"] + [0] 
        labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [-100]

        if len(input_ids) > self.max_seq_length:
            input_ids = input_ids[:self.max_seq_length]
            attention_mask = attention_mask[:self.max_seq_length]
            labels = labels[:self.max_seq_length]
            if input_ids[-1] != self.tokenizer.pad_token_id:
                input_ids[-1] = self.tokenizer.pad_token_id
                attention_mask[-1] = 0
                labels[-1] = -100
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
