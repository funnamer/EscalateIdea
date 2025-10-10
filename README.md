# 医疗问答系统 - Qwen3 0.6B模型微调与部署

本项目基于医疗问答数据集，对Qwen3 0.6B模型进行微调，并通过FastAPI实现模型部署，支持下游任务调用。

## 项目结构
```
EscalateIdea/
├── api/
│   ├── qwen3_api.py
│   └── qwen3_embedding_api.py
├── data/
│   ├── Chinese-medical-dialogue-data-master/
│   ├── train.json
│   └── val.json
├── embedding/                # 存放 Qwen3 0.6B-embedding 模型
│   └── Qwen3-Embedding-0.6B/
├── finetune_res/
│   ├── logs/
│   └── output/               # 微调过程中自动生成的临时输出目录（train.py 运行后产生）
├── lora_fine-tune/           # 模型微调相关代码
│   ├── combie.py             # （注：推测为 LoRA 权重与原模型融合脚本，原结构中对应 combine.py）
│   ├── lora_instruction.py
│   ├── model_info.py
│   ├── qa_dataset.py
│   ├── test.py               # 微调后模型测试脚本
│   └── train.py              # 微调训练主脚本
└── model/                    # 存放 Qwen3 0.6B 基础模型
    ├── lora_output/          # 模型融合后生成的最终微调模型目录（combie.py 运行后产生）
    └── Qwen3-0.6B/
```

## 环境准备

### 模型下载

1. **Qwen3 0.6B基础模型**：
   [- 从ModelScope下载Qwen3 0.6B模型](https://www.modelscope.cn/models/Qwen/Qwen3-0.6B)
   - 放置在 `model` 文件夹下

2. **Qwen3 0.6B-embedding模型**：
   [- 从ModelScope下载Qwen3 0.6B-embedding模型](https://www.modelscope.cn/models/Qwen/Qwen3-Embedding-0.6B)
   - 放置在 `embedding` 文件夹下

### 数据集下载
**Chinese medical dialogue data 中文医疗问答数据集**

```https://github.com/Toyhom/Chinese-medical-dialogue-data```
### 环境配置

## 快速开始

### 模型微调

**运行微调训练**：
    python train.py
微调完成后会在项目根目录生成 output 文件夹
**合并LoRA权重**：
    python combie.py
合并完后存放在lora_output文件夹，生成最终的微调模型 lora_output
**测试模型输出**：
    python test.py

### 模型部署
项目使用FastAPI框架进行模型部署，代码在api文件夹下
运行qwen3_api.py和qwen3_embedding_api.py即可分别启动Qwen3模型和Qwen3嵌入模型的api服务


### 测试
可以打开 本地8000和8001端口 查看对应接口和测试

### langchain中调用
如要在langchain中使用，可参考我另外的langchainQA项目，其中已封装好 模型和嵌入模型的代码
