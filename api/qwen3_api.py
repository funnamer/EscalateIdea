from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Optional, List
import os

# 初始化FastAPI应用
app = FastAPI(
    title="Qwen3-0.6B 本地模型API",
    description="封装本地Qwen3-0.6B模型为API，支持基础对话功能",
    version="1.0.0"
)

# 本地模型路径  这里要放绝对路径，根据需要替换为自己模型的路径
LOCAL_MODEL_PATH = "C:/Users/27946/Desktop/EscalateIdea/model/Qwen3-0.6B"
if not os.path.exists(LOCAL_MODEL_PATH):
    raise FileNotFoundError(f"模型路径不存在：{LOCAL_MODEL_PATH}")

# 加载分词器和模型（全局初始化，仅加载一次）
tokenizer = AutoTokenizer.from_pretrained(
    LOCAL_MODEL_PATH,
    trust_remote_code=True,
    local_files_only=True
)

model = AutoModelForCausalLM.from_pretrained(
    LOCAL_MODEL_PATH,
    dtype=torch.float16,  # 适配GPU，CPU环境可改为torch.float32
    device_map="auto",
    trust_remote_code=True,
    local_files_only=True
)
model.eval()  # 推理模式


# 数据格式定义（Pydantic校验）
class Message(BaseModel):
    """单轮对话消息格式"""
    role: str = Field(..., pattern="^(user|assistant)$", description="角色，仅支持'user'或'assistant'")
    content: str = Field(..., description="消息内容")


class Qwen3Request(BaseModel):
    """API请求参数（保留核心参数）"""
    messages: List[Message] = Field(..., description="对话历史列表（至少包含1条user消息）")
    enable_thinking: bool = Field(default=False, description="是否启用思考模式")
    max_new_tokens: int = Field(default=1024, ge=1, le=32768, description="最大生成token数（1-32768）")
    temperature: float = Field(default=0.7, ge=0.1, le=1.0, description="随机性（0.1-1.0，值越高越随机）")


class Qwen3Response(BaseModel):
    """API响应格式"""
    response: str = Field(..., description="模型生成的回复内容")
    thinking_content: Optional[str] = Field(default=None, description="思考模式下的思考过程")


# 核心API接口
@app.post("/qwen3/local/generate", response_model=Qwen3Response)
async def generate(request: Qwen3Request):
    try:
        # 校验对话历史（最后一条必须是用户消息）
        if not request.messages or request.messages[-1].role != "user":
            raise HTTPException(status_code=400, detail="对话历史最后一条必须是'user'角色的消息")

        # 转换对话格式为模型要求的输入
        messages_list = [{"role": m.role, "content": m.content} for m in request.messages]
        input_text = tokenizer.apply_chat_template(
            messages_list,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=request.enable_thinking
        )

        # 转换为模型输入张量（自动适配设备）
        model_inputs = tokenizer(
            [input_text],
            return_tensors="pt",
            truncation=True,
            max_length=32768 - request.max_new_tokens  # 控制上下文长度
        ).to(model.device)

        # 模型生成（仅保留核心参数）
        with torch.no_grad():  # 禁用梯度计算，节省资源
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_p=0.8,  # 固定官方推荐值（非思考模式）
                top_k=20,   # 固定官方推荐值
                do_sample=True  # 启用采样，避免重复
            )

        # 解析生成结果（分离输入和输出）
        input_len = model_inputs["input_ids"].shape[1]
        output_ids = generated_ids[0][input_len:].tolist()  # 仅取模型生成部分
        thinking_end_token_id = 151668  # 思考内容结束符

        # 处理思考模式/非思考模式的结果
        thinking_content = None
        if request.enable_thinking and thinking_end_token_id in output_ids:
            # 思考模式：分离思考内容和最终回复
            idx = output_ids.index(thinking_end_token_id)
            thinking_content = tokenizer.decode(output_ids[:idx], skip_special_tokens=True).strip()
            response_content = tokenizer.decode(output_ids[idx+1:], skip_special_tokens=True).strip()
        else:
            # 非思考模式：直接解码回复
            response_content = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        return Qwen3Response(
            response=response_content,
            thinking_content=thinking_content
        )

    except HTTPException:
        raise  # 传递已知错误
    except Exception as e:
        # 捕获未知错误并返回
        raise HTTPException(status_code=500, detail=f"生成失败：{str(e)}")


# 启动服务
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app="qwen3_api:app",  # 对应文件名api.py
        host="0.0.0.0",  # 允许局域网访问
        port=8000,       # 端口（可修改）
        reload=False     # 生产环境关闭热重载
    )