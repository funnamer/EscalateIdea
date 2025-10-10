from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import torch
import numpy as np  # 导入 numpy
from typing import List, Optional

app = FastAPI(title="Qwen3-0.6B-Embedding API (GPU 版)")

# 替换为你的本地模型路径（Windows 用双斜杠或反斜杠）
MODEL_PATH = "C:/Users/27946/Desktop/EscalateIdea/embedding/Qwen3-Embedding-0.6B"  # 示例："C:\\models\\Qwen3-Embedding-0.6B"

# 关键：加载模型到 GPU（禁用 flash-attn，用默认注意力）
model = SentenceTransformer(
    model_name_or_path=MODEL_PATH,
    model_kwargs={
        "attn_implementation": None,  # 彻底禁用 flash-attn，避免编译依赖
        "device_map": "cuda:0",  # 强制加载到第1块 GPU（0 表示第1块，多GPU可改1、2等）
        "dtype": torch.float16  # GPU 用 float16 节省显存（0.6B 模型仅需 ~1.5GB 显存）
    },
    tokenizer_kwargs={"padding_side": "left"}  # 官方推荐，优化推理
)

# 验证模型是否成功加载到 GPU
print(f"✅ 模型加载完成！当前设备：{model.device}")  # 应输出 "cuda:0"
print(f"✅ 模型显存占用：{torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")  # 查看初始显存占用


# 定义请求体
class EmbeddingRequest(BaseModel):
    texts: List[str]  # 待嵌入的文本列表（支持批量）
    normalize: bool = True  # 归一化向量（推荐开启，用于相似度计算）
    dim: Optional[int] = 1024  # 自定义嵌入维度（32-1024）


# 核心嵌入接口
@app.post("/embed", summary="生成文本嵌入向量（GPU 加速）")
def embed_texts(request: EmbeddingRequest):
    try:
        # 输入校验
        if not request.texts:
            raise HTTPException(status_code=400, detail="texts 不能为空（需传入文本列表）")
        if not (32 <= request.dim <= 1024):
            raise HTTPException(status_code=400, detail="dim 必须在 32~1024 之间")

        # GPU 推理生成嵌入（速度比 CPU 快 5~10 倍）
        with torch.no_grad():  # 禁用梯度计算，节省显存、加速推理
            embeddings = model.encode(
                sentences=request.texts,
                normalize_embeddings=request.normalize,
                convert_to_tensor=False  # 转为 numpy 数组，便于后续处理
            )

        # 将 numpy 数组转换为 Python list
        embeddings = embeddings.tolist()

        # 截断到指定维度
        if request.dim != 1024:
            embeddings = [vec[:request.dim] for vec in embeddings]

        # 返回结果（包含显存占用，方便监控）
        current_mem = torch.cuda.memory_allocated() / 1024 ** 3
        return {
            "code": 200,
            "message": "GPU 嵌入成功",
            "data": {
                "embeddings": embeddings,
                "normalize": request.normalize,
                "dim": request.dim,
                "text_count": len(request.texts),
                "gpu_mem_used": f"{current_mem:.2f} GB"  # 当前 GPU 显存占用
            }
        }
    except Exception as e:
        # 捕获 GPU 相关错误（如显存不足）
        if "out of memory" in str(e).lower():
            raise HTTPException(status_code=500, detail="GPU 显存不足，建议减少批量文本数量")
        raise HTTPException(status_code=500, detail=f"嵌入失败：{str(e)}")


# 健康检查接口（验证 GPU 状态）
@app.get("/health", summary="服务健康检查（含 GPU 信息）")
def health_check():
    gpu_info = {
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "无 GPU",
        "total_mem": f"{torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB" if torch.cuda.is_available() else "无",
        "used_mem": f"{torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB" if torch.cuda.is_available() else "无"
    }
    return {
        "code": 200,
        "status": "healthy",
        "model": "Qwen3-0.6B-Embedding",
        "gpu_info": gpu_info
    }


# 启动服务
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")