from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import torch
import numpy as np  # 导入 numpy
from typing import List, Optional

app = FastAPI(title="Qwen3-0.6B-Embedding API (GPU 版)")

MODEL_PATH = "C:/Users/27946/Desktop/EscalateIdea/embedding/Qwen3-Embedding-0.6B"  
model = SentenceTransformer(
    model_name_or_path=MODEL_PATH,
    model_kwargs={
        "attn_implementation": None,  
        "device_map": "cuda:0",  
        "dtype": torch.float16  
    },
    tokenizer_kwargs={"padding_side": "left"} 
)

print(f"✅ 模型加载完成！当前设备：{model.device}") 
print(f"✅ 模型显存占用：{torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")  



class EmbeddingRequest(BaseModel):
    texts: List[str]  # 待嵌入的文本列表
    normalize: bool = True  # 归一化向量
    dim: Optional[int] = 1024  # 自定义嵌入维度


@app.post("/embed", summary="生成文本嵌入向量（GPU 加速）")
def embed_texts(request: EmbeddingRequest):
    try:
        # 输入校验
        if not request.texts:
            raise HTTPException(status_code=400, detail="texts 不能为空（需传入文本列表）")
        if not (32 <= request.dim <= 1024):
            raise HTTPException(status_code=400, detail="dim 必须在 32~1024 之间")

        with torch.no_grad(): 
            embeddings = model.encode(
                sentences=request.texts,
                normalize_embeddings=request.normalize,
                convert_to_tensor=False 
            )

        embeddings = embeddings.tolist()

        if request.dim != 1024:
            embeddings = [vec[:request.dim] for vec in embeddings]

        current_mem = torch.cuda.memory_allocated() / 1024 ** 3
        return {
            "code": 200,
            "message": "GPU 嵌入成功",
            "data": {
                "embeddings": embeddings,
                "normalize": request.normalize,
                "dim": request.dim,
                "text_count": len(request.texts),
                "gpu_mem_used": f"{current_mem:.2f} GB" 
            }
        }
    except Exception as e:
        if "out of memory" in str(e).lower():
            raise HTTPException(status_code=500, detail="GPU 显存不足，建议减少批量文本数量")
        raise HTTPException(status_code=500, detail=f"嵌入失败：{str(e)}")


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



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
