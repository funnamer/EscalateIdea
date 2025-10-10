from transformers import AutoModelForCausalLM

model_path = "../model/Qwen3-0.6B"
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
print(model)
