import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "../model/lora_output"  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("🚀 Loading model from:", model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.to(device)
model.eval()


system_prompt = (
    "你是一位临床医学专家，擅长根据患者描述提供初步诊断和健康建议。"
    "请用简洁、逻辑清晰的中文回答，不超过150字。”。"
)
user_input = "5月至今上腹靠右隐痛，右背隐痛带酸，便秘，喜睡，时有腹痛，头痛，腰酸症状？"

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_input}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,enable_thinking=False)

# 生成输入张量
model_inputs = tokenizer([text], return_tensors="pt", padding=True).to(device)

generation_config = dict(
    max_new_tokens=512,        
    temperature=0.4,         
    top_p=0.8,              
    top_k=8,              
    repetition_penalty=1.1,    
    do_sample=True,            
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id,
    attention_mask=model_inputs["attention_mask"],  
)
with torch.no_grad():
    output_ids = model.generate(
        input_ids=model_inputs.input_ids,
        **generation_config
    )

generated_ids = [
    output[len(input):] for input, output in zip(model_inputs.input_ids, output_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


if "谢谢" in response:
    response = response.split("谢谢")[0] + "谢谢你的提问。"
elif len(response) > 200:
    response = response[:200] + "……谢谢你的提问。"

print("\n===============================")
print("✅ 模型输出：\n")
print(response)
print("===============================")

