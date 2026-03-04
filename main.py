from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

model_name = "Ppear/Qwen2.5-3B-GRPO-Physics_expo"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "'<think> reasoning process here </think><answer> answer here </answer>' If the question provides options, use "
    "the exact wordings in the options. Otherwise, answer with a numerical number and units, with a space between them."
)

app = FastAPI()

@app.get("/")
def health_check():
    return {"health_check": "OK"}

@app.get("/info")
def info():
    return {"name":"Pysics-search", "description":"Let's ask the high school pysics problem"}

@app.get("/search")
def search(query: str):
    messages = [
        {"role":"system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query}
    ]
    
    # แปลง messages → text ที่โมเดลเข้าใจ
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    
    # tokenize
    inputs = tokenizer(text, return_tensors="pt")
    
    # generate
    outputs = model.generate(**inputs, max_new_tokens=512)
    
    # decode → string
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    thinks = re.findall(r"<think>(.+?)</think>", result, re.DOTALL)
    answers = re.findall(r"<answer>(.+?)</answer>", result, re.DOTALL)

    return {
        "reasoning": thinks[-1].strip() if thinks else "",
        "answer": answers[-1].strip() if answers else result
    }
    
# uvicorn main:app --reload
# pip show fastapi transformers torch peft accelerate --> สร้าง requirements.txt