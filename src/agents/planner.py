# planner.py

# def plan(query: str) -> list[str]:
#     """
#     Decompose a complex query into sub-queries.  
#     Currently a no-op: returns the original query as single task.
#     """
#     return [query]

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load once during module import (recommended for performance)
from src.config import MODEL_NAME
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")

def plan(query: str) -> list[str]:
    """
    Decompose a complex query into sub-queries using DeepSeek-R1-Distill-Llama-8B.
    """
    prompt = f"""Decompose the following question into clear, focused sub-questions. 
Return one sub-question per line. Only output the sub-questions.

Question: {query}
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only sub-questions from generated text
    lines = response.split("\n")
    subqueries = [line.strip("-•. ") for line in lines if line.strip()]
    return subqueries
