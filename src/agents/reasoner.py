# src/agents/reasoner.py
import torch
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList

# Lazy‑loaded references
_tokenizer_llm = None
_llm = None

def _lazy_load_llm():
    """
    Load LLM & tokenizer only once, on first use.
    """
    global _tokenizer_llm, _llm
    if _tokenizer_llm is None or _llm is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from src.config import MODEL_NAME

        # Instantiate tokenizer & model
        _tokenizer_llm = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True
        )
        _llm = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True
        )
        _llm.eval()

# Custom stopping criterion: stop when the model begins to emit "Question:"
class StopOnQuestion(StoppingCriteria):
    def __call__(self, input_ids, scores, **kwargs):
        text = _tokenizer_llm.decode(input_ids[0], skip_special_tokens=True)
        return text.endswith("Question:")

# Custom stopping criterion: stop when chain-of-thought cues appear
class StopOnCue(StoppingCriteria):
    def __init__(self, cues):
        self.cues = cues
    def __call__(self, input_ids, scores, **kwargs):
        text = _tokenizer_llm.decode(input_ids[0], skip_special_tokens=True)
        return any(text.endswith(cue) for cue in self.cues)


def reason(query: str, context: list[str]) -> str:
    """
    Generate an answer given the user query and retrieved context.
    """
    # Ensure model & tokenizer loaded
    _lazy_load_llm()

    # Assemble prompt
    prompt = (
        """
Use the following context to answer the question directly.
Do not ask follow-up questions.
If the answer cannot be found in the context, respond "I don’t know."

{context_blocks}

Question: {query}
Answer:
"""
        .format(
            context_blocks="\n---\n".join(context),
            query=query
        )
    )

    # Tokenize and move to model device
    inputs = _tokenizer_llm(
        prompt,
        return_tensors="pt",
        truncation=True,
        padding=True
    ).to(_llm.device)

    # Setup stopping criteria
    cues = [
        " I think", "Let's", "Therefore", "Thus", "Because", "So,",
        "Alright", "Remember", "I remember", "---"
    ]
    stopping_criteria = StoppingCriteriaList([StopOnQuestion(), StopOnCue(cues)])

    # Generate
    outputs = _llm.generate(
        **inputs,
        max_new_tokens=512,
        pad_token_id=_tokenizer_llm.eos_token_id,
        no_repeat_ngram_size=3,
        early_stopping=True,
        stopping_criteria=stopping_criteria
    )

    # Decode and extract answer
    full = _tokenizer_llm.decode(outputs[0], skip_special_tokens=True)
    answer = full.split("Answer:", 1)[1].strip()
    answer = answer.split("---", 1)[0].rstrip()
    if not answer.startswith("I don't"):
        answer = answer.split("I don't know", 1)[0].rstrip()
    return answer
