from typing import List
import torch

# Lazy-loaded tokenizer and model references
_tokenizer = None
_model = None

def _lazy_load():
    """
    Load tokenizer & model only once, on first use.
    """
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        from transformers import AutoTokenizer, AutoModel
        from src.config import MODEL_NAME

        # Determine GPU availability
        gpu_count = torch.cuda.device_count()
        device_map = "auto"

        # Load tokenizer
        _tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True
        )

        # Load model with HF offloading
        _model = AutoModel.from_pretrained(
            MODEL_NAME,
            device_map=device_map,
            offload_folder="offload",
            offload_state_dict=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16 if gpu_count > 0 else torch.float32,
            trust_remote_code=True
        )
        _model.eval()


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embed a list of texts using the loaded model with mean pooling.

    :param texts: List of input strings.
    :return: List of embedding vectors.
    """
    # Ensure model & tokenizer are initialized
    _lazy_load()

    # Tokenize inputs
    enc = _tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    # Move inputs to model device
    first_device = next(_model.parameters()).device
    input_ids = enc.input_ids.to(first_device)
    attention_mask = enc.attention_mask.to(first_device)

    # Forward pass
    with torch.no_grad():
        outputs = _model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

    # Mean pooling
    mask = attention_mask.unsqueeze(-1)
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1)
    embeddings = (summed / counts).cpu().tolist()
    return embeddings


def embed_text(text: str) -> List[float]:
    """
    Embed a single text string.
    """
    return embed_texts([text])[0]
