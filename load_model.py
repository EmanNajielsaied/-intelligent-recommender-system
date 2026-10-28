# load_model.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

def _check_meta(model):
    for _, p in model.named_parameters():
        if hasattr(p, "is_meta") and p.is_meta:
            return True
    for _, b in model.named_buffers():
        if hasattr(b, "is_meta") and b.is_meta:
            return True
    return False


def load_sentiment_model(model_path="."):
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model folder not found.")

    required = [
        "config.json",
        "pytorch_model.bin",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.txt"
    ]
    for f in required:
        if not os.path.isfile(os.path.join(model_path, f)):
            raise FileNotFoundError(f"Missing required file: {f}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            local_files_only=True,
            device_map=None,
            low_cpu_mem_usage=False
        )
    except Exception as e:
        raise RuntimeError(f"Model loading failed:\n{e}")

    if _check_meta(model):
        raise RuntimeError(
            "MODEL CONTAINS META TENSORS. This means the weights did not load."
        )

    model.eval()
    return tokenizer, model


def predict_text(text, tokenizer, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0].cpu()

    label = int(torch.argmax(probs).numpy().tolist())
    confidence = float(probs[label])
    return label, confidence