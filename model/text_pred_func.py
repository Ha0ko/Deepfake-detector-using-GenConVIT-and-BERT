import os
import torch
import torch.nn as nn
from transformers import (
    BertTokenizer, 
    BertModel, 
    BertForSequenceClassification,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
)
from pathlib import Path
import gc

# Force CPU to avoid GPU memory issues, can be overridden
device = "cpu"  # Use CPU by default to avoid memory issues
# Note: Uses DistilBERT from transformers library (no local BERT directory needed)
BERT_DIR = None  # Local BERT directory not used - defaults to DistilBERT


class BertTextClassifier(nn.Module):
    """BERT-based classifier for AI-generated text detection."""
    
    def __init__(self, bert_model_path=None, num_labels=2, use_distil=False):
        super().__init__()
        if use_distil:
            # Use DistilBERT which is much smaller and faster
            model_name = "distilbert-base-uncased"
            self.bert = DistilBertForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float32
            )
            self.is_distil = True
        elif bert_model_path and os.path.exists(bert_model_path):
            # Load from local BERT model with memory-efficient options
            try:
                self.bert = BertModel.from_pretrained(
                    bert_model_path,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.float32
                )
                self.is_distil = False
            except Exception:
                # Fallback to DistilBERT if local model fails
                model_name = "distilbert-base-uncased"
                self.bert = DistilBertForSequenceClassification.from_pretrained(
                    model_name,
                    num_labels=num_labels,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.float32
                )
                self.is_distil = True
        else:
            # Use DistilBERT as default (smaller, faster, less memory)
            model_name = "distilbert-base-uncased"
            self.bert = DistilBertForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float32
            )
            self.is_distil = True
        
        if not self.is_distil:
            self.dropout = nn.Dropout(0.1)
            self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask=None):
        if self.is_distil:
            return self.bert(input_ids=input_ids, attention_mask=attention_mask)
        else:
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.pooler_output
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
            return type('Output', (), {'logits': logits})()


def load_text_model(bert_dir=None, use_distil=True, force_cpu=True):
    """Load BERT model and tokenizer for text classification with memory optimization."""
    global device
    if force_cpu:
        device = "cpu"
    
    # Clear cache before loading
    gc.collect()
    if torch.cuda.is_available() and not force_cpu:
        torch.cuda.empty_cache()
    
    if bert_dir is None:
        bert_dir = BERT_DIR
    
    bert_path = str(bert_dir) if os.path.exists(bert_dir) else None
    
    try:
        # Prefer DistilBERT for lower memory usage
        if use_distil or bert_path is None:
            tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
            model = DistilBertForSequenceClassification.from_pretrained(
                "distilbert-base-uncased",
                num_labels=2,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float32
            )
            model.to(device)
            model.eval()
            return model, tokenizer
        
        # Try to load local BERT model
        tokenizer = BertTokenizer.from_pretrained(bert_path)
        try:
            model = BertForSequenceClassification.from_pretrained(
                bert_path,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float32
            )
        except Exception:
            # If that fails, use our custom classifier
            model = BertTextClassifier(bert_model_path=bert_path, use_distil=False)
            # Initialize classifier weights
            nn.init.xavier_uniform_(model.classifier.weight)
            nn.init.zeros_(model.classifier.bias)
        
        model.to(device)
        model.eval()
        return model, tokenizer
    except Exception as e:
        # Fallback to DistilBERT (much smaller)
        print(f"Warning: Could not load BERT model: {e}")
        print("Falling back to DistilBERT (memory-efficient)")
        gc.collect()
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=2,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float32
        )
        model.to(device)
        model.eval()
        return model, tokenizer


def predict_text(text, model, tokenizer, max_length=256):
    """Predict if text is AI-generated or human-written with memory optimization."""
    # Use shorter max_length to reduce memory
    # Tokenize input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length
    )
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get prediction with no_grad to save memory
    with torch.no_grad():
        try:
            outputs = model(**inputs)
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
        except Exception:
            # Fallback for custom models
            logits = model(inputs["input_ids"], attention_mask=inputs.get("attention_mask"))
            if hasattr(logits, 'logits'):
                logits = logits.logits
    
    # Get probabilities
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    confidence, prediction = torch.max(probabilities, dim=-1)
    
    # Move to CPU and convert to Python types to free GPU memory
    probabilities = probabilities.cpu()
    prediction = prediction.cpu()
    confidence = confidence.cpu()
    
    # Map prediction to label (0 = human, 1 = AI-generated)
    label = "AI-Generated" if prediction.item() == 1 else "Human-Written"
    ai_probability = probabilities[0][1].item() if probabilities.shape[1] > 1 else 0.0
    
    # Clear inputs from memory
    del inputs
    gc.collect()
    
    return {
        "label": label,
        "ai_probability": float(ai_probability),
        "confidence": float(confidence.item()),
        "human_probability": float(probabilities[0][0].item()) if probabilities.shape[1] > 1 else 1.0 - ai_probability
    }


def is_ai_generated(text, model, tokenizer, threshold=0.5):
    """Simple binary classification based on threshold."""
    result = predict_text(text, model, tokenizer)
    return result["ai_probability"] >= threshold

