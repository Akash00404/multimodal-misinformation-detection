import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
MODEL_PATH = "./bert_module/bert_fake_news_model"

tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)

model.eval()  # inference mode
def bert_predict(text, max_length=256):
    """
    Predict Fake / Real using trained BERT model
    """
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probs = torch.softmax(logits, dim=1).numpy()[0]

    prediction = np.argmax(probs)
    confidence = probs[prediction]

    label = "Fake" if prediction == 0 else "Real"

    return {
        "prediction": label,
        "confidence_percent": round(confidence * 100, 2)
    }
if __name__ == "__main__":
    sample_text = """
    The Reserve Bank of India kept its benchmark interest rates unchanged
    after reviewing inflation trends and global economic conditions.
    """

    result = bert_predict(sample_text)

    print("\n--- BERT PREDICTION RESULT ---")
    for k, v in result.items():
        print(f"{k}: {v}")
