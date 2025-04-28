from transformers import BertTokenizer, BertForSequenceClassification
import torch

tokenizer = BertTokenizer.from_pretrained("./models/bert_finetuned")
model = BertForSequenceClassification.from_pretrained("./models/bert_finetuned")
model.eval()

def predict_email(email_text):
    inputs = tokenizer(email_text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    confidence, pred = torch.max(probs, dim=-1)
    labels = ["urgent", "non-urgent", "needs human"]
    return labels[pred.item()], round(confidence.item(), 2)

# Example
email = "Hi team, server is down and customers are affected urgently."
label, confidence = predict_email(email)
print(f"Prediction: {label}, Confidence: {confidence * 100}%")
