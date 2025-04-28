from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from utils.preprocessing import preprocess_email


app = FastAPI()

# Load original fine-tuned model (NOT quantized)
tokenizer = BertTokenizer.from_pretrained("./models/bert_finetuned")
model = BertForSequenceClassification.from_pretrained("./models/bert_finetuned")
model.eval()

class EmailRequest(BaseModel):
    email_text: str

@app.post("/predict")
def predict(request: EmailRequest):
    # Preprocess the email text
    clean_text = preprocess_email(request.email_text)
    inputs = tokenizer(clean_text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        confidence, prediction_idx = torch.max(probs, dim=-1)

    labels = ["urgent", "non-urgent", "needs human"]
    predicted_label = labels[prediction_idx.item()]
    confidence_score = round(confidence.item() * 100, 2)
    
    return {
        "prediction": predicted_label,
        "confidence": f"{confidence_score}%"
    }
