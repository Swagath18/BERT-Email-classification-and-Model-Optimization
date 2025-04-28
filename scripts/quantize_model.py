import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os

# Load fine-tuned model
model = BertForSequenceClassification.from_pretrained("./models/bert_finetuned")

# Apply dynamic quantization
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Save manually
save_path = "./models/bert_finetuned_quantized/"
os.makedirs(save_path, exist_ok=True)

# Save model using torch.save
torch.save(quantized_model.state_dict(), save_path + "pytorch_model.bin")

# Save config and tokenizer separately
tokenizer = BertTokenizer.from_pretrained("./models/bert_finetuned")
tokenizer.save_pretrained(save_path)
model.config.save_pretrained(save_path)

print("\n✅ Quantized model saved successfully to ./models/bert_finetuned_quantized/")
