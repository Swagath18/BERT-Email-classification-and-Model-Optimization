import pandas as pd
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import classification_report
import torch

# Mapping text labels to integer IDs
label2id = {"urgent": 0, "non-urgent": 1, "needs human": 2}
id2label = {0: "urgent", 1: "non-urgent", 2: "needs human"}

# Load datasets
train_df = pd.read_csv('./data/train.csv')
test_df = pd.read_csv('./data/test.csv')

# Encode labels
train_df['Label'] = train_df['Label'].map(label2id)
test_df['Label'] = test_df['Label'].map(label2id)

# Convert to Huggingface Datasets
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# Tokenization function (fixed)
def preprocess_function(examples):
    tokens = tokenizer(examples['Email Text'], truncation=True, padding="max_length", max_length=256)
    tokens["labels"] = examples["Label"]
    return tokens

train_dataset = train_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.map(preprocess_function, batched=True)

# Training Arguments
training_args = TrainingArguments(
    output_dir="./models/bert_finetuned",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train
trainer.train()

# Evaluate manually
predictions = trainer.predict(test_dataset)
preds = predictions.predictions.argmax(axis=-1)
labels = predictions.label_ids

report = classification_report(labels, preds, target_names=["urgent", "non-urgent", "needs human"])
print("\nClassification Report:\n", report)

# Save
with open("./models/bert_finetuned/metrics.txt", "w") as f:
    f.write(report)

model.save_pretrained("./models/bert_finetuned")
tokenizer.save_pretrained("./models/bert_finetuned")

print("\n✅ Training complete and model saved!")
