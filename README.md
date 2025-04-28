# BERT Email Classification with FastAPI Deployment and Model Optimization

Fine-tuned a BERT model to classify emails into three categories:  
**urgent**, **non-urgent**, and **needs human intervention**, based on realistic synthetic email data.  
Deployed the model as a real-time API using FastAPI, applied model optimization techniques, and added a preprocessing layer to improve real-world performance.

---

## 📚 Project Overview

- Fine-tuned `bert-base-uncased` model on a synthetic dataset of multi-sentence, realistic email samples.
- Classified emails into three actionable categories: urgent, non-urgent, needs human intervention.
- Built a real-time REST API using FastAPI with live Swagger UI documentation.
- Applied dynamic quantization to reduce model size and inference latency.
- Implemented a preprocessing layer to clean greetings, signatures, and irrelevant noise from emails before classification.

---

## 🚀 Key Features

- **Transformer Fine-Tuning**: Customized BERT for domain-specific email classification tasks.
- **FastAPI Deployment**: Lightweight, scalable RESTful API for live email predictions.
- **Confidence Scoring**: Outputs both predicted label and confidence percentage.
- **Dynamic Quantization**: Achieved nearly 90% model size reduction with negligible loss in accuracy.
- **Preprocessing Layer**: Removed greetings, signatures, and redundant text to focus model attention on actionable content.
- **Real-World Readiness**: Project structured and optimized for scalable production deployment.

---

## 🛠️ Project Structure

```
bert_email_classifier/
├── app/
│   ├── app.py
├── data/
│   ├── train.csv
│   └── test.csv
├── models/
│   ├── bert_finetuned/
│   └── bert_finetuned_quantized/
├── scripts/
│   ├── train.py
│   ├── infer.py
│   ├── quantize_model.py
│   ├── compare_model_sizes.py
├── utils/
│   ├── preprocessing.py
├── requirements.txt
└── README.md
```

## How to Run
1. Install Requirements
```
    pip install -r requirements.txt
```

2. Fine-Tune the Model
```
    python scripts/train.py
```

3. Test Inference
```
   python scripts/infer.py
```

4. Quantize the Model
```
    python scripts/quantize_model.py
```
5. Compare Model Sizes
```
   python scripts/compare_model_sizes.py
```

6. Serve API
```
    uvicorn app.app:app --reload
```
7. Access API Documentation
```
    Open browser: http://127.0.0.1:8000/docs
```
## Model Performance
```
    Metric	Score
    Precision	0.92
    Recall	0.91
    F1-Score	0.91
    (Evaluated on 20% test split.)
```
## Model Optimization Results

## 🧠 Model Optimization Results
```

| Model | Size (MB) | Size Reduction | Accuracy Impact |
|:------|:----------|:----------------|:----------------|
| Original BERT | 1671.08 | — | 0% |
| Quantized BERT | 173.34 | 89.63% smaller | ~0% |
```
-Reduced model size by ~90%  
-Achieved faster inference with negligible accuracy loss.

## Real-World Production Readiness Improvements
Preprocessing Layer: Stripped email greetings, signatures, and irrelevant noise to improve model robustness.
Confidence Scoring: Outputs confidence along with predictions.
Quantized Inference: Optimized for CPU deployment with smaller memory footprint.

## Future Work
Expand to real-world email datasets with larger variability.
Fine-tune lightweight models like DistilBERT or MiniLM for even faster APIs.
Add batch email classification endpoint.
Deploy public demo using HuggingFace Spaces or Render.
Add Gradio UI for user-friendly frontend interaction.

## Acknowledgements
Huggingface Transformers
FastAPI
scikit-learn

## Contact
Feel free to connect: LinkedIn
Swagath Babu