# Customer Sentiment Analysis AI

A professional end-to-end **Customer Sentiment Analysis System** built using **TF-IDF, Logistic Regression, DistilBERT, PyTorch, FastAPI, and JavaScript**.

The system analyzes customer reviews and classifies them into three sentiment categories:

* 🟢 **Positive**
* 🟡 **Neutral**
* 🔴 **Negative**

The final system provides sentiment prediction, confidence score, and class probabilities through a REST API and a modern web interface.

---

## 📌 Project Overview

Customer reviews contain valuable information about how customers feel about products and services.

Manually analyzing thousands of reviews is difficult and time-consuming. This project aims to automate that process using Natural Language Processing and Machine Learning.

The project follows a complete machine-learning workflow:

```text
Customer Reviews
       ↓
Data Preparation
       ↓
Text Processing
       ↓
TF-IDF Representation
       ↓
Traditional ML Baseline
       ↓
Model Evaluation
       ↓
DistilBERT Fine-Tuning
       ↓
Model Comparison
       ↓
Final Model Selection
       ↓
FastAPI REST API
       ↓
Web Frontend
```

---

# 🎯 Objective

The primary objective is to build an automated system capable of analyzing customer reviews and predicting their sentiment.

The system supports three classes:

```text
Negative
Neutral
Positive
```

### Performance Target

The target for the project was:

> **Macro F1 ≥ 0.85**

Macro F1 was selected because the task contains three sentiment classes and each class should contribute equally to the overall evaluation.

---

# 🧠 Model Development

Instead of directly using a transformer model, the project first established a traditional machine-learning baseline.

Two main approaches were evaluated:

1. **TF-IDF + Logistic Regression**
2. **Fine-Tuned DistilBERT**

This allows the final transformer model to be compared against a strong traditional NLP baseline.

---

# 1️⃣ TF-IDF + Logistic Regression Baseline

The first model used **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert customer reviews into numerical feature vectors.

These features were then provided to a **Logistic Regression** classifier.

TF-IDF + Logistic Regression was selected as the baseline because it is:

* Fast
* Computationally efficient
* Easy to interpret
* Strong for traditional text classification
* Suitable for comparison with transformer-based models

## Baseline Performance

| Metric          |      Score |
| --------------- | ---------: |
| Accuracy        |   **0.88** |
| Precision       |   **0.88** |
| Recall          |   **0.88** |
| Macro F1        | **0.8780** |
| Target Macro F1 | **≥ 0.85** |
| Target Achieved |  ✅ **Yes** |

The baseline achieved a **Macro F1 of 0.8780**, successfully exceeding the target of **0.85**.

## Baseline Classification Report

| Sentiment        | Precision |   Recall | F1-score |   Support |
| ---------------- | --------: | -------: | -------: | --------: |
| Negative         |      0.87 |     0.89 |     0.88 |     2,500 |
| Neutral          |      0.84 |     0.81 |     0.82 |     2,579 |
| Positive         |      0.93 |     0.94 |     0.93 |     2,592 |
| **Macro Avg**    |  **0.88** | **0.88** | **0.88** | **7,671** |
| **Weighted Avg** |  **0.88** | **0.88** | **0.88** | **7,671** |

The baseline performed best on the **positive** class.

The **neutral** class was more difficult to classify, achieving an F1-score of **0.82**.

---

# 2️⃣ Fine-Tuned DistilBERT

After establishing the traditional machine-learning baseline, a pretrained **DistilBERT** model was fine-tuned for the same three-class sentiment classification task.

DistilBERT was selected because it provides transformer-based language understanding while being smaller and faster than larger BERT-family models.

The model predicts:

```text
Negative
Neutral
Positive
```

## DistilBERT Performance

| Metric          |      Score |
| --------------- | ---------: |
| Accuracy        |   **0.90** |
| Precision       |   **0.90** |
| Recall          |   **0.90** |
| Macro F1        |   **0.90** |
| Target Macro F1 | **≥ 0.85** |
| Target Achieved |  ✅ **Yes** |

## DistilBERT Classification Report

| Sentiment        | Precision |   Recall | F1-score |   Support |
| ---------------- | --------: | -------: | -------: | --------: |
| Negative         |      0.88 |     0.91 |     0.90 |     2,500 |
| Neutral          |      0.87 |     0.84 |     0.85 |     2,579 |
| Positive         |      0.95 |     0.95 |     0.95 |     2,592 |
| **Macro Avg**    |  **0.90** | **0.90** | **0.90** | **7,671** |
| **Weighted Avg** |  **0.90** | **0.90** | **0.90** | **7,671** |

---

# 📊 Baseline vs DistilBERT

Both models were evaluated on the same evaluation set containing **7,671 samples**.

| Model                        | Accuracy | Precision |   Recall |   Macro F1 |
| ---------------------------- | -------: | --------: | -------: | ---------: |
| TF-IDF + Logistic Regression | **0.88** |  **0.88** | **0.88** | **0.8780** |
| **Fine-Tuned DistilBERT**    | **0.90** |  **0.90** | **0.90** |   **0.90** |
| Target                       |        — |         — |        — | **≥ 0.85** |

## Performance Improvement

Macro F1 improved from:

```text
TF-IDF + Logistic Regression
        0.8780
           ↓
Fine-Tuned DistilBERT
        0.9000
```

### Absolute improvement

```text
0.9000 - 0.8780 = 0.0220
```

The final model achieved an absolute Macro F1 improvement of:

> **+0.0220**

This demonstrates that the fine-tuned transformer model provided better overall sentiment classification performance than the traditional baseline.

---

# 🔎 Class-Level Comparison

| Sentiment | Baseline F1 | DistilBERT F1 | Improvement |
| --------- | ----------: | ------------: | ----------: |
| Negative  |        0.88 |      **0.90** |       +0.02 |
| Neutral   |        0.82 |      **0.85** |       +0.03 |
| Positive  |        0.93 |      **0.95** |       +0.02 |

The largest class-level improvement occurred for the **neutral** class.

Its F1-score increased from:

```text
0.82 → 0.85
```

This indicates that DistilBERT handled ambiguous or moderately expressed sentiment better than the TF-IDF + Logistic Regression baseline.

---

# 🏆 Final Model Selection

The experiments demonstrate that traditional NLP techniques can provide a strong baseline.

However, the fine-tuned DistilBERT model achieved better overall performance across the main evaluation metrics.

Therefore:

> **Fine-Tuned DistilBERT was selected as the final production model.**

The final development pipeline was:

```text
Customer Reviews
       ↓
     TF-IDF
       ↓
Logistic Regression
       ↓
Baseline
Macro F1 = 0.8780
       ↓
   DistilBERT
       ↓
   Fine-Tuning
       ↓
Final Model
Macro F1 = 0.90
       ↓
    FastAPI
       ↓
 Web Frontend
```

---

# 🤖 Model Configuration

The final model is based on **DistilBERT** and was fine-tuned for three-class sentiment classification.

### Training configuration

| Parameter               | Value      |
| ----------------------- | ---------- |
| Architecture            | DistilBERT |
| Number of Classes       | 3          |
| Epochs                  | 3          |
| Batch Size              | 16         |
| Learning Rate           | 5e-5       |
| Maximum Sequence Length | 128        |
| Inference Device        | CPU        |

---

# 🚀 API

The trained model is integrated into a **FastAPI REST API**.

The API provides:

* Sentiment prediction
* Confidence score
* Probability for each sentiment class
* Input validation
* Automatic API documentation

## API Architecture

```text
Client
  │
  │ POST /predict
  ▼
FastAPI
  │
  ▼
Pydantic Validation
  │
  ▼
Tokenizer
  │
  ▼
DistilBERT
  │
  ▼
Softmax
  │
  ├── Negative probability
  ├── Neutral probability
  └── Positive probability
  │
  ▼
JSON Response
```

---

# 📡 API Endpoints

## GET `/`

Checks whether the API is running.

Example response:

```json
{
  "message": "Sentiment Analysis API is running locally!"
}
```

---

## POST `/predict`

Analyzes a customer review.

### Request

```json
{
  "text": "This product is amazing and I love it!"
}
```

### Response

```json
{
  "sentiment": "positive",
  "confidence": 0.9547,
  "probabilities": {
    "negative": 0.0012,
    "neutral": 0.044,
    "positive": 0.9547
  }
}
```

---

# 🧪 Example Predictions

### Positive

Input:

```text
This product is amazing and I love it!
```

Output:

```json
{
  "sentiment": "positive",
  "confidence": 0.9547
}
```

### Neutral

Input:

```text
The product is okay, nothing special.
```

Output:

```json
{
  "sentiment": "neutral",
  "confidence": 0.913
}
```

### Negative

Input:

```text
Very bad quality, I regret buying it.
```

Output:

```json
{
  "sentiment": "negative",
  "confidence": 0.9881
}
```

---

# 🌐 Web Frontend

A modern web interface was developed using:

* HTML
* CSS
* JavaScript

The frontend allows users to:

1. Enter a customer review.
2. Submit the review.
3. Send the review to the FastAPI backend.
4. Receive the prediction.
5. Display the predicted sentiment.
6. Display confidence.
7. Display probabilities for all three classes.

```text
┌─────────────────────────────────────┐
│     Customer Sentiment Analysis     │
│                                     │
│  Enter Customer Review              │
│  ┌───────────────────────────────┐  │
│  │ This product is excellent!    │  │
│  │                               │  │
│  └───────────────────────────────┘  │
│                                     │
│       [ Analyze Sentiment ]         │
│                                     │
│  Sentiment              POSITIVE   │
│  Confidence               95.65%   │
│                                     │
│  Negative                0.09%     │
│  Neutral                 4.26%     │
│  Positive               95.65%     │
└─────────────────────────────────────┘
```

---

# 🧪 Testing

The API was tested using **pytest** and FastAPI's testing utilities.

The test suite verifies:

* API health
* Positive sentiment prediction
* Negative sentiment prediction
* Neutral sentiment prediction
* Input validation
* API response structure

Run the tests with:

```bash
python -m pytest -v
```

Example:

```text
4 passed
```

---

# 📁 Project Structure

```text
sentiment_analysis_project/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   └── ...
│
├── tests/
│   └── test_api.py
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── Model/
│   └── distilbert_final/
│       ├── config.json
│       ├── model.safetensors
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       └── training_args.bin
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# ⚙️ Technologies

## Machine Learning

* Python
* PyTorch
* Hugging Face Transformers
* DistilBERT
* Scikit-learn

## NLP

* TF-IDF
* Text classification
* Tokenization
* Transformer-based NLP

## Backend

* FastAPI
* Uvicorn
* Pydantic

## Frontend

* HTML
* CSS
* JavaScript

## Testing

* pytest
* FastAPI TestClient

---

# 💻 Installation

Clone the repository:

```bash
git clone <your-github-repository-url>
cd sentiment_analysis_project
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the API

From the project root:

```bash
python -m uvicorn app.main:app --reload
```

The API will run at:

```text
http://127.0.0.1:8000
```

---

# 📖 API Documentation

FastAPI automatically provides interactive API documentation.

Open:

```text
http://127.0.0.1:8000/docs
```

This allows you to test `/predict` directly from the browser.

---

# 🌐 Running the Frontend

Open:

```text
frontend/index.html
```

in your browser.

For development, VS Code **Live Server** can be used.

The frontend communicates with:

```text
http://127.0.0.1:8000/predict
```

---

# 📦 Dependencies

The main environment used for the project includes:

```text
torch==2.8.0+cpu
transformers==4.56.2
fastapi==0.117.1
uvicorn==0.36.0
pydantic==2.11.7
```

Additional packages required for training and testing should be included in `requirements.txt`.

---

# ⚠️ Limitations

Although the final model achieved strong evaluation performance, sentiment classification is not perfect.

The model may struggle with:

* Sarcasm
* Very short reviews
* Ambiguous statements
* Mixed sentiment
* Unusual spelling
* Domain-specific terminology
* Text that differs significantly from the training data

The confidence score represents the model's predicted probability and should not be interpreted as guaranteed correctness.

---

# 🔮 Future Improvements

Potential future improvements include:

* Batch CSV sentiment analysis
* Customer sentiment dashboard
* Sentiment distribution visualization
* Docker deployment
* Cloud deployment
* API authentication
* Request logging
* Monitoring
* Model versioning
* Larger and more diverse datasets
* Hyperparameter optimization
* Comparison with additional transformer models
* Explainable AI features

---

# 📈 Project Workflow Summary

```text
                    DATASET
                       │
                       ▼
              Data Preparation
                       │
                       ▼
                Text Processing
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
          TF-IDF             DistilBERT
             │                   │
             ▼                   ▼
       Logistic Regression    Fine-Tuning
             │                   │
             ▼                   ▼
        Baseline Model       Transformer Model
             │                   │
             └─────────┬─────────┘
                       ▼
                 Model Evaluation
                       │
                       ▼
                Model Comparison
                       │
                       ▼
             Final DistilBERT Model
                       │
                       ▼
                    FastAPI
                       │
                       ▼
                 Web Frontend
                       │
                       ▼
              Customer Sentiment
```

---

# 🎓 Key Results

The project successfully developed an end-to-end customer sentiment analysis system.

### Baseline

**TF-IDF + Logistic Regression**

> Macro F1 = **0.8780**

### Final Model

**Fine-Tuned DistilBERT**

> Macro F1 = **0.90**

### Target

> Macro F1 ≥ **0.85**

### Result

> ✅ **Target exceeded**

The final DistilBERT model improved Macro F1 by **0.0220** compared with the TF-IDF + Logistic Regression baseline.

---

# 👨‍💻 Author

**JR. Joy**

Computer Science & Engineering

This project demonstrates an end-to-end machine-learning workflow covering:

```text
Data
→ NLP
→ Traditional ML
→ Transformer Fine-Tuning
→ Evaluation
→ API Development
→ Testing
→ Frontend Integration
```

---

## ⭐ Project Status

**Completed core ML pipeline and API integration.**

Current system includes:

* ✅ Dataset preparation
* ✅ TF-IDF baseline
* ✅ Logistic Regression
* ✅ Baseline evaluation
* ✅ DistilBERT fine-tuning
* ✅ Model evaluation
* ✅ Model comparison
* ✅ FastAPI backend
* ✅ REST API
* ✅ Frontend
* ✅ API testing
* ✅ Local CPU inference

Future work focuses on productionization, deployment, monitoring, and additional model improvements.
