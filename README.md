# 🤖 Customer Sentiment Analysis AI

A professional end-to-end **Customer Sentiment Analysis System** built using **TF-IDF, Logistic Regression, DistilBERT, XLM-RoBERTa, PyTorch, FastAPI, and JavaScript**.

The system analyzes customer reviews and classifies them into three sentiment categories:

* 🟢 **Positive**
* 🟡 **Neutral**
* 🔴 **Negative**

The system provides:

* Sentiment prediction
* Confidence score
* Probability for each sentiment class
* REST API
* Input validation
* Automated API testing
* Modern web frontend
* English sentiment analysis using DistilBERT
* Bangla sentiment analysis using XLM-RoBERTa

---

# 📌 Project Overview

Customer reviews contain valuable information about how customers feel about products and services.

Manually analyzing thousands of reviews is difficult and time-consuming. This project aims to automate that process using Natural Language Processing and Transformer-based Machine Learning.

The project follows a complete machine-learning workflow:

```text
Customer Reviews
       ↓
Dataset Preparation
       ↓
Data Cleaning & Validation
       ↓
Text Processing
       ↓
TF-IDF Representation
       ↓
Traditional ML Baseline
       ↓
Model Evaluation
       ↓
Transformer Fine-Tuning
       ↓
DistilBERT Evaluation
       ↓
XLM-RoBERTa Evaluation
       ↓
Model Comparison
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

## Performance Target

The target for the main sentiment classification system was:

> **Macro F1 ≥ 0.85**

Macro F1 was selected because this is a three-class classification problem where each sentiment class should contribute equally to the overall evaluation.

---

# 🧠 Model Development

The project uses multiple approaches so that traditional machine learning can be compared with transformer-based NLP models.

The main approaches are:

1. **TF-IDF + Logistic Regression**
2. **Fine-Tuned DistilBERT**
3. **Fine-Tuned XLM-RoBERTa**

The models serve different purposes.

| Model                        | Purpose                               | Main Language   | Role                 |
| ---------------------------- | ------------------------------------- | --------------- | -------------------- |
| TF-IDF + Logistic Regression | Traditional baseline                  | English/general | Baseline             |
| DistilBERT                   | Main high-performance sentiment model | English/general | **Primary model**    |
| XLM-RoBERTa                  | Bangla/multilingual sentiment         | Bangla          | **Additional model** |

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

The baseline achieved a **Macro F1 of 0.8780**, exceeding the target of 0.85.

## Baseline Classification Report

| Sentiment        | Precision |   Recall | F1-score |   Support |
| ---------------- | --------: | -------: | -------: | --------: |
| Negative         |      0.87 |     0.89 |     0.88 |     2,500 |
| Neutral          |      0.84 |     0.81 |     0.82 |     2,579 |
| Positive         |      0.93 |     0.94 |     0.93 |     2,592 |
| **Macro Avg**    |  **0.88** | **0.88** | **0.88** | **7,671** |
| **Weighted Avg** |  **0.88** | **0.88** | **0.88** | **7,671** |

The baseline performed best on the **positive** class.

The **neutral** class was more difficult to classify, with an F1-score of approximately **0.82**.

---

# 2️⃣ Fine-Tuned DistilBERT

After establishing the traditional baseline, a pretrained **DistilBERT** model was fine-tuned for the same three-class sentiment classification task.

DistilBERT was selected because it provides transformer-based language understanding while being smaller and more efficient than full-sized BERT.

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

The final evaluation produced approximately:

* Accuracy: **0.8991**
* Precision: **0.8986**
* Recall: **0.8991**
* Macro F1: **0.8987**

This makes DistilBERT the **best-performing model in the current project**.

## DistilBERT Classification Report

| Sentiment        | Precision |   Recall | F1-score |   Support |
| ---------------- | --------: | -------: | -------: | --------: |
| Negative         |      0.88 |     0.91 |     0.90 |     2,500 |
| Neutral          |      0.87 |     0.84 |     0.85 |     2,579 |
| Positive         |      0.95 |     0.95 |     0.95 |     2,592 |
| **Macro Avg**    |  **0.90** | **0.90** | **0.90** | **7,671** |
| **Weighted Avg** |  **0.90** | **0.90** | **0.90** | **7,671** |

---

# 3️⃣ XLM-RoBERTa for Bangla Sentiment Analysis

To extend the system beyond the primary English/general sentiment model, an additional **XLM-RoBERTa** model was fine-tuned for Bangla sentiment classification.

XLM-RoBERTa is a multilingual transformer model and is more suitable for multilingual text than an English-only DistilBERT model.

The Bangla model predicts:

```text
Negative
Neutral
Positive
```

## XLM-RoBERTa Dataset

The XLM-RoBERTa experiment used a separate Bangla sentiment dataset.

The final dataset validation showed:

```text
Train samples: 17,095

Labels:
0 → 5,659
1 → 5,761
2 → 5,675

Missing text: 0
Missing label: 0
Duplicate texts: 0
```

The dataset was therefore:

* Free from missing text
* Free from missing labels
* Free from duplicate texts
* Reasonably balanced across the three classes

## XLM-RoBERTa Training

The model was trained for:

```text
Epochs: 3
```

The validation results were:

| Epoch | Validation Loss |   Accuracy | Precision | Recall |         F1 |
| ----- | --------------: | ---------: | --------: | -----: | ---------: |
| 1     |          0.6285 |     0.7651 |    0.7916 | 0.7661 |     0.7658 |
| 2     |          0.5618 |     0.7843 |    0.8014 | 0.7845 |     0.7848 |
| 3     |          0.5937 | **0.7880** |    0.7947 | 0.7882 | **0.7883** |

The best reported final validation result was approximately:

> **Validation Macro F1 = 0.7883**

---

# ⚠️ Dataset Considerations

One important issue discovered during development was the difference in dataset size and composition between the main sentiment model and the Bangla XLM-RoBERTa experiment.

The main DistilBERT evaluation was performed using a much larger dataset, while the Bangla XLM-RoBERTa experiment used **17,095 training samples**.

This difference matters because transformer models generally benefit from having sufficient high-quality training data that represents the language and domain they will encounter.

The Bangla dataset was balanced and had no missing or duplicate text in the final check, but it was still considerably smaller than the dataset used for the main sentiment model.

Therefore, the lower XLM-RoBERTa score should **not automatically be interpreted as XLM-RoBERTa being a worse architecture**.

The result is affected by factors such as:

* Dataset size
* Dataset quality
* Bangla language coverage
* Domain differences
* Vocabulary
* Review-writing style
* Training configuration
* Amount of training data
* Distribution of sentiment expressions

A larger and more diverse Bangla dataset could potentially improve the XLM-RoBERTa model substantially.

---

# 📊 Model Performance Comparison

The current experiments show that **DistilBERT provides the best measured performance** in this project.

| Model                        | Language / Use |   Accuracy |  Precision |     Recall |   Macro F1 |
| ---------------------------- | -------------- | ---------: | ---------: | ---------: | ---------: |
| TF-IDF + Logistic Regression | Main baseline  | **0.8786** | **0.8778** | **0.8787** | **0.8780** |
| **Fine-Tuned DistilBERT**    | Main model     | **0.8991** | **0.8986** | **0.8991** | **0.8987** |
| XLM-RoBERTa                  | Bangla model   | **0.7880** | **0.7947** | **0.7882** | **0.7883** |

> **Note:** The XLM-RoBERTa numbers above are validation results, while the DistilBERT numbers are from the final evaluation/test workflow. They should therefore be interpreted as indicative model results rather than a perfectly controlled apples-to-apples comparison.

---

# 🏆 Current Best Model

Based on the measured results, **Fine-Tuned DistilBERT is currently the best-performing model**.

Its Macro F1 is approximately:

```text
DistilBERT
    0.8987
```

compared with:

```text
TF-IDF + Logistic Regression
    0.8780
```

Therefore, DistilBERT remains the **primary production model** for the main sentiment-analysis pipeline.

XLM-RoBERTa is included as an additional model specifically to extend the system to **Bangla/multilingual sentiment analysis**.

---

# 📈 DistilBERT vs TF-IDF

The Macro F1 improvement was:

```text
0.8987 - 0.8780
= 0.0207
```

Therefore, DistilBERT improved the measured Macro F1 by approximately:

> **+0.0207**

Using rounded values:

```text
0.90 - 0.8780
= 0.0220
```

So the rounded improvement is approximately:

> **+0.0220**

---

# 🔎 Class-Level Comparison

| Sentiment | TF-IDF F1 | DistilBERT F1 | Improvement |
| --------- | --------: | ------------: | ----------: |
| Negative  |      0.88 |      **0.90** |       +0.02 |
| Neutral   |      0.82 |      **0.85** |       +0.03 |
| Positive  |      0.93 |      **0.95** |       +0.02 |

The largest improvement occurred in the **neutral** class.

Its F1-score increased approximately from:

```text
0.82 → 0.85
```

This indicates that the transformer model was better at handling ambiguous or moderately expressed sentiment than the traditional TF-IDF baseline.

---

# 🏗️ Current System Architecture

The current system contains two transformer models:

```text
                         Customer Review
                               │
                               ▼
                         FastAPI API
                               │
                         Model Selection
                         ┌─────┴─────┐
                         │           │
                         ▼           ▼
                    DistilBERT   XLM-RoBERTa
                    Main Model   Bangla Model
                         │           │
                         └─────┬─────┘
                               ▼
                         Sentiment
                               │
                ┌──────────────┼──────────────┐
                ▼              ▼              ▼
             Negative       Neutral        Positive
                │              │              │
                └──────────────┼──────────────┘
                               ▼
                       Confidence Score
                               │
                               ▼
                       Class Probabilities
                               │
                               ▼
                         JSON Response
                               │
                               ▼
                         Web Frontend
```

---

# 🤖 Model Configuration

## DistilBERT

The primary model is based on:

```text
Architecture: DistilBERT
Classes: 3
Epochs: 3
Batch Size: 16
Learning Rate: 5e-5
Maximum Sequence Length: 128
Inference Device: CPU
```

Model:

```text
mdjrjoy/distilbert-fine-tuned
```

---

## XLM-RoBERTa

The additional Bangla model is loaded from:

```text
mdjrjoy/xlm-roberta-based
```

It is used for multilingual/Bangla sentiment prediction.

---

# 🚀 FastAPI REST API

The trained models are integrated into a **FastAPI REST API**.

The API provides:

* Sentiment prediction
* Model selection
* Confidence score
* Probability for each sentiment class
* Input validation
* Automatic API documentation
* Error handling

---

# 📡 API Endpoints

## GET `/`

Checks whether the API is running.

Example:

```json
{
  "message": "Sentiment Analysis API is running locally!"
}
```

---

# POST `/predict`

Analyzes a customer review.

The API supports model selection.

## DistilBERT Request

```json
{
  "text": "I really love this product!",
  "model": "distilbert"
}
```

## XLM-RoBERTa Request

```json
{
  "text": "এই পণ্যটি অনেক ভালো",
  "model": "xlm-roberta"
}
```

## XLM-RoBERTa Example Response

```json
{
  "text": "এই পণ্যটি অনেক ভালো",
  "sentiment": "POSITIVE",
  "confidence": 0.9434,
  "probabilities": {
    "NEUTRAL": 0.0418,
    "POSITIVE": 0.9434,
    "NEGATIVE": 0.0148
  },
  "model": "xlm-roberta"
}
```

The exact confidence and probability values can vary depending on the input.

---

# 🧠 Prediction Process

The prediction pipeline works approximately as follows:

```text
Input Review
     ↓
Select Model
     ↓
Select Tokenizer
     ↓
Tokenization
     ↓
Transformer Model
     ↓
Logits
     ↓
Softmax
     ↓
Class Probabilities
     ↓
Highest Probability Class
     ↓
Sentiment + Confidence
```

For example:

```text
Input:
এই পণ্যটি অনেক ভালো

        ↓

XLM-RoBERTa Tokenizer

        ↓

XLM-RoBERTa

        ↓

Logits

        ↓

Softmax

        ↓

Positive = 0.9434
Neutral  = 0.0418
Negative = 0.0148

        ↓

Prediction:
POSITIVE
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

The frontend communicates with the FastAPI backend through:

```text
POST /predict
```

---

# 🧪 API Testing

The API was tested using:

* **pytest**
* **FastAPI TestClient**

The final test suite successfully passed:

```text
8 passed
```

The tests verify:

* API health
* Positive sentiment prediction
* Negative sentiment prediction
* Neutral sentiment prediction
* DistilBERT model selection
* XLM-RoBERTa model selection
* Empty input validation
* Maximum text-length validation

The test suite can be executed using:

```bash
python -m pytest -v
```

---

# 📁 Project Structure

```text
sentiment_analysis_project/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── model.py
│   ├── schemas.py
│   └── tests/
│       └── test_api.py
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── requirements.txt
├── .gitignore
└── README.md
```

The trained models are hosted separately and loaded by the application.

---

# ⚙️ Technologies

## Machine Learning

* Python
* PyTorch
* Hugging Face Transformers
* DistilBERT
* XLM-RoBERTa
* Scikit-learn

## NLP

* TF-IDF
* Text Classification
* Tokenization
* Transformer-based NLP
* Multilingual NLP

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

## Model Hosting

* Hugging Face Hub

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

The Swagger interface allows you to directly test the `/predict` endpoint.

---

# 🌐 Running the Frontend

Open:

```text
frontend/index.html
```

in a browser.

For development, VS Code **Live Server** can be used.

The frontend sends requests to:

```text
http://127.0.0.1:8000/predict
```

---

# 📦 Dependencies

The project uses packages including:

```text
torch
transformers
fastapi
uvicorn
pydantic
scikit-learn
pytest
```

The exact versions used for deployment should be maintained in:

```text
requirements.txt
```

---

# ⚠️ Limitations

Although the models achieve strong performance, sentiment classification is not perfect.

The system may struggle with:

* Sarcasm
* Very short reviews
* Ambiguous statements
* Mixed sentiment
* Unusual spelling
* Bangla spelling variations
* Banglish text
* Domain-specific terminology
* Unseen vocabulary
* Text significantly different from the training data

The confidence score represents the model's predicted probability and should not be interpreted as a guarantee of correctness.

---

# 📊 Important Dataset Limitation

A major consideration in the project is **dataset quality and size**.

The main DistilBERT model was trained and evaluated using a substantially larger dataset than the Bangla XLM-RoBERTa experiment.

The Bangla XLM-RoBERTa training dataset contained:

```text
17,095 training samples
```

Although the dataset was balanced and had:

```text
Missing text: 0
Missing labels: 0
Duplicate texts: 0
```

the overall amount of training data was still relatively limited compared with the main sentiment dataset.

As a result, the XLM-RoBERTa model achieved a lower validation Macro F1:

```text
0.7883
```

compared with the main DistilBERT model:

```text
≈ 0.8987
```

This is one of the main areas that can be improved in future development.

A larger, cleaner, and more diverse Bangla customer-review dataset would likely provide a stronger basis for improving Bangla sentiment classification.

---

# 🔮 Future Improvements

Potential future improvements include:

* Larger Bangla sentiment dataset
* More diverse Bangla customer reviews
* Banglish sentiment support
* Batch CSV sentiment analysis
* Customer sentiment dashboard
* Sentiment distribution visualization
* Docker deployment
* Cloud deployment
* API authentication
* Request logging
* Monitoring
* Model versioning
* Hyperparameter optimization
* Better multilingual model comparison
* Improved Bangla preprocessing
* Explainable AI features

---

# 📈 Complete Project Workflow

```text
                         DATASET
                            │
                            ▼
                  Data Preparation
                            │
                            ▼
                 Data Validation
                            │
                            ▼
                    Text Processing
                            │
             ┌──────────────┴──────────────┐
             │                             │
             ▼                             ▼
           TF-IDF                     Transformers
             │                       ┌─────┴─────┐
             ▼                       ▼           ▼
     Logistic Regression         DistilBERT   XLM-RoBERTa
             │                       │           │
             ▼                       ▼           ▼
       Baseline Model          Main Model    Bangla Model
             │                       │           │
             └──────────────┬────────┴───────────┘
                            ▼
                     Model Evaluation
                            │
                            ▼
                     Model Comparison
                            │
             ┌──────────────┴──────────────┐
             │                             │
             ▼                             ▼
        Best Main Model              Bangla Extension
         DistilBERT                  XLM-RoBERTa
             │                             │
             └──────────────┬──────────────┘
                            ▼
                         FastAPI
                            │
                            ▼
                      REST API
                            │
                            ▼
                      Web Frontend
                            │
                            ▼
                  Customer Sentiment
```

---

# 🎓 Key Results

## Baseline

**TF-IDF + Logistic Regression**

> Macro F1 = **0.8780**

## Main Model

**Fine-Tuned DistilBERT**

> Macro F1 ≈ **0.8987**

## Bangla Model

**Fine-Tuned XLM-RoBERTa**

> Validation Macro F1 = **0.7883**

## Target

> Macro F1 ≥ **0.85**

## Main Result

> ✅ **The primary DistilBERT model exceeded the target.**

The DistilBERT model currently provides the strongest measured performance and is therefore used as the **primary model**.

XLM-RoBERTa extends the project toward **Bangla/multilingual sentiment analysis**, but its current performance is lower due in part to the smaller Bangla training dataset and different dataset characteristics.

---

# 👨‍💻 Author

**JR. Joy**

Computer Science & Engineering

This project demonstrates an end-to-end machine-learning workflow covering:

```text
Dataset Preparation
        ↓
NLP
        ↓
Traditional Machine Learning
        ↓
TF-IDF
        ↓
Logistic Regression
        ↓
Transformer Fine-Tuning
        ↓
DistilBERT
        ↓
XLM-RoBERTa
        ↓
Model Evaluation
        ↓
Model Comparison
        ↓
FastAPI
        ↓
API Testing
        ↓
Web Frontend
```

---

# ⭐ Project Status

**Core sentiment-analysis system completed.**

Current system includes:

* ✅ Dataset preparation
* ✅ Dataset validation
* ✅ TF-IDF baseline
* ✅ Logistic Regression
* ✅ Baseline evaluation
* ✅ DistilBERT fine-tuning
* ✅ DistilBERT evaluation
* ✅ XLM-RoBERTa fine-tuning
* ✅ Bangla sentiment model
* ✅ Model comparison
* ✅ Hugging Face model hosting
* ✅ FastAPI backend
* ✅ REST API
* ✅ Model selection through API
* ✅ Frontend
* ✅ API testing
* ✅ 8 automated tests passing
* ✅ Local CPU inference

The next stage focuses on **improving the Bangla dataset/model, deployment, monitoring, and productionization**.
