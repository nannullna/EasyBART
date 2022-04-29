# EasyBART
Extractive/Abstractive summarY BART

## 0. Abstract

We present EasyBART, which namely suggests

## 1. Introduction

Transformer-based language models have shown tremendous performances on various natural language processing tasks including text classification, machine translation (MT), question answering (QA), and even natural language generation (NLG). These methods rely entirely on training an end-to-end deep learning model. The model is initially trained on a large corpus of a target language using masked language modeling (MLM), which aims at learning representations and semantic structures of the langauge (pre-training stage). Weights of the trained model are be used as a warm starting point, which can be fine-tuned with a relatively small dataset for various downstream tasks (fine-tuning stage).

Among many promising language models, BART successfully combines BERT's encoder architecture with GPT's autoregressive decoder architecture. This allows BART to focus more on language generation rather than classification and perform well on language generation task compared to the previous methods. 

---

## 2. Prerequisite

---

## 3. Method

### 3-1. Recursive Extraction Module

Use binary cross entropy loss (rather than cross entropy loss) to individual models

Linear on `<eos>` tokens

LSTM 

TCN

### 3-2. Loss Prediction Module (LPM)

### 3-3. ROUGE Prediction Module (RPM)

### 3-4. Fine-tuning -- One-step Training, Two-step Inference

---

## 4. Experiments

### 4-1. Dataset

Ai-hub

### 4-2. 

---

## 5. Ablation Study

### 5-1. Recursive Extraction Module

### 5-2. LPM, RPM

### 5-3. Fine-tuning

---

## 6. Conclusion

---

## Appendix

### References
