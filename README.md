# Analysis of COVID-19 Dataset Using Neural Networks – School Project

**Course:** ZNEUS (FIIT STU)  
**Dataset:** https://www.kaggle.com/datasets/meirnizri/covid19-dataset/data (≈1.05 M patient records from the Mexican federal COVID-19 surveillance program)

COVID-19 Classification — ZNEUS Project (FIIT STU)

This project was created as part of the ZNEUS course at FIIT STU. It focuses on binary classification of COVID-19 cases using a feedforward neural network (MLP) built with PyTorch. Patients labeled by the Mexican authorities are reorganized into a simple HAS-COVID vs. NO-COVID target to showcase supervised-learning workflows.

The notebook includes:

- Data preprocessing and feature engineering (cleaning categorical codes, enforcing domain rules, derived risk indicators)
- Model architecture definition with configurable hidden layers and dropout
- Training and evaluation with key metrics (loss, ROC-AUC, F1, etc.) and class-imbalance handling
- Threshold tuning and performance visualization, including classification reports and confusion matrices

The goal of the project is to demonstrate practical implementation of supervised learning, model optimization, and evaluation of classification performance on a real, noisy healthcare dataset.
