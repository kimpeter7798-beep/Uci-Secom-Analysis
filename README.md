# Semiconductor Wafer Fault Diagnosis

## Problem
In semiconductor manufacturing, detecting wafer defects accurately is critical for quality control. This project aims to classify wafers as either faulty or normal based on sensor data, addressing the challenge of highly imbalanced datasets.

## Dataset
- **Source:** UCI SECOM dataset.
- **Content:** Sensor readings from semiconductor manufacturing processes, including labels for fault/normal.

## Methodology
- **Preprocessing:** Data cleaning, handling missing values, and feature selection to isolate the top 20 relevant features.
- **Handling Imbalance:** Applied over-sampling, SMOTE, and under-sampling techniques.
- **Modeling:** Compared various classifiers including Decision Trees, Logistic Regression, Random Forest, SVM, Naive Bayes, Gradient Boosting, and MLP.

## Results
- Evaluated performance using confusion matrices and metrics like Recall.
- Identified the impact of different sampling techniques on model performance.

## Tech Stack
- Python
- scikit-learn
- pandas
- matplotlib

## How to Run
1. Ensure the dataset (`uci-secom.csv`) is in the `Data set/` directory.
2. Run the preprocessing script: `python "code set/Semiconductor_Fault_Diagnosis/preprocess.py"`
3. Run model evaluation: `python "code set/Semiconductor_Fault_Diagnosis/model_evaluation.py"`
