# Semiconductor Wafer Fault Diagnosis: Optimizing Recall for Industrial Quality Assurance

## 📌 Project Overview
Semiconductor manufacturing is a highly complex environment generating hundreds of sensor data points. The goal of this project is to analyze sensor data to **detect wafer faults in advance**, thereby minimizing yield loss and reducing manufacturing costs.

In semiconductor fabrication, the cost of a **'False Negative' (misclassifying a fault as normal)** is significantly higher than that of a 'False Positive'. Therefore, this project prioritized the **maximization of Recall (Sensitivity)** over simple Accuracy to ensure high-reliability quality control.

## 🛠 Technical Challenges & Solutions
### 1. Extreme Class Imbalance
- **Challenge:** The severe imbalance between normal and faulty wafers led to the 'Accuracy Paradox,' where a model could achieve high accuracy simply by predicting all samples as 'normal.'
- **Solution:** Implemented **SMOTE (Synthetic Minority Over-sampling Technique)** to generate synthetic samples for the minority (faulty) class, enabling the model to effectively learn fault patterns.

### 2. High-Dimensional Sensor Data
- **Challenge:** The presence of numerous noise variables with low correlation to faults increased the risk of overfitting.
- **Solution:** Conducted statistical analysis to select the **top 20 most relevant features**, enhancing model efficiency and generalization performance.

## 📈 Experimental Results

Performance was validated using 5-Fold Stratified Cross-Validation across various machine learning models.

| Model | Accuracy | Precision | **Recall (Sensitivity)** | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.7288 | 0.1452 | **0.6362** $\uparrow$ | 0.2363 |
| Support Vector Machine | 0.7530 | 0.1356 | 0.4900 | 0.2116 |
| Gradient Boosting | 0.8251 | 0.1429 | 0.3371 | 0.2002 |
| Decision Tree | 0.8194 | 0.1195 | 0.2605 | 0.1636 |
| Random Forest | **0.8979** $\uparrow$ | 0.1494 | 0.1524 | 0.1502 |
| MLP | 0.8915 | 0.1765 | 0.1914 | 0.1798 |

### 💡 Engineering Insight: Why Logistic Regression?
- While **Random Forest** achieved the highest overall accuracy (89.79%), its **Recall was only 15.24%**, meaning it would miss over 84% of actual defects—a critical failure in a production environment.
- **Logistic Regression**, despite lower overall accuracy, achieved a **Recall of 63.62%**, demonstrating a far superior ability to detect potential faults.
- **Conclusion:** From a manufacturing QA perspective, a **"Conservative Detection Strategy (High Recall)"** is mandatory. Consequently, Logistic Regression was selected as the optimal model for this industrial application.

## 🖼 Visual Analysis
- **Confusion Matrix:** Visualization of the trade-off between False Positives and False Negatives for the Logistic Regression model.
- **Feature Importance Plot:** Identification of the top 10 sensor variables most critical for fault detection.
- **Recall-Precision Trade-off Curve:** Analysis of performance shifts based on decision threshold adjustments.

## 💻 Tech Stack
- **Language:** Python
- **Libraries:** scikit-learn, pandas, numpy, imbalanced-learn (SMOTE), matplotlib, seaborn
