import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE

# Import models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

import warnings
warnings.filterwarnings('ignore')

# 1. Load data
try:
    df = pd.read_csv('final_data.csv')
except FileNotFoundError:
    print("Error: 'final_data.csv' not found. Please make sure the file is in the correct directory.")
    exit()

# 2. Separate features and target
if 'Unnamed: 0' in df.columns:
    df = df.drop('Unnamed: 0', axis=1)

# Convert target variable 'Pass_Fail' from (1, 2) to (1, 0)
df['Pass_Fail'] = df['Pass_Fail'].replace({1: 1, 2: 0})

X = df.drop('Pass_Fail', axis=1)
y = df['Pass_Fail']

# Define models
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Support Vector Machine": SVC(random_state=42, probability=True),
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Naive Bayes": GaussianNB(),
    "Multilayer Perceptron": MLPClassifier(random_state=42, max_iter=1000)
}

# 3. Set up 5-fold stratified cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 4. Iterate through each model and evaluate
for model_name, model in models.items():
    acc_scores = []
    prec_scores = []
    rec_scores = []
    f1_scores = []

    print(f"================= {model_name} ==================")

    for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
        # Split data for the current fold
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Standard Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Apply SMOTE only to the training data of the current fold
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

        # Train the model
        model.fit(X_train_resampled, y_train_resampled)

        # Make predictions
        y_pred = model.predict(X_test_scaled)

        # Calculate metrics (using zero_division=0 to handle cases with no predicted positives)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='binary', zero_division=0)
        rec = recall_score(y_test, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)

        # Append scores
        acc_scores.append(acc)
        prec_scores.append(prec)
        rec_scores.append(rec)
        f1_scores.append(f1)

        print(f"  Fold {fold+1}: Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1-score={f1:.4f}")

    # Print average and std deviation of scores
    print("\n  [Average Scores]")
    print(f"  Accuracy:  {np.mean(acc_scores):.4f} (+/- {np.std(acc_scores):.4f})")
    print(f"  Precision: {np.mean(prec_scores):.4f} (+/- {np.std(prec_scores):.4f})")
    print(f"  Recall:    {np.mean(rec_scores):.4f} (+/- {np.std(rec_scores):.4f})")
    print(f"  F1-score:  {np.mean(f1_scores):.4f} (+/- {np.std(f1_scores):.4f})\n")