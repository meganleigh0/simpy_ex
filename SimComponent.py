import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import LabelBinarizer

import matplotlib.pyplot as plt
import seaborn as sns

##############################################################################
# 1. Load Data
##############################################################################
df = pd.read_csv("your_final_data.csv")
# We assume the DataFrame has columns: ["report_text", "final_symptom", "name"]

X = df[["report_text", "name"]]
y = df["final_symptom"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

##############################################################################
# 2. Define ColumnTransformer for text columns
##############################################################################
text_transformer = TfidfVectorizer(ngram_range=(1,2), min_df=2)
name_transformer = TfidfVectorizer(ngram_range=(1,2), min_df=1)

column_transform = ColumnTransformer(
    [
        ("text_tfidf", text_transformer, "report_text"),
        ("name_tfidf", name_transformer, "name")
    ],
    remainder="drop"
)

##############################################################################
# 3. Define a Pipeline with RandomForest
##############################################################################
pipeline = Pipeline([
    ("features", column_transform),
    ("clf", RandomForestClassifier(random_state=42))
])

##############################################################################
# 4. Hyperparameter Tuning with RandomizedSearchCV
##############################################################################
# Define the parameter distributions we want to sample from:
param_distributions = {
    "clf__n_estimators": [100, 200, 300, 500],
    "clf__max_depth": [None, 10, 20, 30, 50],
    "clf__min_samples_split": [2, 5, 10],
    "clf__min_samples_leaf": [1, 2, 4],
    "clf__bootstrap": [True, False]
}

# RandomizedSearchCV allows specifying n_iter to control how many random combos we try
search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_distributions,
    n_iter=20,               # Try 20 different parameter combinations
    cv=5,                    # 5-fold cross-validation
    scoring="accuracy",      # You can also use "f1_macro" if you prefer
    n_jobs=-1,               # Use all available cores
    random_state=42,
    verbose=1
)

search.fit(X_train, y_train)

print(f"Best params found: {search.best_params_}")
print(f"Best CV accuracy: {search.best_score_:.3f}")

# Retrieve the best pipeline
best_pipeline = search.best_estimator_

##############################################################################
# 5. Evaluate on Test Set
##############################################################################
y_pred = best_pipeline.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {test_accuracy:.3f}")

print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

##############################################################################
# 6. Multi-class ROC–AUC Curve
##############################################################################
# For multi-class, we compute one-vs-rest curves.
#  - First, get the probability scores
#  - Binarize y_test
#  - For each class, compute fpr/tpr, then AUC
##############################################################################

# 6a. Predict probabilities on test
y_score = best_pipeline.predict_proba(X_test)

# 6b. Binarize the labels for one-vs-rest ROC
lb = LabelBinarizer()
lb.fit(y_train)  # Learn classes from train set
y_test_bin = lb.transform(y_test)  # shape: (num_samples, num_classes)

n_classes = y_test_bin.shape[1]
class_names = lb.classes_

# 6c. Plot the ROC curve for each class
plt.figure(figsize=(8, 6))

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], lw=1.5, 
             label=f"Class {class_names[i]} (AUC = {roc_auc[i]:.2f})")

# 6d. Macro-average AUC
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

mean_tpr /= n_classes
macro_auc = auc(all_fpr, mean_tpr)
plt.plot(all_fpr, mean_tpr, color="navy", lw=2, linestyle="--",
         label=f"Macro-average (AUC = {macro_auc:.2f})")

# Chance line
plt.plot([0, 1], [0, 1], "r--", label="Chance", lw=1)

plt.title("Multi-class ROC Curve (One-vs-Rest)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.show()

# (Optional) If you only want an aggregated AUC metric:
macro_roc_score = roc_auc_score(y_test_bin, y_score, average="macro", multi_class="ovr")
print(f"Macro-average ROC-AUC: {macro_roc_score:.3f}")