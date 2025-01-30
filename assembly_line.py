import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Assume you've already done train_test_split and have X_train, X_test, y_train, y_test

# 1. Build the pipeline with your best params
final_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_df=0.8, ngram_range=(1,2))),
    ('clf', LogisticRegression(C=10, solver='liblinear'))
])

# 2. Train (fit) the final model
final_pipeline.fit(X_train, y_train)

# 3. Predict on test set
y_pred = final_pipeline.predict(X_test)

# 4. Print classification report and accuracy
print("Final Model: LogisticRegression (Tuned)")
print("Test Accuracy:", np.round(final_pipeline.score(X_test, y_test), 4))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# 5. Confusion Matrix & Visualization
cm = confusion_matrix(y_test, y_pred)
labels = ['Class 0', 'Class 1']
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix - Final Model")
plt.ylabel("True Class")
plt.xlabel("Predicted Class")
plt.show()

# 6. ROC Curve (for binary classification)
y_probs = final_pipeline.predict_proba(X_test)[:,1]  # Probability of class 1
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, color='darkorange', label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - Final Model')
plt.legend(loc="lower right")
plt.show()