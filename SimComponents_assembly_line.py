1. Imports & Basic Setup
python
Copy
Edit
import pandas as pd
import numpy as np

# For splitting data
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

# Vectorizers & Transformers
from sklearn.feature_extraction.text import TfidfVectorizer

# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Evaluation metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Pipeline
from sklearn.pipeline import Pipeline
2. Load and Inspect Your Data
python
Copy
Edit
# Replace 'your_data.csv' with the path to your actual file
df = pd.read_csv('your_data.csv')

# Inspect the first few rows
print(df.head())

# Ensure there are no missing values in the critical columns
df = df.dropna(subset=['block_90', 'failure'])

# Optional: convert 'failure' column to integers 0 and 1 if they are not already
df['failure'] = df['failure'].astype(int)

# Define feature (X) and target (y)
X = df['block_90']
y = df['failure']

print(f"Number of samples: {len(df)}")
print(df['failure'].value_counts())
3. Train/Test Split
We split the dataset into training and test sets. This allows us to evaluate how the model performs on unseen data.

python
Copy
Edit
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2,   # 20% of data will be held out for testing
    random_state=42, # for reproducibility
    stratify=y       # keep class distribution similar in train/test
)

print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
4. Create a Preprocessing + Classifier Pipeline
We will create a pipeline that:

Vectorizes the text using TF-IDF.
Feeds the vectorized features into the classifier.
4.1. Logistic Regression Pipeline
python
Copy
Edit
pipeline_lr = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(solver='liblinear'))  # solver='liblinear' works well for smaller datasets
])
4.2. Random Forest Pipeline
python
Copy
Edit
pipeline_rf = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
])
4.3. SVC Pipeline
python
Copy
Edit
pipeline_svc = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', SVC(kernel='linear'))
])
5. Hyperparameter Tuning (Optional)
You can perform a grid search or randomized search for hyperparameters. Here’s an example of GridSearchCV for the Logistic Regression pipeline:

python
Copy
Edit
param_grid_lr = {
    'tfidf__max_df': [0.8, 1.0],
    'tfidf__ngram_range': [(1,1), (1,2)],
    'clf__C': [0.1, 1, 10]
}

grid_search_lr = GridSearchCV(
    pipeline_lr, 
    param_grid_lr, 
    scoring='accuracy', 
    cv=5,        # 5-fold cross-validation
    n_jobs=-1    # use all available CPU cores
)

grid_search_lr.fit(X_train, y_train)
print("Best LogisticRegression parameters:", grid_search_lr.best_params_)
print("Best CV score (LogisticRegression):", grid_search_lr.best_score_)

# Evaluate on test set with best found parameters
best_lr_model = grid_search_lr.best_estimator_
lr_test_preds = best_lr_model.predict(X_test)

print("Test Accuracy (LogisticRegression):", accuracy_score(y_test, lr_test_preds))
print("Classification Report (LogisticRegression):")
print(classification_report(y_test, lr_test_preds))
You can replicate a similar approach for the RandomForest and SVC pipelines with their respective hyperparameters.

6. Training and Evaluating Multiple Models
6.1. Baseline Training (No Grid Search)
If you want a quick comparison without extensive tuning:

python
Copy
Edit
models = {
    "LogisticRegression": pipeline_lr,
    "RandomForest": pipeline_rf,
    "SVC": pipeline_svc
}

for model_name, model_pipeline in models.items():
    # Train (fit) the model
    model_pipeline.fit(X_train, y_train)
    
    # Predict on test data
    y_pred = model_pipeline.predict(X_test)
    
    # Evaluate accuracy
    acc = accuracy_score(y_test, y_pred)
    
    print(f"Model: {model_name}")
    print(f"Accuracy: {acc:.4f}")
    
    # Print confusion matrix
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("-"*50)
This approach will give you a quick look at how each model performs without hyperparameter optimization.

7. Cross-Validation for More Reliable Comparison
If you want a more robust estimate of how each model performs, you can use cross-validation on the training set for each pipeline:

python
Copy
Edit
for model_name, model_pipeline in models.items():
    cv_scores = cross_val_score(model_pipeline, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
    print(f"Model: {model_name}, CV Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
8. Interpreting Results & Next Steps
Accuracy gives an overall idea of correctness but might be misleading if the classes are imbalanced.
Confusion Matrix helps see how many failures (1) are correctly detected vs. missed.
Classification Report provides precision, recall, and F1-scores for each class, which may be more informative than accuracy if your dataset has significant class imbalance.
Cross-Validation results help gauge stability. If results vary widely across folds, you may need more data or better text preprocessing.
Possible improvements:

Clean text more thoroughly (e.g., remove special characters, handle spelling mistakes, or apply lemmatization/stemming).
Use advanced NLP approaches like pre-trained embeddings (Word2Vec, GloVe) or transformer-based models (BERT, RoBERTa) via libraries like Hugging Face Transformers.
Tune hyperparameters thoroughly with GridSearchCV or RandomizedSearchCV.
Complete Example Combined
Below is a more concise, combined script. You can adapt it as needed.

python
Copy
Edit
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline

# 1. Load data
df = pd.read_csv('your_data.csv')

# 2. Basic preprocessing
df.dropna(subset=['block_90', 'failure'], inplace=True)
df['failure'] = df['failure'].astype(int)

X = df['block_90']
y = df['failure']

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

# 4. Define pipelines
pipeline_lr = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(solver='liblinear'))
])

pipeline_rf = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', RandomForestClassifier(random_state=42))
])

pipeline_svc = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', SVC(kernel='linear'))
])

# 5. Simple training without hyperparameter search
models = {
    "LogisticRegression": pipeline_lr,
    "RandomForest": pipeline_rf,
    "SVC": pipeline_svc
}

for model_name, model_pipeline in models.items():
    # Train
    model_pipeline.fit(X_train, y_train)
    
    # Predict
    y_pred = model_pipeline.predict(X_test)
    
    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    print(f"Model: {model_name}")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("-"*50)

# 6. (Optional) Perform grid search for LogisticRegression
param_grid_lr = {
    'tfidf__max_df': [0.8, 1.0],
    'tfidf__ngram_range': [(1,1), (1,2)],
    'clf__C': [0.1, 1, 10]
}

grid_search_lr = GridSearchCV(pipeline_lr, param_grid_lr, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_lr.fit(X_train, y_train)

print("Best LogisticRegression parameters:", grid_search_lr.best_params_)
print("Best CV score (LogisticRegression):", grid_search_lr.best_score_)

best_lr_model = grid_search_lr.best_estimator_
lr_test_preds = best_lr_model.predict(X_test)
print("Test Accuracy (LogisticRegression, tuned):", accuracy_score(y_test, lr_test_preds))
print("Classification Report (LogisticRegression, tuned):")
print(classification_report(y_test, lr_test_preds))
9. Conclusion
With this approach:

You can quickly experiment with multiple classifiers.
You can perform grid searches to find the best hyperparameters.
You get detailed evaluations (accuracy, confusion matrix, classification report).
This should help you understand how well each model identifies failure (1) vs. no failure (0) based on the text data in block_90. From here, you can refine the pipeline with more elaborate text preprocessing or try advanced NLP models to further boost performance.
