import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, classification_report
import warnings

# 1. Load your data
#    Ensure that df has columns: "report_text", "final_symptom", and "name"
df = pd.read_csv("your_final_data.csv")

# For demonstration, let's rename the columns if needed:
# df.rename(columns={'pname': 'name'}, inplace=True)

X = df[['report_text', 'name']]  # We'll use BOTH columns as features
y = df['final_symptom']

# 2. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

# 3. Define the column transformer for the features
#    We'll apply TfidfVectorizer to both the 'report_text' and 'name' columns.

text_transformer = TfidfVectorizer(ngram_range=(1,2), min_df=2)
name_transformer = TfidfVectorizer(ngram_range=(1,2), min_df=1)  # or min_df=2

column_transform = ColumnTransformer(
    [
        ('text_tfidf', text_transformer, 'report_text'),
        ('name_tfidf', name_transformer, 'name')
    ],
    remainder='drop'  # Only transform these two columns
)

# 4. Define multiple models to compare
models = [
    ("Logistic Regression", LogisticRegression(solver='lbfgs', max_iter=1000)),
    ("Random Forest",       RandomForestClassifier(n_estimators=100, random_state=42)),
    ("SVM",                 SVC(kernel='linear', probability=True, random_state=42))
]

# 5. Train & Evaluate each model in a loop
for model_name, model in models:
    print("="*60)
    print(f"Training model: {model_name}")
    
    # Build a pipeline: (ColumnTransformer => Classifier)
    pipeline = Pipeline([
        ('features', column_transform),
        ('clf', model)
    ])
    
    # Train
    pipeline.fit(X_train, y_train)
    
    # Predict
    y_pred = pipeline.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.3f}")

    # classification_report can produce warnings if some classes have zero predictions.
    # We specify zero_division=0 to handle "divide by zero" gracefully.
    report = classification_report(y_test, y_pred, zero_division=0)
    print("Classification Report:")
    print(report)