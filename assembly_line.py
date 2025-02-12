import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, classification_report

# 1. Load your data
df = pd.read_csv("your_final_data.csv")  # has columns: [report_text, final_symptom, ...]

# 2. Split into train/test
X = df['report_text']
y = df['final_symptom']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Define the models you want to compare
models = [
    ("Logistic Regression", LogisticRegression(max_iter=1000, solver='lbfgs')),
    ("Random Forest",       RandomForestClassifier(n_estimators=100, random_state=42)),
    ("SVM",                 SVC(kernel='linear', probability=True, random_state=42))
]

# 4. Loop over each model, train, and evaluate
for model_name, model in models:
    print("="*60)
    print(f"Training model: {model_name}")
    
    # Create pipeline: TF-IDF => model
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), min_df=2)),
        ('clf', model)
    ])
    
    # Train
    pipeline.fit(X_train, y_train)
    
    # Predict
    y_pred = pipeline.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.3f}")
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))