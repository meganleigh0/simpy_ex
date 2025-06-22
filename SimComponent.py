import shap
import matplotlib.pyplot as plt

# Get TF-IDF vectorizer and transform the validation set
tfidf = model.named_steps['preprocessor'].named_transformers_['text_filtered'].named_steps['tfidf']
X_val_tfidf = tfidf.transform(split_val_df['filtered'])
X_val_dense = X_val_tfidf.toarray()

# Get the classifier
classifier = model.named_steps['classifier']

# Initialize SHAP Explainer
explainer = shap.Explainer(classifier.predict_proba, X_val_dense)

# Compute SHAP values
shap_values = explainer(X_val_dense)

# Get feature names
feature_names = tfidf.get_feature_names_out()

# Plot summary for each class
for i, class_name in enumerate(classifier.classes_):
    print(f"Top words influencing prediction for: {class_name}")
    shap.summary_plot(shap_values.values[:, :, i], X_val_dense, feature_names=feature_names)