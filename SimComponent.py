import shap

# Get TF-IDF vectorizer and transformed features
tfidf = model.named_steps['preprocessor'].named_transformers_['text_filtered'].named_steps['tfidf']
X_val_tfidf = tfidf.transform(split_val_df['filtered'])

# Initialize SHAP explainer for LightGBM multiclass
explainer = shap.TreeExplainer(model.named_steps['classifier'])

# Compute SHAP values for validation set
shap_values = explainer.shap_values(X_val_tfidf)

# Plot SHAP summary for each class
import matplotlib.pyplot as plt
for i, class_name in enumerate(model.named_steps['classifier'].classes_):
    print(f"Top words influencing prediction for: {class_name}")
    shap.summary_plot(shap_values[i], X_val_tfidf, feature_names=tfidf.get_feature_names_out(), show=True)