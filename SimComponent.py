import shap

# TF-IDF and validation data
tfidf = model.named_steps['preprocessor'].named_transformers_['text_filtered'].named_steps['tfidf']
X_val_tfidf = tfidf.transform(split_val_df['filtered'])

# Get classifier
classifier = model.named_steps['classifier']

# Use SHAP's general Explainer interface
explainer = shap.Explainer(classifier.predict_proba, X_val_tfidf)

# Compute SHAP values (returns list: one array per class)
shap_values = explainer(X_val_tfidf)

# Plot SHAP summary for each class
import matplotlib.pyplot as plt
feature_names = tfidf.get_feature_names_out()

for i, class_name in enumerate(classifier.classes_):
    print(f"Top words influencing prediction for: {class_name}")
    shap.summary_plot(shap_values.values[:, :, i], X_val_tfidf, feature_names=feature_names)