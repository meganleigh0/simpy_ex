sample_idx = np.random.choice(X_val_dense.shape[0], size=200, replace=False)
X_val_sample = X_val_dense[sample_idx]
explainer = shap.Explainer(classifier.predict_proba, X_val_sample, max_evals=2000)
shap_values = explainer(X_val_sample)