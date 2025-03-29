from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Validation predictions
dt_preds = best_dt.predict(X_val_final)
rf_preds = best_rf.predict(X_val_final)
svm_preds = best_svm.predict(X_val_final)

# Evaluate each model
def evaluate(model_name, y_true, y_pred):
    print(f"{model_name} - Accuracy:", accuracy_score(y_true, y_pred))
    print(f"{model_name} - Precision:", precision_score(y_true, y_pred))
    print(f"{model_name} - Recall:", recall_score(y_true, y_pred))
    print(f"{model_name} - F1 Score:", f1_score(y_true, y_pred))

evaluate("Decision Tree", y_val_final, dt_preds)
evaluate("Random Forest", y_val_final, rf_preds)
evaluate("SVM", y_val_final, svm_preds)

# Test predictions
dt_preds_test = best_dt.predict(X_test_selected)
rf_preds_test = best_rf.predict(X_test_selected)
svm_preds_test = best_svm.predict(X_test_selected)

# Evaluate on test set
evaluate("Decision Tree - Test", y_test, dt_preds_test)
evaluate("Random Forest - Test", y_test, rf_preds_test)
evaluate("SVM - Test", y_test, svm_preds_test)

# Meta-model evaluation
y_new_pred = meta_model.predict(X_new_meta)
evaluate("Meta-Model Test", y_test, y_new_pred)
print("\nClassification Report on Test Set:")
print(classification_report(y_test, y_new_pred))

# Confusion Matrices
def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

plot_confusion_matrix(confusion_matrix(y_test, dt_preds_test), "Decision Tree")
plot_confusion_matrix(confusion_matrix(y_test, rf_preds_test), "Random Forest")
plot_confusion_matrix(confusion_matrix(y_test, svm_preds_test), "SVM")
plot_confusion_matrix(confusion_matrix(y_test, y_new_pred), "Hybrid Meta-Model")
