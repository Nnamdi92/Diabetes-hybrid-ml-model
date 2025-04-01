# Process df_test
df_test_cleaned = df_test.copy()

# Replace zeros in specific columns with median values in df_test_cleaned
for column in columns_to_replace:
    median_value = df_test_cleaned[column].median()
    df_test_cleaned[column] = df_test_cleaned[column].replace(0, median_value)

# Normalize the data in df_test_cleaned
X_test = df_test_cleaned.drop(columns=['Outcome'])
y_test = df_test_cleaned['Outcome']
X_test_normalized = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# Select only the features used in training
X_test_selected = X_test_normalized[selected_feature_names]

# Predict on the test set using the best models
dt_preds_test = best_dt.predict(X_test_selected)
rf_preds_test = best_rf.predict(X_test_selected)
svm_preds_test = best_svm.predict(X_test_selected)

# Evaluate Decision Tree
dt_accuracy_test = accuracy_score(y_test, dt_preds_test)
dt_precision_test = precision_score(y_test, dt_preds_test)
dt_recall_test = recall_score(y_test, dt_preds_test)
dt_f1_test = f1_score(y_test, dt_preds_test)
print("Decision Tree - Test Accuracy:", dt_accuracy_test)
print("Decision Tree - Test Precision:", dt_precision_test)
print("Decision Tree - Test Recall:", dt_recall_test)
print("Decision Tree - Test F1 Score:", dt_f1_test)

# Evaluate Random Forest
rf_accuracy_test = accuracy_score(y_test, rf_preds_test)
rf_precision_test = precision_score(y_test, rf_preds_test)
rf_recall_test = recall_score(y_test, rf_preds_test)
rf_f1_test = f1_score(y_test, rf_preds_test)
print("Random Forest - Test Accuracy:", rf_accuracy_test)
print("Random Forest - Test Precision:", rf_precision_test)
print("Random Forest - Test Recall:", rf_recall_test)
print("Random Forest - Test F1 Score:", rf_f1_test)

# Evaluate SVM
svm_accuracy_test = accuracy_score(y_test, svm_preds_test)
svm_precision_test = precision_score(y_test, svm_preds_test)
svm_recall_test = recall_score(y_test, svm_preds_test)
svm_f1_test = f1_score(y_test, svm_preds_test)
print("SVM - Test Accuracy:", svm_accuracy_test)
print("SVM - Test Precision:", svm_precision_test)
print("SVM - Test Recall:", svm_recall_test)
print("SVM - Test F1 Score:", svm_f1_test)

# Combine the predictions of the base models into a single feature matrix
X_new_meta = np.column_stack((dt_preds_test, rf_preds_test, svm_preds_test))

# Make a prediction using the meta-model
y_new_pred = meta_model.predict(X_new_meta)
# Evaluate the meta-model on the test set
meta_accuracy_test = accuracy_score(y_test, y_new_pred)
meta_precision_test = precision_score(y_test, y_new_pred)
meta_recall_test = recall_score(y_test, y_new_pred)
meta_f1_test = f1_score(y_test, y_new_pred)

print("Meta-Model Test Accuracy:", meta_accuracy_test)
print("Meta-Model Test Precision:", meta_precision_test)
print("Meta-Model Test Recall:", meta_recall_test)
print("Meta-Model Test F1 Score:", meta_f1_test)

# Print the classification report for detailed metrics on the test set
print("\nClassification Report on Test Set:")

from sklearn.metrics import confusion_matrix, roc_curve, auc
# Confusion Matrix
def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

dt_cm = confusion_matrix(y_test, dt_preds_test)
rf_cm = confusion_matrix(y_test, rf_preds_test)
svm_cm = confusion_matrix(y_test, svm_preds_test)
meta_cm = confusion_matrix(y_test, y_new_pred)

plot_confusion_matrix(dt_cm, "Decision Tree Confusion Matrix")
plot_confusion_matrix(rf_cm, "Random Forest Confusion Matrix")
plot_confusion_matrix(svm_cm, "SVM Confusion Matrix")
plot_confusion_matrix(meta_cm, "Hybrid Model Confusion Matrix")

# ROC Curve
def plot_roc_curve(y_true, y_pred, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.2f})')

plt.figure(figsize=(10, 8))
plot_roc_curve(y_test, dt_preds_test, 'Decision Tree')
plot_roc_curve(y_test, rf_preds_test, 'Random Forest')
plot_roc_curve(y_test, svm_preds_test, 'SVM')
plot_roc_curve(y_test, y_new_pred, 'Hybrid Model')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

print("\nDecision Tree Classification Report:\n", classification_report(y_test, dt_preds_test))
print("\nRandom Forest Classification Report:\n", classification_report(y_test, rf_preds_test))
print("\nSVM Classification Report:\n", classification_report(y_test, svm_preds_test))