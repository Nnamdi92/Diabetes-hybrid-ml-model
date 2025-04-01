from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Decision Tree hyperparameters
dt_param_grid = {
    'max_depth': range(1, 31),
    'min_samples_leaf': range(1, 31),
    'criterion': ['gini', 'entropy'],
    'min_samples_split': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
}
dt_grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), dt_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
dt_grid_search.fit(X_train_final, y_train_final)
best_dt = dt_grid_search.best_estimator_
print("Best Decision Tree:", best_dt)

# SVM hyperparameters search
svm_param_grid = {
    'kernel': ['rbf', 'poly'],
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': [1, 0.1, 0.01, 0.001, 0.0001]
}
svm_grid_search = GridSearchCV(SVC(random_state=42), svm_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
svm_grid_search.fit(X_train_final, y_train_final)
best_svm = svm_grid_search.best_estimator_
print("Best SVM:", best_svm)
best_params = svm_grid_search.best_params_
print("Best Parameters:", best_params)

# Random Forest hyperparameters search
rf_param_grid = {
    'n_estimators': [10, 50, 100, 200, 500],
    'max_features': ['sqrt', 'log2', None],
    'min_samples_leaf': range(1, 31),
    'max_depth': range(1, 31),
    'criterion': ['gini', 'entropy']
}
rf_grid_search = GridSearchCV(RandomForestClassifier(random_state=42), rf_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
rf_grid_search.fit(X_train_final, y_train_final)
best_rf = rf_grid_search.best_estimator_

print("Best Random Forest:", best_rf)

best_rfparams = rf_grid_search.best_params_
print("Best RF parameters:", best_rfparams)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
# Predict on the validation set using the best models
dt_preds = best_dt.predict(X_val_final)
rf_preds = best_rf.predict(X_val_final)
svm_preds = best_svm.predict(X_val_final)

# Evaluate Decision Tree
dt_accuracy = accuracy_score(y_val_final, dt_preds)
dt_precision = precision_score(y_val_final, dt_preds)
dt_recall = recall_score(y_val_final, dt_preds)
dt_f1 = f1_score(y_val_final, dt_preds)
print("Decision Tree - Accuracy:", dt_accuracy)
print("Decision Tree - Precision:", dt_precision)
print("Decision Tree - Recall:", dt_recall)
print("Decision Tree - F1 Score:", dt_f1)

# Evaluate Random Forest
rf_accuracy = accuracy_score(y_val_final, rf_preds)
rf_precision = precision_score(y_val_final, rf_preds)
rf_recall = recall_score(y_val_final, rf_preds)
rf_f1 = f1_score(y_val_final, rf_preds)
print("Random Forest - Accuracy:", rf_accuracy)
print("Random Forest - Precision:", rf_precision)
print("Random Forest - Recall:", rf_recall)
print("Random Forest - F1 Score:", rf_f1)

# Evaluate SVM
svm_accuracy = accuracy_score(y_val_final, svm_preds)
svm_precision = precision_score(y_val_final, svm_preds)
svm_recall = recall_score(y_val_final, svm_preds)
svm_f1 = f1_score(y_val_final, svm_preds)
print("SVM - Accuracy:", svm_accuracy)
print("SVM - Precision:", svm_precision)
print("SVM - Recall:", svm_recall)
print("SVM - F1 Score:", svm_f1)

# Combine the predictions of the base models into a single feature matrix
X_val_meta = np.column_stack((dt_preds, rf_preds, svm_preds))

# Train the meta-model on the combined feature matrix and the target values
meta_model = LogisticRegression(random_state=0)
meta_model.fit(X_val_meta, y_val_final)