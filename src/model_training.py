from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# SVM grid search
svm_param_grid = {
    'kernel': ['rbf', 'poly'],
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': [1, 0.1, 0.01, 0.001, 0.0001]
}
svm_grid_search = GridSearchCV(SVC(random_state=42), svm_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
svm_grid_search.fit(X_train_final, y_train_final)
best_svm = svm_grid_search.best_estimator_
print("Best SVM:", best_svm)

# Random Forest grid search
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
