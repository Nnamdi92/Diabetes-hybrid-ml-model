from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectFromModel

# Separate features and target variable from training set
X_train = df_train.drop(columns=['Outcome'])
y_train = df_train['Outcome']

# Normalize the data using Min-Max scaling
scaler = MinMaxScaler()
X_train_normalized = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)

# Apply SMote to balance the target variable in the training set
smote = SMOTE(random_state=0)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_normalized, y_train)

# Visualize the class distribution after applying SMOTE
plt.figure(figsize=(8, 6))
sns.countplot(x=y_train_balanced)
plt.title('Class Distribution After Applying SMOTE')
plt.xlabel('Outcome')
plt.ylabel('Count')
plt.show()

# Train a Random Forest model
rf_model = RandomForestClassifier(random_state=0)
rf_model.fit(X_train_balanced, y_train_balanced)

# Train an XGBoost model
xgb_model = XGBClassifier(random_state=0, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train_balanced, y_train_balanced)

# Get feature importances from both models
rf_importances = rf_model.feature_importances_
xgb_importances = xgb_model.feature_importances_

# Combine feature importances into a DataFrame
feature_importances = pd.DataFrame({
    'Feature': X_train.columns,
    'RandomForest': rf_importances,
    'XGBoost': xgb_importances
})

# Calculate the mean importance from both models
feature_importances['MeanImportance'] = feature_importances[['RandomForest', 'XGBoost']].mean(axis=1)

# Rank features based on mean importance
feature_importances = feature_importances.sort_values(by='MeanImportance', ascending=False).reset_index(drop=True)

# Display the ranking and selected features
print(feature_importances)

# Apply SelectFromModel to select important features based on Mean Importance
threshold = feature_importances['MeanImportance'].median()
selector = SelectFromModel(estimator=rf_model, threshold=threshold, prefit=True)
X_train_selected = selector.transform(X_train_balanced)

# Get the indices of the selected features
selected_feature_indices = selector.get_support(indices=True)

# Map the indices to feature names
feature_names = X_train.columns
selected_feature_names = feature_names[selected_feature_indices]

# Display the selected feature names
print("Selected Features:", selected_feature_names)

# Visualize feature importances
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importances, x='MeanImportance', y='Feature')
plt.title('Feature Importances Combined from Random Forest and XGBoost')
plt.show()

from sklearn.model_selection import train_test_split

# Split the selected features into training and validation sets
X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(X_train_selected, y_train_balanced, test_size=0.2, random_state=0)

print("Training set shape:", X_train_final.shape)
print("Validation set shape:", X_val_final.shape)