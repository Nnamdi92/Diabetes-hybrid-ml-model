# Diabetes Prediction Using Hybrid Machine Learning

This project develops a hybrid ensemble machine learning pipeline using StackingClassifier to predict diabetes from the PIMA Indian dataset. It combines Decision Tree, Random Forest, and Support Vector Machine models, using Logistic Regression as the meta-model.

---

## 🚀 Objectives
- Perform data cleaning and normalization
- Handle class imbalance using SMOTE
- Perform feature selection using the mean importances from Random Forest and XGBoost
- Train and tune individual models (DT, RF, SVM)
- Build a stacking model with LR as the meta-classifier
- Evaluate models using accuracy, precision, recall, F1-score, confusion matrix

---

## 📁 Folder Structure


---

## 📊 Key Results
| Model         | Accuracy | Precision | Recall | F1 Score |
|---------------|----------|-----------|--------|----------|
| Decision Tree | 74.0%    | 0.58      | 0.69   | 0.63     |
| Random Forest | 74.6%    | 0.59      | 0.67   | 0.63     |
| SVM           | 75.3%    | 0.60      | 0.65   | 0.63     |
| Hybrid Model  | 74.6%    | 0.59      | 0.67   | 0.63     |

---

## 📦 Installation
### 1. Clone the repo
```bash
git clone https://github.com/Nnamdi92/Diabetes-hybrid-ml-model.git
cd HYBRID MODEL

pip install -r requirements.txt
