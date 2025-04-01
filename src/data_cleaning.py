import warnings
warnings.filterwarnings('ignore')
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("data/diabetes.csv")

# Preview data
print(df.head())

# Check for missing values
print("Missing values:\n", df.isnull().sum())

# Check for duplicates
print("Duplicate rows:", df.duplicated().sum())

# data types and summary stats
df.describe(include="all")

# Plot distributions...
df.hist(figsize = (15,20))

# Bar plot for Outcome
sns.countplot(x='Outcome', data=df)
plt.title('Outcome Distribution')
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.show()

# Create a heatmap of the correlations with the target column
corr = df.corr()
target_corr = corr['Outcome'].drop('Outcome')

# Sort correlation values in descending order
target_corr_sorted = target_corr.sort_values(ascending=False)

sns.set(font_scale=0.8)
sns.set_style("white")
sns.set_palette("PuBuGn_d")
sns.heatmap(target_corr_sorted.to_frame(), cmap="coolwarm", annot=True, fmt='.2f')
plt.title('Correlation with Outcome')
plt.show()

#train/test split so the test data remains "unseen"

df_copy = df.copy()
df_train = df_copy.sample(frac=0.80, random_state=0)
df_test = df_copy.drop(df_train.index)

print(df_train.head())

# Identify columns with zero values and count the number of zero values in each column in the training set
zero_counts = (df_train == 0).sum()

# Filter out columns that contain zero values
columns_with_zeros = zero_counts[zero_counts > 0]

print(columns_with_zeros)

# Replace zero values in specific columns with median
columns_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for column in columns_to_replace:
    median_value = df_train[df_train[column] != 0][column].median()
    df_train[column] = df_train[column].replace(0, median_value)

# Confirm replacement
zero_counts_after_replacement = (df_train[columns_to_replace] == 0).sum()
print("Zero values after replacement:\n", zero_counts_after_replacement)

# Tukey's method for outlier detection
def tukeys_fence(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

columns_to_check = df_train.columns.drop('Outcome')
outliers_dict = {col: tukeys_fence(df_train, col)[0] for col in columns_to_check}
outliers_counts = {col: len(outliers) for col, outliers in outliers_dict.items()}

# Display outlier counts
print("Outliers detected:")
for column, count in outliers_counts.items():
    print(f"{column}: {count} outliers")

# Replace outliers with 5th and 95th percentiles
def replace_outliers_with_quantiles(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    lower_quantile = df[column].quantile(0.05)
    upper_quantile = df[column].quantile(0.95)
    df.loc[df[column] < lower_bound, column] = lower_quantile
    df.loc[df[column] > upper_bound, column] = upper_quantile

for column in columns_to_check:
    replace_outliers_with_quantiles(df_train, column)

# Visualize updated boxplots
plt.figure(figsize=(15, 10))
for i, column in enumerate(columns_to_check, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(data=df_train, y=column)
    plt.title(f'Boxplot of {column} after Quantile Replacement')
plt.tight_layout()
plt.show()

# Preview cleaned training set
print(df_train.head())


