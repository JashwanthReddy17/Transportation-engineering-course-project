from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from scipy.stats import norm
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv(
    '/kaggle/input/traffic-accident-prediction/dataset_traffic_accident_prediction1.csv')
df
df.describe()
df.info()
df.isnull().sum()
columns_with_missing = ['Traffic_Density', 'Speed_Limit', 'Number_of_Vehicles', 'Driver_Alcohol',
                        'Driver_Age', 'Driver_Experience',
                        'Accident']

for i in columns_with_missing:
    print(f"{i} - Среднее: {df[i].mean()}, Медиана: {df[i].median()}")
    df[i].plot(kind='hist', bins=30, title=i)
    plt.axvline(df[i].mean(), color='red', linestyle='dashed', linewidth=2)
    plt.axvline(df[i].median(), color='blue', linestyle='dashed', linewidth=2)
    plt.show()

df['Traffic_Density'].fillna(df['Traffic_Density'].median(), inplace=True)
df['Speed_Limit'].fillna(df['Speed_Limit'].median(), inplace=True)
df['Number_of_Vehicles'].fillna(
    df['Number_of_Vehicles'].median(), inplace=True)
df['Driver_Alcohol'].fillna(df['Driver_Alcohol'].median(), inplace=True)
df['Driver_Age'].fillna(df['Driver_Age'].median(), inplace=True)
df['Driver_Experience'].fillna(df['Driver_Experience'].median(), inplace=True)
df['Accident'].fillna(df['Accident'].median(), inplace=True)

columns_miss = df.columns[df.isnull().sum() > 0]

df[columns_miss].info()

df['Weather'] = df['Weather'].fillna(df['Weather'].mode()[0])
df['Road_Type'] = df['Road_Type'].fillna(df['Road_Type'].mode()[0])
df['Time_of_Day'] = df['Time_of_Day'].fillna(df['Time_of_Day'].mode()[0])
df['Accident_Severity'] = df['Accident_Severity'].fillna(
    df['Accident_Severity'].mode()[0])
df['Road_Condition'] = df['Road_Condition'].fillna(
    df['Road_Condition'].mode()[0])
df['Vehicle_Type'] = df['Vehicle_Type'].fillna(df['Vehicle_Type'].mode()[0])
df['Road_Light_Condition'] = df['Road_Light_Condition'].fillna(
    df['Road_Light_Condition'].mode()[0])

df.isnull().sum()
duplicates_count = df.duplicated().sum()
print(f"Number of duplicates: {duplicates_count}")
df = df.drop_duplicates()
duplicates_count1 = df.duplicated().sum()
print(f"Number of duplicates: {duplicates_count1}")

fig, axes = plt.subplots(3, 2, figsize=(15, 12))

sns.boxplot(x=df['Speed_Limit'], ax=axes[0, 0])
axes[0, 0].set_title('Speed_Limit')

sns.boxplot(x=df['Number_of_Vehicles'], ax=axes[0, 1])
axes[0, 1].set_title('Number_of_Vehicles')

sns.boxplot(x=df['Driver_Age'], ax=axes[1, 0])
axes[1, 0].set_title('Driver_Age')

sns.boxplot(x=df['Driver_Experience'], ax=axes[1, 1])
axes[1, 1].set_title('Driver_Experience')

sns.boxplot(x=df['Traffic_Density'], ax=axes[2, 0])
axes[2, 0].set_title('Traffic_Density')

fig.delaxes(axes[2, 1])

plt.tight_layout()
plt.show()

cols = ['Speed_Limit', 'Number_of_Vehicles', 'Driver_Age', 'Driver_Experience']
sns_plot = sns.pairplot(df[cols])
sns_plot.savefig('pairplot.png')

numeric_df = df.select_dtypes(include=['number'])
spearman_corr = numeric_df.corr(method='spearman')

# Выводим матрицу корреляции
print(spearman_corr)

# Построение тепловой карты для визуализации матрицы корреляции
plt.figure(figsize=(8, 7))
sns.heatmap(spearman_corr, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title('Spearman Correlation Matrix')
plt.show()

df = pd.get_dummies(df, columns=['Weather', 'Road_Type', 'Time_of_Day', 'Accident_Severity',
                                 'Road_Condition', 'Vehicle_Type', 'Road_Light_Condition'], drop_first=True)
df['Age_vs_Experience'] = df['Driver_Age'] - df['Driver_Experience']
df = df.drop(['Driver_Age', 'Driver_Experience'], axis=1)


X = df.drop(['Accident'], axis=1)
y = df['Accident']
feature_names = X.columns

numeric_columns = ['Speed_Limit', 'Number_of_Vehicles', 'Age_vs_Experience']

scaler = RobustScaler()

X_scaled = X.copy()
X_scaled[numeric_columns] = scaler.fit_transform(X[numeric_columns])

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.33, random_state=100)


svm_classifier = SVC()

param_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto'],
    'degree': [3, 5],
    'shrinking': [True, False],
    'class_weight': [None, 'balanced']
}

grid_search_svm = GridSearchCV(svm_classifier, param_svm, cv=5)

grid_search_svm.fit(X_train, y_train)

print("Best Hyperparameters:", grid_search_svm.best_params_)


best_svm_model = grid_search_svm.best_estimator_

y_pred = best_svm_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', cbar=False, xticklabels=[
            'No Accident', 'Accident'], yticklabels=['No Accident', 'Accident'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
