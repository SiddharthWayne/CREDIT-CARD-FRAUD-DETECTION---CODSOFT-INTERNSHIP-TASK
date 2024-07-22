import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv(r"C:\Users\siddh\OneDrive\Desktop\codsoft intern\CREDIT CARD FRUARD\creditcard.csv")

# Display basic information about the dataset
print(df.info())
print(df['Class'].value_counts(normalize=True))

# Separate features and target
X = df.drop('Class', axis=1)
y = df['Class']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Function to train and evaluate model
def train_evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    return precision, recall, f1, y_pred

# Logistic Regression
lr_model = LogisticRegression(random_state=42)
lr_precision, lr_recall, lr_f1, lr_pred = train_evaluate_model(lr_model, X_train_scaled, y_train, X_test_scaled, y_test)

print("\nLogistic Regression Results:")
print(f"Precision: {lr_precision:.4f}")
print(f"Recall: {lr_recall:.4f}")
print(f"F1-score: {lr_f1:.4f}")

# Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_precision, rf_recall, rf_f1, rf_pred = train_evaluate_model(rf_model, X_train_scaled, y_train, X_test_scaled, y_test)

print("\nRandom Forest Results:")
print(f"Precision: {rf_precision:.4f}")
print(f"Recall: {rf_recall:.4f}")
print(f"F1-score: {rf_f1:.4f}")

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

# Retrain models with SMOTE
lr_smote_precision, lr_smote_recall, lr_smote_f1, lr_smote_pred = train_evaluate_model(lr_model, X_train_smote, y_train_smote, X_test_scaled, y_test)

print("\nLogistic Regression with SMOTE Results:")
print(f"Precision: {lr_smote_precision:.4f}")
print(f"Recall: {lr_smote_recall:.4f}")
print(f"F1-score: {lr_smote_f1:.4f}")

rf_smote_precision, rf_smote_recall, rf_smote_f1, rf_smote_pred = train_evaluate_model(rf_model, X_train_smote, y_train_smote, X_test_scaled, y_test)

print("\nRandom Forest with SMOTE Results:")
print(f"Precision: {rf_smote_precision:.4f}")
print(f"Recall: {rf_smote_recall:.4f}")
print(f"F1-score: {rf_smote_f1:.4f}")

# Visualizations
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

# Plot confusion matrices
plot_confusion_matrix(y_test, lr_pred, "Confusion Matrix - Logistic Regression")
plot_confusion_matrix(y_test, rf_pred, "Confusion Matrix - Random Forest")
plot_confusion_matrix(y_test, lr_smote_pred, "Confusion Matrix - Logistic Regression with SMOTE")
plot_confusion_matrix(y_test, rf_smote_pred, "Confusion Matrix - Random Forest with SMOTE")

# Feature importance (for Random Forest)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
plt.title('Top 10 Important Features')
plt.show()

# Function to preprocess and classify new user input
def classify_new_input(input_data, scaler, lr_model, rf_model):
    input_data_scaled = scaler.transform([input_data])
    lr_prediction = lr_model.predict(input_data_scaled)[0]
    rf_prediction = rf_model.predict(input_data_scaled)[0]
    
    print(f"Logistic Regression Prediction: {'Fraud' if lr_prediction == 1 else 'Not Fraud'}")
    print(f"Random Forest Prediction: {'Fraud' if rf_prediction == 1 else 'Not Fraud'}")
    return lr_prediction, rf_prediction

# Function to get user input and classify
def get_user_input_and_classify(scaler, lr_model, rf_model):
    input_data = []
    for feature in X.columns:
        value = float(input(f"Enter value for {feature}: "))
        input_data.append(value)
    
    classify_new_input(input_data, scaler, lr_model, rf_model)

# Example usage
# Uncomment the line below to use the function to get user input and classify
# get_user_input_and_classify(scaler, lr_model, rf_model)
