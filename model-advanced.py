import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv(r"C:\Users\siddh\OneDrive\Desktop\codsoft intern\CREDIT CARD FRUARD\creditcard.csv")

# Separate features and target variable
X = df.drop('Class', axis=1)
y = df['Class']

# Normalize the 'Amount' feature
scaler = StandardScaler()
X['Amount'] = scaler.fit_transform(X['Amount'].values.reshape(-1, 1))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

def predict_fraud(transaction):
    prediction = rf_classifier.predict(transaction)
    return prediction[0]

def evaluate_model(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return precision, recall, f1

# Evaluate the model
y_pred = rf_classifier.predict(X_test)
precision, recall, f1 = evaluate_model(y_test, y_pred)

print("Model Performance:")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

# Feature Importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_classifier.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Important Features:")
print(feature_importance.head(10))

# Visualize feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
plt.title("Top 10 Important Features")
plt.tight_layout()
plt.show()

# Visualize class distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='Class', data=df)
plt.title("Distribution of Fraudulent vs Genuine Transactions")
plt.show()

# Function to get user input for a transaction
def get_transaction_input():
    print("\nEnter transaction details:")
    amount = float(input("Transaction Amount: "))
    time = int(input("Transaction Time: "))
    v_inputs = []
    for i in range(1, 29):
        v_inputs.append(float(input(f"V{i}: ")))
    return [time] + v_inputs + [amount]

# Main loop for prediction
while True:
    user_input = get_transaction_input()
    input_df = pd.DataFrame([user_input], columns=X.columns)
    input_df['Amount'] = scaler.transform(input_df['Amount'].values.reshape(-1, 1))
    
    prediction = predict_fraud(input_df)
    
    if prediction == 0:
        print("Result: Transaction is likely genuine.")
    else:
        print("Result: Transaction is likely fraudulent!")
    
    continue_pred = input("Do you want to predict another transaction? (y/n): ")
    if continue_pred.lower() != 'y':
        break

print("Thank you for using the Credit Card Fraud Detection system.")
