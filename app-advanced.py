import streamlit as st
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
st.title("Credit Card Fraud Detection")

st.sidebar.header("Transaction Information")

# Get user input for transaction details
amount = st.sidebar.number_input("Transaction Amount", min_value=0.0, value=200.0)
time = st.sidebar.number_input("Transaction Time", min_value=0, value=1500)

# Create input fields for V1 to V28
v_inputs = []
for i in range(1, 29):
    v_inputs.append(st.sidebar.number_input(f"V{i}", value=0.0))

# Create a button to trigger prediction
if st.sidebar.button("Predict Fraud"):
    # Prepare the input data
    input_data = [time] + v_inputs + [amount]
    input_df = pd.DataFrame([input_data], columns=X.columns)
    
    # Normalize the Amount
    input_df['Amount'] = scaler.transform(input_df['Amount'].values.reshape(-1, 1))
    
    # Make prediction
    prediction = predict_fraud(input_df)
    
    # Display the result
    if prediction == 0:
        st.success("Transaction is likely genuine.")
    else:
        st.error("Transaction is likely fraudulent!")

# Display model performance metrics
st.header("Model Performance")
y_pred = rf_classifier.predict(X_test)
precision, recall, f1 = evaluate_model(y_test, y_pred)

col1, col2, col3 = st.columns(3)
col1.metric("Precision", f"{precision:.2f}")
col2.metric("Recall", f"{recall:.2f}")
col3.metric("F1-Score", f"{f1:.2f}")

# Visualize feature importance
st.header("Feature Importance")
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_classifier.feature_importances_
}).sort_values('importance', ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance.head(10), ax=ax)
ax.set_title("Top 10 Important Features")
st.pyplot(fig)

# Visualize class distribution
st.header("Class Distribution")
fig, ax = plt.subplots(figsize=(8, 6))
sns.countplot(x='Class', data=df, ax=ax)
ax.set_title("Distribution of Fraudulent vs Genuine Transactions")
st.pyplot(fig)