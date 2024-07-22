import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.title("Credit Card Fraud Detection")

# Define the file path
file_path = 'C:/Users/siddh/OneDrive/Desktop/codsoft intern/CREDIT CARD FRUARD/creditcard.csv'

# Load the dataset
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

credit_card_data = load_data(file_path)

st.subheader("Dataset")
st.write(credit_card_data.head())

# Separating the data for analysis
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]

# Under-Sampling: Build a sample dataset containing similar distribution of normal and fraudulent transactions
legit_sample = legit.sample(n=492)
new_dataset = pd.concat([legit_sample, fraud], axis=0)

# Splitting the data into Features & Targets
X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']

# Split the data into training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Model training using Logistic Regression
model = LogisticRegression()
model.fit(X_train, Y_train)

# Model evaluation: Accuracy score
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
st.subheader('Model Accuracy')
st.write('Accuracy on Training data:', training_data_accuracy)

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
st.write('Accuracy on Test data:', test_data_accuracy)
