
# Credit Card Fraud Detection - CODSOFT INTERNSHIP TASK

Build a machine learning model to identify fraudulent credit card transactions. This project involves preprocessing and normalizing transaction data, handling class imbalance issues, and splitting the dataset into training and testing sets. A classification algorithm, such as logistic regression or random forests, is trained to classify transactions as fraudulent or genuine. The model's performance is evaluated using metrics like precision, recall, and F1-score. Techniques like oversampling or undersampling are considered to improve results.


## Description

The project consists of two main files:

1) model.py: This file contains the code to train and run the machine learning model efficiently.

2) app.py: This file sets up a Streamlit web application for an interactive and user-friendly interface to input transaction details and predict fraud.

The dataset used for this project is credit_card.csv, which contains various features related to credit card transactions.

You can run either model.py for a command-line interface or app.py for a graphical user interface. app.py provides a better user experience by using Streamlit to interact with the model.
## Acknowledgements

 We would like to thank the following resources and individuals for their contributions and support:

Streamlit: For offering an easy-to-use framework for deploying machine learning models.

Scikit-learn: For providing powerful machine learning tools and libraries.

Pandas: For data manipulation and preprocessing.

NumPy: For numerical operations.

## Demo

https://drive.google.com/file/d/1gfrQ9g3wlXaqvD2AhaUmBHTDeZXRLcEj/view?usp=drive_link

You can see a live demo of the application by running the app.py file. The Streamlit app allows you to input transaction details and get a predicted fraud status based on the trained model.
## Features

Data Loading and Preprocessing: The model can load and preprocess data from the credit_card.csv file.

Model Training: Utilizes a classification algorithm to train the model on transaction data.

Interactive User Input: Through the Streamlit app, users can input transaction details and receive a predicted fraud status.

Model Evaluation: Evaluates model performance using metrics like precision, recall, and F1-score.

Class Imbalance Handling: Implements techniques like oversampling or undersampling to handle class imbalance.
## Technologies Used

Python: The programming language used to implement the model and the 
Streamlit app.

Pandas: For data manipulation and preprocessing.

NumPy: For numerical operations.

Scikit-learn: For building and training the machine learning model.

Streamlit: For creating the interactive web application.
## Installation

To get started with this project, follow these steps:

1) Clone the repository:

git clone https://github.com/SiddharthWayne/CREDIT-CARD-FRAUD-DETECTION---CODSOFT-INTERNSHIP-TASK.git

cd credit-card-fraud-detection

2) Install the required packages:

pip install -r requirements.txt

Ensure that requirements.txt includes the necessary dependencies like pandas, numpy, scikit-learn, and streamlit.

3) Download the dataset:

Place the credit_card.csv file in the project directory. Make sure the path in model.py and app.py is correctly set to this file.



## Usage/Examples

1) Running the Model (model.py)

To train and run the model using the command line, execute the following:

python model.py

This will train the model and allow you to input transaction details via the command line interface to get a predicted fraud status.

2) Running the Streamlit App (app.py)

To run the Streamlit app for an interactive experience, execute the following:

streamlit run app.py

This will start the Streamlit server, and you can open your web browser to the provided local URL to use the app.

Example

Once the Streamlit app is running, you can input transaction details such as:


Transaction Amount: $200.00

Transaction Time: 1500

Other relevant transaction features

Click the "Predict Fraud" button to get the predicted fraud status.

