import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv('creditcard.csv')
    return data

# Preprocess the dataset
def preprocess_data(data):
    legit = data[data.Class == 0]
    fraud = data[data.Class == 1]

    # Under-sampling
    legit_sample = legit.sample(n=492)
    new_dataset = pd.concat([legit_sample, fraud], axis=0)
    
    # Splitting the data
    X = new_dataset.drop(columns='Class', axis=1)
    Y = new_dataset['Class']
    return X, Y

# Train the model
def train_model(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    model = LogisticRegression()
    model.fit(X_train, Y_train)
    
    return model, scaler

# Predict using the model
def predict_transaction(model, scaler, user_input):
    input_array = np.array(user_input).reshape(1, -1)
    input_array = scaler.transform(input_array)
    prediction = model.predict(input_array)
    return prediction[0]

# Load and preprocess the data
data = load_data()
X, Y = preprocess_data(data)

# Train the model
model, scaler = train_model(X, Y)

# Streamlit app
st.title('Credit Card Fraud Detection')

# User input form for all features
st.header('Enter Transaction Details')

# Generate input fields for each column except the target column 'Class'
user_input = []
for column in X.columns:
    value = st.number_input(f'{column}', min_value=float(X[column].min()), max_value=float(X[column].max()), step=0.01)
    user_input.append(value)

# Predict the result
if st.button('Predict'):
    result = predict_transaction(model, scaler, user_input)
    if result == 0:
        st.success('The transaction is Legitimate')
    else:
        st.error('The transaction is Fraudulent')

# Option to retrain the model
if st.checkbox('Retrain the model'):
    model, scaler = train_model(X, Y)
    st.write('Model retrained successfully!')
