import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler 
import streamlit as st
import os


st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")

credit_card_data = None
try:
    credit_card_data = pd.read_csv(r'C:\Users\HP\Desktop\college\vs codes\Credit_Card_Fraud_Detection\creditcard.csv')
except FileNotFoundError:
    st.error("Error: 'creditcard.csv' not found. Please ensure the dataset is in the same directory as this script.")
    st.warning("Application cannot proceed without the dataset. Please place 'creditcard.csv' in the correct location.")
    st.stop()

if credit_card_data is not None:

    legit = credit_card_data[credit_card_data.Class == 0]
    fraud = credit_card_data[credit_card_data.Class == 1]

    legit_sample = legit.sample(n=len(fraud), random_state=42)
    credited_card = pd.concat([legit_sample, fraud], axis=0)


    X = credited_card.drop(columns='Class', axis=1)
    Y = credited_card['Class']


    feature_medians = X.median()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

  
    scaler = StandardScaler()


    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)

    feature_medians_scaled = X_train_scaled_df.median()

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train_scaled, Y_train)


    st.title("Credit Card Fraud Detection")
    st.markdown("### Amount-Based Prediction")

    st.write(
        "Enter a transaction amount below to check if it's likely legitimate or fraudulent. "
        "This model predicts based on the transaction amount, using typical values for other "
        "transaction features."
    )

    amount_input = st.number_input(
        "Enter Transaction Amount:",
        min_value=0.0,
        format="%.2f",
        help="The monetary value of the transaction."
    )

    submit = st.button("Check Transaction")

    if submit:
       
        input_features_scaled = feature_medians_scaled.copy()

        temp_input_df = pd.DataFrame([feature_medians.values], columns=X.columns)
        temp_input_df['Amount'] = amount_input

        scaled_single_input = scaler.transform(temp_input_df)
        prediction = model.predict(scaled_single_input)

        st.subheader("Prediction Result:")
        if prediction[0] == 0:
            st.success("Legitimate Transaction ")
            st.write("This transaction is predicted to be legitimate based on the amount provided.")
        else:
            st.error("Fraudulent Transaction ")
            st.write("This transaction is predicted to be fraudulent based on the amount provided.")






