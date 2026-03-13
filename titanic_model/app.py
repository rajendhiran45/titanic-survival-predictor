import streamlit as st
import titanic
import pandas as pd
import os
from sklearn.metrics import accuracy_score

@st.cache_resource

def load_and_train():
    base_path = os.path.dirname(__file__)
    data_path = os.path.join(base_path, "Titanic-Dataset.csv")
    df=titanic.load_data(data_path)
    df=titanic.clean_data(df)
    model,x_test,y_test=titanic.train_model(df)
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    st.info(f"Model Accuracy: {acc*100:.1f}%")
    return model,x_test,y_test

st.divider()

st.title("Titanic Survival Predictor")
st.subheader("Machine Learning Model to Predict Passenger Survival")
st.write("An interactive machine learning application that predicts whether a Titanic passenger would survive based on factors such as class, age, gender, and fare. Built using Python, Pandas, Scikit-learn, and Streamlit.")


st.sidebar.title("Passenger Details")
name=st.sidebar.text_input("Enter Passenger Name")
age=st.sidebar.number_input("Enter Passenger Age")
pclass=st.sidebar.selectbox("Passenger Class", [1, 2, 3])
sex=st.sidebar.radio("Sex", ["Male", "Female"])
fare=st.sidebar.number_input("Fare Paid", min_value=0.0, value=30.0)
sibsp=st.sidebar.number_input("Sibiling/Spouse")
parch=st.sidebar.number_input("Parents/Children")
Embark=st.sidebar.selectbox("Embarked Port",["C","Q", "S"])

model, x_test, y_test = load_and_train()

if st.sidebar.button("Predict Survival"):
    st.subheader(f"Passenger: {name}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Age", age)
    col2.metric("Class", pclass)
    col3.metric("Fare", f"${fare}")
    titanic.pred(model,name,age,pclass,sex,fare,sibsp,parch,Embark)
    titanic.feature_imp(model,x_test)
        