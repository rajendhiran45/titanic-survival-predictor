import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import streamlit as st


def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def clean_data(df):
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df.drop("Cabin", axis=1, inplace=True)
    df["FamilySize"] = df["SibSp"] + df["Parch"]
    df["IsAlone"] = (df["FamilySize"] == 0).astype(int)
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)
    df.drop(["Name", "Ticket", "PassengerId"], axis=1, inplace=True)
    return df

def train_model(df):
    x=df.drop("Survived",axis=1)
    y=df["Survived"]

    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
    print(f"Training Samples:{len(x_train)}")
    print(f"Testing Samples:{len(x_test)}")

    model=RandomForestClassifier(n_estimators=100,random_state=42)
    model.fit(x_train,y_train)
    print("Model trained successfully!")

    return model,x_test,y_test

def pred(model,name,age,pclass,sex,fare,sibsp,parch,Embark):
    sex=1 if sex=="Female" else 0
    familysize = sibsp + parch
    IsAlone=1 if familysize==0 else 0
    Embarked_Q = 1 if Embark == "Q" else 0
    Embarked_S = 1 if Embark == "S" else 0

    passenger = {
            "Pclass": pclass,
            "Sex": sex,
            "Age": age,
            "SibSp": sibsp,
            "Parch": parch,
            "Fare": fare,
            "FamilySize": familysize,
            "IsAlone": IsAlone,
            "Embarked_Q": Embarked_Q,
            "Embarked_S": Embarked_S
        }

    passenger_data=pd.DataFrame([passenger])
    prediction=model.predict(passenger_data)[0]
    if prediction == 1:
        st.success(f"Prediction: {name} would likely SURVIVE")
    elif prediction==0:
        st.error(f"Prediction: {name} would likely NOT SURVIVE")
    else:
        st.write("Anything Here")
    prob=model.predict_proba(passenger_data)[0][1]
    st.progress(int(prob*100))
    st.write(f"Survival Probability: {prob*100:.1f}%")

def feature_imp(model,x_test):
    importance=pd.Series(model.feature_importances_,
        index=x_test.columns
    ).sort_values(ascending=False)
    st.subheader("Feature Importance")
    st.bar_chart(importance)


