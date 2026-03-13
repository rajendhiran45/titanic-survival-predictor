import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import streamlit as st

df = pd.read_csv("Titanic-Dataset.csv")

df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df.drop("Cabin", axis=1, inplace=True)
df["FamilySize"] = df["SibSp"] + df["Parch"]
df["IsAlone"] = (df["FamilySize"] == 0).astype(int)
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)
df.drop(["Name", "Ticket", "PassengerId"], axis=1, inplace=True)

#splitting Features and target
x=df.drop("Survived",axis=1)
y=df["Survived"]
def pred(model,name,age,pclass,sex,fare,sibsp,parch,Embark):
    sex=1 if sex=="female" else 0
    familysize = sibsp + parch
    IsAlone=1 if familysize==0 else 0

    passenger = {
            "Pclass": pclass,
            "Sex": sex,
            "Age": age,
            "SibSp": sibsp,
            "Parch": parch,
            "Fare": fare,
            "FamilySize": familysize,
            "IsAlone": IsAlone,
            "Embarked_Q": 0,
            "Embarked_S": 1
        }

    passenger_data=pd.DataFrame([passenger])
    prediction=model.predict(passenger_data)[0]
    if prediction == 1:
        st.success("Prediction: Passenger would likely SURVIVE")
    elif prediction==0:
        st.error("Prediction: Passenger would likely NOT SURVIVE")
    else:
        st.write("Anything Here")

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
print(f"Training Samples:{len(x_train)}")
print(f"Testing Samples:{len(x_test)}")

model=RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(x_train,y_train)
print("Model trained successfully!")

y_pred=model.predict(x_test)

print(f"Accuracy:{accuracy_score(y_test,y_pred):.2f}")
print("\nConfusion matrix")
print(confusion_matrix(y_test,y_pred))
print("\nClassification Report")
print(classification_report(y_pred,y_test))

#plotting
importance=pd.Series(model.feature_importances_,index=x.columns).sort_values(ascending=False)
importance.plot(kind="bar")
plt.title("Feature Importance")
plt.ylabel("Importance Score")
plt.tight_layout()
plt.show()
