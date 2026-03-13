import pandas as pd
df=pd.read_csv("Titanic-Dataset.csv")
df["FamilySize"]=df["SibSp"]+df["Parch"]
df["IsAlone"]=(df["FamilySize"]==0).astype(int)
df.drop(["Name","Ticket","PassengerId"],axis=1,inplace=True)
df["Sex"]=df["Sex"].map({"male":0,"female":1})
# df["Embarked"]=df["Embarked"].map({"S":0,"C":1,"Q":2})
df=pd.get_dummies(df,columns=["Embarked"],drop_first=True)
print(df.head())
print(df.dtypes)