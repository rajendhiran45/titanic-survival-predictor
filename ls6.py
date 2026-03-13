import pandas as pd
df=pd.read_csv("Titanic-Dataset.csv")
surv_class=df.groupby("Pclass")["Survived"].mean()
avg_fare=df.groupby("Pclass")["Fare"].mean()
df["FamilySize"]=df["SibSp"]+df["Parch"]
df["IsAlone"]=(df["FamilySize"]==0).astype(int)
alone_trv=df.groupby("IsAlone")["Survived"].mean()
pivot=pd.pivot_table(df,values="Survived",index="Pclass",columns="Sex")
ac=df["IsAlone"].value_counts()
print(surv_class)
print(avg_fare)
print(alone_trv)
print(pivot)
print(ac)