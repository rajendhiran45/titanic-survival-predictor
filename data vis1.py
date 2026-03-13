import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("Titanic-Dataset.csv")
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df.drop("Cabin", axis=1, inplace=True)
df["FamilySize"] = df["SibSp"] + df["Parch"]
df["IsAlone"] = (df["FamilySize"] == 0).astype(int)
plt.figure(figsize=(10, 6))
sns.countplot(x="Sex",hue="Survived",data=df)
plt.title("Survived Split By Sex")
plt.show()
sns.countplot(x="Pclass",hue="Survived",data=df)
plt.title(" Pclass split by Survived")
plt.show()
plt.hist(df["Age"],bins=20,edgecolor="black")
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()
sns.boxplot(x="Survived",y="Age",data=df)
plt.title("Age Vs Survived")
plt.show()
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), 
            annot=True, 
            cmap="coolwarm", 
            fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()