import pandas as pd
df=pd.read_csv("Titanic-Dataset.csv")
print("\n Missing Percentage")
print(df.isnull().sum()/len(df)*100)
#Fill missing Age values using median
# Median is preferred because Age data can contain outliers
# (very old or very young passengers), and median is more robust than mean
df["Age"].fillna(df["Age"].median(),inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0],inplace=True)
df.drop("Cabin",axis=1,inplace=True)
print("\nMissing Values After Cleaning:\n")
print(df.isnull().sum())