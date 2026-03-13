import pandas as pd
data=pd.read_csv("Titanic-Dataset.csv")
df=pd.DataFrame(data)
print("\nPassengers In First Class")
print(df[df["Pclass"]==1])
print(df[["Name","Age"]])
print(df[df["Age"]>50])