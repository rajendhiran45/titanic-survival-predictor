import pandas as pd
data=pd.read_csv("Titanic-Dataset.csv")
df=pd.DataFrame(data)
print(f"It Has {df.shape[0]} Columns And {df.shape[1]} Rows")
print(f"The Average Age Of Passengers:{df["Age"].mean()}")
print("\nMissing Values:")
print(df.isnull().sum())