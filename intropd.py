import pandas as pd
data={"Name": ["Alice", "Bob", "Charlie"],
    "Age": [25, 30, 35],
    "City": ["Chennai", "Delhi", "Mumbai"]}

df=pd.DataFrame(data)
print(df)
inf=df.info()
print(inf)
print(f"Desc{df.describe()}")
print(f"Head{df.head()}")
print(f"Tail{df.tail()}")