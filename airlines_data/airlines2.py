import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

df = pd.read_csv("train.csv")

print(df.shape)
print(df.head())
print(df.describe())
print(df.info())
print(df.columns)
print(df.isnull().sum())

plt.figure(figsize=(6,4))
sns.countplot(x="satisfaction",data = df)
plt.title("memnuniyet")
plt.show()


df.drop(columns=["Unnamed: 0","id"], inplace = True)

df["Arrival Delay in Minutes"] = df["Arrival Delay in Minutes"].fillna(0)

df["satisfaction"] = df["satisfaction"].map(
    {
    "neutral or dissatisfied": 0,
    "satisfied": 1
    
    }
)

categorical_cols =["Gender","Customer Type","Type of Travel","Class"]
pd.get_dummies(df, columns=categorical_cols,drop_first=True)

print(df.isnull().sum())










for col in df.columns:
    print(col)