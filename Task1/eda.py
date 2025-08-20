
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("Titanic-Dataset.csv")

print("Shape of dataset:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
print("\nFirst 5 rows:\n", df.head())

sns.countplot(x="Survived", data=df, palette="Set2")
plt.title("Survival Count (0 = No, 1 = Yes)")
plt.show()

sns.countplot(x="Sex", hue="Survived", data=df, palette="husl")
plt.title("Gender vs Survival")
plt.show()

sns.countplot(x="Pclass", hue="Survived", data=df, palette="coolwarm")
plt.title("Passenger Class vs Survival")
plt.show()

sns.histplot(df["Age"].dropna(), bins=30, kde=True)
plt.title("Age Distribution of Passengers")
plt.show()
