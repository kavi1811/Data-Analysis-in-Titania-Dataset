# import the important library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#load the dataset
df=pd.read_csv(r"C:\Users\kavi\Downloads\archive (16)\Titanic-Dataset.csv")
df.head()

#check the data types and missing values
df.info()
df.isnull().sum()

#handle the missing values
df["Age"]=df["Age"].fillna(df["Age"].mean())
df["Cabin"]=df["Cabin"].fillna("Unknown")
df.info()

#analyze the survival rate by different factors
df.groupby("Pclass")["Survived"].mean()

#analyze the survival rate by age groups
df["Age"] = df["Age"].fillna(df["Age"].median())
bins = [0, 12, 18, 35, 60, 100]
labels = ["Child", "Teen", "Young Adult", "Adult", "Senior"]

# Create a new column for age groups
df["AgeGroup"] = pd.cut(df["Age"], bins=bins, labels=labels)
survival_by_age = df.groupby("AgeGroup")["Survived"].mean()
print(survival_by_age)


#analyze the survival rate by gender
survival_by_gender = df.groupby("Sex")["Survived"].mean()
survival_by_gender.plot(kind="bar")

plt.title("Survival Rate by Gender")
plt.ylabel("Survival Rate")
plt.xlabel("Gender")

plt.show()

#analyze the survival rate by passenger class
survival_by_class = df.groupby("Pclass")["Survived"].mean()
survival_by_class.plot(kind="bar")

plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")

plt.show()

#analyze the distribution of ages
ages = df["Age"].fillna(df["Age"].median())
plt.hist(ages, bins=20)

plt.title("Distribution of Passenger Ages")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")

plt.show()
