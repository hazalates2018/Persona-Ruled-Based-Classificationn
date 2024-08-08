
# Calculating Potential Customer Revenue with Rule-Based Classification




# Business Problem

# A gaming company wants to create level-based (persona) new customer profiles using certain characteristics of their customers.
# They also want to segment these profiles and estimate how much new customers might contribute on average to the company based on these segments.
# For example, the goal is to determine the average revenue that a 25-year-old male user from Turkey using an iOS device could potentially generate.

# Price: The amount spent by the customer
# Source: The type of device the customer is using
# Sex: The gender of the customer
# Country: The country of the customer
# Age: The age of the customer

################# Pre-Implementation #####################

#    PRICE   SOURCE   SEX COUNTRY  AGE
# 0     39  android  male     bra   17
# 1     39  android  male     bra   17
# 2     49  android  male     bra   17
# 3     29  android  male     tur   17
# 4     49  android  male     tur   17

################# Post-Implementation #####################

#       customers_level_based        PRICE SEGMENT
# 0   BRA_ANDROID_FEMALE_0_18  1139.800000       A
# 1  BRA_ANDROID_FEMALE_19_23  1070.600000       A
# 2  BRA_ANDROID_FEMALE_24_30   508.142857       A
# 3  BRA_ANDROID_FEMALE_31_40   233.166667       C
# 4  BRA_ANDROID_FEMALE_41_66   236.666667       C


# Project Tasks

# TASK 1: Answer the following questions.

# Question 1: Read the `persona.csv` file and display general information about the dataset.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = pd.read_csv("C:\\Users\\hazal\\Downloads\\proje-230307-154749\\proje\\persona.csv")
df.isnull().values.any()
df.isnull().sum()
df.head()

# Question 2: How many unique `SOURCE` values are there? What are their frequencies?
df["SOURCE"].nunique()
df["SOURCE"].info()

# Question 3: How many unique `PRICE` values are there?
df["PRICE"].nunique()

# Question 4: How many sales were made for each `PRICE` value?
df["PRICE"].value_counts()

# Question 5: How many sales have been made from each country?

df["COUNTRY"].value_counts()
df.groupby("COUNTRY")["PRICE"].count()
df.groupby("COUNTRY").agg({"PRICE":"count"})

# Question 6: How much total revenue has been earned from sales by country?
df.groupby("COUNTRY").agg({"PRICE":"sum"})

# Question 7: What are the sales numbers according to the `SOURCE` types?
df.groupby("SOURCE").agg({"PRICE":"count"})

# Question 8: What are the average `PRICE` values by country?
df.groupby("COUNTRY")["PRICE"].mean()

# Question 9: What are the average `PRICE` values according to `SOURCE`?
df.groupby("SOURCE")["PRICE"].mean()

# Question 10: What are the average `PRICE` values for each `COUNTRY`-`SOURCE` breakdown?
df.groupby(["COUNTRY","SOURCE"])["PRICE"].mean()
df.groupby(by=["COUNTRY", 'SOURCE']).agg({"PRICE": "mean"})

# TASK 2: What are the average earnings broken down by `COUNTRY`, `SOURCE`, `SEX`, and `AGE`?
df.groupby(by=["COUNTRY","SOURCE","SEX","AGE"])["PRICE"].mean()

# TASK 3: Sort the output by `PRICE`.
agg_df = df.groupby(by=["COUNTRY","SOURCE","SEX","AGE"]).agg({}).sort_values("PRICE",ascending=False)

agg_df = df.groupby(by=["COUNTRY", 'SOURCE', "SEX", "AGE"]).agg({"PRICE": "mean"}).sort_values("PRICE", ascending=False)
agg_df.head()

# TASK 4: Convert the names in the index to variable names.
agg_df = agg_df.reset_index()
agg_df.head()

# TASK 5: Convert the `AGE` variable into a categorical variable and add it to `agg_df`.
max_age = agg_df["AGE"].max()
bins=[0,18,23,30,40,max_age]
mylabels = ["0_18","19_23","24_30","31_40","41_" + str(max_age)]

# age'i bölelim
agg_df["AGE_CAT"] = pd.cut(agg_df["AGE"], bins, labels = mylabels )
agg_df.head()


# Task 6: Define new level-based customers and add them as a variable to the dataset.


agg_df["CUSTOMERS_LEVEL_BASED"] = agg_df[["COUNTRY","SOURCE","SEX","AGE_CAT"]].agg(lambda x: "_".join(x).upper(),axis=1)


agg_df["customers_level_based"] = ['_'.join(i).upper() for i in agg_df.drop(["AGE", "PRICE"], axis=1).values]

agg_df["customers_level_based"] = ["_".join(i).upper() for i in agg_df.drop(["AGE","PRICE"],axis=1).values]


agg_df = agg_df[["CUSTOMERS_LEVEL_BASED", "PRICE"]]
agg_df.head()

# Task 7: Segment new customers (e.g., USA_ANDROID_MALE_0_18).
agg_df["SEGMENT"] = pd.qcut(agg_df["PRICE"], 4, labels=["D", "C", "B", "A"])

agg_df["SEGMENT"].head(30)
agg_df["SEGMENT"].tail(30)

agg_df.groupby("SEGMENT").agg({"PRICE":["mean","max","sum"]})


# Task 8: Classify new customers and estimate how much revenue they might generate.

[agg_df["PRICE"].mean() for row in agg_df.iterrows() if agg_df["AGE"]==33 and agg_df["SOURCE"] == "android" and agg_df["SEX"]=="female" and agg_df["COUNTRY"]== "tur"  ]

filtered_df = agg_df[(agg_df["AGE"]==33)&(agg_df["SOURCE"]=="android") & (agg_df["SEX"]=="female") & (agg_df["COUNTRY"] =="tur")]

new_user= agg_df[(agg_df["CUSTOMERS_LEVEL_BASED"]==  "TUR_ANDROID_FEMALE_31_40")]

new_user.mean()

new_user = "TUR_ANDROID_FEMALE_31_40"
agg_df[agg_df["CUSTOMERS_LEVEL_BASED"] == new_user]

new_user = "FRA_IOS_FEMALE_31_40"
agg_df[agg_df["customers_level_based"] == new_user]
