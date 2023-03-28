# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 19:15:56 2023

@author: raque
"""

import pandas as pd
from scipy.stats import ttest_ind, shapiro, levene

# read in the CSV files

df_nmi = pd.read_csv(r"C:\Users\raque\OneDrive\Escritorio\UU\capita_selecta_medical_tue\statistical test\NMI.csv")
df_random = pd.read_csv(r"C:\Users\raque\OneDrive\Escritorio\UU\capita_selecta_medical_tue\statistical test\Random.csv")

# drop the first column
df_nmi_2 = df_nmi.iloc[:, 1:]
df_random_2 = df_random.iloc[:, 1:]


# Shapiro-wilk test to assess the normality of the data
# list to store p-values
p_values = []

# Loop through each row in the dataframe
for index, row in df_nmi_2.iterrows():
    # Perform the Shapiro-Wilk test on the row
    stat, p = shapiro(row)
    p_values.append(p)
    print('Shapiro-Wilk test statistic:', stat)
    print('p-value:', p)
    
    if p > 0.05:
        # If the p-value is greater than your chosen significance level 
        # you can assume that your data is normally distributed.
        print('Data is normally distributed')
    else:
        print('Data is not normally distributed')
    
# all data was normally distributed 

variances1 = df_nmi_2.var(axis=1, ddof=1)
variances2 = df_random_2.var(axis=1, ddof=1)

# perform Levene's test
statistic, pvalue = levene(variances1, variances2)

print(f"Levene's test statistic: {statistic:.4f}")
print(f"p-value: {pvalue:.4f}")

if pvalue < 0.05:
    print("Assumption of equal-variances is violated")

# calculate the mean Dice similarity coefficient for each method
mean_nmi = df_nmi_2.mean(axis=1)
mean_random = df_random_2.mean(axis=1)

# perform two-sample t-test
t_stat, p_value = ttest_ind(mean_nmi, mean_random, equal_var = False)

# print results
print('Mean Dice Similarity Coefficient (NMI):', mean_nmi.mean())
print('Mean Dice Similarity Coefficient (Random):', mean_random.mean())
print("t-statistic: {:.4f}".format(t_stat))
print("p-value: {:.4f}".format(p_value))

# this is per number of atlases
t_stat_var, p_value_var = ttest_ind(df_nmi_2, df_random_2, equal_var=False)
print("t-statistic:", t_stat_var)
print("p-value: {:.4f}", p_value_var)


