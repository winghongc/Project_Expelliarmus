import numpy as np  
import pandas as pd  
import os

# import libraries for plotting
import matplotlib.pyplot as plt
import seaborn as sns



data = '/home/ubuntu/project/7008-Projects/Dataset/Breast_Cancer_Dataset.csv'

df = pd.read_csv(data)
df.drop(df.columns[0], axis=1, inplace=True)
df.drop(df.columns[-1], axis=1, inplace=True)
df.to_csv('/home/ubuntu/project/7008-Projects/Dataset/Breast_Cancer_Dataset_Cleaned.csv', index=False)
print(df.shape)
print(df.dtypes)
print(os.getcwd())