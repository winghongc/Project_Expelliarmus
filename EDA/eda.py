import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

#TO SCALE THE DATA FOR LR Model
from sklearn.preprocessing import StandardScaler

#PCA
from sklearn.decomposition import PCA 
#MACHINE LEARNING MODELS
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import precision_score



data = pd.read_csv('/home/ubuntu/project/7008-Projects/Dataset/Breast_Cancer_Dataset_Cleaned.csv')

#USE THIS FOR TRAIN_TEST_SPLIT
df = pd.DataFrame(data)
df['diagnosis']=df['diagnosis'].apply(lambda x: 0 if x=='M' else 1)

print(df.dtypes)
# Calculate the correlation matrix
correlation_matrix = df.corr()

# Plot the correlation matrix as a heatmap
plt.figure(figsize=(20, 10))
sns.heatmap(correlation_matrix, cmap='twilight',linewidths=0.5, annot=True)
plt.title('Correlation Matrix')
plt.show
plt.savefig("/home/ubuntu/project/7008-Projects/Dataset/Heatmap.png")