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
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.ensemble import GradientBoostingClassifier


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import precision_score
data = pd.read_csv('/home/ubuntu/project/7008-Projects/Dataset/Breast_Cancer_Dataset_Cleaned.csv')
df = pd.DataFrame(data)
df['diagnosis']=df['diagnosis'].apply(lambda x: 0 if x=='M' else 1)
y = df["diagnosis"]
X = df.drop('diagnosis', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, random_state=123)

scalar = StandardScaler()
scalar.fit(X_train)

X_train = pd.DataFrame(scalar.transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(scalar.transform(X_test), columns=X_test.columns)

X_train


#original
LR_original_model = LogisticRegression()
LR_original_model.fit(X_train, y_train)
LR_predict=LR_original_model.predict(X_test)

# Calculate accuracy
LR_accuracy = accuracy_score(y_test, LR_predict)

# Calculate precision
LR_precision = precision_score(y_test, LR_predict)

# Calculate f1-score
LR_f1 = f1_score(y_test, LR_predict)

# Calculate recall
LR_recall = recall_score(y_test, LR_predict)

# Confusion Matrix
LR_cm = confusion_matrix(y_test, LR_predict)

# Classification report
LR_cr = classification_report(y_test, LR_predict)
print('Accuracy Score for LR: ', LR_accuracy)
print('Precision Score for LR: ', LR_precision)
print('F1 score for LR:', LR_f1)
print('Recall for LR:', LR_recall)
print(LR_cm)
print(LR_cr)

#Support Vector Machine
svm_o_model = LinearSVC(loss='hinge', dual=True)
svm_o_model.fit(X_train, y_train)


svm_o_model.score(X_test, y_test)
svm_predict=svm_o_model.predict(X_test)
# Calculate accuracy
SVC_accuracy = accuracy_score(y_test, svm_predict)

# Calculate precision
SVC_precision = precision_score(y_test, svm_predict)

# Calculate f1-score
SVC_f1 = f1_score(y_test, svm_predict)

# Calculate recall
SVC_recall = recall_score(y_test, svm_predict)

# Confusion Matrix
SVC_cm = confusion_matrix(y_test, svm_predict)

# Classification report
SVC_cr = classification_report(y_test, svm_predict)
print('Accuracy Score for SVC: ', SVC_accuracy)
print('Precision Score for SVC: ', SVC_precision)
print('F1 score for SVC: ', SVC_f1)
print('Recall for SVC: ', SVC_recall)
print(SVC_cm)
print(SVC_cr)

#Gradient Boosting
gb_classifer = GradientBoostingClassifier()
gb_classifer.fit(X_train, y_train)

y_pred = gb_classifer.predict(X_test)




# Calculate accuracy
GBC_accuracy = accuracy_score(y_test, y_pred)

# Calculate precision
GBC_precision = precision_score(y_test, y_pred)

# Calculate f1-score
GBC_f1 = f1_score(y_test, y_pred)

# Calculate recall
GBC_recall = recall_score(y_test, y_pred)

# Confusion Matrix
GBC_cm = confusion_matrix(y_test, y_pred)

# Classification report
GBC_cr = classification_report(y_test, y_pred)

print('Accuracy Score for GBC: ', GBC_accuracy)
print('Precision Score for GBC: ', GBC_precision)
print('F1 score for GBC: ', GBC_f1)
print('Recall for SGBC: ', GBC_recall)
print(GBC_cm)
print(GBC_cr)