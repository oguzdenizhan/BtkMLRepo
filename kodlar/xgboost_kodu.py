#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# veri kümesi
dataset = pd. read_csv( 'Churn_Modelling.csv')
X = dataset.iloc[:, 3:13]. values
y = dataset.iloc[:, 13]. values

# On Isleme
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

# Define a ColumnTransformer
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')

# Apply the ColumnTransformer to X
X = np.array(ct.fit_transform(X))

#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.33, random_state=0)

from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train) 

y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred,y_test)
# Extract TP, TN, FP, FN
TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]
TP = cm[1, 1]

# Calculate accuracy
accuracy = (TP + TN) / (TP + TN + FP + FN)

print("Confusion Matrix:")
print(cm)
print("Accuracy:", accuracy)