import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

purchaseData = pd.read_csv('D:/IITK_MLDL/Purchase_Logistic.csv')

X = purchaseData.iloc[:,[2,3]]
Y = purchaseData.iloc[:,4]

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train,y_test = train_test_split(X,Y, test_size=0.25,random_state=0)

cf = GaussianNB()
cf.fit(X_train,y_train)
Ypred = cf.predict(X_test)

cmat = confusion_matrix(y_test, Ypred)
print('Confusion matrix =\n', cmat)

plt.figure(1);  
plt.scatter(X[:, 0], X[:, 1], c = Y)
plt.suptitle('Purchase Data')
plt.xlabel('Scaled Age')
plt.ylabel('Scaled Income')
plt.grid(1,which='both')
plt.axis('tight')
plt.show()

col = cf.predict(X)

plt.figure(2);
plt.scatter(X[:, 0], X[:, 1], c = col)
plt.suptitle('Naive Bayes Purchase Data')
plt.xlabel('Scaled Age')
plt.ylabel('Scaled Income')
plt.grid(1,which='both')
plt.axis('tight')
plt.show()