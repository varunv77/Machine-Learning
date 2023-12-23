import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

def sigmoid(x):
  return 1/(1 + np.exp(-x))

purchaseData = pd.read_csv('D:/IITK_MLDL/Purchase_Logistic.csv')

X = purchaseData.iloc[:,[2,3]]
y = purchaseData.iloc[:,4].values

scaler = StandardScaler();
X = scaler.fit_transform(X)
X_train, X_test, y_train,y_test = train_test_split(X,y, test_size=0.25,random_state=0)

logr = LogisticRegression(random_state=0)
logr.fit(X_train, y_train)
Ypred = logr.predict(X_test)

cmat = confusion_matrix(y_test, Ypred)

plt.figure(1);  
plt.scatter(X[:, 0], X[:, 1], c = y)
plt.suptitle('Purchase Data')
plt.xlabel('Scaled Age')
plt.ylabel('Scaled Income')
plt.grid(1,which='both')
plt.axis('tight')
plt.show()

col = sigmoid(np.dot(X, np.transpose(logr.coef_)) + logr.intercept_) 
cf = logr.coef_;
xplot = np.arange(-1.0,1.2,0.01);
yplot = -(cf[0,0]*xplot + logr.intercept_)/cf[0,1]

plt.figure(2);
plt.scatter(X[:, 0], X[:, 1], c = col)
plt.plot(xplot,yplot,'g')
plt.suptitle('Logistic Regression Purchase Data')
plt.xlabel('Scaled Age')
plt.ylabel('Scaled Income')
plt.grid(1,which='both')
plt.axis('tight')
plt.show()
