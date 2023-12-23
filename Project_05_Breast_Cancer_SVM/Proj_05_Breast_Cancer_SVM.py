from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

bcancer = datasets.load_breast_cancer()

X = bcancer.data
Y = bcancer.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train,y_test = train_test_split(X,Y, test_size=0.25,random_state=0)

#Linear SVM
svmc = SVC(kernel='linear', random_state=0)
svmc.fit(X_train,y_train)
Ypred = svmc.predict(X_test)
svmcscore = accuracy_score(Ypred, y_test)
print('Accuracy score of Linear SVM Classifier is', 100*svmcscore,'%\n')

#Kernel SVM RBF - Gaussian Kernel

ksvmc = SVC(kernel = 'rbf', random_state=0)
ksvmc.fit(X_train,y_train)
y_pred = ksvmc.predict(X_test)
ksvmcscore = accuracy_score(y_pred, y_test)
print('Accuracy score of Kernel SVM Classifier with RBF is', 100*ksvmcscore,'%\n')


