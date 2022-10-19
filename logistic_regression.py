import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pandas as pd
data=pd.read_csv("C:\\Users\DELL-G5\\Downloads\\Social_Network_Ads.csv")
x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values
# print(x)
# print(y)
print(x.shape)
print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)
print("x-train is : ",X_train.shape)
print("y-train is : ",y_train.shape)
print("x-test is : ",X_test.shape)
print("y-train is : ",y_test.shape)
logistic_standered=StandardScaler()
X_train=logistic_standered.fit_transform(X_train)
X_test=logistic_standered.fit_transform(X_test)
print("FIT X_TRAIN : ",X_train[0:10])
print("FIT X_TEST : ",X_train[0:10])
logistic_regression=LogisticRegression(random_state=0)
logistic_regression.fit(X_train,y_train)
print(logistic_regression.predict(logistic_standered.transform([[60,36000]])))
y_pred = logistic_regression.predict(X_test)
y_pred = logistic_regression.predict(X_test)

# print(X_test[:10])
# print('-'*15)
# print(y_pred[:10])
print(y_pred[:20])
print(y_test[:20])
cm = confusion_matrix(y_test, y_pred)
print(cm)
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
print(X1.shape)
print(X2.shape)
plt.contourf(X1, X2, logistic_regression.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.6, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, logistic_regression.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.6, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

