import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("C:\\Users\\DELL-G5\\Downloads\\Social_Network_Ads (1).csv")
X = dataset.iloc[:,:-1]
y = dataset.iloc[:, -1]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)


y_pred = classifier.predict(X_test)
print(y_pred[0:5],"\n",y_test[0:5])

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

import seaborn as sns
sns.heatmap(cm, center=True)
plt.show()