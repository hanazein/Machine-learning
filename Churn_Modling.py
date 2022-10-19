import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score ,confusion_matrix

df = pd.read_csv("D:\\downloads\\Churn_Modelling.csv")

print(df.head())
print(df.describe())
df=df.drop(['Surname','Geography','Gender'],axis=1)
print(df.head())
y = df['Exited']
x = df.drop(['Exited'], axis=1)
print(y.shape)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.25,random_state=27)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

clf = MLPClassifier(hidden_layer_sizes=(100), max_iter=1000, alpha=0.0001,solver='sgd', verbose=10,  random_state=21,)
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
print(y_pred)

print("score is ",accuracy_score(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
print(cm)

sns.heatmap(cm, center=True)
plt.show()