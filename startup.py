import pandas as pd
dataset = pd.read_csv("D:\\downloads\\50_Startups.csv")
dataset=dataset.drop(['State'],axis=1)
print(dataset.head(10))
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]
print(X.shape)
print(y.shape)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
regressor.score(X_train, y_train)
print(regressor.score(X_train, y_train))
regressor.score(X_test, y_test)
print(regressor.score(X_test, y_test))
y_pred = regressor.predict(X_test)
print(y_pred,"\n y test",y_test)
from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(y_test, y_pred))
from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test, y_pred))
from sklearn.metrics import median_absolute_error
print(median_absolute_error(y_test, y_pred))


