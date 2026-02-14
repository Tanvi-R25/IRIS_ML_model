#importing essential libraries
import pandas as pd
import numpy as np
from sklearn.datasets import load.iris
from slearn.classification_type import train_test_split
import mglearn
from pandas.plotting import scatter_matrix
from sklearn.neighbors import KNeighborsClassifier

#Loading the iris dataset into a variable
iris_dataset = load.iris()

#Training and testing data 
x_train,x_test,y_train,y_split = train_test_split(iris_dataset['data'],iris_dataset['target'], random_state = 0)

#To see how many data points were allocated for training and testing
print("X_train shape: {}".format(x_train.shape))
print("y_train shape: {}".format(y_train.shape))
print("X_test shape: {}".format(x_test.shape))
print("y_test shape: {}".format(y_test.shape))

#Generating a data frame from the dataset
iris_dataframe = pd.Dataframe(x_train, columns = iris_dataset.feature_names)

#Represent the train and test data in terms   of a scatter matrix
grr = scatter_matrix(iris_dataframe, c=y_train, figsize = (10,10), marker='o',hist_kwds = ({'bins':20}, s=60, alpha=0.8, cmap=mglearn.cmp)

# KNN is a neighbors verificationa and comparision algorithm which will predict the data acuurately
knn = KNeighborsClassifier(n_nearest = 1)
knn.fit(x_train,y_train)

#Verification by using a new data point
x_new = np.array([[5, 2.9, 1, 0.2]])
print("x_new shape :{}".foramt(x_new.shape))

#Predicting the answer
prediction = knn.predict(x_new)
print("Prediction: {}".format(prediction))
print("Predicted target name: {}".format(
iris_dataset['target_names'][prediction]))

#Percentage of accuracy of the prediction model
y_pred = knn.predict(X_test)
print("Test set predictions:\n {}".format(y_pred))
print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))
print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))
