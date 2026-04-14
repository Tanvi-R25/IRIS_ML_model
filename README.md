# IRIS_ML_model
This code is an implementation of a Machine Learning classification of an Iris flower dataset into 3 catergories - Verginica, Setosa and Versicolor. The ML model classifies the data into these categories taking 4 features into consideration :
1) Sepal length
2) Sepal width
3) Petal length
4) Petal width

The iris model classification uses a supervised learning algorithm, that is, K Nearest Neighbors (KNN)
The workflow of this code involves :
1) Data Loading - Loading the iris dataset from sklearn
2) Data visualization - A scatter matrix is generated
3) Pre-processing - 75% of the data from the dataset is used for training the model, while 25% of the data is used for testing
4) Training - KNN algorithm (neighbors = 10), is used tot train the model
5) Data prediction - A new data is considered to test and predict the output 
6) Evaluation - The respective accuracy and scores for the testing and training datasets are evaluated and analysed
