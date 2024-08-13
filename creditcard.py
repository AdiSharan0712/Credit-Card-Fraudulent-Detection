import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# loading the dataset
credit_card_data = pd.read_csv("creditcard.csv")

# printing the initial 5 rows of the dataset
credit_card_data.head()

# printing the last 5 rows
credit_card_data.tail()

# dataset information
credit_card_data.info()

# check the ditribution of legit and fraudulant transactions
credit_card_data['Class'].value_counts()

#since the dataset is highly unbalanced 
#separating the data for analysis
legit=credit_card_data[credit_card_data.Class==0]
fraud=credit_card_data[credit_card_data.Class==1]

print(legit.shape)
print(fraud.shape)

# statistical measure of the data
legit.Amount.describe()
fraud.Amount.describe()

# compare the values for both transactions
# taking the mean of all columns in the dataset based on legit or fraud
credit_card_data.groupby('Class').mean()

# under-sampling to handle the unbalanced data
# building a sample dataset with similar distribution of the legit and fraudulant transactions
# 492 fraudulant transactions
legit_sample=legit.sample(n=492)
# concatenation of the 2 dataframes
new_dataset=pd.concat([legit_sample, fraud], axis=0)

#first 5 rows of the new dataset
new_dataset.head()
new_dataset.tail()
new_dataset['Class'].value_counts()
new_dataset.groupby('Class').mean()

# splitting the data into features and targets
X=new_dataset.drop(columns='Class', axis=1)
Y=new_dataset['Class']
print(X)
print(Y)

# split the data into training data and testing data
X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

# model training
#logistic regression model
model=LogisticRegression()
#training the logistic regression model with the training data
model.fit(X_train, Y_train)

#model evaluation based on the accuracy score
#accuracy on training data
X_train_prediction=model.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction, Y_train)
print("Accuracy on the training data: ",training_data_accuracy)

#accuracy on the testing data
X_test_prediction=model.predict(X_test)
testing_data_accuracy=accuracy_score(X_test_prediction, Y_test)
print("Accuracy on the testing data: ",testing_data_accuracy)