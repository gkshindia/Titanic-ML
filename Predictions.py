# -*- coding: utf-8 -*-
"""
Created on Sat May  6 12:22:30 2017

@author: KANHAIYA
"""

import pandas as pd
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
#Print the `head` of the train and test dataframes
train.head()
test.head()
train.describe()
train.shape

# Passengers that survived vs passengers that passed away
print(train['Survived'].value_counts())

# As proportions
print(train['Survived'].value_counts(normalize = True))

# Males that survived vs males that passed away
print(train['Survived'][train['Sex'] == 'male'].value_counts())

# Females that survived vs Females that passed away
print(train['Survived'][train['Sex'] == 'female'].value_counts())

# Normalized male survival
print(train['Survived'][train['Sex'] == 'male'].value_counts(normalize = True))

# Normalized female survival
print(train['Survived'][train['Sex'] == 'female'].value_counts(normalize = True))
train["Child"] = float('NaN')
train['Age'] = train['Age'].fillna(train['Age'].median())
# Assign 1 to passengers under 18, 0 to those 18 or older. Print the new column.
train['Child'][train['Age'] < 18] = 1
train['Child'][train['Age'] > 18] = 0
print(train['Child'])
# Print normalized Survival Rates for passengers under 18
print(train["Survived"][train["Child"] == 1].value_counts(normalize = True))

# Print normalized Survival Rates for passengers 18 or older
print(train["Survived"][train["Child"] == 0].value_counts(normalize = True))
'''
In one of the previous exercises you discovered that in your training set, 
females had over a 50% chance of surviving and males had less than a 50% 
chance of surviving. Hence, you could use this information for your first 
prediction: all females in the test set survive and all males in the test 
set die. '''

# Create a copy of test: test_one
test_one = test
# Initialize a Survived column to 0
test_one['Survived'] = 0
# Set Survived to 1 if Sex equals "female" and print the `Survived` column from `test_one`
test_one['Survived'][test_one['Sex'] == 'female'] = 1
print(test_one['Survived'])

# Import the Numpy library
# Cleaning and Converting the categorical data to numerical format and Imputing the
# missing values
train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"] = 1
# Impute the Embarked variable
train["Embarked"] = train['Embarked'].fillna('S')

# Convert the Embarked classes to integer form
train["Embarked"][train["Embarked"] == "S"] = 0
train["Embarked"][train["Embarked"] == "C"] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2
     
     # Print the train data to see the available features
print(train)
import numpy as np
# Import 'tree' from scikit-learn library
from sklearn import tree
# Create the target and features numpy arrays: target, features_one
target = train['Survived'].values
features_one = train[["Pclass", "Sex", "Age", "Fare"]].values

# Fit your first decision tree: my_tree_one
my_tree_one = tree.DecisionTreeClassifier()
my_tree_one = my_tree_one.fit(features_one, target)

# Look at the importance and score of the included features
print(my_tree_one.feature_importances_)
print(my_tree_one.score(features_one, target))



# Impute the missing value with the median
test.Fare[152] = test.Fare.median()
test["Sex"][test["Sex"] == "male"] = 0
test["Sex"][test["Sex"] == "female"] = 1
# Extract the features from the test set: Pclass, Sex, Age, and Fare.
test_features = test[["Pclass", "Sex", "Age", "Fare"]].values
test['Age'] = test['Age'].fillna(test['Age'].median())
# Assign 1 to passengers under 18, 0 to those 18 or older. Print the new column.
test['Child'][test['Age'] < 18] = 1
test['Child'][test['Age'] > 18] = 0
# Make your prediction using the test set and print them.
my_prediction = my_tree_one.predict(test_features)
print(my_prediction)

# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
PassengerId =np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])
print(my_solution)

# Check that your data frame has 418 entries
print(my_solution.shape)

# Write your solution to a csv file with the name my_solution.csv
my_solution.to_csv("my_solution_one.csv", index_label = ["PassengerId"])


'''Overfitting Avoid'''

#Create a new array with the added features: features_two
features_two = train[["Pclass","Age","Sex","Fare", "SibSp", "Parch", "Embarked"]].values

#Control overfitting by setting "max_depth" to 10 and "min_samples_split" to 5 : my_tree_two
max_depth = 10
min_samples_split = 5
my_tree_two = tree.DecisionTreeClassifier(max_depth = 10, min_samples_split = 5, random_state = 1)
my_tree_two = my_tree_two.fit(features_two, target)

#Print the score of the new decison tree
print(my_tree_two.score(features_two, target))

'''
While feature engineering is a discipline in itself, too broad to be covered here
 in detail, you will have a look at a simple example by creating your own new
 predictive attribute: family_size.

A valid assumption is that larger families need more time to get together on 
a sinking ship, and hence have lower probability of surviving. Family size is 
determined by the variables SibSp and Parch, which indicate the number of family 
members a certain passenger is traveling with. So when doing feature engineering,
 you add a new variable family_size, which is the sum of SibSp and Parch plus one 
 (the observation itself), to the test and train set.'''
 
 # Create train_two with the newly defined feature
train_two = train.copy()
train_two["family_size"] = train_two['SibSp'] + train_two['Parch'] + 1

# Create a new feature set and add the new feature
features_three = train_two[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "family_size"]].values

# Define the tree classifier, then fit the model
my_tree_three = tree.DecisionTreeClassifier()
my_tree_three = my_tree_three.fit(features_three, target)

# Print the score of this decision tree
print(my_tree_three.score(features_three, target))


''' Random Forest classifier'''

# Import the `RandomForestClassifier`
from sklearn.ensemble import RandomForestClassifier

# We want the Pclass, Age, Sex, Fare,SibSp, Parch, and Embarked variables
features_forest = train[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values

# Building and fitting my_forest
forest = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = 100, random_state = 1)
my_forest = forest.fit(features_forest, target)

# Print the score of the fitted random forest
print(my_forest.score(features_forest, target))

# Compute predictions on our test set features then print the length of the prediction vector
test_features = test[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
pred_forest = my_forest.predict(test_features)
print(len(pred_forest))


'''Interpreting and Comparing'''

#Request and print the `.feature_importances_` attribute
print(my_tree_two.feature_importances_)
print(my_forest.feature_importances_)

#Compute and print the mean accuracy score for both models
print(my_tree_two.score(features_two, target))
print(my_forest.score(features_forest, target))
