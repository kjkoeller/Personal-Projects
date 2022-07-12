"""
Logistical Regression Exercise

Author: Kyle Koeller
Created: 7/11/2022
Last Updated: 7/11/2022
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin
import pickle

data_preprocessed = pd.read_csv("df_preprocessed.csv")

median = data_preprocessed["Absenteeism Time in Hours"].median()
# print(median)

# checks whether the absenteeism time is greater than the median value and true is 1 and false is 0
targets = np.where(data_preprocessed["Absenteeism Time in Hours"] > median, 1, 0)

# add the targets to the dataframe
data_preprocessed["Excessive Absenteeism"] = targets

# around 46% of the targets are 1
# usually 45-55 split is sufficient for this type of machine learning
# print(targets.sum() / targets.shape[0])

# remove the absenteeism time as it is no longer needed
data_with_targets = data_preprocessed.drop(["Absenteeism Time in Hours", "Day of the Week", "Daily Work Load Average",
                                            "Distance to Work"], axis=1)

# checks if the objects are the same
# print(data_with_targets is data_preprocessed)

# isolate the inputs not including excessive absenteeism
unscaled_inputs = data_with_targets.iloc[:, :-1]

"""
Standardize the Data
"""
absenteeism_scaler = StandardScaler()


class CustomScaler(BaseEstimator, TransformerMixin):
    def __init__(self, columns, copy=True, with_mean=True, with_std=True):
        self.scaler = StandardScaler()
        self.columns = columns
        self.mean = None
        self.var = None

    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns], y)
        self.mean = np.mean(X[self.columns])
        self.var = np.var(X[self.columns])
        return self

    def transform(self, X, y=None, copy=None):
        init_col_order = X.columns
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns)
        X_not_scaled = X.loc[:, ~X.columns.isin(self.columns)]
        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]


# columns_to_scale = ['Month Value', 'Day of the Week', 'Transportation Expense', 'Distance to Work', 'Age',
#                     'Daily Work Load Average', 'Body Mass Index', 'Children', 'Pets']

# this picks the columns to omit
columns_to_omit = ["Reason_1", "Reason_2", "Reason_3", "Reason_4", "Education"]

# list comprehension to create a new list from a previous list
columns_to_scale = [x for x in unscaled_inputs.columns.values if x not in columns_to_omit]
absenteeism_scaler = CustomScaler(columns_to_scale)
absenteeism_scaler.fit(unscaled_inputs)

# most common way is to "transform" the raw data from the import csv file at the beginning
scaled_inputs = absenteeism_scaler.transform(unscaled_inputs)

# splits the data into train and test data into 3:1 ratio of training to testing
# most common method is to specify 80-20 or 90-10 between train and test data
x_train, x_test, y_train, y_test = train_test_split(scaled_inputs, targets)

# makes this an 80-20 split
# random_state makes the random shuffling sudo-random and 20 makes the random shuffle the same shuffle every time
x_train, x_test, y_train, y_test = train_test_split(scaled_inputs, targets, train_size=0.8, random_state=20)

"""
Logistic Regression
"""
reg = LogisticRegression()
reg.fit(x_train, y_train)
# checks the accuracy of the model
print(reg.score(x_train, y_train))

model_outputs = reg.predict(x_train)
# checks whether there is a match or not within the guessed values
# print(np.sum(model_outputs == y_train))
# this manually checks the reg.score value (essentially) they should be the exact same
# print(np.sum(model_outputs == y_train) / model_outputs.shape[0])

"""
Finding the intercept and coefficients
"""

intercepts = reg.intercept_
coeff = reg.coef_

# create a new dataframe containing the intercept and coefficients with their corresponding columns
feature_name = unscaled_inputs.columns.values
summary_table = pd.DataFrame(columns=["Feature name"], data=feature_name)
summary_table["Coefficient"] = np.transpose(coeff)

# move all the indexes up by 1 and add the intercept to the 0th index
summary_table.index = summary_table.index + 1
summary_table.loc[0] = ["Intercept", intercepts[0]]
# print(summary_table.sort_index())

summary_table["Odds_ratio"] = np.exp(summary_table.Coefficient)
"""
Sorts by most important to least important by coefficients
A feature is not that important if the coefficient is around 0 or the ration is around 1
    might be useful to get rid of these features later on
"""

# since we standardized all values including the dummy ones, the output is not useful atm
print(summary_table.sort_values("Odds_ratio", ascending=False))

"""
Testing the Model
"""

# if the test value is higher than the trained data value than we got lucky or made a mistake somewhere along the line
print(reg.score(x_test, y_test))

# column one is the prob of being 0 and column 1 is prob of being 1
# added together should equal a value of 1
predicted_proba = reg.predict_proba(x_test)

# outputs only column 1 which is probability of being excessively absent for this example
print(predicted_proba[:, 1])

"""
Save the model

save the reg object in essence
"""

with open("model", "wb") as file:
    # .dump is the save
    pickle.dump(reg, file)

with open("scalar", "wb") as file:
    pickle.dump(absenteeism_scaler, file)
