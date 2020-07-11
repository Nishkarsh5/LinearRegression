import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# Importing LabourEarningPrediction.csv
LabourData = pd.read_csv('.\LabourEarningPrediction.csv')


# Looking at first 5 rows
print(LabourData.head())


# Looking at the last five rows
print(LabourData.tail())


# What type of values are stored in columns?
print(LabourData.info())


# Statistical information about our dataframe.
print(LabourData.describe())


# plotting a pair plot of all numerical variables in our dataframe.
print(sns.pairplot(LabourData))


# Visualise the relationship between the features and the response using scatterplots
print(sns.pairplot(LabourData, x_vars=['Age','Earnings_1974','Earnings_1975'], y_vars='Earnings_1978', size=7, aspect=0.7, kind='scatter'))
             

print(sns.boxplot(data = LabourData["Age"]))
print(sns.boxplot(data = LabourData["Earnings_1975"]))
print(sns.boxplot( x=LabourData["Race"], y=LabourData["Earnings_1978"]))
print(sns.boxplot( x=LabourData["MaritalStatus"], y=LabourData["Earnings_1978"]))


LabourData_num = LabourData[['Age', 'Nodeg', 'Earnings_1974', 'Earnings_1975', 'Earnings_1978']]

LabourData_dummies = pd.get_dummies(LabourData[['Race', 'Hisp', 'MaritalStatus', 'Eduacation']])
print(LabourData_dummies.head())


LabourData_combined = pd.concat([LabourData_num, LabourData_dummies], axis=1)
print(LabourData_combined.head())


#putting feature variable to X
X = LabourData_combined[['Age', 'Earnings_1974', 'Earnings_1975', 'Race_NotBlack', 'Race_black', 
                         'Hisp_NotHispanic', 'Hisp_hispanic','MaritalStatus_Married', 
                         'MaritalStatus_NotMarried', 'Eduacation_HighSchool', 'Eduacation_Intermediate',
                         'Eduacation_LessThanHighSchool', 'Eduacation_PostGraduate', 'Eduacation_graduate']]


#putting response variable to y
y = LabourData['Earnings_1978']


#Splitting the data in Training and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, random_state = 100)
