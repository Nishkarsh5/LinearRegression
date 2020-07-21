import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


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


#Performing Linear Regression
#creating linearreggression object
lr = LinearRegression()
lr.fit(X_train, y_train)

i = lr.intercept_
print("Intercept = " + str(i))

#coefficients
coeff_df = pd.DataFrame(lr.coef_, X_test.columns, columns = ['coefficient'])
print(coeff_df)


#Making predictions using the model
y_pred = lr.predict(X_test)


#Model performance metrics
#Coefficients of determination(R square)
mse = mean_squared_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)

rmse = sqrt(mse)

print('Mean squared error      : ',mse)
print('Root mean squared error : ',rmse)
print('r squared value         : ',r_squared)


#Cheacking P-value using statsmodels
X_train_sm = X_train
X_train_sm = sm.add_constant(X_train_sm)

lr_1 = sm.OLS(y_train, X_train_sm).fit()

print(lr_1.params)

print(lr_1.summary())


#Variance Inflation Factor
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values,i) for i in range(X.shape[1])]
vif["features"] = X.columns

print(vif.round(2))