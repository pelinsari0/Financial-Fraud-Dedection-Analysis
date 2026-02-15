## importing pandas, numpy, matplotlib:
import seaborn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## importing sklearn.linear_model stuff:
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

## Loading the data:
transactions = pd.read_csv('financial_fraud.csv')

## Just for lookup:

#print(transactions.head(10))
#print(transactions.columns)
#print(transactions.info())

## Summary statistics:
summary_stat1 = transactions['amount'].describe()
#print(summary_stat1)


## isPayment column using type transaction
transactions['isPayment'] = transactions['type'].isin(['Payment', 'Debit']).astype(int)


## isMovement capture if money moved out of the origin account
transactions['isMovement'] = transactions['type'].isin(['CASH_OUT', 'TRANSFER']).astype(int)


## Finding fraud or suspicious thing between the origin and destination account. abs() turns negative into positive. Significant different value can be a problem
transactions['accountDiff'] = abs(transactions['oldbalanceOrg'] - transactions['oldbalanceDest'])


## Defining feature and labels for machine learning
features = transactions[['amount', 'isPayment', 'isMovement', 'accountDiff']]
label = transactions['isFraud']


## Defining train test split
X_train, X_test, y_train, y_test = train_test_split(features, label,  train_size = 0.7, test_size = 0.3)


## StandardScaler using
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


## Finding best coefficients with using .fit()
model = LogisticRegression()
model.fit(X_train_scaled, y_train)


## Running model score
training_score = model.score(X_train_scaled, y_train)
print("Training score is: " + str(training_score))


## Running test score
test_score = model.score(X_test_scaled, y_test)
print("Test score is: " + str(test_score))


## Finding coef (if high means more influence, low otherwise)
print(model.coef_)


## New transaction data
transaction1 = np.array([123456.78, 0.0, 1.0, 54670.1])
transaction2 = np.array([98765.43, 1.0, 0.0, 8524.75])
transaction3 = np.array([543678.31, 1.0, 0.0, 510025.5])


## Combining transactions
sample_transactions = np.stack([transaction1, transaction2, transaction3])

sample_transactions = scaler.transform(sample_transactions)

## Dedecting which transaction is fradulent
print(model.predict(sample_transactions))
print(model.predict_proba(sample_transactions))









