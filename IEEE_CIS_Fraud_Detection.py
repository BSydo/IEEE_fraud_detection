import sys
import warnings
import pandas as pd
import numpy as np
import xgboost
from sklearn import model_selection, preprocessing, ensemble, metrics, linear_model
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE

warnings.simplefilter('ignore')

input_train = f'{sys.path[0]}/Input/ieee-fraud-detection/train_transaction.csv'
input_test = f'{sys.path[0]}/Input/ieee-fraud-detection/test_transaction.csv'
output_path = f'{sys.path[0]}/output/sample_submission.csv'
output = pd.DataFrame()

train_transaction = pd.read_csv(input_train)
test_transaction = pd.read_csv(input_test)

non_num_cols = [train_transaction.columns[n] for n,i in enumerate(train_transaction.dtypes)
                if i not in ('int64','float64')]

# label encode the target variable
encoder = preprocessing.LabelEncoder()

for i in non_num_cols:
    train_transaction['mod_'+i] = encoder.fit_transform(train_transaction[i].fillna(train_transaction[i].mode()[0]))
    test_transaction['mod_'+i] = encoder.fit_transform(test_transaction[i].fillna(test_transaction[i].mode()[0]))

col_list = [x for x in train_transaction.columns if x not in non_num_cols+['isFraud']]

x_train = train_transaction[col_list].fillna(0)
y_train = test_transaction[col_list].fillna(0)
x_valid = train_transaction['isFraud']

scaler = MinMaxScaler(feature_range = (0,1))

scaler.fit(x_train)
x_train = scaler.transform(x_train)
y_train = scaler.transform(y_train)

from imblearn.over_sampling import RandomOverSampler
#ros = RandomOverSampler(random_state=0)
#x_train, x_valid = ros.fit_resample(x_train, x_valid)
x_train, x_valid = SMOTE().fit_resample(x_train, x_valid)

# RF ----- 0.7139
#RF = ensemble.RandomForestClassifier(n_jobs=-1)
#RF.fit(x_train, x_valid)
#predictions = RF.predict(y_train)

#XGboost - 0,68
XGb = xgboost.XGBClassifier(n_estimators=100)
XGb.fit(x_train, x_valid)
predictions = XGb.predict(y_train)

# LR
#LR = linear_model.LogisticRegression(penalty = 'l2', C = 100, n_jobs=-1)

#LR.fit(x_train, x_valid)
#predictions = LR.predict(y_train)

output['TransactionID'] = test_transaction[col_list].fillna(0)['TransactionID']
output['isFraud'] = predictions
output.to_csv(output_path, index=False)
