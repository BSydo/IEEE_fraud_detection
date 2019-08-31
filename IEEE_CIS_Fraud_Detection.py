import sys
import warnings
import pandas as pd
import numpy as np
import xgboost
from sklearn import model_selection, preprocessing, ensemble, metrics

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
    test_transaction['mod_'+i] = encoder.fit_transform(test_transaction[i].fillna(test_transaction[i].mode()[0])

col_list = [x for x in train_transaction.columns if x not in non_num_cols+['isFraud']]

x_train = train_transaction[col_list].fillna(0)
y_train = test_transaction[col_list].fillna(0)
x_valid = train_transaction['isFraud']

# RF
# n_estimators = 31     ----- 0.7139
# min_samples_leaf = 25 ----- 0.6894
RF = ensemble.RandomForestClassifier(n_estimators = 31
                                     , n_jobs=-1
                                     , max_depth=31
                                     , min_samples_split = 0.3
                                    )
RF.fit(x_train, x_valid)
predictions = RF.predict(y_train)

#XGboost - 0,68
#XGb = xgboost.XGBClassifier()
#XGb.fit(x_train, x_valid)
#predictions = XGb.predict(y_train)


output['TransactionID'] = y_train['TransactionID']
output['isFraud'] = predictions
output.to_csv(output_path, index=False)
