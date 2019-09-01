import sys
import warnings
import copy
import pandas as pd
import numpy as np
from sklearn import model_selection, preprocessing, ensemble, metrics, linear_model
#, naive_bayes, metrics, svm, decomposition
import xgboost
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE, RandomOverSampler

warnings.simplefilter('ignore')

input_train_tr = f'{sys.path[0]}/Input/ieee-fraud-detection/train_transaction.csv'
input_test_tr = f'{sys.path[0]}/Input/ieee-fraud-detection/test_transaction.csv'
input_train_id = f'{sys.path[0]}/Input/ieee-fraud-detection/train_identity.csv'
input_test_id = f'{sys.path[0]}/Input/ieee-fraud-detection/test_identity.csv'
output_path = f'{sys.path[0]}/output/sample_submission.csv'

output = pd.DataFrame()

train_transaction = pd.read_csv(input_train_tr)
test_transaction = pd.read_csv(input_test_tr)
train_identity = pd.read_csv(input_train_id)
test_identity = pd.read_csv(input_test_id)

train_ident = train_transaction['TransactionID'].isin(train_identity['TransactionID'])
test_ident = test_transaction['TransactionID'].isin(test_identity['TransactionID'])

train_tr = train_transaction[train_ident == False]
train_id = pd.merge(train_transaction[train_ident == True], train_identity, on='TransactionID')

test_tr = test_transaction[test_ident == False]
test_id = pd.merge(test_transaction[test_ident == True], test_identity, on='TransactionID')

# label encode the target variable
def encode(dataset):

    encoder = preprocessing.LabelEncoder()

    non_num_cols = [dataset.columns[n] for n,i in enumerate(dataset.dtypes)
                    if i not in ('int64','float64')]

    #Filling missing string values with most common value
    for i in non_num_cols:
        dataset[i+'_mod'] = encoder\
                            .fit_transform(dataset[i].fillna(dataset[i]\
                                                             .replace(np.nan
                                                                      , '0'
                                                                      , regex=True).mode()[0]))

    col_list = [x for x in train_transaction.columns if x not in non_num_cols+['isFraud']]

    dataset = dataset[col_list].fillna(0)

    return dataset, col_list

# feature engineering
def feature_eng(init_ds, train_dataset, test_dataset):

    x_train = train_dataset.fillna(0)
    y_train = test_dataset.fillna(0)
    x_valid = init_ds['isFraud']

    scaler = MinMaxScaler(feature_range = (0,1))

    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    y_train = scaler.transform(y_train)


    #ros = RandomOverSampler(random_state=0)
    #x_train, x_valid = ros.fit_resample(x_train, x_valid)
    x_train, x_valid = SMOTE().fit_resample(x_train, x_valid)

    return x_train, y_train, x_valid

# algorithm testing
def test_alg(x_train, y_train, x_valid):

    RF = ensemble.RandomForestClassifier(n_jobs=-1) #97.25

    RF.fit(x_train, x_valid)
    predictions = RF.predict(y_train)

    return predictions

# mapped id
train_id_enc, col_list = encode(train_id)
test_id_enc, col_list = encode(test_id)

x_train, y_train, x_valid = feature_eng(pd.merge(train_transaction[train_ident == True]
                                                          , train_identity
                                                          , on='TransactionID')
                                                 , train_id_enc
                                                 , test_id_enc
                                                )
id_predictions = test_alg(x_train, y_train, x_valid)

# no id
train_tr_enc, col_list = encode(train_tr)
test_tr_enc, col_list = encode(test_tr)

x_train, y_train, x_valid = feature_eng(train_transaction[train_ident == False]
                                        , train_tr_enc
                                        , test_tr_enc
                                       )
tr_predictions = test_alg(x_train, y_train, x_valid)

id_id = pd.merge(test_transaction[test_ident == True]
                           , test_identity
                           , on='TransactionID')['TransactionID'].tolist()

tr_id = test_transaction[test_ident == False]['TransactionID'].tolist()

output = pd.DataFrame()
output['TransactionID'] = (id_id + tr_id)
output['isFraud'] = list(id_predictions) + list(tr_predictions)
output.to_csv(output_path, index=False)
