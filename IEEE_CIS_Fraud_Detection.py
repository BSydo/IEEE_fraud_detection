import sys
import warnings
import copy
import pandas as pd
import numpy as np
from sklearn import model_selection, preprocessing, ensemble, metrics, linear_model
# , naive_bayes, metrics, svm, decomposition
import xgboost
from sklearn.preprocessing import MinMaxScaler
from imblearn.under_sampling import RandomUnderSampler

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

#train_ident = train_transaction['TransactionID'].isin(train_identity['TransactionID'])
#test_ident = test_transaction['TransactionID'].isin(test_identity['TransactionID'])

#train_tr = train_transaction[train_ident == False]
tr_dataset = pd.merge(train_transaction, train_identity,
                      how='left', on='TransactionID')

#test_tr = test_transaction[test_ident == False]
fin_test_id = pd.merge(test_transaction, test_identity,
                       how='left', on='TransactionID')

# label encode the target variable


def encode(dataset):

    encoder = preprocessing.LabelEncoder()

    non_num_cols = [dataset.columns[n] for n, i in enumerate(dataset.dtypes)
                    if i not in ('int64', 'float64')]

    # Filling missing string values with most common value
    for i in non_num_cols:
        dataset[i+'_mod'] = encoder\
            .fit_transform(dataset[i].fillna(dataset[i]
                                             .replace(np.nan, '0', regex=True).mode()[0]))

    col_list = [
        x for x in train_transaction.columns if x not in non_num_cols+['isFraud']]

    #y = dataset['isFraud']

    dataset = dataset[col_list].fillna(0)

    return dataset

# feature engineering


def feature_eng(dataset, y):

    x_train = dataset.fillna(0)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)

    rus = RandomUnderSampler(random_state=42)
    x_train, y = rus.fit_resample(x_train, y)

    return x_train, y

def feature_eng_t(dataset):

    x_train = dataset.fillna(0)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)

    return x_train

# algorithm testing
def clf(x_train, y_train, x_test):

    RF = ensemble.RandomForestClassifier(n_estimators=32#31
                                        , n_jobs=-1
                                        , max_depth=8#31
                                        , min_samples_split = 0.1#0.3
                                        ) #76,71

    #LR = linear_model.LogisticRegression(n_jobs=-1)#70.48

    RF.fit(x_train, y_train)
    predictions = RF.predict(x_test)

    #LR.fit(x_train, y_train)
    #predictions = LR.predict(x_test)

    #accuracy = metrics.accuracy_score(predictions, y_test)
    # print(f"RF:{accuracy}")
    return predictions

x_train = encode(tr_dataset)
x_train, y = feature_eng(x_train, tr_dataset['isFraud'])

x_test = encode(fin_test_id)
x_test = feature_eng_t(x_test)

predictions = clf(x_train, y, x_test)

output = pd.DataFrame()
output['TransactionID'] = fin_test_id['TransactionID']
output['isFraud'] = list(predictions)
output.to_csv(output_path, index=False)
