import pandas as pd
import numpy as np

from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.feature_extraction import DictVectorizer

import pickle

df = pd.read_csv('data/dune_good_vs_bad_trader.csv')
# Shuffle the entire DataFrame
df = shuffle(df, random_state=42)

# Remove emojis and the space after them
df['target_variable'] = df['target_variable'].str.replace(r'^[🔴🟢]\s*', '', regex=True)
df['trader_activity_status'] = df['trader_activity_status'].str.replace(r'^[🐣🐤🐦]\s*', '', regex=True)
df['trader_volume_status'] = df['trader_volume_status'].str.replace(r'^[🦐🐳🐟]\s*', '', regex=True)
df['trader_weekly_frequency_status'] = df['trader_weekly_frequency_status'].str.replace(r'^[🐣🐤🐦]\s*', '', regex=True)

df['target_variable'] = df['target_variable'].map({'Good Trader': 1, 'Bad Trader': 0})

numerical = ['active_weeks', 'total_volume', 'tx_count_365d']

categorical = ['trader_activity_status', 'trader_weekly_frequency_status']

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

# training
def train(df_train, y_train):
    dicts = df_train[categorical + numerical].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    return dv, model

def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred

# Validation
print('doing validation')
n_splits = 5

kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

scores = []

for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train.target_variable.values
    y_val = df_val.target_variable.values

    dv, model = train(df_train, y_train)
    y_pred = predict(df_val, dv, model)

    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)

print('validation results:')
print('%.3f +- %.3f' % (np.mean(scores), np.std(scores)))

# training final model
print('training the final model')
dv, model = train(df_full_train, df_full_train.target_variable.values)
y_pred = predict(df_test, dv, model)

y_test = df_test.target_variable.values
auc = roc_auc_score(y_test, y_pred)
print(f'auc={auc}')

# save the model
output_file = 'good_bad_trader_log_reg.bin'
with open(output_file, 'wb') as f_out: 
    pickle.dump((dv, model), f_out)



