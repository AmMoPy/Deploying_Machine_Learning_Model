#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction import DictVectorizer
import pickle
from tqdm import tqdm
import os

# user input
C = int(input('Pls specify value for C parameter: '))
n_folds = int(input('Pls specify CV n_folds: '))
file_name = input('Pls specify file name to save model: ').strip().lower().replace(" ", "")

print('\n', '-' * 40, '\n', 'Preprocessing data.....', '\n', sep = '')
# preprocessing

df = pd.read_csv(''.join(os.path.dirname(os.path.abspath(__file__)) + '\WA_Fn-UseC_-Telco-Customer-Churn.csv'))

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

df['TotalCharges'] = df['TotalCharges'].fillna(0)

df.columns = df.columns.str.lower().str.replace(' ', '_')

string_columns = list(df.dtypes[df.dtypes == 'object'].index)

for col in string_columns:
    df[col] = df[col].str.lower().str.replace(' ', '_')

df.churn = (df.churn == 'yes').astype(int)

print('Done.', '\n', '-' * 40, '\n', sep = '')

# excluding customer ID and target values
cols = df.columns[1:-1]

# validation framework

# split train and test sets
df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=1)

y_train_f = df_train_full.churn.values
y_test = df_test.churn.values

# helper functions

def train(df, y, C = C):
    feats = df[cols].to_dict(orient='records')
    
    dv = DictVectorizer(sparse=False)
    dv.fit(feats)

    X = dv.transform(feats)

    model = LogisticRegression(solver='liblinear', C = C)
    model.fit(X, y)

    return dv, model

def predict(df, dv, model):
    feats = df[cols].to_dict(orient='records')
    
    X = dv.transform(feats)

    y_pred = model.predict_proba(X)[:, 1]

    return y_pred

# cross validate train set
kfold = KFold(n_splits = n_folds, shuffle = True, random_state = 1)

aucs = []

print(f'{n_folds} fold CV....', '\n',  sep = '')

for train_idx, val_idx in tqdm(kfold.split(df_train_full)):
    df_train = df_train_full.iloc[train_idx]
    y_train = df_train.churn.values

    df_val = df_train_full.iloc[val_idx]
    y_val = df_val.churn.values

    dv, model = train(df_train, y_train)
    y_pred = predict(df_val, dv, model)

    rocauc = roc_auc_score(y_val, y_pred)
    aucs.append(rocauc)

print('\n', f'Cval results: C = {C}, avg_auc = {round(np.mean(aucs), 3)} ± {round(np.std(aucs), 3)}', 
'\n', '-' * 40, '\n', sep = '')

print('Training final model.....', '\n', sep = '')
# final model

dv, model = train(df_train_full, y_train_f, C = C)
y_pred = predict(df_test, dv, model)

auc = roc_auc_score(y_test, y_pred)

print(f'Final model results: auc = {auc.round(3)}', '\n', '-' * 40, '\n', sep = '')

print('Saving model.....', '\n', sep = '')
# saving the model

path = ''.join(os.path.dirname(os.path.abspath(__file__)) + '\\' + file_name + '.bin')

with open(path, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print(f'Model saved to {path} ', '\n', '-' * 40, '\n', sep = '')