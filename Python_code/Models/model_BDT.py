import xgboost as xgb
import numpy as np
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, f1_score
import matplotlib.pyplot as plt

def BDT_model(X_train, X_test, y_train, y_test):
    # Definizione e addestramento del modello xgboost

    params = {'objective': 'binary:logistic', 'eval_metric': 'logloss', 'nthread': 4}
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    evals_result = {}
    num_round = 100
    model = xgb.train(params, dtrain, num_round)
    model = xgb.train(params, dtrain, num_round, evals=[(dtrain, 'train'), (dtest, 'test')], evals_result=evals_result)
    return model, evals_result

