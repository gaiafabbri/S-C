import xgboost as xgb
import numpy as np
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, f1_score
import matplotlib.pyplot as plt

def BDT_model(X_train, X_test, y_train, y_test):
    # Definition and training of the xgboost model
    '''
    Parameters for the xgboost model:
    'objective': 'binary:logistic' - Specifies the objective function for binary classification,
    where the model outputs probabilities. This is suitable for binary classification tasks.
    'eval_metric': 'logloss' - Evaluation metric used to assess the model's performance during training.
    'nthread': 4 - Number of threads to use for parallel computation. Setting it to 4 enables the model
    to utilize multiple CPU cores for faster training.
    '''
    params = {'objective': 'binary:logistic', 'eval_metric': 'logloss', 'nthread': 4}
    
    '''
    Convert training and testing data to DMatrix format for xgboost:
    xgb.DMatrix is a data interface provided by xgboost to efficiently handle large datasets.
    It converts input arrays (X_train, X_test) along with corresponding labels (y_train, y_test) 
    into a format suitable for xgboost's internal computation.
    '''
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # Dictionary to store evaluation results
    evals_result = {}
    
    # Number of boosting rounds
    num_round = 100
    
    # Train the xgboost model
    model = xgb.train(params, dtrain, num_round, evals=[(dtrain, 'train'), (dtest, 'test')], evals_result=evals_result)
    
    return model, evals_result
