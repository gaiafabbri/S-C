import xgboost as xgb
import numpy as np
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

'''
This script defines several functions for evaluating and visualizing XGBoost models:

BDT_eval: This function evaluates an XGBoost model using test data. It calculates accuracy, precision, and F1 score based on the model's predictions and the true labels, and then prints these metrics.

print_ROC_BDT: This function calculates and prints the ROC curve and the confusion matrix based on the model's predictions and the true labels. It plots the ROC curve, calculates the area under the curve (AUC), and prints the confusion matrix.

plot_training_curves_BDT: This function plots the training and test log loss over the iterations of XGBoost training.

'''
    
def BDT_eval(model, X_test, y_test):
    # Evaluation of the model
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # Test set prediction
    y_pred = model.predict(dtest)
    y_pred_binary = np.round(y_pred)
    
    # Metrics calculation
    accuracy = round(accuracy_score(y_test, y_pred_binary), 3) #accuracy
    precision = round(precision_score(y_test, y_pred_binary), 3) #precision
    f1 = round(f1_score(y_test, y_pred_binary), 3) #f1 score

    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('F1 Score:', f1)
    
    return y_pred, y_pred_binary, precision, f1, accuracy
    
def print_ROC_BDT(y_test, y_pred, y_pred_binary):
    # Calculating the ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    
    # Calculating background rejection
    background_rejection = 1 - fpr
    roc_auc = auc(background_rejection, tpr)
    
    # Plot
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(background_rejection, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    ax1.plot([1, 0], [0, 1], color='navy', lw=2, linestyle='--')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('Background Rejection')
    ax1.set_ylabel('Signal Efficiency')
    ax1.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax1.legend(loc="lower right")

    # Calculation of the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred_binary)
    print("Matrice di Confusione:")
    print(conf_matrix)
    return background_rejection, tpr
    
def plot_training_curves_BDT(evals_result):
    # Plot
    ax2 = plt.subplot(1, 2, 2)
    eval_metric = 'logloss'
    train_metric = evals_result['train'][eval_metric]
    test_metric = evals_result['test'][eval_metric]
    epochs = len(train_metric)
    x_axis = range(0, epochs)
    ax2.plot(x_axis, train_metric, label='Train')
    ax2.plot(x_axis, test_metric, label='Test')
    ax2.legend()
    ax2.set_ylabel(eval_metric)
    ax2.set_xlabel('Numero di iterazioni')
    ax2.set_title('History delle metriche di valutazione')



