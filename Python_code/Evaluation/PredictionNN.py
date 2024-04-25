import numpy as np
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


'''
This script defines several functions for evaluating and visualizing neural network models:

eval_Neural_Networks: This function evaluates a neural network model using test data. It calculates test accuracy, predicts classes for the test data, calculates precision, F1 score, and accuracy, and then prints these metrics.

print_ROC: This function calculates and prints the ROC curve and the confusion matrix based on the model's predictions and the true labels. It plots the ROC curve, calculates the area under the curve (AUC), and prints the confusion matrix.

plot_training_curves_KERAS: This function plots the training and validation accuracy and loss curves from the training history of a Keras model.
'''

def eval_Neural_Networks(model, X_test, y_test):
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print('Test accuracy:', test_acc)

    # Test set prediction
    y_pred = model.predict(X_test)
    y_pred_classes = np.round(y_pred).astype(int)

    # Metrics calculation
    precision = round(precision_score(y_test, y_pred_classes), 3)  # precision
    f1 = round(f1_score(y_test, y_pred_classes), 3)  # f1 score
    accuracy = round(accuracy_score(y_test, y_pred_classes), 3)  # accuracy

    print('Precision:', precision)
    print('F1 Score:', f1)
    print('Accuracy:', accuracy)
    
    return y_pred, y_pred_classes, precision, f1, accuracy
    
    
def print_ROC(y_test, y_pred, y_pred_classes):
    # Calculating the ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred)

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
    conf_matrix = confusion_matrix(y_test, y_pred_classes)
    print("Matrice di Confusione:")
    print(conf_matrix)
    return background_rejection, tpr


def plot_training_curves_KERAS(history):
    # Plot
    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(history.history['accuracy'], label='Training Accuracy')
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax2.plot(history.history['loss'], label='Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Training History')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Metrics')
    ax2.legend()
    ax2.set_title('Loss Curve')



