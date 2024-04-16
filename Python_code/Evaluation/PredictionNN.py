import numpy as np
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def eval_Neural_Networks(model, X_test, y_test):
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print('Test accuracy:', test_acc)

    # Test set prediction
    y_pred = model.predict(X_test)
    y_pred_classes = np.round(y_pred).astype(int)

    # Metrics calculation
    precision = precision_score(y_test, y_pred_classes)  # precision
    f1 = f1_score(y_test, y_pred_classes)  # f1 score
    accuracy = accuracy_score(y_test, y_pred_classes)  # accuracy

    print('Precision:', precision)
    print('F1 Score:', f1)
    print('Accuracy:', accuracy)
    
    return y_pred, y_pred_classes
    
    
def print_ROC(y_test, y_pred, y_pred_classes):
    # Calcolo della curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_pred)

    # Calcolo del background rejection
    background_rejection = 1 - fpr
    roc_auc = auc(background_rejection, tpr)
    
    # Plot della curva ROC
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(background_rejection, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    ax1.plot([1, 0], [0, 1], color='navy', lw=2, linestyle='--')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('Background Rejection')
    ax1.set_ylabel('Signal Efficiency')
    ax1.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax1.legend(loc="lower right")

    # Calcolo della matrice di confusione
    conf_matrix = confusion_matrix(y_test, y_pred_classes)
    print("Matrice di Confusione:")
    print(conf_matrix)
    return background_rejection, tpr


def plot_training_curves_KERAS(history):
    # Plot della loss curve
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



