import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, accuracy_score, precision_score, auc
from sklearn.metrics import confusion_matrix

device = torch.device('cpu')

'''------------------ MODEL TRAINING & EVALUATION------------------'''
''' The trained_model_PCA function is defined to train the model: within each epoch the training and the validation are carried on
1)The training is performed on mini-batch until the whole train set is explored
2) The prediction of the model on the input data are computed together with loss and accuracy
3) The backpropagation of errors and the update of parameters is done after optimization; the train statistics are updated and printed
4) During the model evaluation, the BatchNomr and Drop modules do not affect the previsions
5) The loss and the accuracy are evaluated, even in this case using a mini-batch; the mean accuracy and the mean loss are then evaluated on the entire set and printed.
'''

'''------------------  ACCURACY ------------------'''
def accuracy(outputs, targets):
    # Converts probabilities to class predictions (0 or 1)
    predicted = outputs > 0.5
    # Compare the predictions with the target labels.
    correct = predicted.eq(targets.view_as(predicted))
    # Calculate the accuracy.
    accuracy = correct.float().mean()
    return accuracy.item()
    

# Function for training the model
def trained_model(model, train_loader, val_loader, num_epochs, batch_size, optimizer, criterion, save_best, scheduler):
    
    # Initialize the optimizer
    trainer = optimizer(model.parameters(), lr=0.01)
    best_val = None
    
    # History dictionary to store training and validation metrics
    history = {'train_loss': [], 'train_accuracy': [], 'val_loss': [], 'val_accuracy': []}
    
    # Iterate over epochs
    for epoch in range(num_epochs):
        # Training Loop
        # Set to train mode
        model.train()
        running_train_loss = 0.0
        running_train_accuracy = 0.0
        for i, (X, y) in enumerate(train_loader):
            # Zero the parameter gradients
            trainer.zero_grad()
            X, y = X.to(device), y.to(device)
            X = X.float()
            y = y.float()
            
            # Forward pass
            output = model(X)
            target = y
            # Compute loss
            train_loss = criterion(output, target)
            # Compute accuracy
            train_accuracy = accuracy(output, target)
            # Backward pass and optimize
            train_loss.backward()
            trainer.step()

            # print train statistics
            running_train_loss += train_loss.item()
            running_train_accuracy += train_accuracy
            if i % 4 == 3:    # print every 4 mini-batches
                print(f"[{epoch+1}, {i+1}] train loss: {running_train_loss / 4 :.3f}, train accuracy: {running_train_accuracy / 4 :.3f}")
                
        # After each epoch, calculate average training loss and accuracy, and append them to history
        avg_train_loss = running_train_loss / len(train_loader)
        avg_train_accuracy = running_train_accuracy / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        history['train_accuracy'].append(avg_train_accuracy)
        
        # Validation Loop
        # Set to eval mode
        model.eval()
        with torch.no_grad():
            running_val_loss = 0.0
            running_val_accuracy = 0.0
            for i, (X, y) in enumerate(val_loader):
                X, y = X.to(device), y.to(device)
                X = X.float()
                y = y.float()

                output = model(X)
                target = y
                # Compute validation loss
                val_loss = criterion(output, target)
                # Compute validation accuracy
                val_accuracy = accuracy(output, target)
                running_val_loss += val_loss.item()
                running_val_accuracy += val_accuracy

            # Calculate average validation loss and accuracy
            curr_val_loss = running_val_loss / len(val_loader)
            curr_val_accuracy = running_val_accuracy / len(val_loader)
            history['val_loss'].append(curr_val_loss)
            history['val_accuracy'].append(curr_val_accuracy)
            if save_best:
               if best_val is None:
                   best_val = curr_val_loss
               best_val = save_best(model, curr_val_loss, best_val)

            # print val statistics per epoch
            print(f"[{epoch+1}] val loss: {curr_val_loss :.3f}, val accuracy: {curr_val_accuracy :.3f}")

    print(f"Finished Training on {epoch+1} Epochs!")
    

    return model, history

def test_eval(model, test_loader, criterion):
    # Initialize test loss and accuracy
    test_loss, test_acc = 0.0, 0.0
    num_batches = len(test_loader)
    
   # Iterate over batches in the test_loader
    with torch.no_grad():
        for inputs, labels in test_loader:
            # Move inputs and labels to device
            inputs, labels = inputs.to(device), labels.to(device)
            # Forward pass to get model outputs
            outputs = model(inputs)
            # Compute test loss
            test_loss += criterion(outputs.float(), labels.float()).item()
            # Compute test accuracy
            test_acc += accuracy(outputs, labels)

    # Calculate average test loss and accuracy
    test_loss /= num_batches
    test_acc /= num_batches

    # Print test loss and accuracy
    print('Test loss:', test_loss)
    print('Test accuracy:', test_acc)



'''------------------  PREDICTION ------------------'''


def predict(model, test_loader):
    # Set device
    model = model.to(device)
    # Set model to evaluation mode
    model.eval()
    predictions = []
    # Iterate over batches in the test_loader
    with torch.no_grad():
        for inputs, _ in test_loader:
            # Move inputs to the appropriate device
            inputs = inputs.to(device)
            # Forward pass to get model outputs
            outputs = model(inputs)
            # Append predictions for each batch
            predictions.append(outputs)
        # Concatenate predictions from all batches
        predictions = torch.cat(predictions)
    # Move predictions to CPU and convert to NumPy array
    return predictions.cpu().numpy()

# Predictions on the test set
def torch_eval(trained_model, test_loader):
    # Get predictions
    y_pred = predict(trained_model, test_loader)
    y_pred_classes = (y_pred > 0.5).astype(int)  # Round probabilities to binary classes

    # Extract true labels from DataLoader
    true_labels = []
    for _, labels in test_loader:
        true_labels.extend(labels.numpy())  # Convert tensors to numpy array and append to the list
    true_labels = np.array(true_labels)  # Convert the list to a NumPy array

    # Calculate metrics
    precision = round(precision_score(true_labels, y_pred_classes), 3)
    f1 = round(f1_score(true_labels, y_pred_classes), 3)
    accuracy = round(accuracy_score(true_labels, y_pred_classes), 3)

    print('Precision:', precision)
    print('F1 Score:', f1)
    print('Accuracy:', accuracy)

    return true_labels, y_pred, y_pred_classes, precision, f1, accuracy


def plot_training_curves_TORCH(history):
    # Number of epochs
    epochs = range(1, len(history['train_loss']) + 1)

    # Plot accuracy curves
    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(epochs, history['train_loss'], 'g', label='Training loss')
    ax2.plot(epochs, history['train_accuracy'], 'r', label='Training accuracy')
    ax2.plot(epochs, history['val_loss'], 'b', label='Validation loss')
    ax2.plot(epochs, history['val_accuracy'], 'k', label='Validation accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
