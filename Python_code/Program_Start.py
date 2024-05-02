'''------------------ IMPORT ------------------'''
import os
import time
import torch

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
from torch.optim import Adam
from tensorflow.keras.callbacks import EarlyStopping

from DataPreparation.Data_Preparation import load_and_normalize_images, create_features_labels
from DataPreparation.Preparing_Dataset import Control_PCA


from Models.model_keras_PCA import keras_PCA
from Models.model_DNN_PCA import DNN_model_PCA
from Models.model_Torch_PCA import torch_PCA
from Models.model_BDT import BDT_model


from Evaluation.PredictionNN import eval_Neural_Networks, print_ROC, plot_training_curves_KERAS
from Evaluation.TrainTorch import trained_model, test_eval, torch_eval, plot_training_curves_TORCH
from Evaluation.PredictionBDT import BDT_eval, print_ROC_BDT, plot_training_curves_BDT
from Evaluation.Table import save_results_table




'''------------------ END IMPORT ------------------'''


#Setting the environment to use the CPU only
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Check whether the 'plot_results' folder exists
if not os.path.exists("plot_results"):
    os.makedirs("plot_results") ## If it does not exist, create it

'''------------------ DATA LOADING & NORMALIZATION ------------------'''
    
folder_path = "images"
file_name = "images_data_16x16_100000.root"

# Extracting dimensions
dimensions_str = file_name.split("_")[2]
dimensions_without_extension = dimensions_str.split(".")[0]
height, width = map(int, dimensions_without_extension.split("x"))
event_number = int(file_name.split("_")[-1].split(".")[0])

# Loading data and normalization
signal_images, background_images = load_and_normalize_images(folder_path, file_name)

print("Lunghezza immagini di segnale:", len(signal_images))
print("Lunghezza immagini di segnale:", len(signal_images))

'''------------------ DATA PREPARATION ------------------'''

signal_immages_numpy=np.array(signal_images)
background_images_numpy=np.array(background_images)
print("Signal images and background images dimension: ", signal_immages_numpy.shape, background_images_numpy.shape)

# Defining the features and the labels
X,y=create_features_labels(signal_immages_numpy, background_images_numpy)


'''------------------ MODEL CHOICE ------------------'''

available_options = {
    "1": "CNN with torch",
    "2": "CNN with tensorflow-keras",
    "3": "BDT",
    "4": "DNN",
    "5": "Plot comparative ROC and exit"
}
# Define an empty results dictionary outside the loop
results_dict = {}
results_df = pd.DataFrame(columns=['Modello', 'Precision', 'Accuracy', 'F1 Score', 'Training Time (s)'])


    
while True:
# Ask the user to choose an option
    print("Choose an option:")
    for key, value in available_options.items():
        print(key, "=", value)
    choice = input("Enter the number corresponding to the desired option:")
    
    
    if choice not in available_options:
        print("Option already explored or invalid. Please enter a number from the remainder.")
        continue
    print("You choose", available_options[choice])
    
    

    if choice == "1":
    
        '''------------------ CNN TORCH ------------------'''
        
        X_shuffled,y_shuffled, n_principal_components = Control_PCA (X, y, width, height, choice)
        
        X_tensor = torch.tensor(X_shuffled, dtype=torch.float32)
        y_tensor = torch.tensor(y_shuffled)
        y_tensor = y_tensor.view(-1, 1)
        # Creation of TensorDataset for data and labels
        dataset = TensorDataset(X_tensor, y_tensor)
        # Creation of DataLoader for training set and test set
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        print("Dimensioni del training set:")
        for inputs, labels in train_loader:
            print("X_train:", inputs.shape)
            print("y_train:", labels.shape)
            break

        print("\nDimensioni del test set:")
        for inputs, labels in test_loader:
            print("X_test:", inputs.shape)
            print("y_test:", labels.shape)
            break
            
        #Definition of the model
        model = torch_PCA(width, height, n_principal_components)
        
        
        # Define hyperparameters
        num_epochs = 10
        batch_size = 64
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam
        start_time=time.time() #monitoring the starting time
        
        # Model training
        model, history = trained_model(model, train_loader, test_loader, num_epochs, batch_size, optimizer, criterion, save_best=None, scheduler=None)
        criterion = nn.BCELoss().to(torch.float32)  # Specifies the data type of the loss function

        end_time=time.time() #monitoring the finish time
        training_time = "{:.3f}".format(end_time - start_time)  # Calculate total training time
        print(f"Tempo di addestramento: {training_time} secondi")
        
        test_eval(model, test_loader, criterion)
        # Creating the figure
        fig = plt.figure(figsize=(15, 5))
        true_labels, y_pred, y_pred_classes, precision, f1, accuracy=torch_eval(model, test_loader)
        # Call to the function to plot the ROC curve
        background_rejection_CNN_torch, tpr_CNN_torch = print_ROC(true_labels, y_pred, y_pred_classes)
        # Call to the function to plot the training curves
        plot_training_curves_TORCH(history)
        
        
    elif choice == "2":
    
        '''------------------ CNN KERAS ------------------'''
        
        X_shuffled,y_shuffled, n_principal_components = Control_PCA (X, y, width, height, choice)
        
        # Setting the number of threads used by TensorFlow for intra-op and inter-op operations
        tf.config.threading.set_intra_op_parallelism_threads(4)
        tf.config.threading.set_inter_op_parallelism_threads(4)
        
        X_train, X_test, y_train, y_test = train_test_split(X_shuffled, y_shuffled, test_size=0.2, random_state=42)
        
        print("Training set dimension:")
        print("X_train:", X_train.shape)
        print("y_train:", y_train.shape)
        
        print("\nTesting set dimensions:")
        print("X_test:", X_test.shape)
        print("y_test:", y_test.shape)
        
        
        #Definition of the model
        model = keras_PCA(n_principal_components)

        
        # Define EarlyStopping callback
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, min_delta=0.001, restore_best_weights=True)
        
        start_time = time.time() #setting the time when the training starts
        history = model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test), callbacks=[early_stopping])
        end_time = time.time() #monitoring the time when the trainign ends

        training_time = "{:.3f}".format(end_time - start_time)
        print("Training time with the CPU:", training_time, "s")
        # Creating the figure
        fig = plt.figure(figsize=(15, 5))
        y_pred, y_pred_classes, precision, f1, accuracy  =eval_Neural_Networks(model, X_test, y_test)
        # Call to the function to plot the ROC curve
        background_rejection_CNN_keras, tpr_CNN_keras = print_ROC(y_test, y_pred, y_pred_classes)
        # Call to the function to plot the training curves
        plot_training_curves_KERAS(history)

        
        
    elif choice == "3":
    
        '''------------------ BDT ------------------'''
        
        X_BDT,y_BDT, n_principal_components = Control_PCA (X, y, width, height, choice)
        
        X_train, X_test, y_train, y_test = train_test_split(X_BDT, y_BDT, test_size=0.2, random_state=42)

        print("Training set dimensions:")
        print("X_train:", X_train.shape)
        print("y_train:", y_train.shape)
        
        print("\nTesting set dimensions:")
        print("X_test:", X_test.shape)
        print("y_test:", y_test.shape)

        
        # Definition and training of the xgboost model
        start_time = time.time() #setting the time when the training starts
        model, evals_result = BDT_model(X_train, X_test, y_train, y_test)


        end_time = time.time() #monitoring the time when the trainign ends
        training_time = "{:.3f}".format(end_time - start_time)
        print("Training time with the CPU:", training_time, "s")

        y_pred, y_pred_binary, precision, f1, accuracy = BDT_eval(model, X_test, y_test)
        
        # Creating the figure
        fig = plt.subplots(figsize=(15, 5))

        # Call to the function to plot the ROC curve
        background_rejection_BDT, tpr_BDT= print_ROC_BDT(y_test, y_pred, y_pred_binary)
        
        # Call to the function to plot the training curves
        plot_training_curves_BDT(evals_result)

        pass
        
    elif choice == "4":
        
        '''------------------ DNN ------------------'''
        
        X_shuffled,y_shuffled, n_principal_components = Control_PCA (X, y, width, height, choice)
        
        # Setting the number of threads used by TensorFlow for intra-op and inter-op operations
        tf.config.threading.set_intra_op_parallelism_threads(4)
        tf.config.threading.set_inter_op_parallelism_threads(4)
        
        # Splitting the dataset in 80% training and 20% testing
        X_train, X_test, y_train, y_test = train_test_split(X_shuffled, y_shuffled, test_size=0.2, random_state=42)
        print("Training set dimensions:")
        print("X_train:", X_train.shape)
        print("y_train:", y_train.shape)
        
        print("\nTesting set dimensions:")
        print("X_test:", X_test.shape)
        print("y_test:", y_test.shape)
        
        #Definition of the model
        model = DNN_model_PCA(n_principal_components)
    
    
        # Define EarlyStopping callback
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, min_delta=0.001, restore_best_weights=True)

        # Model Training
        start_time = time.time()
        history = model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test), callbacks=[early_stopping])
        end_time = time.time()
        training_time = "{:.3f}".format(end_time - start_time)
        print("Training time with the CPU:", training_time, "s")
        y_pred, y_pred_classes, precision, f1, accuracy =eval_Neural_Networks(model, X_test, y_test)
        
        # Creating the figure
        fig = plt.figure(figsize=(15, 5))

        # Call to the function to plot the ROC curve
        background_rejection_DNN, tpr_DNN = print_ROC(y_test, y_pred, y_pred_classes)
        

        # Call to the function to plot the training curves
        plot_training_curves_KERAS(history)
        
    elif choice == "5":
        
        # Creating a dictionary to store ROC curve results for each model
        roc_results = {}

        # Add ROC curve results for the CNN Torch model, if available
        if 'background_rejection_CNN_torch' in locals() and 'tpr_CNN_torch' in locals():
            roc_results['CNN Torch'] = (background_rejection_CNN_torch, tpr_CNN_torch)

        # Add ROC curve results for the CNN Keras model, if available
        if 'background_rejection_CNN_keras' in locals() and 'tpr_CNN_keras' in locals():
            roc_results['CNN Keras'] = (background_rejection_CNN_keras, tpr_CNN_keras)

        # Add ROC curve results for the BDT model, if available
        if 'background_rejection_BDT' in locals() and 'tpr_BDT' in locals():
            roc_results['BDT'] = (background_rejection_BDT, tpr_BDT)

        # Add ROC curve results for the DNN model, if available
        if 'background_rejection_DNN' in locals() and 'tpr_DNN' in locals():
            roc_results['DNN'] = (background_rejection_DNN, tpr_DNN)

        # Creating the figure
        fig = plt.figure(figsize=(7,5))

        # Plot of ROC curves for available models
        for model_name, (background_rejection, tpr) in roc_results.items():
            plt.plot(background_rejection, tpr, label=model_name)

        # Aesthetics
        plt.xlabel('Background Rejection')
        plt.ylabel('Signal Efficiency')
        plt.title('Receiver Operating Characteristic (ROC) Curve for all Models')
        plt.plot([1, 0], [0, 1], color='navy', lw=2, linestyle='--')
        plt.legend()
        
        # Saving
        plt.savefig(f"plot_results/Comparison_among_models_{event_number}_{width}x{height}.png")
        
        # Show figure
        plt.tight_layout()
        plt.show()
        
        
        break
    plt.savefig(f"plot_results/{available_options[choice]}_{event_number}_{width}x{height}.png")
    plt.tight_layout()
    plt.show()
    # Store metrics in a dictionary
    results_dict = {
        available_options[choice]: (accuracy, precision, f1, training_time)
    }

    # Passes the dictionary to the save_results_table function
    results_df = save_results_table(results_dict, results_df)
    
    del available_options[choice]
